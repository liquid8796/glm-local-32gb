"""CPU executor batching with bounded fake I/O and independent tile arithmetic."""
from dataclasses import replace
import math
import struct
from types import MappingProxyType
import unittest
from unittest.mock import patch
import weakref

from glm_local.execution import FULL_MODEL_FLAGS, TensorDescriptor, _vector
from glm_local import nvfp4_execution as execution
from glm_local.nvfp4_execution import (MAX_ROW_BAND_COLUMNS, NVFP4ProjectionDescriptor,
                                      NVFP4_ROW_BAND_SCRATCH_BYTES, execute_nvfp4_projection)
from glm_local.residency import BudgetExceededError, ReservationLedger
from glm_local.safetensor_reader import MAX_READ_BYTES, SafeTensorError, SafeTensorReader, _uint
from test_nvfp4_kernels import f32, oracle


def descriptor(rows=129, cols=144):
    specifications = (("weight", "U8", (rows, cols // 2)),
                      ("scale", "F8_E4M3", (rows, cols // 16)),
                      ("global", "F32", ()), ("input", "F32", ()))
    tensors = []
    for name, dtype, shape in specifications:
        size = math.prod(shape) * (4 if dtype == "F32" else 1)
        tensors.append(TensorDescriptor(name, dtype, shape, size, (0, size), "fixture.safetensors"))
    return NVFP4ProjectionDescriptor(tensors[0], tensors[1], (), "fixture/nvfp4", "a" * 40,
        "fixture", "0" * 64, "fixture", "1" * 64, "fixture", "2" * 64,
        False, "SYNTHETIC", tensors[2], tensors[3])


class CountingReader:
    def __init__(self, desc, *, zeros=False):
        self._tensors = MappingProxyType({item.name: item.info() for item in desc.tensors})
        self.zeros = zeros
        self.reads = []
        self.tile_reads = 0
        self.global_scale = 0.0137
        self.input_scale = 3.25
        self.bad_scale = None
        self.short_name = None
        self.fail_name = None
        self.on_read = None

    @property
    def tensors(self):
        return self._tensors

    _tensor = SafeTensorReader._tensor

    def _assert_unchanged(self):
        pass

    def read_bytes(self, name, offset, count):
        info = self._tensor(name)
        _uint(offset, info.nbytes, "offset"); _uint(count, MAX_READ_BYTES, "count")
        if offset % info.itemsize or count % info.itemsize or offset + count > info.nbytes:
            raise SafeTensorError("Invalid bounded fixture read")
        self.reads.append((name, offset, count))
        if self.on_read is not None:
            self.on_read(name, offset, count)
        if self.fail_name == name:
            raise SafeTensorError("Fixture read failed")
        if name in ("global", "input"):
            raw = struct.pack("<f", self.global_scale if name == "global" else self.input_scale)[offset:offset + count]
        elif name == "weight":
            raw = bytes(count) if self.zeros else bytes((index * 37 + 19) % 256 for index in range(offset, offset + count))
        else:
            raw = bytes([56]) * count if self.zeros else bytes((index * 11 + 7) % 127 for index in range(offset, offset + count))
            if self.bad_scale is not None and offset == 0:
                raw = bytes([self.bad_scale]) + raw[1:]
        return raw[:-1] if self.short_name == name else raw

    def read_matrix_tile(self, name, row, col, rows, cols):
        self.tile_reads += 1
        return SafeTensorReader.read_matrix_tile(self, name, row, col, rows, cols)


class TileKernel:
    max_operations = 4096

    def __init__(self, *, zeros=False):
        self.operations = 0
        self.tile_calls = []
        self.zeros = zeros

    def matvec_nvfp4_tile(self, packed, rows, cols, scales, vector, global_scale):
        self.operations += 1
        self.tile_calls.append((rows, cols))
        return [0.0] * rows if self.zeros else oracle(packed, rows, cols, scales, vector, global_scale)


class BandKernel(TileKernel):
    supports_nvfp4_row_band = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.band_calls = []
        self.bad_output = None
        self.output_references = []

    def matvec_nvfp4_row_band(self, packed, rows, cols, scales, vector, global_scale):
        self.band_calls.append((rows, cols, len(packed), len(scales), id(vector), global_scale))
        if self.bad_output is not None:
            return self.bad_output(rows)
        if self.zeros:
            return [0.0] * rows
        values = vector.values if isinstance(vector, PreparedVector) else vector
        result = [0.0] * rows
        for col in range(0, cols, 128):
            count = min(128, cols - col)
            weights = b"".join(packed[row * (cols // 2) + col // 2:row * (cols // 2) + (col + count) // 2]
                               for row in range(rows))
            blocks = b"".join(scales[row * (cols // 16) + col // 16:row * (cols // 16) + (col + count) // 16]
                              for row in range(rows))
            partial = oracle(weights, rows, count, blocks, values[col:col + count], global_scale)
            result = [f32(left + right) for left, right in zip(result, partial)]
        output = TrackedOutput(result)
        self.output_references.append(weakref.ref(output))
        return output


class PreparedVector:
    def __init__(self, vector):
        self.values = _vector(vector, len(vector))

    def __len__(self):
        return len(self.values)


class PreparedBandKernel(BandKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prepare_calls = 0

    def prepare_nvfp4_vector(self, vector):
        self.prepare_calls += 1
        return PreparedVector(vector)


class TrackedOutput:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class Gate:
    def __init__(self):
        self.calls = 0

    def before_submit(self):
        self.calls += 1


def host_bytes(rows, cols):
    return 3 * (128 * 128 // 2 + 128 * 128 // 16) + 4 * (rows + cols + 4 * 128) + 8


class NVFP4ExecutorRowBandTests(unittest.TestCase):
    def vector(self, cols):
        return [((index * 7) % 23 - 11) * 0.137 for index in range(cols)]

    def test_ragged_rows_and_partial_144_columns_match_old_tile_accumulation_bit_for_bit(self):
        for rows, cols in ((1, 16), (3, 144), (129, 144), (145, 272)):
            with self.subTest(rows=rows, cols=cols):
                desc = descriptor(rows, cols)
                old, new = CountingReader(desc), CountingReader(desc)
                kernel = BandKernel()
                expected, old_counts = execute_nvfp4_projection(old, desc, self.vector(cols), cpu=TileKernel())
                actual, counts = execute_nvfp4_projection(new, desc, self.vector(cols), cpu=kernel)
                self.assertEqual(actual, expected)
                self.assertEqual(counts["cpu_tiles"], old_counts["cpu_tiles"])
                self.assertEqual(counts["cpu_row_bands"], (rows + 127) // 128)
                self.assertEqual(counts["native_row_band_calls"], len(kernel.band_calls))
                self.assertEqual(new.tile_reads, 0)
                self.assertEqual(kernel.tile_calls, [])
                self.assertEqual(counts["packed_weight_bytes"], rows * cols // 2)
                self.assertEqual(counts["block_scale_bytes"], rows * cols // 16)
                self.assertEqual(counts["row_band_read_bytes"], rows * (cols // 2 + cols // 16))
                self.assertEqual(counts["row_band_scalar_reads"], 2 * len(kernel.band_calls))
                self.assertFalse(counts["native_w4a4_parity_verified"])
                self.assertEqual(counts["activation_quantization"], "none")
                self.assertTrue(counts["input_scale_validated"])
                for flag in FULL_MODEL_FLAGS:
                    self.assertFalse(counts[flag])

    def test_prepared_vector_is_validated_once_and_reused_across_row_bands(self):
        desc = descriptor(257, 144)
        kernel = PreparedBandKernel()
        output, _ = execute_nvfp4_projection(CountingReader(desc), desc, self.vector(144), cpu=kernel)
        expected, _ = execute_nvfp4_projection(CountingReader(desc), desc, self.vector(144), cpu=TileKernel())
        self.assertEqual(output, expected)
        self.assertEqual(kernel.prepare_calls, 1)
        self.assertEqual(len(kernel.band_calls), 3)
        self.assertEqual(len({call[4] for call in kernel.band_calls}), 1)

    def test_band_outputs_are_not_retained_during_the_next_band_read(self):
        desc = descriptor(129, 144)
        source, kernel = CountingReader(desc), BandKernel()
        scalar_reads = 0
        def inspect(name, offset, count):
            nonlocal scalar_reads
            if name == "global":
                scalar_reads += 1
                if scalar_reads > 1:
                    self.assertIsNone(kernel.output_references[-1]())
        source.on_read = inspect
        execute_nvfp4_projection(source, desc, self.vector(144), cpu=kernel)
        self.assertEqual(scalar_reads, 2)
        self.assertTrue(all(reference() is None for reference in kernel.output_references))

    def test_max_width_uses_bounded_reads_and_the_five_mib_scratch_lease(self):
        desc = descriptor(129, MAX_ROW_BAND_COLUMNS)
        source, kernel = CountingReader(desc, zeros=True), BandKernel(zeros=True)
        required = host_bytes(*desc.logical_shape) + NVFP4_ROW_BAND_SCRATCH_BYTES
        ledger = ReservationLedger(required)
        observed = []
        source.on_read = lambda *_: observed.append(ledger.snapshot()["cpu"]["used_bytes"])
        output, counts = execute_nvfp4_projection(source, desc, [0.1] * MAX_ROW_BAND_COLUMNS, cpu=kernel, ledger=ledger)
        self.assertEqual(output, [0.0] * 129)
        self.assertEqual(counts["cpu_tiles"], 256)
        self.assertEqual(counts["native_row_band_calls"], 2)
        self.assertEqual(counts["row_band_read_calls"], 20)
        self.assertEqual(counts["row_band_scalar_reads"], 4)
        self.assertEqual(len(source.reads), 24)
        self.assertEqual(counts["max_encoded_band_bytes"], 1_179_648)
        self.assertEqual(counts["max_packed_band_bytes"], 1_048_576)
        self.assertEqual(counts["max_row_band_read_bytes"], MAX_READ_BYTES)
        self.assertTrue(all(1 <= call[2] <= MAX_READ_BYTES for call in source.reads))
        self.assertEqual(set(observed), {required})
        self.assertEqual(counts["logical_host_buffer_bytes"], required)
        self.assertEqual(counts["row_band_scratch_bytes"], 5 * 1024**2)
        self.assertEqual(counts["retained_encoded_weight_bytes"], 0)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)
        self.assertEqual(ledger.snapshot()["cpu"]["peak_bytes"], required)

    def test_missing_or_disabled_band_api_and_oversized_width_fall_back(self):
        for mode in ("missing", "disabled", "wide"):
            with self.subTest(mode=mode):
                cols = MAX_ROW_BAND_COLUMNS + 16 if mode == "wide" else 144
                desc = descriptor(1, cols)
                source = CountingReader(desc, zeros=True)
                kernel = TileKernel(zeros=True) if mode == "missing" else BandKernel(zeros=True)
                if mode == "disabled":
                    kernel.supports_nvfp4_row_band = False
                ledger = ReservationLedger(host_bytes(*desc.logical_shape))
                output, counts = execute_nvfp4_projection(source, desc, [1.0] * cols, cpu=kernel, ledger=ledger)
                self.assertEqual(output, [0.0])
                self.assertFalse(counts["native_row_band_batching"])
                self.assertEqual(counts["native_row_band_calls"], 0)
                self.assertEqual(counts["row_band_scratch_bytes"], 0)
                self.assertEqual(len(kernel.tile_calls), (cols + 127) // 128)
                self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_hybrid_keeps_cpu_gpu_tiles_and_gate_instead_of_cpu_batching(self):
        desc = descriptor(129, 144)
        source, cpu, gpu, gate = CountingReader(desc), PreparedBandKernel(), BandKernel(), Gate()
        output, counts = execute_nvfp4_projection(source, desc, self.vector(144), backend="hybrid", cpu=cpu, gpu=gpu, gate=gate)
        expected, _ = execute_nvfp4_projection(CountingReader(desc), desc, self.vector(144), cpu=TileKernel())
        self.assertEqual(output, expected)
        self.assertEqual((counts["cpu_tiles"], counts["gpu_tiles"], gate.calls), (2, 2, 2))
        self.assertEqual(cpu.tile_calls, [(128, 128), (128, 16)])
        self.assertEqual(gpu.tile_calls, [(1, 128), (1, 16)])
        self.assertEqual(cpu.band_calls + gpu.band_calls, [])
        self.assertEqual(cpu.prepare_calls, 0)
        self.assertFalse(counts["native_row_band_batching"])

    def test_invalid_global_and_input_scales_fail_before_payload_or_kernel(self):
        for name in ("global_scale", "input_scale"):
            for value in (0.0, -1.0, math.nan, math.inf):
                with self.subTest(name=name, value=value):
                    desc = descriptor(1, 144)
                    source, kernel = CountingReader(desc), BandKernel()
                    setattr(source, name, value)
                    ledger = ReservationLedger(8 * 1024**2)
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        execute_nvfp4_projection(source, desc, self.vector(144), cpu=kernel, ledger=ledger)
                    self.assertEqual(kernel.band_calls, [])
                    self.assertTrue(all(read[0] in ("global", "input") for read in source.reads))
                    self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_invalid_scale_bytes_are_rejected_before_injected_band_kernel(self):
        for code in (127, 128, 255):
            with self.subTest(code=code):
                desc = descriptor(3, 144)
                source, kernel = CountingReader(desc), BandKernel(zeros=True)
                source.bad_scale = code
                ledger = ReservationLedger(8 * 1024**2)
                with self.assertRaisesRegex(ValueError, "scale"):
                    execute_nvfp4_projection(source, desc, self.vector(144), cpu=kernel, ledger=ledger)
                self.assertEqual(kernel.band_calls, [])
                self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_input_calibration_is_checked_but_does_not_change_weight_only_output(self):
        desc = descriptor(3, 144)
        source = CountingReader(desc)
        baseline, _ = execute_nvfp4_projection(source, desc, self.vector(144), cpu=BandKernel())
        source.input_scale = 123.0
        actual, _ = execute_nvfp4_projection(source, desc, self.vector(144), cpu=BandKernel())
        self.assertEqual(actual, baseline)

    def test_short_scalar_and_encoded_reads_and_read_failure_release_leases(self):
        for name in ("global", "input", "weight", "scale"):
            for failure in ("short", "exception"):
                with self.subTest(name=name, failure=failure):
                    desc = descriptor(3, 144)
                    source, kernel = CountingReader(desc), BandKernel()
                    setattr(source, "short_name" if failure == "short" else "fail_name", name)
                    ledger = ReservationLedger(8 * 1024**2)
                    with self.assertRaises(ValueError):
                        execute_nvfp4_projection(source, desc, self.vector(144), cpu=kernel, ledger=ledger)
                    self.assertEqual(kernel.band_calls, [])
                    self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_nonfinite_invalid_vector_fails_before_read_with_and_without_preparer(self):
        for backend in (BandKernel, PreparedBandKernel):
            for value in (math.nan, math.inf, 1e40, True, "1"):
                with self.subTest(backend=backend.__name__, value=value):
                    desc = descriptor(3, 144)
                    source, kernel = CountingReader(desc), backend()
                    ledger = ReservationLedger(8 * 1024**2)
                    with self.assertRaises(ValueError):
                        execute_nvfp4_projection(source, desc, [value] * 144, cpu=kernel, ledger=ledger)
                    self.assertEqual(source.reads, [])
                    self.assertEqual(ledger.snapshot()["active_leases"], 0)
        source = CountingReader(descriptor(3, 144))
        kernel = PreparedBandKernel()
        kernel.prepare_nvfp4_vector = lambda _: []
        with self.assertRaisesRegex(ValueError, "Prepared"):
            execute_nvfp4_projection(source, descriptor(3, 144), self.vector(144), cpu=kernel)
        self.assertEqual(source.reads, [])

    def test_invalid_kernel_output_or_arithmetic_failure_releases_both_leases(self):
        outputs = [lambda rows: [0] * (rows - 1), lambda rows: [math.nan] * rows,
                   lambda rows: [math.inf] * rows, lambda rows: [1e40] * rows, lambda rows: [True] * rows]
        def fail(_):
            raise ValueError("kernel arithmetic overflow")
        outputs.append(fail)
        for output in outputs:
            with self.subTest(output=output):
                desc = descriptor(3, 144)
                kernel = BandKernel(); kernel.bad_output = output
                ledger = ReservationLedger(8 * 1024**2)
                with self.assertRaises(ValueError):
                    execute_nvfp4_projection(CountingReader(desc), desc, self.vector(144), cpu=kernel, ledger=ledger)
                self.assertEqual(len(kernel.band_calls), 1)
                self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_scratch_budget_is_reserved_before_input_preparation_or_payload(self):
        desc = descriptor(3, 144)
        source, kernel = CountingReader(desc), PreparedBandKernel()
        required = host_bytes(*desc.logical_shape) + NVFP4_ROW_BAND_SCRATCH_BYTES
        ledger = ReservationLedger(required - 1)
        with self.assertRaises(BudgetExceededError):
            execute_nvfp4_projection(source, desc, self.vector(144), cpu=kernel, ledger=ledger)
        self.assertEqual(source.reads, [])
        self.assertEqual(kernel.prepare_calls, 0)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_descriptor_shape_and_vector_bounds_are_checked_before_reads(self):
        desc = descriptor(3, 144)
        source, kernel = CountingReader(desc), BandKernel()
        wrong = replace(desc, weight=replace(desc.weight, shape=(3, 80)))
        with self.assertRaisesRegex(ValueError, "metadata"):
            execute_nvfp4_projection(source, wrong, self.vector(144), cpu=kernel)
        for bad in ([], "1" * 144, b"1" * 144, iter([1] * 144)):
            with self.subTest(vector=type(bad).__name__), self.assertRaises(ValueError):
                execute_nvfp4_projection(source, desc, bad, cpu=kernel)
        self.assertEqual(source.reads, [])
        malformed = descriptor(3, 136)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            execute_nvfp4_projection(CountingReader(malformed), malformed, [1] * 136, cpu=kernel)

    def test_allocation_failure_releases_host_and_row_band_scratch(self):
        desc = descriptor(3, 144)
        ledger = ReservationLedger(8 * 1024**2)
        kernel = BandKernel()
        class FailedAllocation(bytearray):
            def __new__(cls, *_args, **_kwargs):
                raise MemoryError("fixture")
        with patch("glm_local.nvfp4_execution.bytearray", FailedAllocation, create=True):
            with self.assertRaises(MemoryError):
                execute_nvfp4_projection(CountingReader(desc), desc, self.vector(144), cpu=kernel, ledger=ledger)
        self.assertEqual(kernel.band_calls, [])
        self.assertEqual(ledger.snapshot()["active_leases"], 0)


if __name__ == "__main__":
    unittest.main()
