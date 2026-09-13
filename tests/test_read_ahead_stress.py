"""Deterministic ownership and error interleavings for bounded read-ahead."""
from array import array
import gc
import threading
import unittest
import weakref

from glm_local.residency import BudgetExceededError, MAX_BYTES, ReservationLedger
from glm_local.runtime_read_ahead import ReadAheadPool
from glm_local.runtime_linear import project_many
from glm_local.safetensor_reader import TensorInfo


class TrackedPayload(bytearray):
    pass


class CheckingLease:
    def __init__(self, lease):
        self.lease = lease
        self.payload = None
        self.releases = 0

    def release(self):
        if self.payload is not None and self.payload() is not None:
            raise AssertionError("Encoded payload is still referenced when its lease is released")
        self.releases += 1
        if self.releases != 1:
            raise AssertionError("Read-ahead lease released more than once")
        self.lease.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class CheckingLedger:
    def __init__(self, budget=256):
        self.inner = ReservationLedger(budget)
        self.leases = []
        self.by_thread = {}

    def reserve(self, count, **kwargs):
        lease = CheckingLease(self.inner.reserve(count, **kwargs))
        self.leases.append(lease)
        self.by_thread[threading.get_ident()] = lease
        return lease

    def attach(self, value):
        self.by_thread[threading.get_ident()].payload = weakref.ref(value)


class ReadAheadStressTests(unittest.TestCase):
    def test_release_drops_packet_payload_before_lease_even_with_packet_reference_kept(self):
        pool, ledger = ReadAheadPool(), CheckingLedger()
        retained_packets, payload_refs = [], []
        def load(item):
            value = TrackedPayload([item] * 16)
            ledger.attach(value)
            payload_refs.append(weakref.ref(value))
            return value
        try:
            with pool.bands(range(6), load, lambda _: 16, ledger=ledger) as packets:
                for packet in packets:
                    self.assertEqual(packet.payload[0], packet.item)
                    retained_packets.append(packet)
                    packet.release()
                    packet.release()
                    self.assertIsNone(packet.payload)
            gc.collect()
            self.assertTrue(all(reference() is None for reference in payload_refs))
            self.assertTrue(all(lease.releases == 1 for lease in ledger.leases))
            self.assertEqual(ledger.inner.snapshot()["active_leases"], 0)
        finally:
            pool.close()

    def test_stopping_with_both_slots_full_drops_queue_without_loading_third_item(self):
        pool, ledger = ReadAheadPool(), CheckingLedger()
        second_loaded = threading.Event()
        loaded, references = [], []
        def load(item):
            value = TrackedPayload([item] * 16)
            ledger.attach(value)
            loaded.append(item)
            references.append(weakref.ref(value))
            if item == 1:
                second_loaded.set()
            return value
        try:
            with pool.bands(range(100), load, lambda _: 16, ledger=ledger) as packets:
                iterator = iter(packets)
                first = next(iterator)
                self.assertTrue(second_loaded.wait(3), "Producer did not fill its second slot")
                # Keep the packet/iterator alive across context teardown to expose
                # hidden references; the stream must clear the packet itself.
            self.assertEqual(loaded, [0, 1])
            self.assertIsNone(first.payload)
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            self.assertEqual(ledger.inner.snapshot()["active_leases"], 0)
            with pool.bands([7], load, lambda _: 16, ledger=ledger) as packets:
                for packet in packets:
                    self.assertEqual(packet.payload[0], 7)
                    packet.release()
        finally:
            pool.close()

    def test_primary_consumer_exception_survives_concurrent_producer_failure(self):
        pool, ledger = ReadAheadPool(), CheckingLedger()
        producer_inside_second, release_failure = threading.Event(), threading.Event()
        first_ref = []
        def load(item):
            if item == 1:
                producer_inside_second.set()
                if not release_failure.wait(3):
                    raise RuntimeError("Test did not release producer failure")
                raise OSError("secondary producer failure")
            value = TrackedPayload([item] * 16)
            ledger.attach(value)
            first_ref.append(weakref.ref(value))
            return value
        try:
            with self.assertRaisesRegex(ArithmeticError, "primary consumer failure"):
                with pool.bands(range(10), load, lambda _: 16, ledger=ledger) as packets:
                    iterator = iter(packets)
                    first = next(iterator)
                    self.assertTrue(producer_inside_second.wait(3))
                    release_failure.set()
                    raise ArithmeticError("primary consumer failure")
            self.assertIsNone(first.payload)
            gc.collect()
            self.assertIsNone(first_ref[0]())
            self.assertEqual(ledger.inner.snapshot()["active_leases"], 0)
        finally:
            release_failure.set()
            pool.close()

    def test_keyboard_interrupt_stops_future_reads_and_returns_reader_ownership(self):
        pool, ledger = ReadAheadPool(), ReservationLedger(256)
        reader_threads, reads = [], []
        def load(item):
            reader_threads.append(threading.get_ident())
            reads.append(item)
            return bytearray(16)
        owner = threading.get_ident()
        try:
            with self.assertRaises(KeyboardInterrupt):
                with pool.bands(range(100), load, lambda _: 16, ledger=ledger) as packets:
                    iterator = iter(packets)
                    next(iterator)
                    raise KeyboardInterrupt()
            completed = len(reads)
            self.assertLessEqual(completed, 2)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
            self.assertNotIn(owner, reader_threads)
            with pool.bands([101], load, lambda _: 16, ledger=ledger) as packets:
                for packet in packets:
                    packet.release()
            self.assertEqual(reads[-1], 101)
            self.assertEqual(len(set(reader_threads)), 1)
        finally:
            pool.close()

    def test_close_from_own_active_context_rejects_self_deadlock(self):
        pool, ledger = ReadAheadPool(), ReservationLedger(256)
        try:
            with pool.bands([0, 1], lambda _: bytearray(16), lambda _: 16, ledger=ledger) as packets:
                iterator = iter(packets)
                packet = next(iterator)
                with self.assertRaises(RuntimeError):
                    pool.close()
                packet.release()
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
        finally:
            pool.close()
        with self.assertRaises(RuntimeError):
            with pool.bands([0], lambda _: bytearray(16), lambda _: 16, ledger=ledger):
                pass

    def test_external_close_finishes_only_after_active_context_hands_reader_back(self):
        pool, ledger = ReadAheadPool(), ReservationLedger(256)
        closing, closed = threading.Event(), threading.Event()
        failures = []
        def close():
            closing.set()
            try:
                pool.close()
            except BaseException as error:
                failures.append(error)
            finally:
                closed.set()
        worker = None
        try:
            with pool.bands([0, 1], lambda _: bytearray(16), lambda _: 16, ledger=ledger) as packets:
                iterator = iter(packets)
                packet = next(iterator)
                worker = threading.Thread(target=close, name="test-read-ahead-close", daemon=True)
                worker.start()
                self.assertTrue(closing.wait(3))
                self.assertFalse(closed.wait(0.05), "Pool close returned before active reader handoff")
                packet.release()
            self.assertTrue(closed.wait(3), "Pool close did not observe active context teardown")
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
        finally:
            if worker is not None:
                worker.join(3)
            pool.close()
        with self.assertRaises(RuntimeError):
            with pool.bands([2], lambda _: bytearray(16), lambda _: 16, ledger=ledger):
                pass

    def test_all_concurrent_close_callers_wait_and_active_owner_still_rejects_close(self):
        pool, ledger = ReadAheadPool(), ReservationLedger(256)
        completed = [threading.Event(), threading.Event()]
        workers, failures = [], []
        def close(index):
            try:
                pool.close()
            except BaseException as error:
                failures.append(error)
            finally:
                completed[index].set()
        try:
            with pool.bands([0, 1], lambda _: bytearray(16), lambda _: 16, ledger=ledger) as packets:
                iterator = iter(packets)
                packet = next(iterator)
                workers.append(threading.Thread(target=close, args=(0,), daemon=True))
                workers[0].start()
                # This internal stop event is set inside close(), after it owns
                # the lifecycle lock; it avoids a scheduler-dependent delay.
                self.assertTrue(packets._stop.wait(3))
                workers.append(threading.Thread(target=close, args=(1,), daemon=True))
                workers[1].start()
                self.assertFalse(completed[1].wait(0.05), "Second close bypassed active reader handoff")
                with self.assertRaises(RuntimeError):
                    pool.close()
                packet.release()
            for done in completed:
                self.assertTrue(done.wait(3))
            self.assertEqual(failures, [])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
        finally:
            for worker in workers:
                worker.join(3)
            pool.close()

    def test_producer_reentrant_close_raises_without_waiting_for_its_own_completion(self):
        pool, ledger = ReadAheadPool(), ReservationLedger(256)
        calls = []
        def load(item):
            calls.append(item)
            pool.close()
            raise AssertionError("Producer close should have raised")
        try:
            with self.assertRaisesRegex(RuntimeError, "active band stream"):
                with pool.bands([0, 1], load, lambda _: 16, ledger=ledger) as packets:
                    list(packets)
            self.assertEqual(calls, [0])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
        finally:
            pool.close()

    def test_producer_budget_failure_preserves_structured_error_before_any_load(self):
        pool, ledger = ReadAheadPool(), ReservationLedger(15)
        calls = []
        def load(item):
            calls.append(item)
            return bytearray(16)
        try:
            with self.assertRaises(BudgetExceededError) as caught:
                with pool.bands([0, 1], load, lambda _: 16, ledger=ledger) as packets:
                    list(packets)
            self.assertEqual((caught.exception.device, caught.exception.required_bytes, caught.exception.budget_bytes),
                             ("cpu", 16, 15))
            self.assertEqual(calls, [])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
            with pool.bands([2], lambda _: bytearray(8), lambda _: 8, ledger=ledger) as packets:
                for packet in packets:
                    packet.release()
        finally:
            pool.close()

    def test_budget_error_preserves_required_sum_above_signed_limit_without_allocating(self):
        pool = ReadAheadPool()
        ledger = ReservationLedger(MAX_BYTES, base_cpu_bytes=MAX_BYTES)
        loads = []
        try:
            with self.assertRaises(BudgetExceededError) as caught:
                with pool.bands([0], lambda item: loads.append(item), lambda _: 1, ledger=ledger) as packets:
                    list(packets)
            self.assertEqual(caught.exception.required_bytes, MAX_BYTES + 1)
            self.assertEqual(caught.exception.budget_bytes, MAX_BYTES)
            self.assertEqual(loads, [])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
        finally:
            pool.close()

    def test_project_many_hands_back_reader_and_drops_both_encoded_bands(self):
        pool, ledger = ReadAheadPool(), CheckingLedger(64 * 1024**2)
        owner, reader_threads, kernel_threads, references = threading.get_ident(), [], [], []
        info = TensorInfo("BF16", (257, 16), (0, 257 * 16 * 2), 257 * 16 * 2, 2)
        class Reader:
            def read_span(self, name, offset, count):
                reader_threads.append(threading.get_ident())
                value = TrackedPayload(count)
                ledger.attach(value)
                references.append(weakref.ref(value))
                return value
        class Cpu:
            def prepare_dense_vector(self, value):
                return value
            def matvec_dense_row_band(self, raw, rows, cols, vector, dtype):
                kernel_threads.append(threading.get_ident())
                return array("f", [1]) * rows
        try:
            result, stats = project_many(Reader(), "w", info, [array("f", [1]) * 16],
                                         Cpu(), ledger=ledger, read_ahead=pool)
            self.assertEqual(result, [array("f", [1]) * 257])
            self.assertTrue(stats["read_ahead"]["enabled"])
            self.assertLessEqual(stats["read_ahead"]["peak_live_bands"], 2)
            self.assertEqual(stats["read_ahead"]["live_bands"], 0)
            self.assertEqual(set(kernel_threads), {owner})
            self.assertEqual(len(set(reader_threads)), 1)
            self.assertNotIn(owner, reader_threads)
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            self.assertEqual(ledger.inner.snapshot()["active_leases"], 0)
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
