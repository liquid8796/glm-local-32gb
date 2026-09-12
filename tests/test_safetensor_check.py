import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from glm_local.safetensor_check import compare_raw_tensors, run_blocks, launch_check, validate_check
from glm_local.winjob import InstalledLimits


class StorageCheckTests(unittest.TestCase):
    def test_format_byte_comparison_streams_aligned_bounded_chunks(self):
        payload=b"a"*(65536+4)
        reference={"tensors":{"w":{"shape":[16385],"dtype":"F32","raw_bytes":payload},
                              "e":{"shape":[0],"dtype":"F32","raw_bytes":b""}}}
        reader=SimpleNamespace(tensors={"w":SimpleNamespace(shape=(16385,),dtype="F32",nbytes=len(payload)),
                                       "e":SimpleNamespace(shape=(0,),dtype="F32",nbytes=0)})
        reads=[]
        def read(name,offset,count):
            reads.append((name,offset,count))
            return reference["tensors"][name]["raw_bytes"][offset:offset+count]
        reader.read_bytes=read
        result=compare_raw_tensors(reader,reference)
        self.assertTrue(result["passed"])
        self.assertEqual(reads,[("w",0,65536),("w",65536,4),("e",0,0)])
        self.assertEqual(result["tensor_sha256"]["w"],hashlib.sha256(payload).hexdigest())
        reader.read_bytes=lambda *args:b"wrong"
        with self.assertRaises(RuntimeError):compare_raw_tensors(reader,reference)

    def test_dtype_disagreement_fails_before_read(self):
        reader=SimpleNamespace(tensors={"w":SimpleNamespace(shape=(1,),dtype="F32",nbytes=4)},read_bytes=Mock())
        reference={"tensors":{"w":{"shape":[1],"dtype":"I32","raw_bytes":b"\0"*4}}}
        with self.assertRaises(RuntimeError):compare_raw_tensors(reader,reference)
        reader.read_bytes.assert_not_called()

    def test_hybrid_routes_rows_and_multiplies_each_block_scale(self):
        blocks=[SimpleNamespace(row_start=0,col_start=0,rows=1,cols=1,scale=.25,weights=b"\x38"),
                SimpleNamespace(row_start=128,col_start=0,rows=1,cols=1,scale=.5,weights=b"\x38")]
        matrix=SimpleNamespace(rows=129,cols=1,iter_blocks=lambda:iter(blocks))
        events=[]
        class Backend:
            def __init__(self,label):self.label=label
            def matvec_tile(self,weights,rows,cols,vector,scale):
                events.append(self.label)
                return [vector[0]*scale] # 0x38 = exact 1 in E4M3FN
        gate=SimpleNamespace(before_submit=lambda:events.append("gate"))
        result,counts=run_blocks(matrix,[4.0],Backend("cpu"),Backend("gpu"),gate)
        self.assertEqual(events,["cpu","gate","gpu"])
        self.assertEqual((result[0],result[128]),(1.,2.))
        self.assertEqual((counts["cpu_blocks"],counts["gpu_blocks"]),(1,1))

    def test_gpu_gate_failure_does_not_fallback(self):
        block=SimpleNamespace(row_start=128,col_start=0,rows=1,cols=1,scale=.5,weights=b"\x38")
        matrix=SimpleNamespace(rows=129,cols=1,iter_blocks=lambda:iter([block]))
        cpu,gpu=Mock(),Mock()
        gate=SimpleNamespace(before_submit=Mock(side_effect=RuntimeError("unavailable")))
        with self.assertRaisesRegex(RuntimeError,"unavailable"):run_blocks(matrix,[1.],cpu,gpu,gate)
        cpu.matvec_tile.assert_not_called();gpu.matvec_tile.assert_not_called()

    def test_invalid_parameters_and_missing_venv(self):
        for backend,seed in (("auto",1),("cpu",-1),("hybrid",True),("cpu",2**32)):
            with self.assertRaises(ValueError):validate_check(backend,seed)
        settings={"model_directory":"unused","ram_budget_bytes":32_000_000_000,
                  "cpu_job_percent":70,"gpu_average_target":.6,"gpu_window_seconds":10,
                  "gpu_index":0,"disk_reserve_bytes":0}
        with tempfile.TemporaryDirectory() as tmp, patch("glm_local.safetensor_check.run_local_process") as runner:
            with self.assertRaisesRegex(RuntimeError,"setup-reference"):launch_check(tmp,settings)
            runner.assert_not_called()

    def test_worker_failure_cannot_publish_pass_and_job_is_installed(self):
        settings={"model_directory":"unused","ram_budget_bytes":32_000_000_000,
                  "cpu_job_percent":70,"gpu_average_target":.6,"gpu_window_seconds":10,
                  "gpu_index":0,"disk_reserve_bytes":0}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);exe=root/".venv-reference/Scripts/python.exe"
            exe.parent.mkdir(parents=True);exe.touch()
            def run(args,**kwargs):
                self.assertEqual(kwargs["limits"].committed_memory_bytes,32_000_000_000)
                self.assertEqual(kwargs["limits"].cpu_percent,70)
                kwargs["on_policy"](InstalledLimits(70,32_000_000_000,True,True,True))
                (Path(args[-1]).parent/"result.json").write_text('{"status":"PASS"}')
                return 1
            with patch("glm_local.safetensor_check.run_local_process",side_effect=run),patch("builtins.print"):
                self.assertEqual(launch_check(root,settings),1)
            report=json.loads((root/"reports/storage-latest.json").read_text())
            self.assertEqual(report["status"],"ERROR")
            self.assertFalse(report["synthetic_storage_verified"])
            self.assertTrue(report["job_policy_verified"])
