# Runtime timeout diagnosis — core 0.10.1 / ModelDesk 1.0.3

The user's CPU generation request used the downloaded NVFP4 checkpoint, prompt `chào bạn`, context 4096, eight new tokens and a 300-second deadline. The Windows job reached that deadline and stopped its worker. The original report contained no stage information, so timeout alone did not identify a deadlock, a damaged payload, or the exact operation that was running.

The patch removes concrete overhead in the implementation while preserving the existing decoded-weight FP32 arithmetic:

- BF16/F16/F32 dense tiles use a native CPU primitive instead of Python per-element conversion, multiplication and addition. The additive ABI entry point retains 128-square bounds and strict sequential FP32 reductions.
- CPU NVFP4 projections batch up to 128 rows and 16384 logical columns per native call. Each row still reduces 128-column subtotals and adds them in the original order. Legacy and hybrid tile execution remain available.
- A reader cache retains at most 8 MiB of encoded row bands, one partial band per tensor. It does not retain complete matrices. Source reads and returned tiles remain at most 64 KiB, with identity checks before/after cache hits and allocation leases released on eviction/close.
- The planner accounts for the 8 MiB cache and a separate 5 MiB NVFP4 batching scratch allowance. The 32,000,000,000-byte setting limit and Windows job enforcement remain unchanged.

Stage diagnostics retain one bounded `progress.json` (at most 16 KiB), with elapsed time, phase, token/layer counters and the active tensor name. Heartbeats persist once per second and normally print every five seconds. Setup/phase boundaries flush immediately. Timeout reports recover only progress matching the run, model, revision and action; they retain exit code 1 and false inference-success flags. The GUI/CLI classify a fresh matching timeout report as **Timed out**. No timeout is extended or retried automatically.

## Validation

- `test-reference.bat`: **788/788 tests passed**, no skips, including native CPU/CUDA and the pinned miniature Transformers reference. Full-suite runtime was 45.795 seconds.
- `build-studio.bat -Test`: **146/146 tests passed**, no skips; Release build had zero warnings/errors. Actual XAML rendered headlessly across seven tabs, two sizes and both themes (28 frames).
- Native dense synthetic 128×128 BF16 tile: median 31.099 ms in the previous scalar path versus 0.260 ms in the native path, bit-identical results.
- Native NVFP4 synthetic 128×6144 band: median 30.901 ms across the prior tile calls/gather versus 5.456 ms in one band call, bit-identical results.
- A synthetic NVFP4 row band required 7 coalesced payload reads versus 12,288 matrix-row reads, with identical returned bytes. These kernel/read-count comparisons do not measure full-model throughput.

## Bounded real-checkpoint probe

The updated CLI ran one input token ID and requested one output token with context 128, CPU backend and the same 300-second deadline. Metadata validation finished around 8.5 seconds, weights initialized around 32.3 seconds, and layer/projection progress continued until timeout. The last persisted stage was layer 21/78, `model.layers.20.mlp.experts.186.up_proj.weight`. No output token had been generated. The worker tree was stopped and the C# result correctly reported `TimedOut: true` and `Status: Timed out`.

A mid-run process sample reported peak working set 1,126,273,024 bytes and about one CPU core of accumulated work. This is a sample from the bounded run, not full-model RAM or throughput acceptance. Even after removing the measured overhead, the current CPU implementation remains too slow to claim interactive use of this checkpoint on the tested i7-11800H laptop. A context memory estimate does not establish a usable generation rate.

Local evidence: `reports/studio/runtime-timeout-v0.10.1-tests.log`, `reports/studio/runtime-timeout-v1.0.3-tests.log`, `reports/studio/ui/headless-*.png`, `reports/nvfp4/generate/20260912T201345Z-bc6414b2/{request,progress,result}.json`, and `reports/modeldesk/runs/20260912T201344972Z-258fee63/`.

## Completed single-token diagnostic

A second CPU run used the literal prompt `Hi` (one tokenizer token), context 128, one requested new token and a 1800-second deadline. It completed all 78 backbone layers and the output head in **1033.816 seconds** (CLI wall time 1034.227 seconds), returning exit 0 and `GENERATED_UNVERIFIED`.

The only generated token was **154820, `<|endoftext|>`**, which is an EOS token in the pinned config. Decoding with special tokens removed produced an empty string. This demonstrates completion of the bounded execution path; it does **not** demonstrate a useful chat answer, numerical agreement with the complete reference model, or API/provider readiness.

The worker recorded peak working set **1,179,160,576 bytes**, peak private commit **1,171,648,512 bytes**, and average CPU utilization about **6.15% across 16 logical processors** (approximately one busy core). The planner required 2,206,099,780 bytes; peak declared allocations were 2,203,872,584 bytes. There were 1,079,808 native dense tiles and zero scalar dense tiles. Reader payload calls returned 48,040,045,056 bytes; this is logical reader traffic and does not establish physical disk throughput. Windows job readback retained the 70% CPU cap and 32,000,000,000-byte committed-memory limit.

The tested 300-second configuration is therefore insufficient even for the smallest complete forward pass on this implementation/hardware. The 1800-second setting allowed this one-token diagnostic to finish; it is not a recommended chat configuration or a guarantee for longer prompts/output. Runtime performance and chat formatting/behavior require further work before this checkpoint can serve Jarvis interactively.

Completed-run evidence: `reports/nvfp4/generate/20260912T202144Z-eed99984/{request,progress,result}.json` and `reports/modeldesk/runs/20260912T202143970Z-994cc646/`.
