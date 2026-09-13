# Direct answers and bounded read-ahead — 0.11.1 / ModelDesk 1.1.1

Windows validation on 2026-09-13, using the same pinned NVFP4 profile, native DLLs and reference dependencies as [0.11.0](conversation-performance-v0.11.0.md). This patch changes Python/C# code; it does not require new native exports beyond the 0.11.0 DLLs.

## Conversation behavior

The optional `--direct-answer` extension appends the single closing-think token `154842` to the assistant generation prefix produced from the hash-verified template. It supplies no answer text. A normal `Hi`/low prompt has 13 tokens; the direct prefix has 14. The original template's 60 oracle combinations remain unchanged; 12 additional checks compare the extension with that original output plus the closing marker.

The option defaults to false in Python and the shared C# DTO. GUI and CLI opt in explicitly. Low/high/max effort choices remain intact. Streaming starts in the visible channel but can detect later thinking re-entry, keeping reasoning and visible text separate through final decoding and partial recovery. Raw/token input rejects the option before a worker is started.

## CPU pipeline

Native CPU row-band methods now also expose owned FP32 array outputs. The original list-returning methods remain available. Both variants share the same native call, locking and validation. This removes repeated Python FP32 pack/unpack work from single-token output handling.

A lazy, reusable producer thread reads the next encoded band while the consumer computes the current band. All weight/scale/scalar reads belong to the producer during that projection. Admission is bounded before allocation to two live bands; each has a ledger lease, with at most 16 MiB additional capacity in the planner. Each actual read remains at most 64 KiB and the catalogue retains at most two open shard handles.

Shutdown releases payload references before leases, drains queued bands and joins the producer before closing or reusing the reader. Tests exercise concurrent close, producer/consumer errors, KeyboardInterrupt, both slots full, reentrant close, budget rejection and injected adapters. They compare native outputs bit for bit against sequential access.

The read-ahead prototype measured warm BF16 projections with identical output hashes:

| Projection scope | Sequential median | Read-ahead median |
|---|---:|---:|
| Synthetic 2048×16384, five alternating pairs | 46.249 ms | 35.532 ms |
| First 2048 rows of real `layers.0.mlp.down_proj.weight`, 12288 columns, two alternating pairs | 36.6445 ms | 27.3174 ms |

The real projection trial read 240 MiB including warmup. These short measurements are approximately 1.30–1.34× improvements under their recorded warm-file conditions, not claims about full-model speed. Local evidence: `reports/profiling/20260913-read-ahead-prototype.json`.

A separate, instrumented one-layer diagnostic with synthetic finite input identified 146,880 Python scalar-conversion calls in output handling before this change. The diagnostic wall time changed from 1.756 s to 0.861 s with array outputs/read-ahead. Profiling overhead and file-cache state affect those times; this is bottleneck evidence, not a calibrated full-model comparison. The full decoder layers were not executed by that diagnostic.

An earlier experiment increasing individual span reads to 1 MiB did not show a repeatable material benefit and was not integrated. The bounded 64-KiB read policy remains intact.

## Release validation

`test-reference.bat` passed **934/934 tests**, with no errors or skips. The pinned template, input-batch benchmark and row-SIMD opt-ins were enabled in the same run. Main log SHA-256: `7dc794ef7c4ca0825d982b68499a164216c14b3ab7f09c55fbdfa9a1bc2d47a0`.

ModelDesk Release build passed **238/238 tests**, no skips, warnings or errors. Headless rendering checked all 28 combinations of seven tabs, two sizes and two themes, including the unchecked direct-answer option and usable composer.

The complete Python suite ran concurrently with a full-checkpoint probe. Its benchmark timings are explicitly not treated as calibrated comparisons. Detailed receipts remain under `reports/validation/core-v0.11.1-final*` and `reports/studio/tests/studio-tests.trx`.

For context 4096 and 256 new tokens, the CPU planner estimates 5,449,468,740 bytes including cache capacities and runtime headroom. `reader.cpu_parallelism` reports configured kernel/read-ahead limits, not sustained CPU utilization. The model remains subject to full-checkpoint numerical and broader capability acceptance; these release tests alone do not establish those claims.

## Full-checkpoint functional smoke

The local NVFP4 checkpoint completed one short history-recall test through all 78 backbone layers. The supplied history was user `Call me An.`, assistant `OK.`, then user `My name? One word.`. Direct-answer mode produced **`An.`**, followed by EOS, with no generated reasoning, `assistant_response_complete=true` and exit code 0. This is one replayed history, not a measurement of several live GUI sessions.

The input contained 29 tokens; output IDs were `[2082, 13, 154827]` (two text tokens plus EOS). Initialization took 44.455 s, prefill 904.078 s, time to the first token 948.532 s, and two decode steps 145.804 s. Total worker time was 1094.522 s (about 18 minutes 15 seconds). Observed decode rate was 0.01372 tokens/s across only those two steps. The model still has very high interactive latency on this machine.

Peak working set was 1,181,495,296 bytes and peak private commit 1,173,565,440 bytes. Measured worker CPU averaged 13.642% of 16 logical processors; both native row-band backends were configured for 8 compute threads. Read-ahead ran 31,848 projections and reached a peak of 2 live bands. The reader returned 344,876,226,936 bytes over the entire test, including reads served by the OS cache; this is not a physical-disk-byte measurement. Read sizes stayed at most 64 KiB and at most 2 shards remained open.

The installed Windows Job confirmed the 70% CPU hard cap, committed-memory limit of 32,000,000,000 bytes and kill-on-close policy. Initial tests/builds overlapped prefill, so these are observed functional-run timings, not an isolated before/after comparison. A single correct reply does not establish full-checkpoint numerical parity, long-context quality or a broader conversation benchmark. [Portable result receipt](direct-answer-full-checkpoint-v0.11.1.json).
