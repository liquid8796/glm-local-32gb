# Conversation and CPU performance — core 0.11.0 / ModelDesk 1.1.0

Measured on Windows, 2026-09-13: Intel i7-11800H (8 cores, 16 logical processors), 32 GB RAM and RTX 3070 Laptop 8 GB. The measurements below exercise native CPU execution. The reference environment remains pinned, including CPU-only Torch; this release does not install or claim a new GPU inference backend.

## Changes checked

- Native CPU pools use at most eight active compute threads per projection. SSE2 lanes cover four independent input vectors, or four independent output rows for batches of one to three vectors. Each output retains the original 128-column subtotal and FP32 addition order; no FMA is introduced.
- Layer-wise prefill shares one encoded band across up to 16 inputs, skips unnecessary intermediate output heads and avoids replaying MoE routing/FFN work. Attention remains causal and preserves the established selection/softmax order.
- Exact FP32 arrays are copied directly into owned native buffers and finite-checked; generic inputs retain their existing coercion and rejection rules. Returned native arrays are validated without scalar pack/unpack round trips.
- Windows readers hold deny-write/delete handles for active shards and config/index. Reopened shards are rebound to the original identity/header proof. Portable readers retain identity polling. Reads remain at most 64 KiB, with spans/band cache bounded to 8 MiB and at most two active shard handles.
- Expanded K/V reuse is bounded to 256 tokens per layer with LRU eviction, source validation, rollback and ledger accounting. Reset/close release retained data. This is activation caching, not full weight residency.

## Projection measurements

These measurements isolate different stages and must not be multiplied together as a full-model speedup.

| Scope | Before | After | Equality |
|---|---:|---:|---|
| One real BF16 `layers.0.self_attn.o_proj.weight`, 6144×16384, tile versus row-band path | 4.190326 s | 0.236900 s | Bit identical |
| Warm synthetic NVFP4 file, 2048×6144, 13 inputs, before/after FP32 buffer preparation | 142.15 ms | 35.90 ms | Same output SHA-256 |
| Warm synthetic BF16 file, 6144×16384, 13 inputs, before/after FP32 buffer preparation | 648.54 ms | 300.50 ms | Same output SHA-256 |
| Resident BF16 band, 128×6144, 1–3 inputs, 8 threads, before/after row SIMD | — | 2.58–3.03× speed ratio | Bit identical |
| Resident NVFP4 band, same geometry and thread count | — | 3.47–3.76× speed ratio | Bit identical |

The real projection timing includes local weight access but excludes model initialization. Warm synthetic file measurements use three samples; resident row SIMD uses seven alternating before/after samples and excludes input preparation/file I/O. Results vary with workload and system activity.

Local detailed evidence: `reports/studio/real-dense-band-vnext.json`, `reports/profiling/runtime-linear-before/measurements.json`, `reports/profiling/runtime-linear-after/measurements.json` and `reports/profiling/20260913-row-simd-benchmark.json`.

## Conversation checks

The downloaded template is 10,464 bytes with SHA-256 `4a4b64df09bd4f54fb18a2cfc86b99c329d37362a8c289d466b443a99bac0645`. Its pinned Git blob is `4c431fa97c2b53e795fdc10cee6bbc9847f8e39a`. The manual text formatter matches 60 combinations evaluated against that approved template. Production does not evaluate repository Jinja/Python. Tests cover English, Vietnamese, emoji and token boundaries.

Structured events distinguish reasoning and visible output, use UTF-16 replacement offsets and reconcile against the final report. Incomplete responses are not added to future conversation context. Per-run messages files and recovered output are bound to the selected model/revision and run directory. These tests establish protocol behavior, not the model's conversational quality.

Small native FP8/NVFP4 decoder checks compare scalar versus batched prefill at chunk boundaries and with expanded cache enabled/disabled. They compare logits and stored cache bytes, including rollback and retry. The six-token cache fixture reduces linear calls from 186 to 160 without changing those results. The 13-token, three-layer memo fixture reduces replay lookups from 1100 to 229 with no cache misses and unchanged projection counts.

## Acceptance boundary

Final Python validation covered **899 distinct tests, all passed**: 898 in `test-reference.bat` plus the one row-SIMD benchmark opt-in executed separately. The main run also enabled the pinned chat-template oracle and input-batch benchmark. Main log SHA-256: `8fa0d9fb48d76cf64d149602c8bf44e9380b25e6b48e0bf642273710d3ced6b1`; supplementary log SHA-256: `f4719b875d53719bd5021830f41100b507230bb5b88a50f2cf8151bf38840404`. These validation runs overlapped a real-model probe; their benchmark times are not the controlled measurements in the table above.

Final ModelDesk Release validation passed **225/225 tests**, with no skips, build warnings or errors. Headless WPF rendering checked 28 frames across seven tabs, two sizes and two themes, including the visible response/composer, collapsed reasoning and Ctrl+Enter binding. The service tests cover stale partial recovery, Unicode, empty/incomplete results and terminal error normalization. `publish-studio.bat -SkipTests` then built the self-contained x64 package from the validated source without repeating the test suite.

The local NVFP4 inventory has all 282 shard names/sizes matching the pinned manifest (464,822,689,680 bytes). This inventory did not reread every payload hash. Metadata, native kernels, small graph parity and GUI/CLI tests do not establish full-checkpoint W4A4/BF16 parity or useful interactive throughput. Capability flags remain unverified until the corresponding acceptance work succeeds.

For context 4096 and 256 new tokens, the current CPU planner estimates 5,432,691,524 bytes including cache capacities and 2 GiB runtime headroom. It is not a measured peak. Windows Job limits remain 32,000,000,000 bytes committed memory and 70% CPU under the default profile.

The historical raw `Hi` probe on core 0.10.1 took 1033.816 seconds to emit a single EOS token with empty visible text; see [the original report](runtime-timeout-v0.10.1.md). It is not a successful chat baseline. An intermediate 13-token chat prefill on this development branch hit its explicit 900-second diagnostic deadline at layer 68/78, before the final buffer/replay/row-SIMD changes. That interrupted probe is excluded from the projection comparisons above.
