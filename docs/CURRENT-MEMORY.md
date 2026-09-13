# Current Memory Snapshot — GLM Local 32GB

Current release: v0.11.0. Historical verification artifacts retain their original release and dependency versions.

ModelDesk1.1.0 adds structured text conversation, pinned chat-template verification, incremental Unicode-safe assistant/reasoning output, complete-pair history, retries and CLI chat. Incomplete output stays visible and is excluded from subsequent context. Python remains the inference owner; the C# adapter writes per-run messages and validates report ownership. Each turn starts a worker and replays its bounded history; there is no persistent inference server or HTTP provider endpoint.

Core0.11.0 adds eight-thread native CPU row bands, SIMD across up to16 independent vectors, layer-wise bounded prefill, protected Windows read handles and expanded MLA output reuse (up to256 tokens/layer). Planner and runtime leases include the caches and scratch space. CPU is the default generation/planning backend; hybrid dense operations also use CPU row bands. New paths preserve the established FP32 arithmetic and require rebuilt native DLLs. Full-checkpoint output fidelity remains a separate acceptance boundary.

Final release validation:899 distinct Python tests passed (898 main suite plus the one separately enabled row-SIMD benchmark),225 C# tests passed with0buildwarnings/errors, and28 headless WPF frames. A self-contained1.1.0 package was published under artifacts/ModelDesk/win-x64. See verification/conversation-performance-v0.11.0.md for scoped measurements and acceptance limits.

Active default profile is nvfp4: dealignai/GLM-5.3-ABLITERATED-NVFP4 pinned371bdb985d0124e76348c91e4a8fcf3a9d719d09. User set the former FP8 model aside; it remains selectable with --profile fp8. The user completed the NVFP4 download; on2026-09-13 all282 shard names/sizes matched the pinned manifest and the tokenizer passed at the selected E: directory. Full payload hashes were not reread in that inventory.

Core0.10.1 adds native BF16/F16/F32 tile matvecs preserving sequential FP32 arithmetic, bounded8MiB encoded row-band caching counted by the planner/ledger, and bounded stage progress retained on timeout. The user's300-second CPU generation expired; elapsed time alone does not establish corruption or a deadlock. Full-model correctness and throughput remain distinct from metadata and kernel checks.

Post-fix real diagnostic: context128, CPU, promptHi (one input token), one requested output and1800s timeout completed in1033.816s. Output154820 isEOS/endoftext, decoded text empty. Peak working set1,179,160,576B and private commit1,171,648,512B; no numerical/chat/provider acceptance. The earlier300s probe stopped during layer21/78. CPU NVFP4 row-band batching additionally leases5MiB scratch. Full suites:788 Python and146 C# tests passed. See verification/runtime-timeout-v0.10.1.md.

NVFP4 snapshot is docs/models/abliterated-nvfp4/model-metadata.json; reports are under reports/nvfp4. All282 headers/232,385 tensors/57,600 NVFP4 quadruples verified, architecture has zero findings. Metadata HTTP timed out after279headers; new --resume reverified cache and fetched only3missingheaders. Original evidence is unchanged.

Implemented packedE2M1 low-even/high-odd reader, row-group16E4M3 scales, F32global/input calibration scalars, nativeCPUandCUDAFP32fallback, descriptors and runtime dispatch. Input calibration is verified but not multiplied into weights; activation_quantization=none and nativeW4A4parity=false. Attention/shared/dense/MTP stayBF16. Same config-derived decoder and resource planner; MTP speculative execution remains excluded. Both model tokenizer artifacts have identical verified hashes and were reused without network.

See [NVFP4.md](NVFP4.md) for current commands and [CHECKPOINTS.md](CHECKPOINTS.md) for historical FP8 acceptance boundaries.

## Historical v0.9.0 handoff

The model/revision remain dealignai/GLM-5.3-CYBERSECURITY-FP8 at 5915c1b88f998a9c1e1a0c83688e285a08ae3ca5. User selected code completion and small checks only; this does not authorize a 756 GB download.

All 282 headers and 118,629 tensors pass metadata/architecture checks. Exact header accounting proves that this index.total_size includes complete shard files. Reviewed config defaults apply only to the pinned identity; 78 backbone layers and one MTP inventory are supported.

Implemented: selected FP8 descriptors/readers/execution; RAM/VRAM planner and leases; complete-catalogue BF16/F16/F32/FP8 streaming reader with two open shards; config-derived compressed MLA/DSA/grouped-MoE decoder; pinned native tokenizer verification; Windows-job CLI and report workflow. Real q_a2048x6144 CPU384/GPU384 tiles matched independent FP32 reference with zero error. Native small full-stack decoder cases match pinned Transformers. These checks do not verify full-model BF16 fidelity, physical RAM or GPU utilization.

Runtime uses FP32 caches/linear and scalar reductions, portable tie rules, default RoPE and sequential disk streaming. MTP speculative execution is excluded. Full checkpoint shards remain absent and doctor stays BLOCKED. See [CHECKPOINTS.md](CHECKPOINTS.md) and [RUNTIME.md](RUNTIME.md).

## Historical v0.8.1 handoff

The synthetic decoder has CPU/GPU, official Transformers parity and bounded sharded safetensors validation. Those results do not establish full-checkpoint inference or resource limits.

The metadata audit reads only config/index/shard headers for the pinned checkpoint and produces a referenced JSONL catalogue with source evidence. A complete audit whose declared index size differs from observed tensor payload size returns `ERROR` before catalogue/FP8 review and cannot verify the structure. Partial audits defer equality but still enforce the declared total as an upper bound. Accounting diagnostics and received header evidence remain available on failure.

The architecture mapper validates the source report's scope/status/model/revision and embedded config, plus the referenced catalogue SHA-256, byte count and coverage before assessing a supported config-based profile. It reads only `reports/metadata-latest.json`; the catalogue must stay within a referenced run under `reports/metadata/`. Tensor roles use complete name patterns; projection gates, MoE routers and scale tensors are separate. Shape, dtype and required layer/expert inventory must all match. Unsupported profiles or incomplete/unknown inventories require review. Invalid reports, digests or identity produce errors rather than a fallback PASS from the saved model manifest. Architecture outputs JSON/Markdown with exit codes PASS 0, REVIEW_REQUIRED 2 and ERROR 1.

All metadata-only results leave checkpoint compatibility, inference, payload-value and full-model-limit capability flags false. `doctor` remains blocked. A structural architecture result is not an executable FP8 descriptor.

Historical baseline evidence: [Windows v0.6.1](verification/windows-acceptance-v0.6.1.md), [Linux v0.7.0 tests](verification/unit-tests-linux-v0.7.0.txt) and [the v0.7.0 failed network attempt](verification/metadata-online-attempt-linux-v0.7.0.json). The Linux v0.7.0 environment used safetensors 0.7.0; a release bump does not change that history.

Next steps:

1. Audit a complete catalogue from the pinned checkpoint and resolve unsupported tensor/layout/schedule findings.
2. Build bounded FP8 execution descriptors and validate selected real projections against the reference.
3. Develop a CPU/GPU residency planner, then verify numerical behavior and RAM/CPU/GPU limits under full-checkpoint inference.
