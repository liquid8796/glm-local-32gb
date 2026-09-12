# Current Memory Snapshot — GLM Local 32GB

Current release: v0.8.1. This is a project handoff document; historical verification artifacts retain their original release and dependency versions.

The synthetic decoder has CPU/GPU, official Transformers parity and bounded sharded safetensors validation. Those results do not establish full-checkpoint inference or resource limits.

The metadata audit reads only config/index/shard headers for the pinned checkpoint and produces a referenced JSONL catalogue with source evidence. A complete audit whose declared index size differs from observed tensor payload size returns `ERROR` before catalogue/FP8 review and cannot verify the structure. Partial audits defer equality but still enforce the declared total as an upper bound. Accounting diagnostics and received header evidence remain available on failure.

The architecture mapper validates the source report's scope/status/model/revision and embedded config, plus the referenced catalogue SHA-256, byte count and coverage before assessing a supported config-based profile. It reads only `reports/metadata-latest.json`; the catalogue must stay within a referenced run under `reports/metadata/`. Tensor roles use complete name patterns; projection gates, MoE routers and scale tensors are separate. Shape, dtype and required layer/expert inventory must all match. Unsupported profiles or incomplete/unknown inventories require review. Invalid reports, digests or identity produce errors rather than a fallback PASS from the saved model manifest. Architecture outputs JSON/Markdown with exit codes PASS 0, REVIEW_REQUIRED 2 and ERROR 1.

All metadata-only results leave checkpoint compatibility, inference, payload-value and full-model-limit capability flags false. `doctor` remains blocked. A structural architecture result is not an executable FP8 descriptor.

Historical baseline evidence: [Windows v0.6.1](verification/windows-acceptance-v0.6.1.md), [Linux v0.7.0 tests](verification/unit-tests-linux-v0.7.0.txt) and [the v0.7.0 failed network attempt](verification/metadata-online-attempt-linux-v0.7.0.json). The Linux v0.7.0 environment used safetensors 0.7.0; a release bump does not change that history.

Next steps:

1. Audit a complete catalogue from the pinned checkpoint and resolve unsupported tensor/layout/schedule findings.
2. Build bounded FP8 execution descriptors and validate selected real projections against the reference.
3. Develop a CPU/GPU residency planner, then verify numerical behavior and RAM/CPU/GPU limits under full-checkpoint inference.
