# Implementation plan: offline GLM5Next Flash W4A16

Spec: `docs/superpowers/specs/2026-10-16-glm5next-flash-local-design.md`.
Current branch `feature/flash-w4a16-offline-core`; no checkpoint download.

## Task 1 — W4A16 projection reference (completed 2026-10-16)
Files: `glm_local/flash_w4a16.py`, `tests/test_flash_w4a16.py`.
Test RED: missing module; test GREEN: packed signed INT4, two group scales,
BF16/F32, invalid layouts, bounded local safetensors read. Completion requires
unit suite, Python full regression suite and `git diff --check`.
Verification on Windows: 7 W4A16 tests passed, complete Python suite 934 tests (65 skipped), .NET ModelDesk suite 309 passed, and staged diff check passed. Explicitly *not* full model generation.

## Task 2 — Real tensor verification (partial: structure and one expert-row probe on 2026-10-16)
Files: `glm_local/flash_w4a16.py`, new isolated reader adapter + tests.
Add synthetic arbitrary-shape catalogue fixtures first; inspect locally
downloaded pinned shard headers and compare a small real projection against
independent compressed-tensors dequantization if appropriate dependencies
exist. Verified local 120/120 shard headers and pinned config/index/tokenizer hashes,
113,074 index-to-shard tensor mappings, and index total_size accounting. One
real 2048-wide INT4 down_proj row agreed with a NumPy unpack/reference
calculation (zero absolute difference). See
docs/verification/flash-w4a16-local-checkpoint-2026-10-16.md. Full weight
payload SHA-256 and upstream compressed-tensors numerical parity are still
pending. Never claim full checkpoint payload fidelity from headers alone.
The user has downloaded all model shards to their chosen E: directory; no
duplicate download is required.

## Task 3 — GLM5Next text-only architecture mapper (future)
Build strict tensor inventory shape/role checks, config-preserving routing,
mHC/linear-attention/sparse-attention operators with small official-reference
numerical fixtures. Test unsupported/ambiguous config failure modes.
Do not reuse legacy decoder architecture assumptions.

## Task 4 — CPU/native optimization with bounded residency (future)
Port validated reference kernels to native SIMD/threads. Start with one
projection; compare elementwise to CPU oracle including extremes/group
boundaries. Track read bandwidth, cache behavior and peak private commit.
Add hybrid RTX3070 path only if it demonstrates measured benefit/parity.

## Task 5 — Offline one-token full-checkpoint acceptance (future)
Only with all 120 shards locally present: full header and hash validation,
one text prompt / context128 / one generated token, CPU first, network
disabled; record RAM/VRAM/CPU/SSD metrics and output parity limitations.
Do not enable GUI native Flash chat until this gate passes.
