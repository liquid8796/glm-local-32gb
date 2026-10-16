# GLM5Next Flash W4A16 — local/offline execution design (2026-10-16)

## Goal and boundary

Target `dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16` at revision
`4c2ee763ff75f3ba50652ee0f024b0b630cd54a0`, inference on the user's
Windows 11 machine (i7-11800H, 64 GB RAM, RTX 3070 Laptop 8 GB VRAM), without
external inference services at runtime. Initial acceptance is **text-only,
context 128, one output token**; vision and MTP speculative decoding are
explicitly out of scope until their own numerical parity gates pass.

The pinned public manifest lists 120 safetensors shards (194,660,869,128 bytes).
The Flash weights are **not downloaded** in this checkout. Obtaining the weights
is a separate one-time prerequisite; local/offline refers to *inference* with
weights already on local storage. Do not download the 195 GB checkpoint or
change unrelated projects as part of preliminary kernel work.

## Architecture and invariants

Model profile reports `Glm5NextForConditionalGeneration`/`glm5_next`,
45 text backbone layers (linear attention, DeepSeek sparse attention), mHC and
288 routed experts; MTP is separate. The existing `glm_moe_dsa` decoder must
continue to reject this checkpoint. Explicit native Flash implementation is
the only route to local inference, not selecting ModelDesk's `openai` backend.

Compressed-tensors v0.18.0 `pack-quantized` applies to expert
`gate_proj/up_proj/down_proj`: signed INT4, group size 32 along input
features, packed eight nibbles per INT32, with per-group scales. Dense,
attention and other ignored tensors remain higher precision. References:
`docs/models/flash-cybersecurity-w4a16/model-metadata.json` and upstream
compressed-tensors `pack_quantized/helpers.py`.

### Proposed components

1. **Independent W4A16 projection reference (first bounded milestone).**
   `glm_local/flash_w4a16.py` validates `weight_shape`, packed and scale
   metadata; only reads bounded row groups from an existing safetensors reader.
   Signed nibble decoding + BF16/F32 scales + float64 reference sum. It does
   not claim numerical parity with BF16 inference, CUDA or real weight payloads.
2. **Verified local shard and tensor catalogue.** Reuse protected bounded
   safetensors reads, revision-pinned metadata and payload hashes. Stream at
   most two shards; never eagerly construct the expert bank. Validate the
   manifest's file-size/index-accounting convention against real headers.
3. **Text-only GLM5Next graph.** Implement embedding, mHC, linear-attention
   recurrent state, sparse attention/selection, routing, experts, norm and
   logits with config-derived shapes. Fail closed for unsupported layers,
   vision requests and MTP. Compare each subgraph to a pinned independent
   reference using synthetic small finite fixtures before full weights.
4. **Native CPU throughput path.** Replace Python per-element loops using
   tested native INT4 x FP32 kernels; add thread scheduling and constrained
   GPU offload where correct. Never mutate original weights.
   Account and lease temporary buffers; report measured RAM/VRAM/CPU, reads
   and per-stage timings. Any GPU path remains opt-in after parity.
5. **End-to-end offline acceptance.** With complete verified weights in a
   user-selected local directory, run one prompt/one token with no network,
   report numerical parity scope, peak commit/working set and first token
   latency. Only then consider longer context/chat/UI integration.

## Gates and stop conditions

- Keep `flash-w4a16` external-server profile behavior unchanged until a
  separate `native_flash` backend is capable; no misleading automatic fallback.
- No physical-memory extrapolation from a *planner estimate*. Windows Job
  budget is committed process memory, not a hard limit on all OS cache.
- Reject tensor mismatches and nonfinite activations/scales before compute;
  weight reads bounded to <=64 KiB per call.
- Do not claim model loaded, generated tokens, security quality or usable
  performance from tests of a projection or mock HTTP server.
- A prior **different** NVFP4 model produced two text tokens after ~18 minutes;
  that measurement is a caution, not a prediction for Flash.
- If full checkpoint integrity or numerical-parity gates fail, stop rather
  than silently substitute a different model.

## Current status (2026-10-16)

The first projection reference and synthetic safetensors regression tests are
implemented on branch `feature/flash-w4a16-offline-core`. Full Flash
architecture mapping, real checkpoint parity and end-to-end generation remain
**unimplemented and unverified**.
