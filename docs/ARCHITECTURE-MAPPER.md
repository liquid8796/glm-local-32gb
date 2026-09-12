# GLM Architecture Mapper v0.10.1

NVFP4 now has a separate reviewed ModelOpt0.45.0 profile for `dealignai/GLM-5.3-ABLITERATED-NVFP4` at `371bdb985d0124e76348c91e4a8fcf3a9d719d09`. Its complete inventory matches232,385 tensors:57,600 routed projections each have U8[N,K/2] weight, E4M3[N,K/16] block scale and two F32scalar ancillaries. Protected attention/shared/dense/MTP tensors remain BF16. Roles, logical/stored shapes, producer/config groups/ignore coverage and scalar shapes are independently checked; unsupported profiles still require review.

The NVFP4 metadata flag verifies stored structure only. Activation W4A4 and FP8 KV-cache quantization remain unverified; current runtime uses FP32 inputs and caches. Reports are isolated by the selected model profile. See [NVFP4.md](NVFP4.md).

The mapper checks checkpoint metadata against a supported GLM structure profile. It does not read tensor payloads, execute a graph or prove numerical compatibility.

Run the metadata audit first, then the mapper:

```powershell
.\glm.bat metadata-check
.\glm.bat architecture-check
```

The mapper reads only `reports/metadata-latest.json`, uses the config embedded in that report and follows its referenced `tensor-catalogue.jsonl`. It writes `reports/architecture-latest.json` and `reports/architecture-latest.md`. Catalogue rows are not embedded in the report. There is no fallback to `docs/model-metadata.json` or an inline tensor array: a manifest does not contain the checkpoint tensor catalogue. Older reports lacking the required provenance must be regenerated with `metadata-check`; an existing snapshot can be replayed offline as described in [CHECKPOINT-METADATA.md](CHECKPOINT-METADATA.md).

## Source and completeness checks

Before approving a structure, the mapper validates the report's metadata-only scope, status, model/revision, config, baseline comparison and coverage. Model/revision must agree with the pinned project configuration. It checks the catalogue byte count and SHA-256 against the report descriptor, then validates record counts, inspected shards, tensor dtype/shape/nbytes/offsets and summed payload accounting. A missing or invalid report, malformed catalogue, changed artifact or identity mismatch produces `ERROR`. A valid partial audit or upstream `REVIEW_REQUIRED` remains architecture `REVIEW_REQUIRED`; matching some layer names cannot promote it to `PASS`.

The referenced run must resolve inside `reports/metadata/`; the catalogue filename is fixed to `tensor-catalogue.jsonl` within that run. The source report is capped at 4 MiB, catalogue at 128 MiB, each JSONL record at 16 KiB and inventory at 262,144 records. These are parsing/input limits, not a full-process or system RAM guarantee.

Offsets must form contiguous, non-overlapping ranges starting at zero within each shard, including valid zero-length boundary entries. The result records the captured source report's byte count and SHA-256. If the source report is replaced during analysis, the command fails instead of publishing a verdict against a different latest report.

This is a local consistency check. A user who can rewrite all local evidence can also recalculate its digests; the mapper does not independently authenticate remote checkpoint contents.

## Supported structure

Roles use complete tensor-name patterns. Embeddings, norms, attention projections/indexer, dense MLP projections, routed experts, shared experts, router and FP8 scales are distinct. For example, `mlp.gate_proj.weight` is an MLP projection; it is not the MoE router `mlp.gate.weight`. A `weight_scale_inv` tensor is a scale, not its paired projection or router.

The supported profile requires config dimensions, dtype/shape and complete layer/expert inventory. For the exact reviewed CYBERSECURITY model/revision only, missing `layer_types` and `mlp_bias` use the pinned official Transformers configuration rules, with source hashes recorded. Generic profiles remain strict. One declared MTP layer is structurally supported with its full indexer, MoE and four BF16 projection/norm extras. MTP results prove inventory only; ordinary generation executes the 78-layer backbone. Other MTP profiles, packed experts and unknown tensors require review.

Live verification read 282 headers and 118,629 tensors: 59,044 FP8 pairs, 78 backbone layers plus one MTP layer, zero findings. For this checkpoint's complete-file `total_size`, the mapper reconstructs exact header overhead from hashed snapshot evidence before acceptance.

Layer numbers appearing in a name alone cannot verify a layer. Unknown names, unsupported profiles and incomplete inventories return `REVIEW_REQUIRED` and retain findings for review. A structural match does not validate scale values, kernel behavior, tokenizer, RoPE values or activation/cache memory.

## Result boundary

| Architecture status | Exit code | Meaning |
|---|---:|---|
| `PASS` | 0 | Required source metadata and supported architecture structure checks passed. |
| `REVIEW_REQUIRED` | 2 | Valid source metadata or the architecture inventory remains incomplete or unsupported. |
| `ERROR` | 1 | The source report, referenced catalogue or their consistency checks are invalid. |

Read the findings as well as the status. The exit code 2 here belongs to `architecture-check`; `metadata-check` has its own documented status/code mapping.

Metadata-only results leave `real_checkpoint_compatible`, `full_model_loaded`, `inference_verified`, `full_model_limits_verified` and `payload_values_verified` false. The metadata audit also leaves `architecture_mapping_verified` false; the separate mapper can verify only its documented structural profile. `doctor` remains blocked until the runtime and full checkpoint are independently validated.

Separate `projection-check`, `runtime-plan`, `tokenizer-check` and `generate` commands implement the execution path. See [RUNTIME.md](RUNTIME.md). This command remains metadata-only.
