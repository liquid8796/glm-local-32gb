# GLM Architecture Mapper v0.8.1

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

The supported `glm_moe_dsa_unpacked_fp8_metadata_v1` profile requires config dimensions rather than miniature defaults. It checks dtype and expected shape and requires the complete tensor inventory for the configured layers, dense/MoE schedule and experts. It requires an explicit sparse-attention and full/shared-indexer schedule, unpacked experts, untied embeddings, bias-free attention/MLP, and FP8 E4M3 weights with F32 scales on a 128×128 grid. Required configuration cannot be inferred from tensor names. Packed experts, extra/MTP layers and unknown tensors need further review.

Layer numbers appearing in a name alone cannot verify a layer. Unknown names, unsupported profiles and incomplete inventories return `REVIEW_REQUIRED` and retain findings for review. A structural match does not validate scale values, kernel behavior, tokenizer, RoPE values or activation/cache memory.

## Result boundary

| Architecture status | Exit code | Meaning |
|---|---:|---|
| `PASS` | 0 | Required source metadata and supported architecture structure checks passed. |
| `REVIEW_REQUIRED` | 2 | Valid source metadata or the architecture inventory remains incomplete or unsupported. |
| `ERROR` | 1 | The source report, referenced catalogue or their consistency checks are invalid. |

Read the findings as well as the status. The exit code 2 here belongs to `architecture-check`; `metadata-check` has its own documented status/code mapping.

Metadata-only results leave `real_checkpoint_compatible`, `full_model_loaded`, `inference_verified`, `full_model_limits_verified` and `payload_values_verified` false. The metadata audit also leaves `architecture_mapping_verified` false; the separate mapper can verify only its documented structural profile. `doctor` remains blocked until the runtime and full checkpoint are independently validated.

No executable FP8 descriptors or residency planner are delivered by this command. The next step is to review a complete catalogue from the pinned checkpoint, resolve unsupported layout/dtype/schedule findings, and then implement bounded execution descriptors and numerical parity for selected real projections.
