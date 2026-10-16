# GLM-5.3-Flash-CYBERSECURITY-W4A16

Profile: `flash-w4a16`. Checkpoint revision: `4c2ee763ff75f3ba50652ee0f024b0b630cd54a0`.
The profile, metadata and reports are isolated from the existing FP8 and NVFP4 models.
The default RAM budget remains 32,000,000,000 bytes.

## Architecture and storage

The pinned config declares `Glm5NextForConditionalGeneration`, model type `glm5_next`,
with nested text and vision configs. The text backbone has 45 layers: 34 linear
attention layers and 11 sparse attention layers, mHC, 288 routed experts and an
additional MTP layer at index 45. This differs from the existing `glm_moe_dsa` decoder.

Routed expert projections use `compressed-tensors` 0.18.0 `pack-quantized`
symmetric INT4 weights with group size 32. The index stores `weight_packed`,
`weight_scale` and `weight_shape` tensors. Other matrices use BF16. This is not the
existing ModelOpt NVFP4 representation or its E2M1/E4M3 scale scheme.

The manifest contains 120 weight shards totaling 194,660,869,128 bytes (194.66 GB;
181.29 GiB). The index lists 113,074 tensors. Its declared `total_size` equals the
sum of complete shard file sizes; shard headers were not read, so the byte-accounting
convention has not been verified. The API's displayed parameter count includes packed
I32 tensor elements and must not be treated as a decoded parameter count.

Sources: [pinned config](https://huggingface.co/dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16/resolve/4c2ee763ff75f3ba50652ee0f024b0b630cd54a0/config.json),
[manifest](https://huggingface.co/api/models/dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16/revision/4c2ee763ff75f3ba50652ee0f024b0b630cd54a0?blobs=true),
[index](https://huggingface.co/dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16/resolve/4c2ee763ff75f3ba50652ee0f024b0b630cd54a0/model.safetensors.index.json).

## Execution boundary

The bundled Python/native decoder does not execute this model. Registering a profile
does not implement INT4 unpacking, linear attention, mHC, the vision tower or MTP.
Native architecture checks must continue to reject this config.

The profile declares `inference_backend: openai` and
`backend_status: external_openai_required` for an external serving process such as
a compatible local vLLM deployment. Here, OpenAI-compatible describes the HTTP
request/response protocol; a server on this machine does not require the OpenAI
cloud or an OpenAI account. The default served model name is the full Hugging Face
model ID, and must match the configured server's advertised model name.

The server is responsible for loading this architecture and quantization format,
its CPU/GPU allocation, and its inference limits. ModelDesk's local decoder resource
settings do not configure a separate server. A usable server build, deployment,
hardware requirements and real generation still require runtime verification.

## Baseline evidence

`model-metadata.json` records the pinned public manifest, complete config, tokenizer
config, index summary, metadata hashes and explicit verification limits. No weight
payloads or shard headers were downloaded for this baseline. The checkpoint has not
been loaded or evaluated here, and no full-model parity or cybersecurity capability
claim is made.

Local weight downloads belong under `models/GLM-5.3-Flash-CYBERSECURITY-W4A16` and
must remain untracked. Reports use `reports/flash-w4a16`; the metadata baseline uses
`docs/models/flash-cybersecurity-w4a16/model-metadata.json`.
