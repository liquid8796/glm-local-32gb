# Flash W4A16: read-only local checkpoint verification — 2026-10-16

## Scope

The user-provided Flash checkpoint was inspected locally, with **no HTTP
requests, downloads, weight writes, model initialization or inference**.
The repository workspace selection remains separate from the model directory.

Model: `dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16`
Pinned baseline revision: `4c2ee763ff75f3ba50652ee0f024b0b630cd54a0`.

Command (from repository root, using the repository's Python environment):

```powershell
.\.venv-reference\Scripts\python.exe -m glm_local.flash_checkpoint_verify --model-dir "E:\Program Files\models\glm-5.3\GLM-5.3-Flash-CYBERSECURITY-W4A16"
```

## Evidence

- 120 out of 120 safetensors shard filenames and byte sizes match the pinned
  public manifest. Combined file length: **194,660,869,128 bytes**.
- Config, safetensors index, tokenizer config and tokenizer hashes match the
  pinned snapshot's SHA-256 values. The index has 113,074 tensor mappings.
- All 120 local shard headers parse with `SafeTensorReader`. All 113,074
  tensor descriptors match the index-to-shard mapping. Zero duplicates,
  missing names or extra names. The precise byte accounting is:
  - Encoded tensor payload: **194,645,249,400 bytes**.
  - Safetensors headers including 8-byte lengths: **15,619,728 bytes**.
  - Sum: **194,660,869,128 bytes**, equal to the index's
    `metadata.total_size`. The index counts full shard file sizes.
- A real
  `model.language_model.layers.10.mlp.experts.0.down_proj`
  projection was inspected: packed tensor `I32 (4096, 256)`,
  scale tensor `BF16 (4096, 64)` and shape tensor `I64 (2,)` containing
  `(4096, 2048)`.
- For first output row, a fixed 2048-element finite activation vector
  `x[i]=((i % 17)-8)/32` produced output **0.208183289**. The
  `W4A16Projection.matvec(..., first_row=0, row_count=1)` streaming
  reference and an independent NumPy uint32 nibble-extraction / BF16
  scale / float64 dot calculation had **0 absolute difference** for this
  sample. This demonstrates agreement on one row using the same assumed
  packing convention, **not** upstream compressed-tensors numerical parity.

## Limitations and next gates

- **No full weight payload SHA-256** verification was performed. A file may
  match size and header while containing corrupted weight values. The receipt
  reports `weight_payload_hashes_verified: false`.
- No full GLM5Next layers, sparse/linear attention, mHC, 288-expert routing,
  tokenizer prompting or end-to-end token generation were executed.
- This does not establish whether 64GB RAM/8GB VRAM can serve the model at
  an acceptable speed or with correct end-to-end numerical results.
- Before claiming model-load integrity, compare full checkpoint shard
  hashes against trusted LFS hashes where available, with bounded streaming,
  then validate real operators against an independent reference.
- Keep the existing external `flash-w4a16` server profile unchanged until a
  separately tested native backend is ready.

## Repeatable tests

`tests/test_flash_checkpoint_verify.py` exercises correct metadata, bad
config hash, missing pinned hash, bad shard size, missing index mapping,
invalid shard path, reported tokenizer digest and correct payload-vs-header
accounting using **small synthetic local safetensors**.

Output receipt (ignored by Git):
`reports/flash-w4a16-local-audit-20261016.json`.
