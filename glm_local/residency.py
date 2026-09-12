"""Config-derived residency estimates and bounded allocation accounting.

This module allocates no model, cache or CUDA buffers. It models sequential
single-token execution with CPU-resident compressed MLA latents, full-indexer
keys and streamed 128-square weight tiles. The plan is an estimate, not evidence
that a checkpoint executes correctly or fits in a measured process. Allocation
leases account declared payloads before the executor allocates them; they do
not observe Python, native-library, driver or operating-system allocations.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from threading import Lock

from .architecture.mapper import _profile
from .checkpoint_schema import Findings, quantization_format
from .runtime_io import DEFAULT_ROW_BAND_CACHE_BYTES
from .nvfp4_execution import NVFP4_ROW_BAND_SCRATCH_BYTES

MAX_BYTES = 2**63 - 1
TILE_EDGE = 128
ENCODED_ROW_BAND_CACHE_BYTES = DEFAULT_ROW_BAND_CACHE_BYTES


class ResidencyError(ValueError):
    """An unsupported configuration, invalid request or residency budget."""


class BudgetExceededError(ResidencyError):
    def __init__(self, device, required_bytes, budget_bytes):
        self.device = device
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes
        super().__init__(f"{device} residency requires {required_bytes} bytes; "
                         f"budget is {budget_bytes} bytes")


def _integer(value, name, *, minimum=0, maximum=MAX_BYTES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ResidencyError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _device(device):
    if not isinstance(device, str) or device not in ("cpu", "cuda"):
        raise ResidencyError("Allocation device must be cpu or cuda")
    return device


@dataclass(frozen=True)
class PlannerSettings:
    ram_budget_bytes: int = 32_000_000_000
    vram_budget_bytes: int = 0
    device: str = "cpu"
    context_tokens: int = 4096
    max_new_tokens: int = 32
    block_rows: int = TILE_EDGE
    cache_dtype: str = "float32"
    runtime_headroom_bytes: int = 2 * 1024**3
    gpu_headroom_bytes: int = 256 * 1024**2


class AllocationLease:
    """An immediate reservation, released explicitly or by a context manager."""

    __slots__ = ("_ledger", "_token", "_entered")

    def __init__(self, ledger, token):
        self._ledger = ledger
        self._token = token
        self._entered = False

    def release(self):
        self._ledger.release(self)

    @property
    def active(self):
        return self._ledger._active(self)

    def __enter__(self):
        if self._entered or not self.active:
            raise ResidencyError("Allocation lease cannot be reused")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.active:
            self.release()
        return False


class ReservationLedger:
    """Thread-safe payload reservations with fixed, nonreleasable baselines.

    Reserve before allocating; keep the lease live until every referenced
    buffer is freed or reusable. The baseline represents caller-owned cache
    payloads and runtime headroom. No GPU initialization occurs here, and a
    CUDA lease does not grant permission to bypass the executor's GPU gate.
    """

    def __init__(self, ram_budget_bytes, vram_budget_bytes=0, *, base_cpu_bytes=0,
                 base_cuda_bytes=0, max_active_leases=4096):
        self._budgets = {
            "cpu": _integer(ram_budget_bytes, "ram_budget_bytes"),
            "cuda": _integer(vram_budget_bytes, "vram_budget_bytes"),
        }
        self._baseline = {
            "cpu": _integer(base_cpu_bytes, "base_cpu_bytes"),
            "cuda": _integer(base_cuda_bytes, "base_cuda_bytes"),
        }
        self._maximum = _integer(max_active_leases, "max_active_leases", minimum=1,
                                 maximum=65536)
        for device in self._budgets:
            if self._baseline[device] > self._budgets[device]:
                raise BudgetExceededError(device, self._baseline[device], self._budgets[device])
        self._used = dict(self._baseline)
        self._peak = dict(self._baseline)
        self._leases = {}
        self._next_token = 0
        self._lock = Lock()

    def reserve(self, nbytes, *, device="cpu", label="temporary"):
        """Charge payload bytes now, before any associated allocation occurs."""
        _integer(nbytes, "reservation bytes", minimum=1)
        _device(device)
        if (not isinstance(label, str) or not 1 <= len(label) <= 128
                or any(ord(char) < 32 for char in label)):
            raise ResidencyError("Reservation label must be 1 to 128 printable characters")
        with self._lock:
            required = self._used[device] + nbytes
            if required > self._budgets[device]:
                raise BudgetExceededError(device, required, self._budgets[device])
            if len(self._leases) >= self._maximum:
                raise ResidencyError("Active allocation lease limit reached")
            token = self._next_token
            lease = AllocationLease(self, token)
            self._leases[token] = (lease, device, nbytes, label)
            self._next_token += 1
            self._used[device] = required
            self._peak[device] = max(self._peak[device], required)
            return lease

    def _active(self, lease):
        with self._lock:
            record = self._leases.get(lease._token)
            return record is not None and record[0] is lease

    def release(self, lease):
        if not isinstance(lease, AllocationLease) or lease._ledger is not self:
            raise ResidencyError("Allocation lease belongs to another ledger")
        with self._lock:
            record = self._leases.get(lease._token)
            if record is None or record[0] is not lease:
                raise ResidencyError("Allocation lease was already released or is invalid")
            self._used[record[1]] -= record[2]
            del self._leases[lease._token]

    def snapshot(self):
        with self._lock:
            return {
                "accounting_only": True,
                "active_leases": len(self._leases),
                "maximum_active_leases": self._maximum,
                **{device: {"budget_bytes": self._budgets[device],
                            "baseline_bytes": self._baseline[device],
                            "used_bytes": self._used[device],
                            "peak_bytes": self._peak[device],
                            "available_bytes": self._budgets[device] - self._used[device]}
                   for device in ("cpu", "cuda")},
            }


@dataclass(frozen=True)
class ResidencyPlan:
    settings: PlannerSettings
    model_context_tokens: int
    layer_count: int
    mtp_layers_excluded: int
    latent_width: int
    index_key_width: int
    indexer_owners: tuple[int, ...]
    max_projection_rows: int
    max_projection_columns: int
    cpu_components: tuple[tuple[str, int], ...]
    gpu_components: tuple[tuple[str, int], ...]
    configuration_resolution_json: str = "{}"

    @property
    def ram_required_bytes(self):
        return sum(value for _, value in self.cpu_components)

    @property
    def vram_required_bytes(self):
        return sum(value for _, value in self.gpu_components)

    @property
    def cache_bytes(self):
        return sum(value for key, value in self.cpu_components
                   if key in ("mla_latent_cache", "dsa_index_key_cache"))

    def device_for_block(self, index):
        """Alternate whole tile rows; every column tile in that row stays put."""
        _integer(index, "row-block index")
        return "cuda" if self.settings.device == "hybrid" and index % 2 else "cpu"

    def validate_prompt(self, prompt_tokens, max_new_tokens=None):
        _integer(prompt_tokens, "prompt_tokens", minimum=1)
        generated = self.settings.max_new_tokens if max_new_tokens is None else max_new_tokens
        _integer(generated, "max_new_tokens")
        if generated > self.settings.max_new_tokens:
            raise ResidencyError("Generation request exceeds the planned max_new_tokens")
        if prompt_tokens + generated > self.settings.context_tokens:
            raise ResidencyError("prompt_tokens + max_new_tokens exceeds planned context_tokens")

    def allocator(self, *, include_cache=True):
        """Reserve cache/headroom baselines; executor must lease its temporaries."""
        if type(include_cache) is not bool:
            raise ResidencyError("include_cache must be boolean")
        return ReservationLedger(
            self.settings.ram_budget_bytes,
            self.settings.vram_budget_bytes if self.settings.device == "hybrid" else 0,
            base_cpu_bytes=(self.cache_bytes if include_cache else 0) + self.settings.runtime_headroom_bytes,
            base_cuda_bytes=(self.settings.gpu_headroom_bytes
                             if self.settings.device == "hybrid" else 0),
        )

    def to_dict(self):
        return {
            "status": "ESTIMATE_FITS",
            "estimate_only": True,
            "full_model_loaded": False,
            "inference_verified": False,
            "full_model_limits_verified": False,
            "scope": "Sequential single-token backbone with CPU latent caches and streamed tiles",
            "context_tokens": self.settings.context_tokens,
            "max_new_tokens": self.settings.max_new_tokens,
            "model_context_tokens": self.model_context_tokens,
            "backbone_layers": self.layer_count,
            "mtp_layers_excluded": self.mtp_layers_excluded,
            "speculative_decoding": False,
            "cache": {
                "device": "cpu", "dtype": self.settings.cache_dtype, "itemsize": 4,
                "representation": "compressed MLA: normalized KV latent + rotated shared RoPE key",
                "mla_values_per_token_per_layer": self.latent_width,
                "mla_latent_bytes": dict(self.cpu_components)["mla_latent_cache"],
                "dsa_representation": "one index key per token per full indexer; shared layers reuse indices",
                "dsa_full_indexer_count": len(set(self.indexer_owners)),
                "dsa_index_values_per_token_per_full_layer": self.index_key_width,
                "dsa_index_key_bytes": dict(self.cpu_components)["dsa_index_key_cache"],
                "indexer_owner_by_layer": list(self.indexer_owners),
                "expanded_per_head_kv_retained": False,
            },
            "streaming": {
                "tile_rows": TILE_EDGE, "tile_columns": TILE_EDGE,
                "maximum_encoded_itemsize": 4,
                "max_projection_rows": self.max_projection_rows,
                "max_projection_columns": self.max_projection_columns,
                "persistent_decoded_weight_bytes": 0,
                "persistent_encoded_weight_bytes": ENCODED_ROW_BAND_CACHE_BYTES,
                "max_encoded_row_band_cache_bytes": ENCODED_ROW_BAND_CACHE_BYTES,
                "weights_residency": "Active decoded tiles and bounded encoded row bands; no retained full matrix or expert bank",
                "row_block_assignment": "cpu,cuda alternating" if self.settings.device == "hybrid" else "cpu",
                "gpu_gate_required_before_work": self.settings.device == "hybrid",
            },
            "ram": {"required_bytes": self.ram_required_bytes,
                    "budget_bytes": self.settings.ram_budget_bytes,
                    "remaining_bytes": self.settings.ram_budget_bytes - self.ram_required_bytes,
                    "components": dict(self.cpu_components)},
            "vram": {"required_bytes": self.vram_required_bytes,
                     "budget_bytes": self.settings.vram_budget_bytes if self.settings.device == "hybrid" else 0,
                     "remaining_bytes": ((self.settings.vram_budget_bytes - self.vram_required_bytes)
                                         if self.settings.device == "hybrid" else 0),
                     "components": dict(self.gpu_components)},
            "configuration_resolution": json.loads(self.configuration_resolution_json),
            "limitations": [
                "Configuration estimates do not establish complete metadata or payload verification",
                "Headroom is an explicit allowance; Python/native/driver/OS memory is not measured",
                "FP32 caches and one-token scheduling are required by this estimate",
                "Attention scores and expanded selected K/V must be processed in bounded token tiles",
                "Reservations constrain declared allocations only; executors must lease every buffer",
                "No throughput, whole-machine RAM, GPU utilization or full-model inference guarantee",
            ],
        }


def build_plan(config, settings, *, prompt_tokens=None, model_id=None, revision=None):
    """Validate all required dimensions before producing a no-allocation plan.

    Config provenance and a complete metadata report belong to the caller.
    Reviewed model identity is passed through to the architecture profile for
    explicitly source-verified defaults; other missing dimensions are errors.
    """
    if not isinstance(settings, PlannerSettings):
        raise ResidencyError("settings must be PlannerSettings")
    _integer(settings.ram_budget_bytes, "ram_budget_bytes", minimum=1)
    _integer(settings.vram_budget_bytes, "vram_budget_bytes")
    _integer(settings.context_tokens, "context_tokens", minimum=1, maximum=2**31 - 1)
    _integer(settings.max_new_tokens, "max_new_tokens", maximum=settings.context_tokens)
    _integer(settings.runtime_headroom_bytes, "runtime_headroom_bytes")
    _integer(settings.gpu_headroom_bytes, "gpu_headroom_bytes")
    if not isinstance(settings.device, str) or settings.device not in ("cpu", "hybrid"):
        raise ResidencyError("Planner device must be cpu or hybrid")
    if type(settings.block_rows) is not int or settings.block_rows != TILE_EDGE:
        raise ResidencyError("The streamed projection contract requires block_rows=128")
    if settings.cache_dtype != "float32":
        raise ResidencyError("Only explicit float32 compressed-cache accounting is supported")
    findings = Findings()
    identity = {} if model_id is None and revision is None else {
        "model_id": model_id, "revision": revision,
    }
    profile = _profile(config, findings, **identity)
    if profile is None:
        codes = ", ".join(sorted(findings.counts))
        raise ResidencyError(f"Unsupported architecture configuration: {codes}")
    context_max = _integer(config.get("max_position_embeddings"), "max_position_embeddings",
                           minimum=1, maximum=2**31 - 1)
    topk = _integer(config.get("index_topk"), "index_topk", minimum=1, maximum=2**31 - 1)
    experts_per_token = _integer(config.get("num_experts_per_tok"), "num_experts_per_tok",
                                minimum=1, maximum=profile["experts"])
    if settings.context_tokens > context_max:
        raise ResidencyError("context_tokens exceeds max_position_embeddings")
    if config["qk_rope_head_dim"] % 2:
        raise ResidencyError("qk_rope_head_dim must be even for RoPE pairs")
    if config["index_head_dim"] < config["qk_rope_head_dim"]:
        raise ResidencyError("index_head_dim cannot be smaller than qk_rope_head_dim")

    layers, heads = profile["layers"], config["num_attention_heads"]
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    qrank, kvrank = config["q_lora_rank"], config["kv_lora_rank"]
    nope, rope, value = (config[key] for key in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
    index_heads, index_dim = config["index_n_heads"], config["index_head_dim"]
    dense = config["intermediate_size"] if "dense" in profile["mlps"] else 0
    moe = config["moe_intermediate_size"] if "moe" in profile["mlps"] else 0
    shared = moe * config["n_shared_experts"]
    largest_ffn = max(dense, moe, shared)
    largest_rows = max(hidden, vocab, qrank, heads * (nope + rope), kvrank + rope,
                       heads * (nope + value), index_heads * index_dim, index_dim,
                       index_heads, profile["experts"], largest_ffn)
    largest_cols = max(hidden, qrank, kvrank, heads * value, largest_ffn)
    owners, owner = [], 0
    for layer, kind in enumerate(profile["indexers"]):
        if kind == "full":
            owner = layer
        owners.append(owner)

    context = settings.context_tokens
    selected = min(topk, context)
    # A single token's live activations, including residuals, normalization
    # gains, queries, routing scores and a sequential (not all-experts) FFN.
    activations = 4 * (8 * hidden + 4 * qrank + 3 * (kvrank + rope)
                       + 3 * heads * (nope + rope + value)
                       + 3 * index_heads * index_dim + 4 * index_dim
                       + 4 * largest_ffn + 4 * profile["experts"]
                       + 2 * experts_per_token + 2 * vocab)
    # Full sequence index-score vector plus deterministic selection workspace;
    # attention expands at most 128 selected latent rows at a time. No S^2 mask.
    scratch = (context * (4 * 3 + 8 * 2) + selected * (8 * 2 + 4 * 2)
               + 4 * min(selected, TILE_EDGE) * heads * (nope + rope + value)
               + 4 * heads * value + 8 * context)
    # Reader/native adapters reserve fixed capacity, including on partial
    # tiles and small vectors. Planning only the active prefix would approve
    # exact budgets that the actual fixed-capacity leases then reject.
    tile_elements = TILE_EDGE * TILE_EDGE
    # BF16/F16/F32 nonquantized matrices also stream. Four encoded bytes per
    # element and three copies conservatively cover FP8/read/native handoffs.
    encoded_tiles = 3 * tile_elements * 4
    decoded_tile = tile_elements * 4
    vectors = 4 * (2 * largest_rows + 2 * largest_cols + 4 * TILE_EDGE)
    cpu_components = (
        ("mla_latent_cache", layers * context * (kvrank + rope) * 4),
        ("dsa_index_key_cache", len(set(owners)) * context * index_dim * 4),
        ("encoded_row_band_cache", ENCODED_ROW_BAND_CACHE_BYTES),
        ("nvfp4_row_band_scratch", NVFP4_ROW_BAND_SCRATCH_BYTES if quantization_format(config) == "nvfp4" else 0),
        ("token_activations", activations),
        ("attention_and_selection_scratch", scratch),
        ("encoded_tile_copies", encoded_tiles),
        ("decoded_tile_scratch", decoded_tile),
        ("projection_vectors", vectors),
        ("scale_scratch", 4),
        ("runtime_headroom", settings.runtime_headroom_bytes),
    )
    gpu_components = ()
    if settings.device == "hybrid":
        gpu_components = (
            ("encoded_tile", tile_elements * 4),
            ("decoded_tile_scratch", decoded_tile),
            ("tile_vectors", 4 * TILE_EDGE * 4),
            ("scale_scratch", 4),
            ("runtime_headroom", settings.gpu_headroom_bytes),
        )
    mtp = profile.get("mtp_layers", 0)
    if isinstance(mtp, (list, tuple)):
        mtp = len(mtp)
    _integer(mtp, "mtp_layers", maximum=256)
    resolution = json.dumps(profile.get("configuration_resolution", {}),
                            sort_keys=True, allow_nan=False)
    plan = ResidencyPlan(settings, context_max, layers, mtp, kvrank + rope, index_dim,
                         tuple(owners), largest_rows, largest_cols, cpu_components,
                         gpu_components, resolution)
    if prompt_tokens is not None:
        plan.validate_prompt(prompt_tokens)
    for device, required, budget in (
        ("cpu", plan.ram_required_bytes, settings.ram_budget_bytes),
        ("cuda", plan.vram_required_bytes, settings.vram_budget_bytes),
    ):
        if required > budget:
            raise BudgetExceededError(device, required, budget)
    return plan
