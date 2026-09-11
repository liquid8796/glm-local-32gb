"""Offline Transformers graph for the one fixed invented miniature fixture.

The existing independent NumPy oracle supplies only validated, bounded fixture
arrays. All forward computation runs through the unmodified installed official
model. This module cannot accept a model identifier, tokenizer, checkpoint
configuration, arbitrary dimensions, or pretrained weights.
"""

from copy import deepcopy
import hashlib
import importlib
import os
from pathlib import Path

from .mini_spec import SOURCE_REVISION, matrix_shapes, vector_lengths


HIDDEN_STATE_NAMES = (
    "embedding",
    *(f"layer.{layer}.{name}" for layer in range(2)
      for name in ("input_norm", "attention_output", "post_attention",
                   "post_attention_norm", "output")),
    "final_norm",
)


def _fixed_config(config_class):
    config = config_class(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_shared_experts=1, n_routed_experts=4, routed_scaling_factor=2.5,
        kv_lora_rank=4, q_lora_rank=8, qk_rope_head_dim=4, v_head_dim=4,
        qk_nope_head_dim=4, n_group=1, topk_group=1, num_experts_per_tok=2,
        norm_topk_prob=True, hidden_act="silu", max_position_embeddings=128,
        rms_norm_eps=1e-5, use_cache=True, pad_token_id=None, bos_token_id=0,
        eos_token_id=1, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        mlp_layer_types=["dense", "sparse"], attention_bias=False,
        attention_dropout=0.0, index_topk=4, index_head_dim=8, index_n_heads=2,
        mlp_bias=False, first_k_dense_replace=1,
        layer_types=["deepseek_sparse_attention", "deepseek_sparse_attention"],
        indexer_types=["full", "shared"],
    )
    config._attn_implementation = "eager"
    return config


def _mapped_state(reference, torch):
    """Map every fixed fixture array, including the router correction buffer."""
    state, used_matrices, used_vectors = {}, set(), set()

    def matrix(name):
        if name in used_matrices:
            raise ValueError("Synthetic matrix mapped more than once")
        used_matrices.add(name)
        return torch.tensor(reference._matrices[name], dtype=torch.float32, device="cpu")

    def vector(name):
        if name in used_vectors:
            raise ValueError("Synthetic vector mapped more than once")
        used_vectors.add(name)
        return torch.tensor(reference._vectors[name], dtype=torch.float32, device="cpu")

    state["model.embed_tokens.weight"] = matrix("embed")
    state["lm_head.weight"] = matrix("lm_head")
    state["model.norm.weight"] = vector("final_norm")
    for layer in range(2):
        source, target = f"layer.{layer}.", f"model.layers.{layer}."
        for name, module in (
            ("q_a", "q_a_proj"), ("q_b", "q_b_proj"),
            ("kv_a", "kv_a_proj_with_mqa"), ("kv_b", "kv_b_proj"), ("o", "o_proj"),
        ):
            state[target + "self_attn." + module + ".weight"] = matrix(source + name)
        for name, module in (
            ("q_norm", "self_attn.q_a_layernorm"),
            ("kv_norm", "self_attn.kv_a_layernorm"),
            ("in_norm", "input_layernorm"), ("post_norm", "post_attention_layernorm"),
        ):
            state[target + module + ".weight"] = vector(source + name)
    indexer = "model.layers.0.self_attn.indexer."
    for name, module in (("index_q", "wq_b"), ("index_k", "wk"),
                         ("index_weight", "weights_proj")):
        state[indexer + module + ".weight"] = matrix("layer.0." + name)
    for suffix in ("weight", "bias"):
        state[indexer + "k_norm." + suffix] = vector("layer.0.index_norm_" + suffix)
    for source, target in (("layer.0.mlp.", "model.layers.0.mlp."),
                           ("layer.1.shared.", "model.layers.1.mlp.shared_experts.")):
        for projection in ("gate", "up", "down"):
            state[target + projection + "_proj.weight"] = matrix(source + projection)
    state["model.layers.1.mlp.gate.weight"] = matrix("layer.1.router")
    state["model.layers.1.mlp.gate.e_score_correction_bias"] = vector("layer.1.router_bias")
    state["model.layers.1.mlp.experts.gate_up_proj"] = torch.stack([
        torch.cat([matrix(f"layer.1.expert.{expert}.gate"),
                   matrix(f"layer.1.expert.{expert}.up")], dim=0)
        for expert in range(4)
    ], dim=0)
    state["model.layers.1.mlp.experts.down_proj"] = torch.stack([
        matrix(f"layer.1.expert.{expert}.down") for expert in range(4)
    ], dim=0)
    if used_matrices != set(matrix_shapes()) or used_vectors != set(vector_lengths()):
        raise ValueError("The official mapping must consume every fixed synthetic array")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Synthetic arrays must remain finite when converted to float32")
    return state


def _source_identity(module):
    path = Path(module.__file__).resolve()
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("Official model source exceeds verification bound")
    with path.open("rb") as source:
        content = source.read(2 * 1024**2 + 1)
    if len(content) > 2 * 1024**2:
        raise ValueError("Official model source exceeds verification bound")
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


class OfficialMiniReference:
    """CPU float32, eager, incremental official inference for invented IDs 0-31.

    Initialization enables the Hugging Face offline flags before optional
    imports. Use the dedicated reference worker to isolate that process policy.
    ``metadata`` records installed source hashes; the caller verifies them
    against its pinned-source lock before reporting revision identity.
    """

    def __init__(self, directory):
        self._handles, self._model = [], None
        self._closed = False
        self._cache = None
        self._position = 0
        self._captured = {}
        self._indices = self._routing = None
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Deliberately lazy: normal project commands need neither dependency.
        import torch
        import transformers
        from .mini_reference import ReferenceDecoder

        module_path = "transformers.models.glm_moe_dsa."
        modeling = importlib.import_module(module_path + "modeling_glm_moe_dsa")
        configuration = importlib.import_module(module_path + "configuration_glm_moe_dsa")
        reference = ReferenceDecoder(directory)
        state = _mapped_state(reference, torch)
        self._torch = torch
        try:
            with torch.device("cpu"):
                self._model = modeling.GlmMoeDsaForCausalLM(
                    _fixed_config(configuration.GlmMoeDsaConfig)).float().eval()
            current = self._model.state_dict()
            if set(state) != set(current):
                raise ValueError("Official model state keys differ from the fixed synthetic mapping")
            if any(value.shape != current[name].shape for name, value in state.items()):
                raise ValueError("Official model state shapes differ from the fixed synthetic mapping")
            self._model.load_state_dict(state, strict=True)
            loaded = self._model.state_dict()
            if any(value.dtype != torch.float32 or value.device.type != "cpu"
                   or not torch.equal(value, state[name]) for name, value in loaded.items()):
                raise ValueError("Official fixed state must be copied exactly as CPU float32")
            self._metadata = {
                "synthetic_only": True, "real_checkpoint_compatible": False,
                "requested_source_revision": SOURCE_REVISION,
                "torch_version": str(torch.__version__),
                "transformers_version": str(transformers.__version__),
                "device": "cpu", "parameter_dtype": "torch.float32",
                "attention_implementation": "eager", "incremental_cache": True,
                "matrix_count": len(reference._matrices),
                "vector_count": len(reference._vectors), "state_dict_count": len(state),
                "state_value_count": sum(value.numel() for value in state.values()),
                "all_state_values_copied_exactly": True,
                "state_dtypes": sorted({str(value.dtype) for value in loaded.values()}),
                "source_files": {"modeling": _source_identity(modeling),
                                 "configuration": _source_identity(configuration)},
                "hidden_state_names": list(HIDDEN_STATE_NAMES),
                "fixture_loader": "ReferenceDecoder bounded NumPy fixture decode only",
                "selection_order": "Native torch.topk order; no tie-rule replacement",
            }
            self._install_hooks()
        except BaseException:
            self.close()
            raise

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def position(self):
        return self._position

    def _require_open(self):
        if self._closed:
            raise ValueError("Official miniature reference is closed")

    def _capture(self, name, value):
        if name in self._captured:
            raise ValueError("Official hidden-state hook fired more than once in a token step")
        if value.shape != (1, 1, 16) or value.dtype != self._torch.float32:
            raise ValueError("Official hidden states must be [1, 1, 16] float32")
        self._captured[name] = value.detach().clone()

    def _install_hooks(self):
        def output_hook(name, index=None):
            def capture(module, inputs, output):
                self._capture(name, output if index is None else output[index])
            return capture

        def input_hook(name):
            def capture(module, inputs):
                self._capture(name, inputs[0])
            return capture

        base = self._model.model
        self._handles.append(base.embed_tokens.register_forward_hook(output_hook("embedding")))
        self._handles.append(base.norm.register_forward_hook(output_hook("final_norm")))
        for index, layer in enumerate(base.layers):
            prefix = f"layer.{index}."
            self._handles.append(layer.input_layernorm.register_forward_hook(
                output_hook(prefix + "input_norm")))
            self._handles.append(layer.self_attn.register_forward_hook(
                output_hook(prefix + "attention_output", 0)))
            self._handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                input_hook(prefix + "post_attention")))
            self._handles.append(layer.post_attention_layernorm.register_forward_hook(
                output_hook(prefix + "post_attention_norm")))
            self._handles.append(layer.register_forward_hook(output_hook(prefix + "output", 0)))

        def capture_indices(module, inputs, output):
            self._indices = output[2].detach().clone()

        def capture_routing(module, inputs, output):
            self._routing = (output[1].detach().clone(), output[2].detach().clone())

        self._handles.append(base.layers[0].self_attn.register_forward_hook(capture_indices))
        self._handles.append(base.layers[1].mlp.gate.register_forward_hook(capture_routing))

    def reset(self):
        self._require_open()
        self._cache = None
        self._position = 0
        self._captured.clear()
        self._indices = self._routing = None

    def step(self, token):
        self._require_open()
        if type(token) is not int or not 0 <= token < 32:
            raise ValueError("Official miniature token must be an integer in [0, 31]")
        if self._position >= 128:
            raise ValueError("Official miniature context limit is 128 tokens; call reset()")
        self._captured.clear()
        self._indices = self._routing = None
        torch = self._torch
        try:
            with torch.inference_mode():
                output = self._model(input_ids=torch.tensor([[token]], dtype=torch.long,
                                                            device="cpu"),
                                     past_key_values=self._cache, use_cache=True,
                                     return_dict=True)
            if set(self._captured) != set(HIDDEN_STATE_NAMES):
                raise ValueError("Official hooks did not capture every fixed hidden state")
            if (output.logits.shape != (1, 1, 32) or output.logits.dtype != torch.float32
                    or not torch.isfinite(output.logits).all().item()
                    or any(not torch.isfinite(value).all().item()
                           for value in self._captured.values())):
                raise ValueError("Official miniature output must be finite float32")
            if self._indices is None or self._routing is None:
                raise ValueError("Official selection hooks did not run")
            weights, experts = self._routing
            if (tuple(self._indices.shape) != (1, 1, min(4, self._position + 1))
                    or tuple(weights.shape) != (1, 2) or tuple(experts.shape) != (1, 2)):
                raise ValueError("Official selections do not match the fixed miniature")
            self._cache = output.past_key_values
            if self._cache is None or self._cache.get_seq_length() != self._position + 1:
                raise ValueError("Official cache did not advance by one token")
            result = {
                "logits": output.logits[0, 0].tolist(),
                "hidden_states": {name: self._captured[name][0, 0].tolist()
                                  for name in HIDDEN_STATE_NAMES},
                "selected_indices": self._indices[0, 0].tolist(),
                "routed_experts": experts[0].tolist(),
                "router_weights": weights[0].tolist(),
            }
            self._position += 1
            return result
        except BaseException:
            # A failed forward may already have modified the official cache.
            self.reset()
            raise

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache = None
        self._captured.clear()
        self._indices = self._routing = None
        self._model = None
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
