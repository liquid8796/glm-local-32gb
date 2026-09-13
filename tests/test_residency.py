"""No-allocation plans and lease failures, using explicit configuration fixtures."""
from copy import deepcopy
from dataclasses import replace
import json
from concurrent.futures import ThreadPoolExecutor
import unittest

from glm_local.residency import (
    BudgetExceededError, PlannerSettings, ReservationLedger, ResidencyError, build_plan,
)


def small_config():
    return {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 2, "hidden_size": 16,
        "vocab_size": 32, "num_attention_heads": 2, "q_lora_rank": 8,
        "kv_lora_rank": 4, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
        "v_head_dim": 4, "index_n_heads": 2, "index_head_dim": 8,
        "intermediate_size": 24, "moe_intermediate_size": 24,
        "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "max_position_embeddings": 128, "index_topk": 4,
        "layer_types": ["deepseek_sparse_attention"] * 2,
        "mlp_layer_types": ["dense", "sparse"], "indexer_types": ["full", "shared"],
        "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
    }


def realistic_config():
    config = small_config()
    config.update(num_hidden_layers=78, hidden_size=6144, vocab_size=154880,
                  num_attention_heads=64, q_lora_rank=2048, kv_lora_rank=512,
                  qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
                  index_n_heads=32, index_head_dim=128, intermediate_size=12288,
                  moe_intermediate_size=2048, n_routed_experts=256, num_experts_per_tok=8,
                  max_position_embeddings=1048576, index_topk=2048,
                  layer_types=["deepseek_sparse_attention"] * 78,
                  mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
                  indexer_types=["full" if i < 3 or (i - 2) % 4 == 0 else "shared"
                                 for i in range(78)])
    return config


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.settings = PlannerSettings(context_tokens=32, max_new_tokens=4,
                                        runtime_headroom_bytes=1024 * 1024)

    def test_latent_cache_uses_latent_width_not_expanded_heads(self):
        report = build_plan(self.config, self.settings).to_dict()
        self.assertEqual(report["cache"]["mla_latent_bytes"], 2 * 32 * (4 + 4) * 4)
        self.assertEqual(report["cache"]["dsa_index_key_bytes"], 32 * 8 * 4)
        self.assertEqual(report["cache"]["indexer_owner_by_layer"], [0, 0])
        self.assertTrue(report["cache"]["expanded_per_head_kv_retained"])
        self.assertEqual(report["cache"]["expanded_cache_tokens_per_layer"], 32)
        self.assertEqual(report["streaming"]["persistent_decoded_weight_bytes"], 0)
        self.assertTrue(report["estimate_only"])
        self.assertFalse(report["full_model_limits_verified"])
        json.dumps(report, allow_nan=False)

    def test_realistic_78_layer_configuration_fits_short_context_as_estimate(self):
        settings = PlannerSettings(context_tokens=4096, max_new_tokens=256,
                                   device="hybrid", vram_budget_bytes=8 * 1024**3)
        plan = build_plan(realistic_config(), settings, prompt_tokens=128)
        report = plan.to_dict()
        self.assertEqual(plan.layer_count, 78)
        self.assertEqual(report["cache"]["mla_latent_bytes"], 78 * 4096 * 576 * 4)
        self.assertEqual(report["cache"]["dsa_full_indexer_count"], 21)
        self.assertEqual(report["cache"]["dsa_index_key_bytes"], 21 * 4096 * 128 * 4)
        self.assertLess(plan.ram_required_bytes, 32_000_000_000)
        self.assertGreater(plan.vram_required_bytes, settings.gpu_headroom_bytes)
        self.assertEqual(plan.max_projection_rows, 154880)
        self.assertEqual(plan.max_projection_columns, 16384)

    def test_requested_maximum_context_rejected_before_cache_allocation(self):
        settings = PlannerSettings(context_tokens=1048576)
        with self.assertRaises(BudgetExceededError) as caught:
            build_plan(realistic_config(), settings)
        self.assertEqual(caught.exception.device, "cpu")
        self.assertGreater(caught.exception.required_bytes, settings.ram_budget_bytes)

    def test_full_shared_full_schedule_retains_distinct_owner_caches(self):
        config = deepcopy(self.config)
        config.update(num_hidden_layers=4, layer_types=["deepseek_sparse_attention"] * 4,
                      mlp_layer_types=["dense"] * 4,
                      indexer_types=["full", "shared", "full", "shared"])
        plan = build_plan(config, self.settings)
        self.assertEqual(plan.indexer_owners, (0, 0, 2, 2))
        self.assertEqual(plan.to_dict()["cache"]["dsa_index_key_bytes"], 2 * 32 * 8 * 4)

    def test_alternating_row_blocks_and_cpu_only_never_allocate_cuda(self):
        plan = build_plan(self.config, replace(self.settings, device="hybrid",
                                              vram_budget_bytes=1024**3))
        self.assertEqual([plan.device_for_block(i) for i in range(6)],
                         ["cpu", "cuda", "cpu", "cuda", "cpu", "cuda"])
        cpu = build_plan(self.config, self.settings)
        self.assertEqual([cpu.device_for_block(i) for i in range(3)], ["cpu"] * 3)
        self.assertEqual(cpu.vram_required_bytes, 0)
        with self.assertRaises(BudgetExceededError):
            cpu.allocator().reserve(1, device="cuda")
        for value in (-1, True, 1.5):
            with self.assertRaises(ResidencyError):
                plan.device_for_block(value)

    def test_ram_and_vram_exact_budget_boundary(self):
        plan = build_plan(self.config, self.settings)
        build_plan(self.config, replace(self.settings, ram_budget_bytes=plan.ram_required_bytes))
        with self.assertRaises(BudgetExceededError):
            build_plan(self.config, replace(self.settings, ram_budget_bytes=plan.ram_required_bytes - 1))
        hybrid = replace(self.settings, device="hybrid", vram_budget_bytes=1024**3)
        gpu = build_plan(self.config, hybrid)
        build_plan(self.config, replace(hybrid, vram_budget_bytes=gpu.vram_required_bytes))
        with self.assertRaises(BudgetExceededError) as caught:
            build_plan(self.config, replace(hybrid, vram_budget_bytes=gpu.vram_required_bytes - 1))
        self.assertEqual(caught.exception.device, "cuda")

    def test_row_band_cache_is_included_in_estimate_and_exact_budget(self):
        from glm_local.runtime_io import DEFAULT_ROW_BAND_CACHE_BYTES
        plan = build_plan(self.config, self.settings)
        self.assertEqual(dict(plan.cpu_components)["encoded_row_band_cache"], DEFAULT_ROW_BAND_CACHE_BYTES)
        self.assertEqual(plan.to_dict()["streaming"]["max_encoded_row_band_cache_bytes"], DEFAULT_ROW_BAND_CACHE_BYTES)
        # A budget that would fit without retained encoded bands must now reject.
        with self.assertRaises(BudgetExceededError):
            build_plan(self.config, replace(self.settings,
                ram_budget_bytes=plan.ram_required_bytes - DEFAULT_ROW_BAND_CACHE_BYTES))

    def test_nvfp4_batch_scratch_is_reserved_in_the_plan(self):
        from glm_local.nvfp4_execution import NVFP4_ROW_BAND_SCRATCH_BYTES
        from test_nvfp4_schema import nvfp4_config
        config = {**self.config, **nvfp4_config()}
        plan = build_plan(config, self.settings)
        self.assertEqual(dict(plan.cpu_components)["nvfp4_row_band_scratch"], NVFP4_ROW_BAND_SCRATCH_BYTES)
        with self.assertRaises(BudgetExceededError):
            build_plan(config, replace(self.settings,
                ram_budget_bytes=plan.ram_required_bytes - NVFP4_ROW_BAND_SCRATCH_BYTES))

    def test_context_and_generation_bounds(self):
        plan = build_plan(self.config, self.settings, prompt_tokens=28)
        plan.validate_prompt(32, max_new_tokens=0)
        for prompt, generated in ((29, 4), (1, 5), (0, 0), (True, 1), (2, -1)):
            with self.subTest(prompt=prompt, generated=generated):
                with self.assertRaises(ResidencyError):
                    plan.validate_prompt(prompt, generated)
        with self.assertRaises(ResidencyError):
            build_plan(self.config, replace(self.settings, context_tokens=129))
        with self.assertRaises(ResidencyError):
            build_plan(self.config, replace(self.settings, max_new_tokens=33))

    def test_invalid_dimensions_and_schedules_are_not_defaulted(self):
        for key, value in (("kv_lora_rank", 0), ("hidden_size", True),
                           ("qk_rope_head_dim", 3), ("index_head_dim", 2),
                           ("num_experts_per_tok", 5), ("index_topk", None),
                           ("max_position_embeddings", None),
                           ("indexer_types", ["shared", "full"]),
                           ("layer_types", None)):
            config = deepcopy(self.config)
            config[key] = value
            with self.subTest(field=key):
                with self.assertRaises(ResidencyError):
                    build_plan(config, self.settings)
        with self.assertRaises(ResidencyError):
            build_plan(None, self.settings)

    def test_setting_types_and_unsupported_storage_are_rejected(self):
        for key, value in (("ram_budget_bytes", True), ("vram_budget_bytes", -1),
                           ("runtime_headroom_bytes", -1), ("context_tokens", 0),
                           ("gpu_headroom_bytes", float("nan")), ("device", "cuda"),
                           ("block_rows", 64), ("block_rows", 128.0),
                           ("cache_dtype", "bfloat16")):
            with self.subTest(field=key):
                with self.assertRaises(ResidencyError):
                    build_plan(self.config, replace(self.settings, **{key: value}))

    def test_plan_and_ledger_snapshots_do_not_mutate_inputs_or_state(self):
        original = deepcopy(self.config)
        plan = build_plan(self.config, self.settings)
        report = plan.to_dict()
        report["ram"]["components"]["mla_latent_cache"] = 0
        self.assertGreater(plan.to_dict()["ram"]["components"]["mla_latent_cache"], 0)
        self.assertEqual(self.config, original)
        ledger = plan.allocator()
        self.assertEqual(ledger.snapshot()["cpu"]["baseline_bytes"],
                         plan.cache_bytes + self.settings.runtime_headroom_bytes)


class ReservationTests(unittest.TestCase):
    def test_reserves_immediately_releases_on_exception_and_keeps_peak(self):
        ledger = ReservationLedger(100, 80, base_cpu_bytes=10)
        lease = ledger.reserve(30)
        self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], 40)
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            with lease:
                with ledger.reserve(70, device="cuda"):
                    raise RuntimeError("allocation failed")
        state = ledger.snapshot()
        self.assertEqual(state["active_leases"], 0)
        self.assertEqual(state["cpu"]["used_bytes"], 10)
        self.assertEqual(state["cpu"]["peak_bytes"], 40)
        self.assertEqual(state["cuda"]["used_bytes"], 0)
        self.assertEqual(state["cuda"]["peak_bytes"], 70)

    def test_failed_reservations_do_not_change_ledger(self):
        ledger = ReservationLedger(10)
        baseline = ledger.snapshot()
        for value in (0, -1, True, 11, 2**63, 1.5):
            with self.assertRaises(ResidencyError):
                ledger.reserve(value)
            self.assertEqual(ledger.snapshot(), baseline)
        for device in ("gpu", "cuda:0", None, True):
            with self.assertRaises(ResidencyError):
                ledger.reserve(1, device=device)
            self.assertEqual(ledger.snapshot(), baseline)

    def test_baseline_cannot_exceed_budget(self):
        for kwargs in ({"base_cpu_bytes": 11}, {"base_cuda_bytes": 1}):
            with self.assertRaises(BudgetExceededError):
                ReservationLedger(10, **kwargs)

    def test_double_release_cross_ledger_and_reentry_fail_without_corruption(self):
        ledger, other = ReservationLedger(10), ReservationLedger(10)
        lease = ledger.reserve(10)
        with self.assertRaises(ResidencyError):
            other.release(lease)
        with lease:
            with self.assertRaises(ResidencyError):
                lease.__enter__()
        before = ledger.snapshot()
        with self.assertRaises(ResidencyError):
            lease.release()
        with self.assertRaises(ResidencyError):
            lease.__enter__()
        self.assertEqual(ledger.snapshot(), before)

    def test_active_lease_count_and_labels_are_bounded(self):
        ledger = ReservationLedger(100, max_active_leases=1)
        with ledger.reserve(1, label="weight tile"):
            before = ledger.snapshot()
            with self.assertRaises(ResidencyError):
                ledger.reserve(1)
            self.assertEqual(ledger.snapshot(), before)
        for label in ("", "x" * 129, "a\nb", 3):
            with self.assertRaises(ResidencyError):
                ledger.reserve(1, label=label)

    def test_concurrent_reservations_cannot_overcommit(self):
        ledger = ReservationLedger(10)
        def reserve_one(_):
            try:
                return ledger.reserve(1)
            except BudgetExceededError:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            leases = [lease for lease in pool.map(reserve_one, range(100)) if lease]
        self.assertEqual(len(leases), 10)
        self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], 10)
        for lease in leases:
            lease.release()
        self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
