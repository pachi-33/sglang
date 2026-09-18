"""Mock-only public CPU-memory expert-pack loader regression tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import expert_pack_hook
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers.moe import expert_offload
from sglang.srt.model_loader import expert_pack_loader
from sglang.srt.model_loader.expert_pack_loader import ExpertPackModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-b-test-cpu-intel")


def _args(**overrides):
    values = dict(
        load_format="expert_pack",
        model_loader_extra_config={"source_backend": "cpu_memory"},
        cpu_offload_gb=0,
        tp_size=1,
        dp_size=1,
        ep_size=1,
        enforce_shared_experts_fusion=False,
        enable_waterfill=False,
        cuda_graph_backend_decode=None,
        cuda_graph_backend_prefill=None,
        cuda_graph_config={},
        device="cuda",
        moe_runner_backend="auto",
        moe_a2a_backend="none",
        enable_eplb=False,
        init_expert_location="trivial",
        ep_num_redundant_experts=0,
        enable_lora=False,
        lora_paths=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestCpuMemoryExpertPackPublicConfig(unittest.TestCase):
    def test_cpu_memory_does_not_require_pack_path(self):
        loader = ExpertPackModelLoader(
            LoadConfig(
                load_format=LoadFormat.EXPERT_PACK,
                model_loader_extra_config={"source_backend": "cpu_memory"},
            )
        )
        self.assertEqual(loader.source_backend, "cpu_memory")
        self.assertIsNone(loader.pack_path)

    def test_ssd_default_still_requires_pack_path(self):
        with self.assertRaisesRegex(ValueError, "requires pack_path"):
            ExpertPackModelLoader(LoadConfig(load_format=LoadFormat.EXPERT_PACK))

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ExpertPackModelLoader(
                LoadConfig(
                    load_format=LoadFormat.EXPERT_PACK,
                    model_loader_extra_config={"source_backend": "unknown"},
                )
            )

    def test_cpu_memory_hook_rejects_conflicts(self):
        for key, value in (
            ("cpu_offload_gb", 1),
            ("moe_a2a_backend", "deepep"),
            ("enable_eplb", True),
            ("enable_lora", True),
            ("device", "cpu"),
        ):
            with (
                self.subTest(key=key),
                patch.object(
                    expert_pack_hook,
                    "resolving_view",
                    return_value=_args(**{key: value}),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "cpu_memory"):
                    expert_pack_hook.handle_expert_pack(object())

    def test_cpu_memory_hook_requires_auto_moe_runner_backend(self):
        with patch.object(
            expert_pack_hook,
            "resolving_view",
            return_value=_args(moe_runner_backend="triton"),
        ):
            with self.assertRaisesRegex(ValueError, "moe-runner-backend auto"):
                expert_pack_hook.handle_expert_pack(object())

    def test_cpu_memory_hook_accepts_indexed_cuda_device(self):
        for device in ("cuda", "cuda:0", "cuda:17"):
            with self.subTest(device=device):
                with (
                    patch.object(
                        expert_pack_hook,
                        "resolving_view",
                        return_value=_args(device=device),
                    ),
                    patch.object(expert_pack_hook, "declare_resolution"),
                ):
                    expert_pack_hook.handle_expert_pack(object())

    def test_cpu_memory_hook_resolves_runtime_overrides(self):
        declared = {}

        def declare(_args, _name, **kwargs):
            declared.update(kwargs)

        with (
            patch.object(expert_pack_hook, "resolving_view", return_value=_args()),
            patch.object(expert_pack_hook, "declare_resolution", side_effect=declare),
        ):
            expert_pack_hook.handle_expert_pack(object())
        self.assertEqual(declared["disable_cuda_graph"], True)
        self.assertEqual(declared["disable_shared_experts_fusion"], True)
        self.assertEqual(declared["max_running_requests"], 1)

    def test_cpu_memory_uses_isolated_auto_checkpoint_loader(self):
        captured = {}

        class FakeDefaultLoader:
            def __init__(self, load_config):
                captured["baseline_config"] = load_config

            def _get_all_weights(self, model_config, model):
                return ()

            @staticmethod
            def load_weights_and_postprocess(model, weights, target_device):
                captured["weights"] = tuple(weights)
                captured["target_device"] = target_device

        class FakeBackend:
            def initialize_cuda(self, **kwargs):
                captured["cache_config"] = kwargs

            def configure_stats(self, path, interval):
                captured["stats_config"] = (path, interval)

            def flush_stats(self, path):
                captured["stats_flush"] = path

        class FakeContext:
            def __init__(self, **kwargs):
                captured["context_config"] = kwargs
                self.backend = FakeBackend()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return None

            def bind_runtime_model(self, model):
                captured["bound_model"] = model

        public_config = LoadConfig(
            load_format=LoadFormat.EXPERT_PACK,
            model_loader_extra_config={
                "source_backend": "cpu_memory",
                "pin_host_experts": True,
                "cache_vram_mib": 64,
                "cache_vram_reserve_mib": 32,
                "stage_slots": 2,
                "stats_flush_interval": 0,
                "stats_path": "/tmp/cpu-memory-test-stats.json",
                "enable_multithread_load": False,
                "num_threads": 1,
            },
        )
        loader = ExpertPackModelLoader(public_config)
        model_config = SimpleNamespace(dtype=torch.float16)
        device_config = SimpleNamespace(device="cuda")
        fake_model = torch.nn.Module()
        safe_parallel = SimpleNamespace(tp_size=1, moe_dp_size=1, moe_ep_size=1)
        safe_exec = SimpleNamespace(
            graph=SimpleNamespace(disable_cuda_graph=True),
            moe=SimpleNamespace(disable_shared_experts_fusion=True),
        )

        with (
            patch.object(expert_pack_loader, "DefaultModelLoader", FakeDefaultLoader),
            patch.object(
                expert_pack_loader, "_get_quantization_config", return_value=object()
            ),
            patch.object(
                expert_pack_loader, "_initialize_model", return_value=fake_model
            ),
            patch.object(expert_offload, "ExpertOffloadContext", FakeContext),
            patch.object(
                expert_pack_loader, "get_parallel", return_value=safe_parallel
            ),
            patch.object(expert_pack_loader, "get_exec", return_value=safe_exec),
        ):
            loaded = loader.load_model(
                model_config=model_config, device_config=device_config
            )

        self.assertIs(loaded, fake_model)
        baseline_config = captured["baseline_config"]
        self.assertEqual(baseline_config.load_format, LoadFormat.AUTO)
        self.assertEqual(
            baseline_config.model_loader_extra_config,
            {"enable_multithread_load": False, "num_threads": 1},
        )
        self.assertIs(public_config.load_format, LoadFormat.EXPERT_PACK)
        self.assertIn("source_backend", public_config.model_loader_extra_config)
        self.assertEqual(captured["cache_config"]["cache_vram_mib"], 64)
        self.assertEqual(
            captured["stats_config"], ("/tmp/cpu-memory-test-stats.json", 0)
        )
        self.assertEqual(captured["context_config"], {"pin_host_experts": True})

    def test_cpu_memory_rejects_non_boolean_pinned_host_expert_config(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            ExpertPackModelLoader(
                LoadConfig(
                    load_format=LoadFormat.EXPERT_PACK,
                    model_loader_extra_config={
                        "source_backend": "cpu_memory",
                        "pin_host_experts": "yes",
                    },
                )
            )

    def test_cpu_memory_loader_rechecks_hook_invariants(self):
        loader = ExpertPackModelLoader(
            LoadConfig(
                load_format=LoadFormat.EXPERT_PACK,
                model_loader_extra_config={"source_backend": "cpu_memory"},
            )
        )
        unsafe_parallel = SimpleNamespace(tp_size=2, moe_dp_size=1, moe_ep_size=1)
        safe_exec = SimpleNamespace(
            graph=SimpleNamespace(disable_cuda_graph=True),
            moe=SimpleNamespace(disable_shared_experts_fusion=True),
        )
        with (
            patch.object(
                expert_pack_loader, "get_parallel", return_value=unsafe_parallel
            ),
            patch.object(expert_pack_loader, "get_exec", return_value=safe_exec),
            self.assertRaisesRegex(RuntimeError, "invariants were not applied"),
        ):
            loader.load_model(
                model_config=SimpleNamespace(dtype=torch.float16),
                device_config=SimpleNamespace(device="cuda"),
            )


if __name__ == "__main__":
    unittest.main()
