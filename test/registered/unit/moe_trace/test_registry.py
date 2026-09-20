import unittest

from torch import nn

from sglang.srt.moe_trace.registry import register_moe_trace_sites


class FusedMoE(nn.Module):
    """Lightweight stand-in preserving the registry's FusedMoE MRO contract."""

    def __init__(self, layer_id: int, *, hidden_size=1024, num_experts=32, top_k=8):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k


class TopK(nn.Module):
    def __init__(self, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.topk_config = type("TopKConfig", (), {"top_k": 8})()


class _GraniteLikeMoE(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        # GraniteMoeMoE deliberately leaves TopK.layer_id unset.
        self.topk = TopK()
        self.experts = FusedMoE(layer_id)


class TestMoeTraceRegistry(unittest.TestCase):
    def test_granite_like_sites_use_sibling_expert_layer_id(self):
        model = nn.Module()
        model.layers = nn.ModuleList([_GraniteLikeMoE(index) for index in range(24)])

        sites = register_moe_trace_sites(
            model, require_activations=True, require_routes=True
        )

        self.assertEqual(len(sites), 24)
        self.assertEqual(
            [(site.layer_id, site.module_path) for site in sites],
            [(index, f"layers.{index}.topk") for index in range(24)],
        )
        self.assertEqual(
            {(site.hidden_size, site.num_experts, site.top_k) for site in sites},
            {(1024, 32, 8)},
        )
        self.assertEqual(
            [layer.topk._moe_trace_site_id for layer in model.layers],
            list(range(24)),
        )

    def test_strict_reports_unbound_fused_moe_path(self):
        model = nn.Module()
        model.unpaired = FusedMoE(3)

        with self.assertRaisesRegex(ValueError, "unpaired"):
            register_moe_trace_sites(
                model, require_activations=False, require_routes=True
            )

    def test_route_only_transformers_bridge_fails_activation_strictness(self):
        transformers_fused_moe = type("TransformersFusedMoE", (nn.Module,), {})
        model = nn.Module()
        model.bridge = transformers_fused_moe()
        model.bridge.num_experts = 8
        model.bridge.top_k = 2
        model.bridge.experts = FusedMoE(1, hidden_size=16, num_experts=8, top_k=2)

        with self.assertRaisesRegex(ValueError, "bridge.experts"):
            register_moe_trace_sites(
                model, require_activations=True, require_routes=True
            )

        sites = register_moe_trace_sites(
            model, require_activations=False, require_routes=True
        )
        self.assertEqual(sites[0].router_type, "transformers-bridge")
        self.assertEqual(sites[0].module_path, "bridge")


if __name__ == "__main__":
    unittest.main()
