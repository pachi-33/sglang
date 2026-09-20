"""Discovery and stable registration of MoE trace decision sites.

The trace recorder intentionally knows only a module carrying
``_moe_trace_site_id``.  Keeping model discovery here avoids coupling the hot
router paths to model-family-specific layer bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.moe_trace.types import MoeTraceSite


@dataclass(frozen=True)
class _MoeTraceBinding:
    """One routed FusedMoE and the module where its routing is observable."""

    fused_module_path: str
    capture_module_path: str
    capture_module: torch.nn.Module
    site: MoeTraceSite
    supports_activation: bool
    supports_route: bool


def _has_class_in_mro(module: torch.nn.Module, class_name: str) -> bool:
    """Recognize model classes without importing every model implementation.

    Registration happens after model construction, and importing FusedMoE (and
    its backend graph) here would make this small metadata helper expensive and
    prone to import cycles.  MRO matching still recognizes FusedMoE subclasses.
    """
    return any(cls.__name__ == class_name for cls in type(module).__mro__)


def _router_kind(module: torch.nn.Module) -> tuple[str, bool, bool]:
    if _has_class_in_mro(module, "TopK"):
        return "topk", True, True
    if _has_class_in_mro(module, "HashTopK"):
        return "hash-topk", True, True
    if _has_class_in_mro(module, "InklingGate"):
        return "inkling-gate", True, True
    # Importing transformers.py from this registry creates an import cycle;
    # this bridge is unique and has the needed route-tap contract by name.
    if type(module).__name__ == "TransformersFusedMoE":
        return "transformers-bridge", False, True
    raise TypeError(f"Unsupported MoE trace router: {type(module).__qualname__}")


def _parent_paths(path: str) -> list[str]:
    """Immediate parent first, ending with the root module path."""
    parents: list[str] = []
    current = path.rpartition(".")[0]
    while True:
        parents.append(current)
        if not current:
            return parents
        current = current.rpartition(".")[0]


def _module_attr(module: torch.nn.Module, name: str, default=None):
    value = getattr(module, name, default)
    return default if value is None else value


def _site_for(
    capture_path: str,
    capture_module: torch.nn.Module,
    fused_module: torch.nn.Module,
) -> MoeTraceSite:
    router_type, _activation, _route = _router_kind(capture_module)
    # A router's layer_id is normally definitive.  GraniteMoe's TopK has no
    # layer_id, however, so fall back to its sibling FusedMoE.
    layer_id = _module_attr(capture_module, "layer_id")
    if layer_id is None:
        layer_id = _module_attr(fused_module, "layer_id")
    if layer_id is None:
        raise ValueError(f"Cannot determine MoE layer_id for router {capture_path!r}.")

    hidden_size = _module_attr(fused_module, "hidden_size")
    num_experts = _module_attr(capture_module, "num_experts")
    if num_experts is None:
        num_experts = _module_attr(fused_module, "num_experts")
    top_k = _module_attr(capture_module, "top_k")
    if _has_class_in_mro(capture_module, "TopK"):
        # Trace records routed decisions only.  Fused shared-expert columns
        # are intentionally excluded at the tap, so do not reserve them in
        # the recorder's fixed route width either.
        top_k = capture_module.topk_config.top_k - getattr(
            capture_module.topk_config, "num_fused_shared_experts", 0
        )
    elif _has_class_in_mro(capture_module, "HashTopK"):
        top_k = capture_module.topk - capture_module.num_fused_shared_experts
    if top_k is None:
        top_k = _module_attr(fused_module, "top_k")
    if hidden_size is None or num_experts is None or top_k is None:
        raise ValueError(
            "Cannot determine MoE trace layout for "
            f"router {capture_path!r} and FusedMoE {fused_module!r}."
        )
    return MoeTraceSite(
        site_id=-1,
        module_path=capture_path,
        layer_id=int(layer_id),
        router_type=router_type,
        hidden_size=int(hidden_size),
        num_experts=int(num_experts),
        top_k=int(top_k),
    )


def _discover_moe_trace_bindings(model: torch.nn.Module) -> list[_MoeTraceBinding]:
    """Find router/FusedMoE pairs without modifying ``model``.

    Public TopK/HashTopK and InklingGate are siblings of their routed experts.
    TransformersFusedMoE is an adapter containing the FusedMoE, so it is a
    parent of the experts rather than a sibling.  Searching ancestor containers
    from nearest to farthest handles both forms deterministically.
    """
    modules = dict(model.named_modules())
    routers = {
        path: module
        for path, module in modules.items()
        if _has_class_in_mro(module, "TopK")
        or _has_class_in_mro(module, "HashTopK")
        or _has_class_in_mro(module, "InklingGate")
        or type(module).__name__ == "TransformersFusedMoE"
    }
    bindings: list[_MoeTraceBinding] = []
    for fused_path, fused_module in modules.items():
        if not _has_class_in_mro(fused_module, "FusedMoE"):
            continue
        if getattr(fused_module, "is_shared_fused_moe", False):
            continue

        candidates: list[tuple[str, torch.nn.Module]] = []
        for parent in _parent_paths(fused_path):
            direct = [
                (path, module)
                for path, module in routers.items()
                if path.rpartition(".")[0] == parent
            ]
            if direct:
                candidates = direct
                break
        if len(candidates) != 1:
            # Keep an invalid binding out of the result: strict registration
            # reports the routed FusedMoE path, while non-strict callers get
            # every unambiguous site.
            continue
        capture_path, capture_module = candidates[0]
        router_type, supports_activation, supports_route = _router_kind(capture_module)
        del router_type
        bindings.append(
            _MoeTraceBinding(
                fused_module_path=fused_path,
                capture_module_path=capture_path,
                capture_module=capture_module,
                site=_site_for(capture_path, capture_module, fused_module),
                supports_activation=supports_activation,
                supports_route=supports_route,
            )
        )
    return bindings


def register_moe_trace_sites(
    model: torch.nn.Module,
    *,
    require_activations: bool,
    require_routes: bool,
    strict: bool = True,
) -> list[MoeTraceSite]:
    """Bind stable site IDs to all traced routed-MoE router modules.

    ``strict`` first guarantees every routed ``FusedMoE`` has exactly one
    observable router, then enforces requested activation/route capabilities.
    """
    modules = dict(model.named_modules())
    expected_fused = [
        path
        for path, module in modules.items()
        if _has_class_in_mro(module, "FusedMoE")
        and not getattr(module, "is_shared_fused_moe", False)
    ]
    bindings = _discover_moe_trace_bindings(model)
    by_fused = {binding.fused_module_path: binding for binding in bindings}
    if strict:
        unbound = sorted(set(expected_fused) - set(by_fused))
        if unbound:
            raise ValueError(
                "MoE trace router coverage requires exactly one router site for "
                "each routed FusedMoE; missing or ambiguous: " + ", ".join(unbound)
            )
        if require_activations:
            missing = sorted(
                binding.fused_module_path
                for binding in bindings
                if not binding.supports_activation
            )
            if missing:
                raise ValueError(
                    "MoE trace activation coverage is unavailable for routed "
                    "FusedMoE modules: " + ", ".join(missing)
                )
        if require_routes:
            missing = sorted(
                binding.fused_module_path
                for binding in bindings
                if not binding.supports_route
            )
            if missing:
                raise ValueError(
                    "MoE trace route coverage is unavailable for routed FusedMoE "
                    "modules: " + ", ".join(missing)
                )
            # Public TopK has route taps only for the STANDARD materialized
            # output.  Opaque backends must fail at startup: computing a
            # second shadow top-k would not necessarily be the route consumed
            # by the experts.
            from sglang.srt.layers.moe import get_moe_runner_backend

            backend = get_moe_runner_backend()
            opaque: list[str] = []
            for binding in bindings:
                module = binding.capture_module
                if not _has_class_in_mro(module, "TopK"):
                    continue
                configured = getattr(module.topk_config, "output_format", None)
                configured_name = getattr(configured, "name", None)
                if configured_name is not None:
                    visible = configured_name == "STANDARD"
                else:
                    visible = not (
                        backend.is_triton_kernels()
                        or backend.is_experimental_sgl_trtllm()
                        or backend.is_flashinfer_trtllm()
                        or (
                            backend.is_flashinfer_mxfp4()
                            and not getattr(module, "is_fp4_experts", False)
                        )
                    )
                if not visible:
                    opaque.append(binding.capture_module_path)
            if opaque:
                raise ValueError(
                    "MoE trace route capture requires materialized logical IDs "
                    "and weights; the active MoE backend is opaque at: "
                    + ", ".join(sorted(opaque))
                )

    ordered = sorted(
        bindings,
        key=lambda binding: (binding.site.layer_id, binding.site.module_path),
    )
    sites: list[MoeTraceSite] = []
    for site_id, binding in enumerate(ordered):
        site = MoeTraceSite(
            site_id=site_id,
            module_path=binding.site.module_path,
            layer_id=binding.site.layer_id,
            router_type=binding.site.router_type,
            hidden_size=binding.site.hidden_size,
            num_experts=binding.site.num_experts,
            top_k=binding.site.top_k,
        )
        binding.capture_module._moe_trace_site_id = site_id
        binding.capture_module._moe_trace_site = site
        sites.append(site)
    return sites
