# SPDX-License-Identifier: Apache-2.0
"""Generic CPU-resident source support for expert-at-a-time MoE loading."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Mapping, Optional

import torch

from sglang.srt.layers.moe.cpu_memory_expert_backend import CpuMemoryExpertBackend
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase


@dataclass(frozen=True)
class ExpertOffloadTensorSpec:
    """One expert-major checkpoint destination and the shards that fill it."""

    name: str
    shard_ids: tuple[str, ...]
    expert_axis: int = 0


@dataclass(frozen=True)
class ExpertOffloadSpec:
    """Capability declaration for a method that can retain experts on CPU.

    The declared tensors must have the local expert as dimension zero both
    before and after ``process_weights_after_loading``.  This keeps the cache
    ABI independent of any particular quantization format.

    Declaring expert-major tensors only guarantees that one expert can be
    sliced and transferred independently.  ``supports_slot_remap`` is the
    separate execution-time guarantee that ``inner.apply`` can consume those
    tensors in a compact cache layout and interpret routed expert IDs as
    physical cache-slot indices instead of logical expert IDs.
    """

    tensors: tuple[ExpertOffloadTensorSpec, ...]
    # True only when every expert-indexed piece of runtime state is either read
    # from the supplied layer on each apply or can be rebuilt against its cache
    # tensors and cache-slot expert count.  A method must leave this False when
    # an opaque runner retains the original weight pointers, expert count, or
    # logical-ID mapping, because CachedExpertLayerView cannot replace that
    # hidden state.
    supports_slot_remap: bool = False


@dataclass(frozen=True)
class ExpertMicrobatchPlan:
    """Contiguous prefill ranges and their one optional CPU routing snapshot.

    The CPU copy is deliberately owned by the plan rather than each routed
    block.  It is transient metadata for cache admission only; kernels still
    receive the original GPU ``topk_ids`` until ``apply`` remaps a block.
    """

    ranges: list[tuple[int, int]]
    ids_cpu: torch.Tensor | None


class ExpertWeightCoverage:
    """Fail-closed coverage accounting around the existing FusedMoE loader."""

    def __init__(self, spec: ExpertOffloadSpec, num_experts: int):
        self._spec = spec
        self._num_experts = num_experts
        self._parameter_names: dict[int, str] = {}
        self._seen: dict[tuple[str, int], set[str]] = {}
        self._invalid_calls: list[str] = []

    def bind(self, layer: torch.nn.Module) -> None:
        """Associate declared tensor names with parameters just created by inner."""
        for tensor_spec in self._spec.tensors:
            parameter = getattr(layer, tensor_spec.name, None)
            if not isinstance(parameter, torch.Tensor):
                raise RuntimeError(
                    f"expert offload expected parameter {tensor_spec.name!r} "
                    f"on {type(layer).__name__}"
                )
            self._parameter_names[id(parameter)] = tensor_spec.name

    def record(
        self,
        layer: torch.nn.Module,
        parameter: torch.Tensor,
        shard_id: str,
        expert_id: Optional[int],
    ) -> None:
        name = self._parameter_names.get(id(parameter))
        if name is None:
            return
        if expert_id is None:
            # A bulk form may be valid for some loaders, but this milestone has
            # no generic proof that every expert and shard was written.
            self._invalid_calls.append(f"{name}: bulk expert load")
            return

        local_expert_id = expert_id
        map_expert = getattr(layer, "_map_global_expert_id_to_local_expert_id", None)
        if callable(map_expert):
            local_expert_id = map_expert(expert_id)
        if not 0 <= local_expert_id < self._num_experts:
            # The original loader may intentionally ignore a non-local expert.
            return
        self._seen.setdefault((name, local_expert_id), set()).add(shard_id)

    def assert_complete(self) -> None:
        if self._invalid_calls:
            raise RuntimeError(
                "expert offload does not support unverified bulk expert loading: "
                + ", ".join(self._invalid_calls)
            )
        missing: list[str] = []
        for tensor_spec in self._spec.tensors:
            expected_shards = set(tensor_spec.shard_ids)
            for expert_id in range(self._num_experts):
                absent = expected_shards - self._seen.get(
                    (tensor_spec.name, expert_id), set()
                )
                if absent:
                    missing.append(
                        f"{tensor_spec.name}[{expert_id}] missing {sorted(absent)}"
                    )
        if missing:
            raise RuntimeError(
                "expert offload checkpoint coverage is incomplete: "
                + "; ".join(missing[:8])
                + ("; ..." if len(missing) > 8 else "")
            )


class CachedExpertLayerView:
    """Read-only layer proxy for GPU cache physical expert slots.

    ``inner.apply`` continues to see its normal attribute names.  The runtime
    will populate ``expert_tensors`` with GPU cache tensors and rewrite routed
    IDs to their physical slots, instead of mutating the CPU source parameters.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        expert_tensors: Mapping[str, torch.Tensor],
        attribute_overrides: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, "_layer", layer)
        overrides = dict(attribute_overrides or {})
        overrides.update(expert_tensors)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_layer"), name)


class OffloadedFusedMoEMethod(torch.nn.Module, FusedMoEMethodBase):
    """CPU-source wrapper for a capability-declared fused MoE method.

    It preserves the model's ``load_weights`` and each parameter's original
    ``weight_loader``.  The wrapper merely observes those calls for exact
    coverage, then delegates existing post-load transforms on a staged device.
    A cache backend may later provide physical slot tensors to ``inner.apply``.
    """

    # Expert sources are pageable; only a runtime staging ring may be pinned.
    post_load_pin_memory = False

    def __init__(self, inner: FusedMoEMethodBase, context: ExpertOffloadContext):
        # Some fused methods (notably the unquantized method) are nn.Modules.
        # FusedMoE has already registered them as ``quant_method`` children,
        # so this wrapper must itself be an nn.Module before it can replace
        # that attribute.  Keep ``inner`` unregistered: it is an implementation
        # delegate, not a second model subtree to traverse during post-load.
        torch.nn.Module.__init__(self)
        object.__setattr__(self, "inner", inner)
        self.context = context
        self.spec = inner.get_expert_offload_spec()
        if self.spec is None:
            raise ValueError(f"{type(inner).__name__} has no expert offload capability")
        if not self.spec.supports_slot_remap:
            raise ValueError(
                f"{type(inner).__name__} does not support physical expert-slot "
                "remapping"
            )
        unsupported_axes = [
            tensor_spec.name
            for tensor_spec in self.spec.tensors
            if tensor_spec.expert_axis != 0
        ]
        if unsupported_axes:
            raise ValueError(
                "expert offload requires expert_axis=0, got "
                + ", ".join(unsupported_axes)
            )
        self.coverage: ExpertWeightCoverage | None = None
        self.source_tensors: dict[str, torch.Tensor] = {}
        self.layer_id: int | None = None
        self.runtime_view: CachedExpertLayerView | None = None
        self.pool = None
        # A ContextVar makes planned IDs exception-safe and does not leak across
        # concurrent task contexts.  ``planned_ids`` restores its token even if
        # dispatch, the MoE kernel, or combine raises.
        self._planned_ids_cpu: ContextVar[torch.Tensor | None] = ContextVar(
            "planned_expert_ids_cpu", default=None
        )

    def __getattr__(self, name: str):
        # FusedMoE and some runners consult method-specific flags.  Preserve
        # that ABI while keeping this wrapper quantization-format agnostic.
        try:
            return torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "inner"), name)

    @property
    def runner(self):
        # FusedMoEMethodBase has ``runner = None`` as a class attribute, so
        # ``__getattr__`` alone cannot forward this lookup to the inner method.
        return self.inner.runner

    @runner.setter
    def runner(self, value) -> None:
        self.inner.runner = value

    def create_weights(self, layer: torch.nn.Module, **kwargs) -> None:
        original_weight_loader = kwargs.get("weight_loader")
        if original_weight_loader is None:
            raise RuntimeError("expert offload requires the original weight_loader")
        if "expert_id" not in inspect.signature(original_weight_loader).parameters:
            raise RuntimeError(
                "expert offload requires a per-expert weight_loader; fused bulk "
                "loading cannot provide exact coverage"
            )
        num_experts = kwargs["num_experts"]
        self.coverage = ExpertWeightCoverage(self.spec, num_experts)

        def covered_weight_loader(
            parameter: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: Optional[int],
        ):
            result = original_weight_loader(
                parameter, loaded_weight, weight_name, shard_id, expert_id
            )
            # Record after the original loader so the original semantics remain
            # authoritative; this wrapper neither repacks nor writes weights.
            assert self.coverage is not None
            self.coverage.record(layer, parameter, shard_id, expert_id)
            # Some model loaders use a weight-loader return value to signal
            # that a checkpoint entry was consumed.  Observing coverage must
            # remain transparent to that established contract.
            return result

        kwargs["weight_loader"] = covered_weight_loader
        # Quant methods normally inherit the surrounding default CUDA device.
        # Sources for this backend must instead be born on CPU before loading.
        with torch.device("cpu"):
            self.inner.create_weights(layer=layer, **kwargs)
        self.coverage.bind(layer)
        self.context.register(self)

    def create_moe_runner(self, layer, moe_runner_config) -> None:
        self.inner.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.inner.process_weights_after_loading(layer)

    def finalize_weight_loading(self, layer: torch.nn.Module) -> None:
        if self.coverage is None:
            raise RuntimeError("expert offload weights were never created")
        self.coverage.assert_complete()

    def finalize_post_load(self, layer: torch.nn.Module) -> None:
        """Capture validated CPU sources after the staging context has exited."""
        names = {tensor_spec.name for tensor_spec in self.spec.tensors}
        named_side_tensors = getattr(layer, "named_per_expert_tensors", None)
        if callable(named_side_tensors):
            names.update(
                name for name, _ in named_side_tensors(layer.num_local_experts)
            )

        sources: dict[str, torch.Tensor] = {}
        for name in sorted(names):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"expert offload source {name!r} was not retained")
            if tensor.device.type != "cpu":
                raise RuntimeError(
                    f"expert offload source {name!r} remained on {tensor.device}"
                )
            if tensor.is_pinned():
                raise RuntimeError(
                    f"expert offload source {name!r} must be pageable CPU memory"
                )
            if tensor.ndim == 0 or tensor.shape[0] != layer.num_local_experts:
                raise RuntimeError(
                    f"expert offload source {name!r} is not expert-major: "
                    f"shape={tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous():
                raise RuntimeError(f"expert offload source {name!r} is not contiguous")
            sources[name] = tensor
        self.source_tensors = sources
        self.layer_id = layer.layer_id
        self.context.register_post_load_sources(layer, sources)

    def make_cached_layer_view(
        self, layer: torch.nn.Module, expert_tensors: Mapping[str, torch.Tensor]
    ) -> CachedExpertLayerView:
        """Construct the proxy a later cache backend will pass to ``inner.apply``."""
        if not expert_tensors:
            raise ValueError("cached expert view requires at least one tensor")
        return CachedExpertLayerView(layer, expert_tensors)

    def bind_runtime(self, layer: torch.nn.Module) -> None:
        """Bind physical cache slots without changing dispatcher expert counts."""
        pool = self.context.backend.pools.get(layer.layer_id)
        if pool is None:
            raise RuntimeError("expert cache pool has not been initialized")
        self.pool = pool
        runtime_config = replace(
            layer.moe_runner_config,
            num_experts=pool.cache_slots,
            num_local_experts=pool.cache_slots,
        )
        self.runtime_view = CachedExpertLayerView(
            layer,
            pool.cache_tensors,
            {
                "num_experts": pool.cache_slots,
                "num_local_experts": pool.cache_slots,
                "moe_runner_config": runtime_config,
            },
        )
        self.inner.create_moe_runner(self.runtime_view, runtime_config)
        runner = self.inner.runner
        if runner is None:
            scheme = getattr(self.runtime_view, "scheme", None)
            runner = getattr(getattr(scheme, "kernel", None), "runner", None)
            if runner is not None:
                self.inner.runner = runner
        layer.runner = runner

    def plan_prefill_microbatches(
        self, topk_output: StandardTopKOutput
    ) -> ExpertMicrobatchPlan:
        """Greedily keep each contiguous token range within cache capacity."""
        if not isinstance(topk_output, StandardTopKOutput):
            raise NotImplementedError("CPU expert offload requires StandardTopKOutput")
        topk_ids = topk_output.topk_ids
        if self.pool is None or topk_ids.numel() <= self.pool.cache_slots:
            # The one-block path does not need routing inspection, so avoid an
            # otherwise needless device-to-host synchronization and copy.
            return ExpertMicrobatchPlan([(0, topk_ids.shape[0])], None)
        ids_cpu = topk_ids.detach().to(device="cpu", copy=True)
        result: list[tuple[int, int]] = []
        start = 0
        active: set[int] = set()
        for token_index, row in enumerate(ids_cpu.tolist()):
            row_ids = {expert_id for expert_id in row if expert_id >= 0}
            if len(row_ids) > self.pool.cache_slots:
                raise RuntimeError("one token requires more experts than cache slots")
            if active and len(active | row_ids) > self.pool.cache_slots:
                result.append((start, token_index))
                start = token_index
                active = set()
            active.update(row_ids)
        if start < topk_ids.shape[0]:
            result.append((start, topk_ids.shape[0]))
        return ExpertMicrobatchPlan(result, ids_cpu)

    @contextmanager
    def planned_ids(self, ids_cpu: torch.Tensor | None):
        """Scope a plan's CPU IDs to exactly one dispatched microbatch."""
        token = self._planned_ids_cpu.set(ids_cpu)
        try:
            yield
        finally:
            self._planned_ids_cpu.reset(token)

    def apply(self, layer, dispatch_output):
        if self.pool is None or self.runtime_view is None:
            raise RuntimeError("CPU expert offload runtime has not been bound")
        if not isinstance(dispatch_output, StandardDispatchOutput) or not isinstance(
            dispatch_output.topk_output, StandardTopKOutput
        ):
            raise NotImplementedError(
                "CPU expert offload supports standard dispatch only"
            )
        topk = dispatch_output.topk_output
        # A prefill plan already made one full GPU->CPU routing copy to find
        # ranges.  Its caller scopes the matching slice here.  Non-planned
        # calls retain the standalone behavior and take their own snapshot.
        ids_cpu = self._planned_ids_cpu.get()
        if ids_cpu is None:
            ids_cpu = topk.topk_ids.detach().to(device="cpu", copy=True)
        elif ids_cpu.shape != topk.topk_ids.shape:
            raise RuntimeError(
                "planned expert IDs do not match the dispatched microbatch"
            )
        valid_ids = ids_cpu[ids_cpu >= 0]
        if valid_ids.numel() == 0:
            return self.inner.apply(self.runtime_view, dispatch_output)
        compute_stream = torch.cuda.current_stream(self.context.backend.target_device)
        lease = self.pool.acquire(
            valid_ids.tolist(),
            self.context.backend.transfer_stream,
            compute_stream,
        )
        try:
            # Remapping can allocate or issue a host-to-device copy.  Keep it
            # inside the lease boundary: a failure before inner.apply must not
            # leave a victim slot permanently protected by an active lease.
            remapped_cpu = ids_cpu.clone()
            # Marlin's expert_map selects entries within its full packed layout;
            # it cannot translate logical router IDs into this cache's compact
            # physical slots, so the dispatcher input must be remapped here.
            for logical, physical in lease.logical_to_physical.items():
                remapped_cpu[ids_cpu == logical] = physical
            remapped_ids = remapped_cpu.to(
                device=topk.topk_ids.device, dtype=topk.topk_ids.dtype
            )
            remapped_topk = topk._replace(topk_ids=remapped_ids)
            remapped_dispatch = dispatch_output._replace(topk_output=remapped_topk)
            return self.inner.apply(self.runtime_view, remapped_dispatch)
        finally:
            lease.release_after(compute_stream)

    def get_expert_offload_spec(self):
        return self.spec


_ACTIVE_EXPERT_OFFLOAD_CONTEXT: ContextVar[ExpertOffloadContext | None] = ContextVar(
    "active_expert_offload_context", default=None
)


class ExpertOffloadContext:
    """Opt-in construction scope for generic CPU-resident expert sources."""

    def __init__(self, backend: CpuMemoryExpertBackend | None = None) -> None:
        self.methods: list[OffloadedFusedMoEMethod] = []
        self.backend = backend or CpuMemoryExpertBackend()
        self._token: Token | None = None

    def __enter__(self) -> ExpertOffloadContext:
        self._token = _ACTIVE_EXPERT_OFFLOAD_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._token is not None
        _ACTIVE_EXPERT_OFFLOAD_CONTEXT.reset(self._token)
        self._token = None

    def wrap(self, method: FusedMoEMethodBase) -> FusedMoEMethodBase:
        if method.get_expert_offload_spec() is None:
            # Silently declining to wrap here would let the normal constructor
            # allocate a full GPU expert tensor despite an active CPU backend.
            raise RuntimeError(
                f"expert offload is active but {type(method).__name__} does not "
                "declare an expert-major offload capability"
            )
        return OffloadedFusedMoEMethod(method, self)

    def register(self, method: OffloadedFusedMoEMethod) -> None:
        self.methods.append(method)

    def register_post_load_sources(
        self, layer: torch.nn.Module, sources: Mapping[str, torch.Tensor]
    ) -> None:
        """Transfer source ownership to the backend after CPU validation."""
        self.backend.register_host_layer(
            layer_id=layer.layer_id,
            top_k=layer.top_k,
            num_experts=layer.num_local_experts,
            tensors=sources,
        )

    def bind_runtime(self, layers: Mapping[int, torch.nn.Module]) -> None:
        """Bind pools after ``backend.initialize_cuda`` has fixed slot counts."""
        for method in self.methods:
            layer = layers.get(method.layer_id)
            if layer is None:
                raise RuntimeError("missing FusedMoE layer for expert-cache binding")
            method.bind_runtime(layer)

    def bind_runtime_model(self, model: torch.nn.Module) -> None:
        for method in self.methods:
            matches = [
                module
                for module in model.modules()
                if getattr(module, "quant_method", None) is method
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "expert-cache runtime binding requires exactly one owning "
                    f"FusedMoE module, found {len(matches)}"
                )
            method.bind_runtime(matches[0])


def get_active_expert_offload_context() -> ExpertOffloadContext | None:
    return _ACTIVE_EXPERT_OFFLOAD_CONTEXT.get()
