"""A small Qwen3.5 layer runner with stateless and single-request paths."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .attention import causal_gqa, causal_gqa_decode
from .checkpoint import Qwen35Checkpoint
from .dense import fp16_embedding, linear_fp16
from .expert_pack.store import ExpertOffloadConfig, ExpertPackStore
from .gdn import (
    chunk_gdn,
    depthwise_conv4_silu,
    depthwise_conv4_silu_decode,
    l2_normalize_qk,
    prepare_gates,
    recurrent_gdn,
    recurrent_gdn_decode,
    recurrent_gdn_short_output,
)
from .model_ops import (
    full_qk_rope_gate,
    gated_attention_fp8,
    gated_gdn_fp8,
    gather_hidden,
    split_full_v,
    split_gdn_qkv,
)
from .moe import MoeWeights, fused_moe
from .ops import gemma_rms_norm, residual_add_gemma_rms_norm
from .quantization import linear_fp8, quantize_fp8
from .weights import Weight


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_offload_checkpoint_identity(
    model_dir: Path, store: ExpertPackStore
) -> None:
    """Bind the manifest to the checkpoint supplying resident weights."""
    source = store.manifest.source
    for filename, field in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ):
        path = model_dir / filename
        expected = source.get(field)
        if not isinstance(expected, str) or _sha256_path(path) != expected:
            raise ValueError(f"ExpertPack source identity does not match {path.name}")
    raw_shards = source.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("ExpertPack manifest has no source shard inventory")
    expected_shards = {
        str(item["file"]): int(item["size"])
        for item in raw_shards
        if isinstance(item, Mapping) and "file" in item and "size" in item
    }
    if len(expected_shards) != len(raw_shards):
        raise ValueError("ExpertPack manifest source shard inventory is malformed")
    for filename, expected_size in expected_shards.items():
        path = model_dir / filename
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(
                f"ExpertPack source shard identity does not match {filename}"
            )


def _require_cuda_matrix(x: torch.Tensor, name: str, columns: int = 2048) -> None:
    if x.ndim != 2 or x.shape[1] != columns or x.dtype != torch.float16:
        raise ValueError(f"{name} must be FP16 [T,{columns}]")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA")


@dataclass
class StatelessLayer:
    layer_id: int
    weights: Mapping[str, torch.Tensor | Weight]

    @property
    def is_gdn(self) -> bool:
        return "linear_attn.in_proj_qkv" in self.weights


@dataclass
class GDNLayerCache:
    """Persistent state for one Gated DeltaNet layer and one request."""

    conv_history: torch.Tensor
    recurrent_state: torch.Tensor


@dataclass
class FullAttentionLayerCache:
    """Continuous, logical-position KV storage for one full-attention layer."""

    key: torch.Tensor
    value: torch.Tensor


LayerCache = GDNLayerCache | FullAttentionLayerCache
RouterCapture = dict[int, tuple[torch.Tensor, torch.Tensor]]


@dataclass
class SingleRequestCache:
    """Explicit cache owned by one runner and one active request at a time."""

    capacity: int
    device: torch.device
    layer_ids: tuple[int, ...]
    layers: dict[int, LayerCache]
    owner_id: int
    consumed_len: int = 0
    poisoned: bool = False


class Qwen35StatelessRunner:
    """Execute a selected original-layer slice.

    ``forward_hidden`` remains the packed, stateless compatibility API.  The
    prefill/decode methods use an explicit :class:`SingleRequestCache`; the
    runner itself never shares prefix state between cache objects or requests.
    """

    def __init__(
        self,
        model_dir: str | Path,
        layer_ids: Iterable[int] = range(4),
        *,
        device: str | torch.device = "cuda",
        load_globals: bool = True,
        expert_offload: ExpertOffloadConfig | None = None,
    ):
        self.model_dir = Path(model_dir)
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("the Qwen3.5 compatibility runner requires CUDA")
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if requested_device.index is None
            else requested_device
        )
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5 compatibility runner requires CUDA")
        capability = torch.cuda.get_device_capability(self.device)
        if capability not in ((7, 0), (8, 9)):
            raise RuntimeError(
                "Qwen3.5 compatibility runner supports only validated "
                f"SM70/SM89 devices, got SM{capability[0]}{capability[1]}"
            )
        if expert_offload is not None:
            if not isinstance(expert_offload, ExpertOffloadConfig):
                raise TypeError("expert_offload must be ExpertOffloadConfig or None")
            if torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "Qwen3.5 expert offload requires exactly one visible CUDA device"
                )
            if capability != (7, 0):
                raise RuntimeError(
                    "Qwen3.5 expert offload is validated only on a single SM70 GPU, "
                    f"got SM{capability[0]}{capability[1]}"
                )
        ids = tuple(layer_ids)
        if (
            not ids
            or tuple(sorted(set(ids))) != ids
            or any(i < 0 or i >= 40 for i in ids)
        ):
            raise ValueError(
                "layer_ids must be unique, ascending original IDs in [0,40)"
            )
        self.layer_ids = ids
        loader = Qwen35Checkpoint(self.model_dir)
        self.expert_offload = expert_offload
        self.expert_store: ExpertPackStore | None = None
        self._closed = False
        store = ExpertPackStore(expert_offload) if expert_offload is not None else None
        try:
            if store is not None:
                _validate_offload_checkpoint_identity(self.model_dir, store)
            self.layers = tuple(
                StatelessLayer(
                    i,
                    loader.load_layer(
                        i,
                        self.device,
                        include_routed_experts=not (
                            expert_offload is not None and 1 <= i <= 38
                        ),
                    ),
                )
                for i in ids
            )
            self._layers_by_id = {layer.layer_id: layer for layer in self.layers}
            self.global_weights = (
                loader.load_global_tensors(self.device) if load_globals else None
            )
            if store is not None:
                store.initialize_device_cache(self.device)
                self.expert_store = store
        except BaseException:
            if store is not None:
                store.close()
            raise

    @property
    def failed(self) -> bool:
        """Whether a fatal ExpertPack error has made this runner unusable."""
        return self.expert_store is not None and self.expert_store.state == "FAILED"

    @property
    def expert_stats(self) -> dict[str, object] | None:
        return None if self.expert_store is None else self.expert_store.snapshot()

    def close(self) -> None:
        """Release ExpertPack resources; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        if self.expert_store is not None:
            self.expert_store.close()

    def _fail_offload(self, error: BaseException) -> None:
        if self.expert_store is not None:
            self.expert_store.fail(error, category="cuda")

    def allocate_request_cache(self, capacity: int = 2048) -> SingleRequestCache:
        """Allocate fixed-capacity, batch-one state for this runner's layers."""
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or not 1 <= capacity <= 2048
        ):
            raise ValueError("cache capacity must be a Python int in [1,2048]")
        caches: dict[int, LayerCache] = {}
        for layer in self.layers:
            if layer.is_gdn:
                caches[layer.layer_id] = GDNLayerCache(
                    conv_history=torch.zeros(
                        (3, 8192), device=self.device, dtype=torch.float16
                    ),
                    recurrent_state=torch.zeros(
                        (32, 128, 128), device=self.device, dtype=torch.float32
                    ),
                )
            else:
                caches[layer.layer_id] = FullAttentionLayerCache(
                    key=torch.empty(
                        (capacity, 2, 256), device=self.device, dtype=torch.float16
                    ),
                    value=torch.empty(
                        (capacity, 2, 256), device=self.device, dtype=torch.float16
                    ),
                )
        return SingleRequestCache(
            capacity=capacity,
            device=self.device,
            layer_ids=self.layer_ids,
            layers=caches,
            owner_id=id(self),
        )

    def _validate_request_cache(self, cache: SingleRequestCache) -> None:
        if not isinstance(cache, SingleRequestCache):
            raise TypeError("cache must be a SingleRequestCache")
        if (
            cache.owner_id != id(self)
            or cache.device != self.device
            or cache.layer_ids != self.layer_ids
        ):
            raise ValueError("cache belongs to a different Qwen3.5 runner")
        if cache.poisoned:
            raise RuntimeError("request cache is poisoned; reset it before reuse")

    def reset_request_cache(self, cache: SingleRequestCache) -> None:
        """Invalidate KV and explicitly clear all recurrent/Conv state."""
        if not isinstance(cache, SingleRequestCache):
            raise TypeError("cache must be a SingleRequestCache")
        if (
            cache.owner_id != id(self)
            or cache.device != self.device
            or cache.layer_ids != self.layer_ids
        ):
            raise ValueError("cache belongs to a different Qwen3.5 runner")
        for layer_cache in cache.layers.values():
            if isinstance(layer_cache, GDNLayerCache):
                layer_cache.conv_history.zero_()
                layer_cache.recurrent_state.zero_()
        cache.consumed_len = 0
        cache.poisoned = False

    def _gdn(
        self,
        hidden: torch.Tensor,
        layer: StatelessLayer,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qkv_z = linear_fp8(a8, w["linear_attn.in_proj_qkv_z"])
        qkv, z = qkv_z[:, :8192], qkv_z[:, 8192:]
        ba = linear_fp16(normed, w["linear_attn.in_proj_ba"])
        b, a = ba[:, :32], ba[:, 32:]
        conv = depthwise_conv4_silu(
            qkv, w["linear_attn.conv1d.weight"], None, cu_seqlens
        )
        q, k, v = split_gdn_qkv(conv)
        q, k = l2_normalize_qk(q, k)
        decay, beta = prepare_gates(
            a, b, w["linear_attn.A_log"], w["linear_attn.dt_bias"]
        )
        if max_seqlen <= 64:
            # The stateless runner only consumes output.  Avoid allocating a
            # [B,32,128,128] state for the ordinary all-short prefill path.
            attended = torch.empty(
                (q.shape[0], 32, 128), device=q.device, dtype=torch.float32
            )
            recurrent_gdn_short_output(
                q, k, v, decay, beta, cu_seqlens, max_seqlen, attended
            )
        else:
            attended, _ = chunk_gdn(q, k, v, decay, beta, cu_seqlens, max_seqlen)
            # A ragged batch's longest document must not choose the numerical
            # recurrence for its short neighbours.  This state-free launch
            # replaces only actual 1..64-token documents in the WY output;
            # long documents retain their WY result and no second state exists.
            recurrent_gdn_short_output(
                q, k, v, decay, beta, cu_seqlens, max_seqlen, attended
            )
        out_a8 = gated_gdn_fp8(attended, z, w["linear_attn.norm.weight"])
        projected = linear_fp8(out_a8, w["linear_attn.out_proj"])
        return projected

    def _full_attention(
        self,
        hidden: torch.Tensor,
        layer: StatelessLayer,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qkv = linear_fp8(a8, w["self_attn.qkv_proj"])
        qgate, k, vflat = qkv[:, :8192], qkv[:, 8192:8704], qkv[:, 8704:]
        q, k, gate = full_qk_rope_gate(
            qgate,
            k,
            positions,
            w["self_attn.q_norm.weight"],
            w["self_attn.k_norm.weight"],
        )
        attended = causal_gqa(
            q, k, vflat.view(vflat.shape[0], 2, 256), cu_seqlens, max_seqlen
        )
        projected = linear_fp8(
            gated_attention_fp8(attended, gate), w["self_attn.o_proj"]
        )
        return projected

    def _gdn_prefill(
        self,
        hidden: torch.Tensor,
        layer: StatelessLayer,
        layer_cache: GDNLayerCache,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Fresh, single-sequence GDN prefill that captures continuation state."""
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qkv_z = linear_fp8(a8, w["linear_attn.in_proj_qkv_z"])
        qkv, z = qkv_z[:, :8192], qkv_z[:, 8192:]
        ba = linear_fp16(normed, w["linear_attn.in_proj_ba"])
        b, a = ba[:, :32], ba[:, 32:]
        conv = depthwise_conv4_silu(
            qkv, w["linear_attn.conv1d.weight"], None, cu_seqlens
        )
        history_rows = min(3, qkv.shape[0])
        layer_cache.conv_history.zero_()
        layer_cache.conv_history[-history_rows:].copy_(qkv[-history_rows:])
        q, k, v = split_gdn_qkv(conv)
        q, k = l2_normalize_qk(q, k)
        decay, beta = prepare_gates(
            a, b, w["linear_attn.A_log"], w["linear_attn.dt_bias"]
        )
        if hidden.shape[0] <= 64:
            attended, final_state = recurrent_gdn(
                q, k, v, decay, beta, cu_seqlens, hidden.shape[0]
            )
        else:
            attended, final_state = chunk_gdn(
                q, k, v, decay, beta, cu_seqlens, hidden.shape[0]
            )
        layer_cache.recurrent_state.copy_(final_state[0])
        out_a8 = gated_gdn_fp8(attended, z, w["linear_attn.norm.weight"])
        return linear_fp8(out_a8, w["linear_attn.out_proj"])

    def _gdn_decode(
        self,
        hidden: torch.Tensor,
        layer: StatelessLayer,
        layer_cache: GDNLayerCache,
    ) -> torch.Tensor:
        """Advance one GDN layer by exactly one logical token."""
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qkv_z = linear_fp8(a8, w["linear_attn.in_proj_qkv_z"])
        qkv, z = qkv_z[:, :8192], qkv_z[:, 8192:]
        ba = linear_fp16(normed, w["linear_attn.in_proj_ba"])
        b, a = ba[:, :32], ba[:, 32:]
        conv = depthwise_conv4_silu_decode(
            qkv,
            w["linear_attn.conv1d.weight"],
            None,
            layer_cache.conv_history,
        )
        q, k, v = split_gdn_qkv(conv)
        q, k = l2_normalize_qk(q, k)
        decay, beta = prepare_gates(
            a, b, w["linear_attn.A_log"], w["linear_attn.dt_bias"]
        )
        attended = recurrent_gdn_decode(
            q, k, v, decay, beta, layer_cache.recurrent_state
        )
        out_a8 = gated_gdn_fp8(attended, z, w["linear_attn.norm.weight"])
        return linear_fp8(out_a8, w["linear_attn.out_proj"])

    def _full_attention_prefill(
        self,
        hidden: torch.Tensor,
        layer: StatelessLayer,
        layer_cache: FullAttentionLayerCache,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Fresh full-attention prefill with continuous KV capture."""
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qkv = linear_fp8(a8, w["self_attn.qkv_proj"])
        qgate, k, vflat = qkv[:, :8192], qkv[:, 8192:8704], qkv[:, 8704:]
        q, k, gate = full_qk_rope_gate(
            qgate,
            k,
            positions,
            w["self_attn.q_norm.weight"],
            w["self_attn.k_norm.weight"],
        )
        value = vflat.view(vflat.shape[0], 2, 256)
        layer_cache.key[: hidden.shape[0]].copy_(k)
        layer_cache.value[: hidden.shape[0]].copy_(value)
        attended = causal_gqa(q, k, value, cu_seqlens, hidden.shape[0])
        return linear_fp8(gated_attention_fp8(attended, gate), w["self_attn.o_proj"])

    def _full_attention_decode(
        self,
        hidden: torch.Tensor,
        layer: StatelessLayer,
        layer_cache: FullAttentionLayerCache,
        position: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        """Append one K/V pair and attend the current Q to the valid prefix."""
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qkv = linear_fp8(a8, w["self_attn.qkv_proj"])
        qgate, k, vflat = qkv[:, :8192], qkv[:, 8192:8704], qkv[:, 8704:]
        q, k, gate = full_qk_rope_gate(
            qgate,
            k,
            position,
            w["self_attn.q_norm.weight"],
            w["self_attn.k_norm.weight"],
        )
        layer_cache.key[prefix_len].copy_(k[0])
        layer_cache.value[prefix_len].copy_(vflat.view(2, 256))
        attended = causal_gqa_decode(
            q, layer_cache.key, layer_cache.value, prefix_len + 1
        )
        return linear_fp8(gated_attention_fp8(attended, gate), w["self_attn.o_proj"])

    def _moe(
        self,
        hidden: torch.Tensor,
        normalized: torch.Tensor,
        layer: StatelessLayer,
        *,
        router_capture: RouterCapture | None = None,
    ) -> torch.Tensor:
        w = layer.weights
        offloaded = self.expert_store is not None and 1 <= layer.layer_id <= 38
        result = fused_moe(
            normalized,
            MoeWeights(
                router=w["mlp.gate"],
                gate_up=None if offloaded else w["mlp.experts.gate_up_proj"],
                down=None if offloaded else w["mlp.experts.down_proj"],
                shared_gate_up=w["mlp.shared_expert.gate_up_proj"],
                shared_down=w["mlp.shared_expert.down_proj"],
                shared_gate=w["mlp.shared_expert_gate"],
                expert_store=self.expert_store if offloaded else None,
                layer_id=layer.layer_id if offloaded else None,
            ),
            residual=hidden,
            capture_router=router_capture is not None,
        )
        if router_capture is None:
            return result
        output, ids, probabilities = result
        # Keep only the last logical token.  This is deliberately a CUDA view:
        # the pipeline's validation worker is the sole caller that later makes
        # an explicit D2H copy for its compact diagnostic frame.
        router_capture[layer.layer_id] = (ids[-1], probabilities[-1])
        return output

    def _finish_layer(
        self,
        hidden: torch.Tensor,
        projected: torch.Tensor,
        layer: StatelessLayer,
        *,
        router_capture: RouterCapture | None = None,
    ) -> torch.Tensor:
        hidden, post_norm = residual_add_gemma_rms_norm(
            hidden, projected, layer.weights["post_attention_layernorm.weight"]
        )
        return self._moe(hidden, post_norm, layer, router_capture=router_capture)

    def prefill_hidden(
        self,
        hidden: torch.Tensor,
        *,
        cache: SingleRequestCache,
        router_capture: RouterCapture | None = None,
    ) -> torch.Tensor:
        """Run one nonempty fresh sequence and populate its continuation cache."""
        _require_cuda_matrix(hidden, "hidden_states")
        self._validate_request_cache(cache)
        if hidden.device != self.device:
            raise ValueError("hidden_states must be on the runner's CUDA device")
        tokens = hidden.shape[0]
        if not 1 <= tokens <= cache.capacity:
            raise ValueError("prefill length must be in [1, cache.capacity]")
        if cache.consumed_len != 0:
            raise RuntimeError("prefill requires an empty request cache")
        positions = torch.arange(tokens, device=self.device, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, tokens], device=self.device, dtype=torch.int32)
        try:
            for layer in self.layers:
                layer_cache = cache.layers[layer.layer_id]
                if layer.is_gdn:
                    if not isinstance(layer_cache, GDNLayerCache):
                        raise TypeError("GDN layer received a full-attention cache")
                    projected = self._gdn_prefill(
                        hidden, layer, layer_cache, cu_seqlens
                    )
                else:
                    if not isinstance(layer_cache, FullAttentionLayerCache):
                        raise TypeError("full-attention layer received a GDN cache")
                    projected = self._full_attention_prefill(
                        hidden, layer, layer_cache, positions, cu_seqlens
                    )
                hidden = self._finish_layer(
                    hidden, projected, layer, router_capture=router_capture
                )
            # Cache progress is transactional: surface asynchronous kernel
            # failures before publishing a new consumed length.  The pipeline
            # immediately performs a D2H boundary copy anyway, so this does not
            # add another synchronization to its production critical path.
            torch.cuda.synchronize(self.device)
        except Exception as error:
            cache.poisoned = True
            self._fail_offload(error)
            raise
        cache.consumed_len = tokens
        return hidden

    def decode_hidden(
        self,
        hidden: torch.Tensor,
        *,
        cache: SingleRequestCache,
        expected_prefix_len: int,
        router_capture: RouterCapture | None = None,
    ) -> torch.Tensor:
        """Advance an existing single-sequence cache by one token."""
        _require_cuda_matrix(hidden, "hidden_states")
        self._validate_request_cache(cache)
        if hidden.device != self.device:
            raise ValueError("hidden_states must be on the runner's CUDA device")
        if hidden.shape[0] != 1:
            raise ValueError("decode_hidden requires exactly one token")
        if isinstance(expected_prefix_len, bool) or not isinstance(
            expected_prefix_len, int
        ):
            raise TypeError("expected_prefix_len must be a Python int")
        if expected_prefix_len != cache.consumed_len:
            raise RuntimeError(
                f"expected prefix {expected_prefix_len}, cache has "
                f"{cache.consumed_len} tokens"
            )
        if cache.consumed_len == 0:
            raise RuntimeError("decode requires a completed prefill")
        if cache.consumed_len >= cache.capacity:
            raise RuntimeError("request cache capacity is exhausted")
        prefix_len = cache.consumed_len
        position = torch.tensor([prefix_len], device=self.device, dtype=torch.int32)
        try:
            for layer in self.layers:
                layer_cache = cache.layers[layer.layer_id]
                if layer.is_gdn:
                    if not isinstance(layer_cache, GDNLayerCache):
                        raise TypeError("GDN layer received a full-attention cache")
                    projected = self._gdn_decode(hidden, layer, layer_cache)
                else:
                    if not isinstance(layer_cache, FullAttentionLayerCache):
                        raise TypeError("full-attention layer received a GDN cache")
                    projected = self._full_attention_decode(
                        hidden, layer, layer_cache, position, prefix_len
                    )
                hidden = self._finish_layer(
                    hidden, projected, layer, router_capture=router_capture
                )
            torch.cuda.synchronize(self.device)
        except Exception as error:
            cache.poisoned = True
            self._fail_offload(error)
            raise
        cache.consumed_len = prefix_len + 1
        return hidden

    def forward_hidden(
        self,
        hidden: torch.Tensor,
        *,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        router_capture: RouterCapture | None = None,
    ) -> torch.Tensor:
        _require_cuda_matrix(hidden, "hidden_states")
        if hidden.device != self.device:
            raise ValueError("hidden_states must be on the runner's CUDA device")
        _validate_metadata(
            hidden.shape[0], positions, cu_seqlens, max_seqlen, hidden.device
        )
        try:
            for layer in self.layers:
                hidden = self.forward_layer(
                    hidden,
                    layer.layer_id,
                    positions=positions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    router_capture=router_capture,
                )
            return hidden
        except Exception as error:
            self._fail_offload(error)
            raise

    def forward_no_cache(
        self,
        hidden: torch.Tensor,
        *,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        router_capture: RouterCapture | None = None,
    ) -> torch.Tensor:
        """Compatibility name for the packed stateless execution path."""
        return self.forward_hidden(
            hidden,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            router_capture=router_capture,
        )

    def forward_layer(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        *,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        router_capture: RouterCapture | None = None,
    ) -> torch.Tensor:
        """Run one explicitly loaded original layer, without global tensors.

        This is the one-layer scan entry used by validation: callers can load
        a large layer independently with ``load_globals=False`` and avoid an
        embedding/head allocation for every original layer.
        """
        _require_cuda_matrix(hidden, "hidden_states")
        if hidden.device != self.device:
            raise ValueError("hidden_states must be on the runner's CUDA device")
        _validate_metadata(
            hidden.shape[0], positions, cu_seqlens, max_seqlen, hidden.device
        )
        try:
            layer = self._layers_by_id[layer_id]
        except KeyError as exc:
            raise KeyError(
                f"layer {layer_id} was not selected for this runner"
            ) from exc
        projected = (
            self._gdn(hidden, layer, cu_seqlens, max_seqlen)
            if layer.is_gdn
            else self._full_attention(hidden, layer, positions, cu_seqlens, max_seqlen)
        )
        return self._finish_layer(
            hidden, projected, layer, router_capture=router_capture
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.global_weights is None:
            raise RuntimeError("embedding is unavailable when load_globals=False")
        if (
            input_ids.ndim != 1
            or input_ids.dtype not in (torch.int32, torch.int64)
            or not input_ids.is_cuda
            or not input_ids.is_contiguous()
        ):
            raise ValueError("input_ids must be contiguous CUDA int32/int64 [T]")
        return fp16_embedding(input_ids, self.global_weights["embed_tokens"])

    def final_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.global_weights is None:
            raise RuntimeError("final norm is unavailable when load_globals=False")
        return gemma_rms_norm(hidden, self.global_weights["final_norm"])

    def logits(
        self, hidden: torch.Tensor, indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project either selected rows or an already gathered `[M,2048]` tensor."""
        if self.global_weights is None:
            raise RuntimeError("LM head is unavailable when load_globals=False")
        if indices is None:
            _require_cuda_matrix(hidden, "selected_hidden")
            return linear_fp16(hidden, self.global_weights["lm_head"])
        if (
            indices.ndim != 1
            or indices.dtype not in (torch.int32, torch.int64)
            or not indices.is_cuda
            or not indices.is_contiguous()
        ):
            raise ValueError("logits_indices must be contiguous CUDA integers")
        chosen = gather_hidden(hidden, indices)
        return linear_fp16(chosen, self.global_weights["lm_head"])


def _validate_metadata(
    tokens: int,
    positions: torch.Tensor,
    cu: torch.Tensor,
    max_seqlen: int,
    device: torch.device,
) -> None:
    if (
        isinstance(max_seqlen, bool)
        or not isinstance(max_seqlen, int)
        or not 0 <= max_seqlen <= 2048
    ):
        raise ValueError("max_seqlen must be a Python int in [0,2048]")
    if tokens > 2048:
        raise ValueError("stateless Qwen3.5 supports at most 2048 packed tokens")
    if tokens and max_seqlen == 0:
        raise ValueError("max_seqlen must be positive for nonempty packed inputs")
    for x, name in ((positions, "positions"), (cu, "cu_seqlens")):
        if (
            not x.is_cuda
            or x.device != device
            or not x.is_contiguous()
            or x.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{name} must be contiguous CUDA int32/int64")
    if positions.shape != (tokens,) or cu.ndim != 1 or cu.numel() < 2:
        raise ValueError(
            "positions must be [T] and cu_seqlens must have at least two entries"
        )
    # cu contents are trusted caller metadata: start=0, end=T,
    # nondecreasing, every length <= max_seqlen, and positions cover the
    # packed tokens.  They remain device-resident; inspecting them here would
    # synchronize the V100 hot path.
