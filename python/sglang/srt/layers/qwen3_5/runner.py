"""A deliberately small stateless runner for selected Qwen3.5 layers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .attention import causal_gqa
from .checkpoint import Qwen35Checkpoint
from .dense import fp16_embedding, linear_fp16
from .gdn import (chunk_gdn, depthwise_conv4_silu, l2_normalize_qk, prepare_gates,
                  recurrent_gdn_short_output)
from .model_ops import full_qk_rope_gate, gated_attention_fp8, gated_gdn_fp8, gather_hidden, split_full_v, split_gdn_qkv
from .moe import MoeWeights, fused_moe
from .ops import gemma_rms_norm, residual_add, residual_add_gemma_rms_norm
from .quantization import linear_fp8, quantize_fp8
from .weights import Weight


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


class Qwen35StatelessRunner:
    """Executes a selected original-layer slice with no KV or recurrent state."""

    def __init__(self, model_dir: str | Path, layer_ids: Iterable[int] = range(4), *,
                 device: str | torch.device = "cuda", load_globals: bool = True):
        self.model_dir = Path(model_dir)
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("the Qwen3.5 compatibility runner requires CUDA")
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}") if requested_device.index is None else requested_device
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(self.device) != (7, 0):
            raise RuntimeError("Qwen3.5 compatibility runner requires an SM70 V100")
        ids = tuple(layer_ids)
        if not ids or tuple(sorted(set(ids))) != ids or any(i < 0 or i >= 40 for i in ids):
            raise ValueError("layer_ids must be unique, ascending original IDs in [0,40)")
        self.layer_ids = ids
        loader = Qwen35Checkpoint(self.model_dir)
        self.layers = tuple(StatelessLayer(i, loader.load_layer(i, self.device)) for i in ids)
        self._layers_by_id = {layer.layer_id: layer for layer in self.layers}
        self.global_weights = loader.load_global_tensors(self.device) if load_globals else None

    def _gdn(self, hidden: torch.Tensor, layer: StatelessLayer, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qkv = linear_fp8(a8, w["linear_attn.in_proj_qkv"])
        z = linear_fp8(a8, w["linear_attn.in_proj_z"])
        # B/A are ordinary FP16 row projections and are intentionally kept
        # independent from the single reused A8 payload above.
        a = linear_fp16(normed, w["linear_attn.in_proj_a"])
        b = linear_fp16(normed, w["linear_attn.in_proj_b"])
        conv = depthwise_conv4_silu(qkv, w["linear_attn.conv1d.weight"], None,
                                    cu_seqlens)
        q, k, v = split_gdn_qkv(conv)
        q, k = l2_normalize_qk(q, k)
        decay, beta = prepare_gates(a, b, w["linear_attn.A_log"], w["linear_attn.dt_bias"])
        if max_seqlen <= 64:
            # The stateless runner only consumes output.  Avoid allocating a
            # [B,32,128,128] state for the ordinary all-short prefill path.
            attended = torch.empty((q.shape[0], 32, 128), device=q.device, dtype=torch.float32)
            recurrent_gdn_short_output(q, k, v, decay, beta, cu_seqlens, max_seqlen, attended)
        else:
            attended, _ = chunk_gdn(q, k, v, decay, beta, cu_seqlens, max_seqlen)
            # A ragged batch's longest document must not choose the numerical
            # recurrence for its short neighbours.  This state-free launch
            # replaces only actual 1..64-token documents in the WY output;
            # long documents retain their WY result and no second state exists.
            recurrent_gdn_short_output(q, k, v, decay, beta, cu_seqlens, max_seqlen, attended)
        out_a8 = gated_gdn_fp8(attended, z, w["linear_attn.norm.weight"])
        projected = linear_fp8(out_a8, w["linear_attn.out_proj"])
        return projected

    def _full_attention(self, hidden: torch.Tensor, layer: StatelessLayer, positions: torch.Tensor,
                        cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        w = layer.weights
        normed = gemma_rms_norm(hidden, w["input_layernorm.weight"])
        a8 = quantize_fp8(normed)
        qgate = linear_fp8(a8, w["self_attn.q_proj"])
        k = linear_fp8(a8, w["self_attn.k_proj"])
        vflat = linear_fp8(a8, w["self_attn.v_proj"])
        q, k, gate = full_qk_rope_gate(qgate, k, positions, w["self_attn.q_norm.weight"], w["self_attn.k_norm.weight"])
        attended = causal_gqa(q, k, vflat.view(vflat.shape[0], 2, 256), cu_seqlens, max_seqlen)
        projected = linear_fp8(gated_attention_fp8(attended, gate), w["self_attn.o_proj"])
        return projected

    def _moe(self, hidden: torch.Tensor, normalized: torch.Tensor, layer: StatelessLayer) -> torch.Tensor:
        w = layer.weights
        result = fused_moe(normalized, MoeWeights(
            router=w["mlp.gate"], gate_up=w["mlp.experts.gate_up_proj"], down=w["mlp.experts.down_proj"],
            shared_gate_up=w["mlp.shared_expert.gate_up_proj"], shared_down=w["mlp.shared_expert.down_proj"],
            shared_gate=w["mlp.shared_expert_gate"],
        ))
        return residual_add(hidden, result)

    def forward_hidden(self, hidden: torch.Tensor, *, positions: torch.Tensor, cu_seqlens: torch.Tensor,
                       max_seqlen: int) -> torch.Tensor:
        _require_cuda_matrix(hidden, "hidden_states")
        if hidden.device != self.device:
            raise ValueError("hidden_states must be on the runner's CUDA device")
        _validate_metadata(hidden.shape[0], positions, cu_seqlens, max_seqlen, hidden.device)
        for layer in self.layers:
            hidden = self.forward_layer(hidden, layer.layer_id, positions=positions,
                                        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return hidden

    def forward_layer(self, hidden: torch.Tensor, layer_id: int, *, positions: torch.Tensor,
                      cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        """Run one explicitly loaded original layer, without global tensors.

        This is the one-layer scan entry used by validation: callers can load
        a large layer independently with ``load_globals=False`` and avoid an
        embedding/head allocation for every original layer.
        """
        _require_cuda_matrix(hidden, "hidden_states")
        if hidden.device != self.device:
            raise ValueError("hidden_states must be on the runner's CUDA device")
        _validate_metadata(hidden.shape[0], positions, cu_seqlens, max_seqlen, hidden.device)
        try:
            layer = self._layers_by_id[layer_id]
        except KeyError as exc:
            raise KeyError(f"layer {layer_id} was not selected for this runner") from exc
        projected = self._gdn(hidden, layer, cu_seqlens, max_seqlen) if layer.is_gdn else self._full_attention(hidden, layer, positions, cu_seqlens, max_seqlen)
        hidden, post_norm = residual_add_gemma_rms_norm(
            hidden, projected, layer.weights["post_attention_layernorm.weight"]
        )
        return self._moe(hidden, post_norm, layer)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.global_weights is None:
            raise RuntimeError("embedding is unavailable when load_globals=False")
        if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.is_cuda or not input_ids.is_contiguous():
            raise ValueError("input_ids must be contiguous CUDA int32/int64 [T]")
        return fp16_embedding(input_ids, self.global_weights["embed_tokens"])

    def final_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.global_weights is None:
            raise RuntimeError("final norm is unavailable when load_globals=False")
        return gemma_rms_norm(hidden, self.global_weights["final_norm"])

    def logits(self, hidden: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Project either selected rows or an already gathered `[M,2048]` tensor."""
        if self.global_weights is None:
            raise RuntimeError("LM head is unavailable when load_globals=False")
        if indices is None:
            _require_cuda_matrix(hidden, "selected_hidden")
            return linear_fp16(hidden, self.global_weights["lm_head"])
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64) or not indices.is_cuda or not indices.is_contiguous():
            raise ValueError("logits_indices must be contiguous CUDA integers")
        chosen = gather_hidden(hidden, indices)
        return linear_fp16(chosen, self.global_weights["lm_head"])


def _validate_metadata(tokens: int, positions: torch.Tensor, cu: torch.Tensor, max_seqlen: int, device: torch.device) -> None:
    if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int) or not 0 <= max_seqlen <= 2048:
        raise ValueError("max_seqlen must be a Python int in [0,2048]")
    if tokens > 2048:
        raise ValueError("stateless Qwen3.5 supports at most 2048 packed tokens")
    if tokens and max_seqlen == 0:
        raise ValueError("max_seqlen must be positive for nonempty packed inputs")
    for x, name in ((positions, "positions"), (cu, "cu_seqlens")):
        if not x.is_cuda or x.device != device or not x.is_contiguous() or x.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be contiguous CUDA int32/int64")
    if positions.shape != (tokens,) or cu.ndim != 1 or cu.numel() < 2:
        raise ValueError("positions must be [T] and cu_seqlens must have at least two entries")
    # cu contents are trusted caller metadata: start=0, end=T,
    # nondecreasing, every length <= max_seqlen, and positions cover the
    # packed tokens.  They remain device-resident; inspecting them here would
    # synchronize the V100 hot path.
