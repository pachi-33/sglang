"""Public Qwen3.5 producer API; Triton implementations live in kernels."""

from .kernels.model_ops import (
    default_last_token_indices,
    full_qk_rope_gate,
    gather_hidden,
    gated_attention_fp8,
    gated_gdn_fp8,
    split_full_v,
    split_gdn_qkv,
)

__all__ = [
    "default_last_token_indices",
    "full_qk_rope_gate",
    "gather_hidden",
    "gated_attention_fp8",
    "gated_gdn_fp8",
    "split_full_v",
    "split_gdn_qkv",
]
