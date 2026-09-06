"""Full-Attention composition without cache or prefix-state inputs."""

from .kernels.attention import causal_gqa, partial_neox_rope
from .ops import gemma_rms_norm



def normalize_and_rope_qk(q, k, q_weight, k_weight, positions, theta=10_000_000.0):
    """Per-head Q/K Gemma norm followed by the checkpoint's partial RoPE."""
    if q.ndim != 3 or k.ndim != 3 or q.shape[-1] != 256 or k.shape[-1] != 256:
        raise ValueError("Q/K must have head dimension 256")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q/K normalization requires contiguous [T,H,256] inputs")
    q_norm = gemma_rms_norm(q.reshape(-1, 256), q_weight).reshape_as(q)
    k_norm = gemma_rms_norm(k.reshape(-1, 256), k_weight).reshape_as(k)
    return partial_neox_rope(q_norm, positions, theta), partial_neox_rope(k_norm, positions, theta)


__all__ = ["causal_gqa", "partial_neox_rope", "normalize_and_rope_qk"]
