"""Independent mathematical layer reference for the M3 checkpoint scan."""

from __future__ import annotations

import math

import torch

from . import moe as moe_reference
from .codec import decode_e2m1, decode_e4m3fn, encode_e2m1, encode_e4m3fn, quantize_a8
from .gdn import recurrent_vectorized


def quantize_a8_rne(x: torch.Tensor):
    """CPU E4M3 A8 codec with the frozen RN32 reciprocal-multiply scale rule."""
    # The common test codec has the frozen explicit float32 reciprocal rule.
    # Keep this wrapper to make the model reference's producer dependency
    # explicit and avoid ever importing runtime quantization here.
    return quantize_a8(x.detach().to("cpu", torch.float32).contiguous())


def quantize_a4_rne(x: torch.Tensor, global_scale: torch.Tensor):
    """CPU NVFP4 codec with static G and RN32 reciprocal-multiply local SF."""
    cpu = x.detach().to("cpu", torch.float32).contiguous()
    g = global_scale.detach().to("cpu", torch.float32).reshape(-1, 1)
    if g.shape[0] == 1:
        g = g.expand(cpu.shape[0], 1)
    blocks = (cpu * g).reshape(cpu.shape[0], -1, 16)
    reciprocal = torch.tensor(1.0 / 6.0, dtype=torch.float32)
    scale = encode_e4m3fn((blocks.abs().amax(-1).float() * reciprocal).float())
    decoded = decode_e4m3fn(scale)[..., None]
    safe = torch.where(decoded == 0, torch.ones_like(decoded), decoded)
    codes = encode_e2m1((blocks / safe).reshape_as(cpu))
    codes = torch.where(
        decoded.repeat_interleave(16, 1).reshape_as(codes) == 0,
        torch.zeros_like(codes),
        codes,
    )
    return (
        (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(),
        scale.contiguous(),
        g[:, 0],
    )


def fp8_weight(weight) -> torch.Tensor:
    """Fully dequantized W8 matrix for the explicit mathematical oracle."""
    raw = decode_e4m3fn(weight.data.detach().cpu()).float()
    row = torch.arange(raw.shape[0])[:, None] // 128
    col = torch.arange(raw.shape[1])[None, :] // 128
    return (raw * weight.block_scale.detach().cpu().float()[row, col]).to(
        weight.data.device
    )


def fp8_linear_math(x: torch.Tensor, weight) -> torch.Tensor:
    """Mathematical W8A8 oracle that dequantizes scales before one GEMM."""
    payload, scale = quantize_a8_rne(x)
    return fp8_linear_payload_math(payload, scale, weight)


def fp8_linear_payload_math(
    data: torch.Tensor, scale: torch.Tensor, weight
) -> torch.Tensor:
    """Mathematical W8A8 oracle for an already-produced A8 operand."""
    decoded_x = (
        decode_e4m3fn(data.detach().cpu()).float()
        * scale.detach().cpu().float().repeat_interleave(128, 1)
    ).to(weight.data.device)
    return (decoded_x @ fp8_weight(weight).t()).to(torch.float16)


def _validate_fp8_payload(
    data: torch.Tensor, scale: torch.Tensor, weight
) -> tuple[int, int, int]:
    if data.ndim != 2 or data.dtype != torch.uint8:
        raise ValueError("FP8 activation data must be uint8 [M,K]")
    if scale.ndim != 2 or scale.dtype != torch.float32:
        raise ValueError("FP8 activation scale must be float32 [M,K/128]")
    if (
        weight.data.ndim != 2
        or weight.data.dtype != torch.uint8
        or weight.block_scale is None
    ):
        raise ValueError("FP8 weight must provide uint8 [N,K] data and block scales")
    m, k = data.shape
    n, weight_k = weight.data.shape
    if k == 0 or k % 128 or weight_k != k:
        raise ValueError(
            "FP8 semantic GEMM requires matching nonempty K divisible by 128"
        )
    if tuple(scale.shape) != (m, k // 128):
        raise ValueError("FP8 activation scale must be exactly [M,K/128]")
    if tuple(weight.block_scale.shape) != ((n + 127) // 128, k // 128):
        raise ValueError("FP8 weight scale must be exactly [ceil(N/128),K/128]")
    return m, n, k


def fp8_linear_payload(data: torch.Tensor, scale: torch.Tensor, weight) -> torch.Tensor:
    """Block-semantic W8A8 reference for an already-produced A8 operand.

    Each K=128 partial decodes E4M3 operands through FP16, accumulates in
    FP32, applies the activation scale first and then the weight scale, and
    is added to the FP32 result in K order.  This models the documented
    block-scale boundaries without claiming bitwise equivalence to HMMA.
    """
    m, n, k = _validate_fp8_payload(data, scale, weight)
    device = weight.data.device
    a = decode_e4m3fn(data.detach().cpu()).to(torch.float16).to(device)
    w = decode_e4m3fn(weight.data.detach().cpu()).to(torch.float16).to(device)
    a_scale = scale.detach().to(device=device, dtype=torch.float32)
    w_scale = weight.block_scale.detach().to(device=device, dtype=torch.float32)
    acc = torch.zeros((m, n), device=device, dtype=torch.float32)
    for group in range(k // 128):
        begin = group * 128
        partial = (
            a[:, begin : begin + 128].float() @ w[:, begin : begin + 128].float().t()
        )
        activation_scaled = partial * a_scale[:, group : group + 1]
        per_row_weight_scale = (
            w_scale[:, group].repeat_interleave(128)[:n].reshape(1, n)
        )
        acc = acc + activation_scaled * per_row_weight_scale
    return acc.to(torch.float16)


def fp8_linear(x: torch.Tensor, weight) -> torch.Tensor:
    """Default block-semantic W8A8 reference used by composed layer scans."""
    payload, scale = quantize_a8_rne(x)
    return fp8_linear_payload(payload, scale, weight)


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * (1 + weight.float())
    ).to(torch.float16)


def _gdn_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for tap in range(4):
            source = t - 3 + tap
            if source >= 0:
                out[t] += x[source].float() * weight[:, tap].float()
    return torch.nn.functional.silu(out).to(torch.float16)


def _l2(x: torch.Tensor) -> torch.Tensor:
    return (
        x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
    ).to(torch.float16)


def _gdn_reference(hidden: torch.Tensor, w: dict, *, details: bool = False):
    n = gemma_rms_norm(hidden, w["input_layernorm.weight"])
    qkv = fp8_linear(n, w["linear_attn.in_proj_qkv"])
    z = fp8_linear(n, w["linear_attn.in_proj_z"])
    a = (n.float() @ w["linear_attn.in_proj_a"].data.float().t()).to(torch.float16)
    b = (n.float() @ w["linear_attn.in_proj_b"].data.float().t()).to(torch.float16)
    conv = _gdn_conv(qkv, w["linear_attn.conv1d.weight"])
    q = _l2(conv[:, :2048].reshape(-1, 16, 128))
    k = _l2(conv[:, 2048:4096].reshape(-1, 16, 128))
    v = conv[:, 4096:].reshape(-1, 32, 128)
    raw = a.float() + w["linear_attn.dt_bias"].float()
    softplus = torch.maximum(raw, torch.zeros_like(raw)) + torch.log1p(
        torch.exp(-raw.abs())
    )
    decay = -torch.exp(w["linear_attn.A_log"].float()) * softplus
    beta = torch.sigmoid(b.float()).to(torch.float16)
    cu = torch.tensor([0, hidden.shape[0]], device=hidden.device, dtype=torch.int32)
    value, state = recurrent_vectorized(q, k, v, decay, beta, cu)
    value16 = value.to(torch.float16)
    norm = (
        value16.float()
        * torch.rsqrt(value16.float().square().mean(-1, keepdim=True) + 1e-6)
        * w["linear_attn.norm.weight"].float()
    )
    gated = (
        norm.reshape(hidden.shape[0], -1) * torch.nn.functional.silu(z.float())
    ).to(torch.float16)
    out = fp8_linear(gated, w["linear_attn.out_proj"])
    return (
        (
            out,
            value,
            state,
            {
                "norm": n,
                "qkv": qkv,
                "z": z,
                "a": a,
                "b": b,
                "conv": conv,
                "q": q,
                "k": k,
                "v": v,
                "decay": decay,
                "beta": beta,
            },
        )
        if details
        else out
    )


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = x.float().clone()
    angle = positions.float()[:, None, None] / torch.exp(
        torch.arange(32, device=x.device, dtype=torch.float32)[None, None, :]
        * math.log(10_000_000.0)
        / 32
    )
    c, s = angle.cos(), angle.sin()
    low, high = x.float()[..., :32], x.float()[..., 32:64]
    out[..., :32], out[..., 32:64] = low * c - high * s, high * c + low * s
    return out.to(torch.float16)


def _full_reference(
    hidden: torch.Tensor, w: dict, positions: torch.Tensor, *, details: bool = False
):
    n = gemma_rms_norm(hidden, w["input_layernorm.weight"])
    qg = fp8_linear(n, w["self_attn.q_proj"])
    k = fp8_linear(n, w["self_attn.k_proj"])
    v = fp8_linear(n, w["self_attn.v_proj"]).reshape(-1, 2, 256)
    qg = qg.reshape(-1, 16, 512)
    q = gemma_rms_norm(
        qg[..., :256].reshape(-1, 256), w["self_attn.q_norm.weight"]
    ).reshape(-1, 16, 256)
    k = gemma_rms_norm(k.reshape(-1, 256), w["self_attn.k_norm.weight"]).reshape(
        -1, 2, 256
    )
    q, k = _rope(q, positions), _rope(k, positions)
    rows = []
    for t in range(hidden.shape[0]):
        keys, values = k[: t + 1], v[: t + 1]
        # GQA groups eight query heads per KV head.  The producer writes K
        # through alternating launch heads, while causal attention consumes
        # the conventional grouped [0..7]->0, [8..15]->1 layout.
        grouped = torch.arange(16, device=q.device) // 8
        score = torch.einsum("hd,shd->hs", q[t].float(), keys[:, grouped].float()) * (
            256**-0.5
        )
        prob = torch.softmax(score, dim=-1)
        rows.append(torch.einsum("hs,shd->hd", prob, values[:, grouped].float()))
    attended = torch.stack(rows).to(torch.float16)
    gate = qg[..., 256:]
    gated = (
        (attended.float() * torch.sigmoid(gate.float()))
        .to(torch.float16)
        .reshape(hidden.shape[0], -1)
    )
    a8_data, a8_scale = quantize_a8_rne(gated)
    out = fp8_linear(gated, w["self_attn.o_proj"])
    if details:
        return out, {
            "q": q,
            "k": k,
            "gate": gate,
            "attended": attended,
            "gated": gated,
            "a8_data": a8_data,
            "a8_scale": a8_scale,
        }
    return out


def layer(hidden: torch.Tensor, w: dict, positions: torch.Tensor):
    gdn = "linear_attn.in_proj_qkv" in w
    gdn_details = _gdn_reference(hidden, w, details=True) if gdn else None
    full_details = (
        _full_reference(hidden, w, positions, details=True) if not gdn else None
    )
    projected = gdn_details[0] if gdn else full_details[0]
    residual = (hidden.float() + projected.float()).to(torch.float16)
    post = gemma_rms_norm(residual, w["post_attention_layernorm.weight"])
    moe, logits, ids, weights = moe_reference.moe(
        post,
        w["mlp.gate"],
        w["mlp.experts.gate_up_proj"],
        w["mlp.experts.down_proj"],
        w["mlp.shared_expert.gate_up_proj"],
        w["mlp.shared_expert.down_proj"],
        w["mlp.shared_expert_gate"],
        quantize_a4=quantize_a4_rne,
    )
    parts = {
        "projected": projected,
        "residual": residual,
        "post_norm": post,
        "router_logits": logits,
        "route_ids": ids,
        "route_weights": weights,
        "moe": moe,
    }
    if gdn_details is not None:
        parts.update(
            {"gdn_value": gdn_details[1], "gdn_state": gdn_details[2], **gdn_details[3]}
        )
    if full_details is not None:
        parts.update(full_details[1])
    return (residual.float() + moe.float()).to(torch.float16), parts
