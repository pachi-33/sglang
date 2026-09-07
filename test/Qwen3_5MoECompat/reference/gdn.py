"""Independent FP32 Gated DeltaNet references used only by tests."""

import torch


def recurrent(q, k, v, decay, beta, cu):
    out = torch.empty(
        (q.shape[0], v.shape[1], 128), device=q.device, dtype=torch.float32
    )
    states = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        state = torch.zeros((32, 128, 128), device=q.device, dtype=torch.float32)
        for t in range(start, end):
            for h in range(32):
                kh = h // 2
                s = state[h] * torch.exp(decay[t, h])
                r = beta[t, h].float() * (v[t, h].float() - s @ k[t, kh].float())
                state[h] = s + r[:, None] * k[t, kh].float()[None, :]
                out[t, h] = (state[h] @ q[t, kh].float()) * (128**-0.5)
        states.append(state)
    return out, torch.stack(states)


def recurrent_vectorized(q, k, v, decay, beta, cu):
    """Independent FP32 recurrence, vectorized across the 32 V heads."""
    out = torch.empty(
        (q.shape[0], v.shape[1], 128), device=q.device, dtype=torch.float32
    )
    states = []
    kh = torch.arange(v.shape[1], device=q.device) // 2
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        state = torch.zeros((32, 128, 128), device=q.device, dtype=torch.float32)
        for t in range(start, end):
            kt = k[t, kh].float()
            qt = q[t, kh].float()
            state = state * torch.exp(decay[t].float()).view(32, 1, 1)
            residual = beta[t].float().view(32, 1) * (
                v[t].float() - torch.bmm(state, kt.unsqueeze(2)).squeeze(2)
            )
            state = state + residual.unsqueeze(2) * kt.unsqueeze(1)
            out[t] = torch.bmm(state, qt.unsqueeze(2)).squeeze(2) * (128**-0.5)
        states.append(state)
    return out, torch.stack(states)


def recurrent_decode(q, k, v, decay, beta, state):
    """Independent one-token FP32 recurrence with a nonzero input state.

    This intentionally mutates neither ``state`` nor the inputs, so tests can
    independently check the cache kernel's required in-place update.
    """
    if (
        q.shape != (1, 16, 128)
        or k.shape != q.shape
        or v.shape != (1, 32, 128)
        or decay.shape != (1, 32)
        or beta.shape != (1, 32)
        or state.shape != (32, 128, 128)
    ):
        raise ValueError("invalid one-token GDN decode reference shapes")
    kh = torch.arange(32, device=q.device) // 2
    kt, qt = k[0, kh].float(), q[0, kh].float()
    updated = state.float() * torch.exp(decay[0].float()).view(32, 1, 1)
    residual = beta[0].float().view(32, 1) * (
        v[0].float() - torch.bmm(updated, kt.unsqueeze(2)).squeeze(2)
    )
    updated = updated + residual.unsqueeze(2) * kt.unsqueeze(1)
    out = torch.bmm(updated, qt.unsqueeze(2)).squeeze(2).unsqueeze(0) * (128**-0.5)
    return out, updated
