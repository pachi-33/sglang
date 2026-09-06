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
