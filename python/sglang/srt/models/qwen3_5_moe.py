"""Text-only, stateless Qwen3.5 MoE entry point for the Volta path."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from sglang.srt.layers.qwen3_5.model_ops import default_last_token_indices, gather_hidden
from sglang.srt.layers.qwen3_5.runner import Qwen35StatelessRunner



class Qwen3_5MoeForConditionalGeneration(nn.Module):
    """Selected-layer Qwen3.5 text model with no cache or persistent state.

    This intentionally is not a normal SGLang ``ModelRunner`` model.  It is a
    compatibility entry point for the NVFP4 checkpoint on one V100 and exposes
    only the packed, call-local stateless API below.
    """

    def __init__(
        self,
        config=None,
        *,
        model_dir: str | Path | None = None,
        selected_layer_ids: Iterable[int] = range(4),
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.config = config
        self.selected_layer_ids = tuple(selected_layer_ids)
        self.runner: Qwen35StatelessRunner | None = None
        if model_dir is not None:
            self.runner = Qwen35StatelessRunner(model_dir, self.selected_layer_ids, device=device)

    @classmethod
    def from_checkpoint(
        cls, model_dir: str | Path, *, selected_layer_ids: Iterable[int] = range(4),
        device: str | torch.device = "cuda", config=None,
    ) -> "Qwen3_5MoeForConditionalGeneration":
        return cls(config, model_dir=model_dir, selected_layer_ids=selected_layer_ids, device=device)

    def _runner(self) -> Qwen35StatelessRunner:
        if self.runner is None:
            raise RuntimeError("construct with model_dir or use from_checkpoint")
        return self.runner

    @torch.inference_mode()
    def forward_no_cache(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        logits_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run selected original layers and return `(final_hidden, logits)`.

        Exactly one of token IDs or precomputed hidden states is accepted.
        The default logit selection is a GPU-produced final token per document;
        a mixed batch represents empty documents by a zero logit row, since its
        selection metadata remains entirely device-resident.
        """
        if (input_ids is None) == (hidden_states is None):
            raise ValueError("provide exactly one of input_ids or hidden_states")
        runner = self._runner()
        hidden = runner.embed(input_ids) if input_ids is not None else hidden_states
        final_hidden = runner.final_hidden(runner.forward_hidden(
            hidden, positions=positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        ))
        if logits_indices is None:
            logits_indices = default_last_token_indices(cu_seqlens, final_hidden.shape[0])
        chosen = gather_hidden(final_hidden, logits_indices)
        return final_hidden, runner.logits(chosen)


EntryClass = Qwen3_5MoeForConditionalGeneration
