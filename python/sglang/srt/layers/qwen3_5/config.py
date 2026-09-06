"""Minimal HuggingFace-compatible configuration for the V100 Qwen3.5 MoE path.

The upstream Qwen3.5 config is newer than the Transformers version used by
this compatibility tree.  Keeping the small text schema here lets
``AutoConfig`` read language-model-only checkpoints without importing the
main SGLang Qwen3.5 implementation (which requires the normal KV runner).
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class Qwen3_5MoeTextConfig(PretrainedConfig):
    """The text backbone contained in a ``qwen3_5_moe`` checkpoint."""

    model_type = "qwen3_5_moe_text"

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2048,
        intermediate_size: int = 512,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000_000.0,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        shared_expert_intermediate_size: int = 512,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        super().__init__(**kwargs)


class Qwen3_5MoeConfig(PretrainedConfig):
    """Language-model-only outer config used by Qwen-AgentWorld NVFP4 files."""

    model_type = "qwen3_5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict[str, Any] | Qwen3_5MoeTextConfig | None = None,
        language_model_only: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            # The exported nested config sometimes still says qwen3_5_text.
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            text_config = Qwen3_5MoeTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3_5MoeTextConfig()
        elif not isinstance(text_config, Qwen3_5MoeTextConfig):
            raise TypeError("text_config must be a dict or Qwen3_5MoeTextConfig")
        self.text_config = text_config
        self.language_model_only = bool(language_model_only)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size


def register_qwen3_5_moe_config() -> None:
    """Install both names before ``AutoConfig.from_pretrained`` is called."""
    from transformers import AutoConfig

    # ``exist_ok`` is absent on some 4.43 point releases.  Re-registering the
    # identical local class is harmless, while a third-party implementation
    # should remain untouched.
    for name, cls in (
        (Qwen3_5MoeConfig.model_type, Qwen3_5MoeConfig),
        (Qwen3_5MoeTextConfig.model_type, Qwen3_5MoeTextConfig),
    ):
        try:
            AutoConfig.register(name, cls)
        except ValueError:
            pass
