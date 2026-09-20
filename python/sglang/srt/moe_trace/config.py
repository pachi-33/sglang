"""Resolved configuration for decode-time MoE tracing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MoeTraceConfig:
    """Immutable runtime view of the MoE trace command-line options."""

    output_dir: Path
    expert_routes: bool
    router_inputs: bool
    max_decode_tokens: int
    activation_group_size: int
    queue_depth: int
    overflow_policy: str

    @property
    def enabled(self) -> bool:
        return self.expert_routes or self.router_inputs

    @classmethod
    def from_observability(cls, observability: Any) -> MoeTraceConfig | None:
        output_dir = observability.moe_trace_output_dir
        routes = bool(observability.moe_trace_expert_routes)
        inputs = bool(observability.moe_trace_router_inputs)
        if output_dir is None and not routes and not inputs:
            return None
        if output_dir is None:
            raise ValueError(
                "--moe-trace-output-dir is required when MoE tracing is enabled"
            )
        config = cls(
            output_dir=Path(output_dir).expanduser().resolve(),
            expert_routes=routes,
            router_inputs=inputs,
            max_decode_tokens=int(observability.moe_trace_max_decode_tokens),
            activation_group_size=int(observability.moe_trace_activation_group_size),
            queue_depth=int(observability.moe_trace_queue_depth),
            overflow_policy=str(observability.moe_trace_overflow_policy),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError(
                "--moe-trace-output-dir requires --moe-trace-expert-routes "
                "and/or --moe-trace-router-inputs"
            )
        if not str(self.output_dir):
            raise ValueError(
                "--moe-trace-output-dir is required when MoE tracing is enabled"
            )
        if self.max_decode_tokens < 0:
            raise ValueError("--moe-trace-max-decode-tokens must be non-negative")
        if self.activation_group_size <= 0:
            raise ValueError("--moe-trace-activation-group-size must be positive")
        if self.activation_group_size % 2:
            raise ValueError("--moe-trace-activation-group-size must be even")
        if self.queue_depth <= 0:
            raise ValueError("--moe-trace-queue-depth must be positive")
        if self.overflow_policy not in {"block", "drop"}:
            raise ValueError(
                "--moe-trace-overflow-policy must be either 'block' or 'drop'"
            )


def validate_moe_trace_server_args(server_args: Any) -> None:
    """Fail early for unsupported v1 execution modes."""

    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    output_dir = cfg.moe_trace_output_dir
    routes = bool(cfg.moe_trace_expert_routes)
    inputs = bool(cfg.moe_trace_router_inputs)
    if output_dir is None and not routes and not inputs:
        return
    if output_dir is None:
        raise ValueError(
            "--moe-trace-output-dir is required when MoE tracing is enabled"
        )
    if not routes and not inputs:
        raise ValueError(
            "--moe-trace-output-dir requires --moe-trace-expert-routes "
            "and/or --moe-trace-router-inputs"
        )
    if cfg.moe_trace_max_decode_tokens < 0:
        raise ValueError("--moe-trace-max-decode-tokens must be non-negative")
    if cfg.moe_trace_activation_group_size <= 0:
        raise ValueError("--moe-trace-activation-group-size must be positive")
    if cfg.moe_trace_activation_group_size % 2:
        raise ValueError("--moe-trace-activation-group-size must be even")
    if cfg.moe_trace_queue_depth <= 0:
        raise ValueError("--moe-trace-queue-depth must be positive")
    if cfg.moe_trace_overflow_policy not in {"block", "drop"}:
        raise ValueError("--moe-trace-overflow-policy must be either 'block' or 'drop'")
    if str(cfg.device).lower() != "cuda":
        raise ValueError("MoE tracing v1 supports CUDA devices only")
    if cfg.speculative_algorithm is not None:
        raise ValueError("MoE tracing v1 does not support speculative decoding")
    if cfg.dllm_algorithm is not None:
        raise ValueError("MoE tracing v1 does not support diffusion LLM decoding")
    if cfg.pp_size > 1:
        raise ValueError("MoE tracing v1 does not support pipeline parallelism")
