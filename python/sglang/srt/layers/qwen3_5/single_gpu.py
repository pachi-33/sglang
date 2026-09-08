"""Single-process Qwen3.5 ExpertPack inference on one SM70 GPU.

This is the narrow production entry point for the V100 compatibility path.  It
owns one complete 40-layer runner and one reusable request cache.  Generation
is batch one, greedy, text only, and deliberately bypasses SGLang's scheduler
and radix cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .expert_pack.store import ExpertOffloadConfig
from .pipeline import (
    EOS_TOKEN_IDS,
    MODEL_DIR_DEFAULT,
    MODEL_VOCAB_SIZE,
    TOKENIZER_VOCAB_SIZE,
    load_tokenizer_compat,
)
from .runner import Qwen35StatelessRunner, SingleRequestCache

EXPERT_PACK_MANIFEST_DEFAULT = (
    "/home/yaozhenyang/huggingface/"
    "Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json"
)
NUM_HIDDEN_LAYERS = 40


def _require_single_sm70() -> torch.device:
    """Return the sole CUDA device after enforcing the deployment contract."""
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 ExpertPack inference requires CUDA")
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(
            "Qwen3.5 ExpertPack inference requires exactly one visible CUDA "
            f"device, got {visible}"
        )
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(
            "Qwen3.5 ExpertPack inference requires an SM70 V100, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return device


def _reset_peak_memory_stats(device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory_stats(device: torch.device) -> dict[str, int]:
    """Capture the memory evidence used by the 16 GiB acceptance gate."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": peak_reserved,
        "total_memory_bytes": int(total_bytes),
        "free_after_bytes": int(free_bytes),
        "peak_reserved_margin_bytes": int(total_bytes) - peak_reserved,
    }


class Qwen35SingleGPU:
    """A complete 40-layer runner with one transactional request cache."""

    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR_DEFAULT,
        *,
        expert_pack_manifest: str | Path = EXPERT_PACK_MANIFEST_DEFAULT,
        expert_cache_mib: int = 7168,
        expert_stage_slots: int = 16,
        expert_io_workers: int = 2,
        capacity: int = 2048,
        stats_path: str | Path | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be a Python int")
        if not 1 <= capacity <= 2048:
            raise ValueError("capacity must be in [1,2048]")

        device = _require_single_sm70()
        config = ExpertOffloadConfig(
            manifest_path=expert_pack_manifest,
            cache_mib=expert_cache_mib,
            stage_slots=expert_stage_slots,
            io_workers=expert_io_workers,
            stats_path=stats_path,
        )
        runner = Qwen35StatelessRunner(
            model_dir,
            range(NUM_HIDDEN_LAYERS),
            device=device,
            load_globals=True,
            expert_offload=config,
        )
        try:
            cache = runner.allocate_request_cache(capacity)
        except BaseException:
            runner.close()
            raise

        self.model_dir = Path(model_dir)
        self.capacity = capacity
        self.device = device
        self.expert_offload = config
        self.runner = runner
        self.cache: SingleRequestCache = cache
        self._closed = False
        self._failed = False
        self._request_count = 0
        self._last_generation: dict[str, Any] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed or bool(getattr(self.runner, "failed", False))

    @property
    def stats(self) -> dict[str, Any]:
        value = getattr(self.runner, "expert_stats", None)
        if callable(value):
            value = value()
        result = {} if value is None else dict(value)
        result["request_count"] = self._request_count
        result["last_generation"] = (
            None if self._last_generation is None else dict(self._last_generation)
        )
        return result

    def _sample(self, hidden: torch.Tensor) -> int:
        last_hidden = hidden[-1:].contiguous()
        logits = self.runner.logits(self.runner.final_hidden(last_hidden))
        if logits.shape != (1, MODEL_VOCAB_SIZE):
            raise RuntimeError(
                "Qwen3.5 LM head returned an invalid shape: "
                f"expected (1,{MODEL_VOCAB_SIZE}), got {tuple(logits.shape)}"
            )
        logits[:, TOKENIZER_VOCAB_SIZE:MODEL_VOCAB_SIZE] = float("-inf")
        return int(torch.argmax(logits[0]).item())

    @staticmethod
    def _validate_token_sequence(tokens: Sequence[int], label: str) -> list[int]:
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in tokens
        ):
            raise TypeError(f"{label} must contain Python ints")
        return list(tokens)

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int] = EOS_TOKEN_IDS,
    ) -> list[int]:
        """Generate greedily and reset the sole cache after every active request."""
        if self._closed:
            raise RuntimeError("single-GPU backend is closed")
        if self.failed:
            raise RuntimeError(
                "single-GPU ExpertPack backend has failed; restart the process"
            )
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be a Python int")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")

        ids = self._validate_token_sequence(prompt_ids, "prompt_ids")
        if not ids or any(token < 0 or token >= TOKENIZER_VOCAB_SIZE for token in ids):
            raise ValueError(
                "prompt_ids must be a nonempty sequence of valid token IDs"
            )
        if len(ids) + max_new_tokens > self.capacity:
            raise ValueError("prompt plus generated tokens exceeds cache capacity")
        if max_new_tokens == 0:
            return []

        eos_values = self._validate_token_sequence(eos_token_ids, "eos_token_ids")
        eos = frozenset(eos_values)
        if any(token < 0 or token >= TOKENIZER_VOCAB_SIZE for token in eos):
            raise ValueError("eos_token_ids contains an invalid token")

        request_started = False
        primary_error: BaseException | None = None
        generated: list[int] = []
        request_started_ns = 0
        first_token_ns: int | None = None
        previous_token_ns: int | None = None
        itl_ms: list[float] = []
        self._request_count += 1
        request_index = self._request_count
        try:
            request_started = True
            request_started_ns = time.perf_counter_ns()
            _reset_peak_memory_stats(self.device)
            with torch.inference_mode():
                input_ids = torch.tensor(
                    ids, device=self.device, dtype=torch.int32
                ).contiguous()
                hidden = self.runner.embed(input_ids)
                hidden = self.runner.prefill_hidden(hidden, cache=self.cache)
                generated = [self._sample(hidden)]
                first_token_ns = time.perf_counter_ns()
                previous_token_ns = first_token_ns

                while len(generated) < max_new_tokens and generated[-1] not in eos:
                    prefix_len = len(ids) + len(generated) - 1
                    decode_id = torch.tensor(
                        [generated[-1]], device=self.device, dtype=torch.int32
                    ).contiguous()
                    hidden = self.runner.embed(decode_id)
                    hidden = self.runner.decode_hidden(
                        hidden,
                        cache=self.cache,
                        expected_prefix_len=prefix_len,
                    )
                    generated.append(self._sample(hidden))
                    token_ns = time.perf_counter_ns()
                    assert previous_token_ns is not None
                    itl_ms.append((token_ns - previous_token_ns) / 1e6)
                    previous_token_ns = token_ns
            return generated
        except BaseException as exc:
            primary_error = exc
            if bool(getattr(self.runner, "failed", False)):
                self._failed = True
            raise
        finally:
            if request_started:
                cleanup_error: BaseException | None = None
                try:
                    if self.failed:
                        # A store failure is process-fatal.  Retain an explicit
                        # poisoned marker so no caller can mistake this cache for
                        # reusable state while shutdown is pending.
                        self.cache.poisoned = True
                    else:
                        self.runner.reset_request_cache(self.cache)
                except BaseException as exc:
                    cleanup_error = exc
                    self._failed = True
                    self.cache.poisoned = True
                finally:
                    finished_ns = time.perf_counter_ns()
                    try:
                        memory = _cuda_memory_stats(self.device)
                    except BaseException as exc:
                        memory = {"measurement_error": f"{type(exc).__name__}: {exc}"}
                    self._last_generation = {
                        "request_index": request_index,
                        "profile": "cold" if request_index == 1 else "warm",
                        "status": (
                            "failed"
                            if primary_error is not None or cleanup_error is not None
                            else "ok"
                        ),
                        "prompt_tokens": len(ids),
                        "completion_tokens": len(generated),
                        "stopped_on_eos": bool(generated and generated[-1] in eos),
                        "ttft_ms": (
                            None
                            if first_token_ns is None
                            else (first_token_ns - request_started_ns) / 1e6
                        ),
                        "itl_ms": itl_ms,
                        "mean_itl_ms": (sum(itl_ms) / len(itl_ms) if itl_ms else None),
                        "total_ms": (finished_ns - request_started_ns) / 1e6,
                        "error": (
                            f"{type(primary_error).__name__}: {primary_error}"
                            if primary_error is not None
                            else (
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                                if cleanup_error is not None
                                else None
                            )
                        ),
                        "memory": memory,
                    }
                if primary_error is None and cleanup_error is not None:
                    raise cleanup_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.runner.close()

    def __enter__(self) -> "Qwen35SingleGPU":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    parser.add_argument("--expert-pack-manifest", default=EXPERT_PACK_MANIFEST_DEFAULT)
    parser.add_argument("--expert-cache-mib", type=int, default=7168)
    parser.add_argument("--expert-stage-slots", type=int, default=16)
    parser.add_argument("--expert-io-workers", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--stats-path")
    parser.add_argument("--prompt", help="user prompt; reads stdin when omitted")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="tokenize prompt literally instead of applying the chat template",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--print-token-ids", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt:
        raise SystemExit("prompt must not be empty")
    tokenizer = load_tokenizer_compat(args.model_dir)
    if args.raw_prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )

    with Qwen35SingleGPU(
        args.model_dir,
        expert_pack_manifest=args.expert_pack_manifest,
        expert_cache_mib=args.expert_cache_mib,
        expert_stage_slots=args.expert_stage_slots,
        expert_io_workers=args.expert_io_workers,
        capacity=args.capacity,
        stats_path=args.stats_path,
    ) as backend:
        try:
            generated = backend.generate_ids(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=EOS_TOKEN_IDS,
            )
        finally:
            # Fatal requests still need diagnostic and memory evidence.  The
            # original exception continues out of ``main`` for a nonzero exit.
            try:
                stats = backend.stats
            except Exception as exc:
                stats = {"stats_error": f"{type(exc).__name__}: {exc}"}
            print(
                "QWEN35_SINGLE_GPU=" + json.dumps(stats, default=str, sort_keys=True),
                file=sys.stderr,
            )
    if args.print_token_ids:
        print("TOKEN_IDS=" + json.dumps(generated), file=sys.stderr)
    visible = [token for token in generated if token not in EOS_TOKEN_IDS]
    print(tokenizer.decode(visible, skip_special_tokens=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
