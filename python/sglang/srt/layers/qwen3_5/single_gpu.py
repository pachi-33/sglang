"""Single-process Qwen3.5 ExpertPack inference on one supported GPU.

This is the narrow production entry point for the validated SM70/SM89
compatibility path.  It owns one complete 40-layer runner and one reusable
request cache.  Generation is batch one, greedy, text only, and deliberately
bypasses SGLang's scheduler and radix cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from .expert_pack.store import ExpertOffloadConfig
from .expert_trace import (
    PHASE_DECODE,
    PHASE_PREFILL_LAST,
    ExpertTraceConfig,
    ExpertTraceSession,
)
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
SUPPORTED_COMPUTE_CAPABILITIES = frozenset({(7, 0), (8, 9)})


def _require_single_supported_gpu() -> torch.device:
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
    if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
        supported = ", ".join(
            f"SM{major}{minor}"
            for major, minor in sorted(SUPPORTED_COMPUTE_CAPABILITIES)
        )
        raise RuntimeError(
            f"Qwen3.5 ExpertPack inference requires one of {supported}, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return device


def _reset_peak_memory_stats(device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory_stats(device: torch.device) -> dict[str, int]:
    """Capture memory evidence for the active single-GPU profile."""
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


def _cuda_device_uuid(device: torch.device) -> str | None:
    """Return the physical GPU UUID on PyTorch versions that omit it publicly."""
    properties = torch.cuda.get_device_properties(device)
    direct = getattr(properties, "uuid", None)
    if direct is not None:
        return str(direct)
    try:
        physical_index = torch.cuda._get_nvml_device_index(device)
        uuids = torch.cuda._raw_device_uuid_nvml()
        if uuids is not None and 0 <= physical_index < len(uuids):
            return str(uuids[physical_index])
    except (AttributeError, OSError, RuntimeError):
        pass
    return None


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
        expert_trace_dir: str | Path | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be a Python int")
        if not 1 <= capacity <= 2048:
            raise ValueError("capacity must be in [1,2048]")
        trace_config = (
            None if expert_trace_dir is None else ExpertTraceConfig(expert_trace_dir)
        )

        device = _require_single_supported_gpu()
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
        self.expert_trace = trace_config
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
    def expert_trace_enabled(self) -> bool:
        return self.expert_trace is not None

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

    def _latch_trace_failure(self) -> None:
        self._failed = True
        self.cache.poisoned = True

    def _begin_trace_step(
        self,
        session: ExpertTraceSession,
        *,
        phase: int,
        model_input_token_id: int,
        model_input_position: int,
    ) -> Any:
        try:
            return session.begin_step(
                phase=phase,
                model_input_token_id=model_input_token_id,
                model_input_position=model_input_position,
            )
        except BaseException:
            self._latch_trace_failure()
            raise

    def _commit_trace_step(
        self, session: ExpertTraceSession, step: Any, sampled_token_id: int
    ) -> None:
        try:
            session.commit_step(step, sampled_token_id)
        except BaseException:
            self._latch_trace_failure()
            raise

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
        expert_trace: bool = False,
        request_id: str | None = None,
    ) -> list[int]:
        """Generate greedily and reset the sole cache after every active request."""
        return self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            token_callback=None,
            expert_trace=expert_trace,
            request_id=request_id,
        )

    def generate_ids_stream(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        token_callback: Callable[[int], None],
        eos_token_ids: Sequence[int] = EOS_TOKEN_IDS,
        expert_trace: bool = False,
        request_id: str | None = None,
    ) -> list[int]:
        """Generate greedily and synchronously report every sampled token."""
        if not callable(token_callback):
            raise TypeError("token_callback must be callable")
        return self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            token_callback=token_callback,
            expert_trace=expert_trace,
            request_id=request_id,
        )

    def _generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        token_callback: Callable[[int], None] | None,
        expert_trace: bool,
        request_id: str | None,
    ) -> list[int]:
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
        if not isinstance(expert_trace, bool):
            raise TypeError("expert_trace must be a Python bool")
        if expert_trace and self.expert_trace is None:
            raise ValueError(
                "expert trace was requested but no expert_trace_dir is configured"
            )

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
        trace_session: ExpertTraceSession | None = None
        try:
            request_started = True
            request_started_ns = time.perf_counter_ns()
            _reset_peak_memory_stats(self.device)
            if expert_trace:
                assert self.expert_trace is not None
                try:
                    trace_session = ExpertTraceSession(
                        self.expert_trace,
                        request_id=request_id or ("trace-" + uuid.uuid4().hex),
                        max_rows=max_new_tokens,
                        prompt_tokens=len(ids),
                        device=self.device,
                        identity=self._expert_trace_identity(),
                    )
                except BaseException:
                    self._latch_trace_failure()
                    raise
            with torch.inference_mode():
                input_ids = torch.tensor(
                    ids, device=self.device, dtype=torch.int32
                ).contiguous()
                hidden = self.runner.embed(input_ids)
                prefill_trace = (
                    None
                    if trace_session is None
                    else self._begin_trace_step(
                        trace_session,
                        phase=PHASE_PREFILL_LAST,
                        model_input_token_id=ids[-1],
                        model_input_position=len(ids) - 1,
                    )
                )
                prefill_kwargs: dict[str, Any] = {"cache": self.cache}
                if prefill_trace is not None:
                    prefill_kwargs["expert_trace_step"] = prefill_trace
                hidden = self.runner.prefill_hidden(hidden, **prefill_kwargs)
                generated = [self._sample(hidden)]
                if trace_session is not None:
                    assert prefill_trace is not None
                    self._commit_trace_step(trace_session, prefill_trace, generated[-1])
                first_token_ns = time.perf_counter_ns()
                previous_token_ns = first_token_ns
                if token_callback is not None:
                    token_callback(generated[-1])

                while len(generated) < max_new_tokens and generated[-1] not in eos:
                    prefix_len = len(ids) + len(generated) - 1
                    decode_id = torch.tensor(
                        [generated[-1]], device=self.device, dtype=torch.int32
                    ).contiguous()
                    decode_trace = (
                        None
                        if trace_session is None
                        else self._begin_trace_step(
                            trace_session,
                            phase=PHASE_DECODE,
                            model_input_token_id=generated[-1],
                            model_input_position=prefix_len,
                        )
                    )
                    hidden = self.runner.embed(decode_id)
                    decode_kwargs: dict[str, Any] = {
                        "cache": self.cache,
                        "expected_prefix_len": prefix_len,
                    }
                    if decode_trace is not None:
                        decode_kwargs["expert_trace_step"] = decode_trace
                    hidden = self.runner.decode_hidden(hidden, **decode_kwargs)
                    generated.append(self._sample(hidden))
                    if trace_session is not None:
                        assert decode_trace is not None
                        self._commit_trace_step(
                            trace_session, decode_trace, generated[-1]
                        )
                    token_ns = time.perf_counter_ns()
                    assert previous_token_ns is not None
                    itl_ms.append((token_ns - previous_token_ns) / 1e6)
                    previous_token_ns = token_ns
                    if token_callback is not None:
                        token_callback(generated[-1])
            return generated
        except BaseException as exc:
            primary_error = exc
            if bool(getattr(self.runner, "failed", False)):
                self._failed = True
            raise
        finally:
            if request_started:
                cleanup_error: BaseException | None = None
                trace_error: BaseException | None = None
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
                    if trace_session is not None:
                        final_error = primary_error or cleanup_error
                        try:
                            trace_session.finalize(
                                status="failed" if final_error is not None else "ok",
                                stopped_on_eos=bool(generated and generated[-1] in eos),
                                error=final_error,
                            )
                        except BaseException as exc:
                            trace_error = exc
                            self._latch_trace_failure()
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
                            if primary_error is not None
                            or cleanup_error is not None
                            or trace_error is not None
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
                                else (
                                    f"{type(trace_error).__name__}: {trace_error}"
                                    if trace_error is not None
                                    else None
                                )
                            )
                        ),
                        "memory": memory,
                    }
                if primary_error is None and cleanup_error is not None:
                    raise cleanup_error
                if (
                    primary_error is None
                    and cleanup_error is None
                    and trace_error is not None
                ):
                    raise trace_error

    def _expert_trace_identity(self) -> dict[str, Any]:
        store = self.runner.expert_store
        if store is None:
            raise RuntimeError("expert trace requires an initialized ExpertPack store")
        manifest = store.manifest
        capability = torch.cuda.get_device_capability(self.device)
        return {
            "model": {
                "directory": str(self.model_dir.resolve()),
                "config_sha256": manifest.source["config_sha256"],
                "index_sha256": manifest.source["index_sha256"],
            },
            "expert_pack": {
                "manifest": str(store.config.manifest_path),
                "format": manifest.format,
                "pack_sha256": manifest.pack_sha256,
            },
            "device": {
                "uuid": _cuda_device_uuid(self.device),
                "name": torch.cuda.get_device_name(self.device),
                "compute_capability": f"{capability[0]}.{capability[1]}",
            },
        }

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
    parser.add_argument(
        "--expert-trace-output",
        help="output basename for this request's .trace.json and .trace.npz",
    )
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

    trace_output = (
        None
        if args.expert_trace_output is None
        else Path(args.expert_trace_output).expanduser().resolve()
    )
    with Qwen35SingleGPU(
        args.model_dir,
        expert_pack_manifest=args.expert_pack_manifest,
        expert_cache_mib=args.expert_cache_mib,
        expert_stage_slots=args.expert_stage_slots,
        expert_io_workers=args.expert_io_workers,
        capacity=args.capacity,
        stats_path=args.stats_path,
        expert_trace_dir=None if trace_output is None else trace_output.parent,
    ) as backend:
        try:
            trace_kwargs: dict[str, Any] = {}
            if trace_output is not None:
                trace_kwargs = {
                    "expert_trace": True,
                    "request_id": trace_output.name,
                }
            generated = backend.generate_ids(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=EOS_TOKEN_IDS,
                **trace_kwargs,
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
