#!/usr/bin/env python3
"""Manual, repeatable acceptance test for the real Qwen3.5 layer pipeline.

This is deliberately a standalone executable rather than a ``test_*.py``
module: it loads the 35B checkpoint into both physical GPUs and takes a
meaningful amount of time.  It emits exactly one JSON document on stdout, so a
CI/job wrapper can retain the evidence without parsing worker logs (which go
to stderr).  ``--output`` writes the same document to a file.

Example:

  PYTHONPATH=python:. /home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \\
    test/Qwen3_5MoECompat/integration/pipeline_acceptance.py \\
    --output /tmp/qwen35-pipeline-acceptance.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sglang.srt.layers.qwen3_5 import pipeline as q35

_MEMORY_FIELDS = (
    "allocated_bytes",
    "reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def _jsonable_memory(header: dict[str, Any]) -> dict[str, int]:
    """Validate and retain the memory counters carried by a worker ACK."""
    result: dict[str, int] = {}
    for key in _MEMORY_FIELDS:
        value = header.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AssertionError(f"worker ACK has invalid {key}: {value!r}")
        result[key] = value
    return result


def _finite_fp16(payload: bytes, label: str) -> None:
    values = np.frombuffer(payload, dtype=np.float16)
    if not values.size or not bool(np.isfinite(values).all()):
        raise AssertionError(f"{label} contains non-finite FP16 values")


def _require_progress(
    front: dict[str, Any], back: dict[str, Any], expected: int
) -> None:
    q35.Qwen35Pipeline._check_progress(front, back, expected)
    for role, header in (("front", front), ("back", back)):
        if header.get("consumed_len") != expected:
            raise AssertionError(
                f"{role} consumed {header.get('consumed_len')}, expected {expected}"
            )


class PipelineAcceptance:
    """Direct-protocol checks kept alongside high-level CLI-equivalent checks."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tokenizer = q35.load_tokenizer_compat(args.model_dir)
        self.pipe: q35.Qwen35Pipeline | None = None
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "model_dir": str(Path(args.model_dir)),
            "capacity": args.capacity,
            "split_layer": args.split_layer,
            "front_uuid": args.front_uuid,
            "back_uuid": args.back_uuid,
            "validation_steps": args.validation_steps,
            "chat_max_new_tokens": args.chat_max_new_tokens,
        }

    def _open(self) -> q35.Qwen35Pipeline:
        if self.pipe is None:
            self.pipe = q35.Qwen35Pipeline(
                self.args.model_dir,
                capacity=self.args.capacity,
                front_uuid=self.args.front_uuid,
                back_uuid=self.args.back_uuid,
                split_layer=self.args.split_layer,
            )
            front, back = self.pipe.worker_info
            self.report["workers"] = {"front": front, "back": back}
        return self.pipe

    def _begin(self) -> tuple[q35.Qwen35Pipeline, int]:
        pipe = self._open()
        # The public generation method owns this counter too.  A direct frame
        # sequence must still respect the worker's strictly-monotonic epoch
        # contract, hence the shared counter is advanced here.
        pipe._epoch += 1
        epoch = pipe._epoch
        pipe._begin(epoch)
        return pipe, epoch

    def _reset(self) -> dict[str, dict[str, int]]:
        pipe = self._open()
        pipe._reset()
        front, back = pipe.worker_stats
        _require_progress(front, back, 0)
        if front.get("epoch") is not None or back.get("epoch") is not None:
            raise AssertionError("RESET did not clear both worker epochs")
        return {
            "front": _jsonable_memory(front),
            "back": _jsonable_memory(back),
        }

    @staticmethod
    def _prefill_headers(epoch: int, ids: Sequence[int]) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "step_id": 0,
            "expected_prefix_len": 0,
            "token_count": len(ids),
        }

    def _prefill(
        self, ids: Sequence[int], epoch: int
    ) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
        pipe = self._open()
        common = self._prefill_headers(epoch, ids)
        front, front_payload = pipe.front.request(
            {
                "command": "PREFILL_IDS",
                "shape": [len(ids)],
                "dtype": "int32",
                **common,
            },
            q35._int32_payload(ids),
        )
        front_hidden = pipe._expect_hidden(front, front_payload, len(ids))
        back, back_payload = pipe.back.request(
            {
                "command": "PREFILL_HIDDEN",
                "shape": [len(ids), 2048],
                "dtype": "float16",
                **common,
            },
            front_hidden,
        )
        back_hidden = pipe._expect_hidden(back, back_payload, 1)
        _require_progress(front, back, len(ids))
        return front_hidden, back_hidden, front, back

    def _decode(
        self, token_id: int, epoch: int, step_id: int, prefix_len: int
    ) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
        pipe = self._open()
        front, front_payload = pipe.front.request(
            {
                "command": "DECODE_ID",
                "epoch": epoch,
                "step_id": step_id,
                "expected_prefix_len": prefix_len,
                "token_count": 1,
                "shape": [1],
                "dtype": "int32",
            },
            q35._int32_payload([token_id]),
        )
        front_hidden = pipe._expect_hidden(front, front_payload, 1)
        back, back_payload = pipe.back.request(
            {
                "command": "DECODE_HIDDEN",
                "epoch": epoch,
                "step_id": step_id,
                "expected_prefix_len": prefix_len,
                "token_count": 1,
                "shape": [1, 2048],
                "dtype": "float16",
            },
            front_hidden,
        )
        hidden = pipe._expect_hidden(back, back_payload, 1)
        _require_progress(front, back, prefix_len + 1)
        return hidden, front, back

    def _direct_prefill_decode(self, ids: Sequence[int]) -> dict[str, Any]:
        """Exercise one cached decode and return both prefill/decode peaks."""
        if not ids:
            raise AssertionError("direct prefill needs at least one token")
        pipe, epoch = self._begin()
        try:
            _, hidden, front_prefill, back_prefill = self._prefill(ids, epoch)
            _finite_fp16(hidden, "back prefill hidden")
            token_id, _ = pipe._sample(hidden, epoch, 0, len(ids))
            decoded, front_decode, back_decode = self._decode(
                token_id, epoch, 1, len(ids)
            )
            _finite_fp16(decoded, "back first decode hidden")
            return {
                "prefill": {
                    "front": _jsonable_memory(front_prefill),
                    "back": _jsonable_memory(back_prefill),
                },
                "first_decode": {
                    "front": _jsonable_memory(front_decode),
                    "back": _jsonable_memory(back_decode),
                },
                "prefix_len": len(ids),
                "decode_token_id": token_id,
                "consumed_len": len(ids) + 1,
            }
        finally:
            self._reset()

    def _ids(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise AssertionError("configured raw prompt tokenized to no IDs")
        return list(ids)

    def _chat_ids(self, text: str) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not ids:
            raise AssertionError("configured chat prompt tokenized to no IDs")
        return list(ids)

    @staticmethod
    def _stop_reason(ids: Sequence[int], max_new_tokens: int) -> str:
        if ids and ids[-1] in q35.EOS_TOKEN_IDS:
            return "eos"
        if len(ids) == max_new_tokens:
            return "max_new_tokens"
        raise AssertionError("generation stopped before EOS or max_new_tokens")

    def _validate_stateless(self, raw_ids: Sequence[int]) -> None:
        pipe = self._open()
        generated = pipe.generate_ids(
            raw_ids,
            max_new_tokens=self.args.validation_steps,
            # This acceptance run must execute all requested steps even if a
            # model happens to choose one of its normal EOS IDs early.
            eos_token_ids=(),
            validate_stateless=True,
        )
        metrics = [dict(metric) for metric in pipe.last_validation]
        if (
            len(generated) != self.args.validation_steps
            or len(metrics) != self.args.validation_steps
        ):
            raise AssertionError(
                "validate_stateless did not produce one result per step"
            )
        for index, metric in enumerate(metrics):
            if metric.get("step_id") != index:
                raise AssertionError(f"validation metric has unexpected step: {metric}")
            if metric.get("cached_token") != metric.get("stateless_token"):
                raise AssertionError(f"cached/stateless greedy mismatch: {metric}")
            for name, value in metric.items():
                if name.endswith(("nrmse", "max_abs")) and not (
                    isinstance(value, (int, float)) and math.isfinite(value)
                ):
                    raise AssertionError(
                        f"non-finite validation metric {name}: {value!r}"
                    )
        self.report["validate_stateless"] = {
            "prompt_token_count": len(raw_ids),
            "generated_token_ids": generated,
            # Preserve every field provided by the implementation, including
            # future router metrics, instead of hard-coding today's schema.
            "steps": metrics,
        }

    def _reset_determinism(self, a_ids: Sequence[int], b_ids: Sequence[int]) -> None:
        pipe = self._open()
        a_first = pipe.generate_ids(a_ids, max_new_tokens=2, eos_token_ids=())
        first_reset = self._worker_reset_snapshot()
        b = pipe.generate_ids(b_ids, max_new_tokens=2, eos_token_ids=())
        second_reset = self._worker_reset_snapshot()
        a_second = pipe.generate_ids(a_ids, max_new_tokens=2, eos_token_ids=())
        third_reset = self._worker_reset_snapshot()
        if a_first != a_second:
            raise AssertionError("A -> reset -> B -> reset -> A was not deterministic")
        self.report["reset_determinism"] = {
            "a_first": a_first,
            "b": b,
            "a_second": a_second,
            "reset_memory": [first_reset, second_reset, third_reset],
        }

    def _worker_reset_snapshot(self) -> dict[str, dict[str, int]]:
        pipe = self._open()
        front, back = pipe.worker_stats
        _require_progress(front, back, 0)
        if front.get("epoch") is not None or back.get("epoch") is not None:
            raise AssertionError("high-level generation did not reset both workers")
        return {"front": _jsonable_memory(front), "back": _jsonable_memory(back)}

    def _chat_determinism(self, chat_ids: Sequence[int]) -> None:
        pipe = self._open()
        first = pipe.generate_ids(
            chat_ids,
            max_new_tokens=self.args.chat_max_new_tokens,
        )
        first_stop = self._stop_reason(first, self.args.chat_max_new_tokens)
        second = pipe.generate_ids(
            chat_ids,
            max_new_tokens=self.args.chat_max_new_tokens,
        )
        second_stop = self._stop_reason(second, self.args.chat_max_new_tokens)
        if first != second or first_stop != second_stop:
            raise AssertionError(
                "two continuous real chat requests were not deterministic"
            )
        self.report["chat_determinism"] = {
            "prompt_token_count": len(chat_ids),
            "first_token_ids": first,
            "second_token_ids": second,
            "stop_reason": first_stop,
            "reset_memory": self._worker_reset_snapshot(),
        }

    def _full_prefill(self, seed_ids: Sequence[int]) -> None:
        repeated = (
            list(seed_ids) * ((self.args.capacity + len(seed_ids) - 1) // len(seed_ids))
        )[: self.args.capacity]
        pipe, epoch = self._begin()
        overflow_error = ""
        reset_memory: dict[str, dict[str, int]] | None = None
        try:
            front_hidden, back_hidden, front, back = self._prefill(repeated, epoch)
            _finite_fp16(front_hidden, "front 2048-token prefill hidden")
            _finite_fp16(back_hidden, "back 2048-token prefill hidden")
            _require_progress(front, back, self.args.capacity)
            self.report["full_prefill_2048"] = {
                "token_count": len(repeated),
                "consumed_len": {
                    "front": front["consumed_len"],
                    "back": back["consumed_len"],
                },
                "front": _jsonable_memory(front),
                "back": _jsonable_memory(back),
                "finite": {"front": True, "back": True},
            }
            try:
                # A 2048-token cache cannot accept a decode.  Exercise the
                # worker-side capacity guard through the real protocol rather
                # than relying only on controller-side input validation.
                pipe.front.request(
                    {
                        "command": "DECODE_ID",
                        "epoch": epoch,
                        "step_id": 1,
                        "expected_prefix_len": self.args.capacity,
                        "token_count": 1,
                        "shape": [1],
                        "dtype": "int32",
                    },
                    q35._int32_payload([0]),
                )
            except q35.PipelineWorkerError as exc:
                overflow_error = str(exc)
            else:
                raise AssertionError(
                    "decode at the full cache capacity unexpectedly succeeded"
                )
        finally:
            reset_memory = self._reset()
        if "capacity" not in overflow_error.lower():
            raise AssertionError(
                f"full-cache decode did not report capacity exhaustion: {overflow_error!r}"
            )
        assert reset_memory is not None
        recovered = pipe.generate_ids(seed_ids, max_new_tokens=1, eos_token_ids=())
        if len(recovered) != 1:
            raise AssertionError("pipeline did not recover after capacity reset")
        self.report["capacity_overflow"] = {
            "error": overflow_error.splitlines()[0],
            "both_reset_memory": reset_memory,
            "recovered_token_ids": recovered,
            "post_recovery_reset_memory": self._worker_reset_snapshot(),
        }

    def _reset_memory_stability(self, ids: Sequence[int]) -> None:
        # Let all allocator workspaces appear before taking the baseline.
        self._direct_prefill_decode(ids)
        snapshots: list[dict[str, dict[str, int]]] = []
        for _ in range(3):
            self._direct_prefill_decode(ids)
            snapshots.append(self._worker_reset_snapshot())
        for role in ("front", "back"):
            for field in ("allocated_bytes", "reserved_bytes"):
                baseline = snapshots[0][role][field]
                observed = [snapshot[role][field] for snapshot in snapshots]
                if any(value > baseline for value in observed[1:]):
                    raise AssertionError(
                        f"{role} {field} grew after warmup: {observed}"
                    )
        self.report["reset_memory_stability"] = {"after_warmup_resets": snapshots}

    def _bad_protocol_and_recovery(self, ids: Sequence[int]) -> None:
        """Poison one real worker with a duplicate step, then prove recovery."""
        pipe, epoch = self._begin()
        error_text = ""
        try:
            _, hidden, _, _ = self._prefill(ids, epoch)
            token_id, _ = pipe._sample(hidden, epoch, 0, len(ids))
            # First delivery is valid, so the front worker advances to step 1.
            front, front_payload = pipe.front.request(
                {
                    "command": "DECODE_ID",
                    "epoch": epoch,
                    "step_id": 1,
                    "expected_prefix_len": len(ids),
                    "token_count": 1,
                    "shape": [1],
                    "dtype": "int32",
                },
                q35._int32_payload([token_id]),
            )
            pipe._expect_hidden(front, front_payload, 1)
            try:
                # Same identity and prefix are deliberately replayed.  This is
                # an actual worker protocol error, not a mocked controller path.
                pipe.front.request(
                    {
                        "command": "DECODE_ID",
                        "epoch": epoch,
                        "step_id": 1,
                        "expected_prefix_len": len(ids),
                        "token_count": 1,
                        "shape": [1],
                        "dtype": "int32",
                    },
                    q35._int32_payload([token_id]),
                )
            except q35.PipelineWorkerError as exc:
                error_text = str(exc)
            else:
                raise AssertionError("duplicate DECODE_ID unexpectedly succeeded")
        finally:
            reset_memory = self._reset()
        if "step" not in error_text.lower():
            raise AssertionError(
                f"duplicate-step error was not reported: {error_text!r}"
            )
        recovered = pipe.generate_ids(ids, max_new_tokens=1, eos_token_ids=())
        if len(recovered) != 1:
            raise AssertionError("pipeline did not recover after protocol reset")
        self.report["protocol_error_recovery"] = {
            "error": error_text.splitlines()[0],
            "both_reset_memory": reset_memory,
            "recovered_token_ids": recovered,
            "post_recovery_reset_memory": self._worker_reset_snapshot(),
        }

    def _bad_epoch_and_recovery(self, ids: Sequence[int]) -> None:
        """Prove an epoch mismatch poisons a real cache until both sides reset."""
        pipe, epoch = self._begin()
        epoch_error = ""
        poison_error = ""
        reset_memory: dict[str, dict[str, int]] | None = None
        request = {
            "command": "DECODE_ID",
            "step_id": 1,
            "expected_prefix_len": len(ids),
            "token_count": 1,
            "shape": [1],
            "dtype": "int32",
        }
        try:
            self._prefill(ids, epoch)
            try:
                pipe.front.request(
                    {**request, "epoch": epoch + 1}, q35._int32_payload([0])
                )
            except q35.PipelineWorkerError as exc:
                epoch_error = str(exc)
            else:
                raise AssertionError("mismatched epoch unexpectedly succeeded")
            try:
                # A correct epoch must now be rejected as well: worker_main
                # marks both its local request state and runner cache poisoned
                # after the preceding protocol error.
                pipe.front.request({**request, "epoch": epoch}, q35._int32_payload([0]))
            except q35.PipelineWorkerError as exc:
                poison_error = str(exc)
            else:
                raise AssertionError("poisoned cache accepted a post-error decode")
        finally:
            reset_memory = self._reset()
        if "epoch" not in epoch_error.lower():
            raise AssertionError(f"epoch mismatch was not reported: {epoch_error!r}")
        if "poison" not in poison_error.lower():
            raise AssertionError(f"cache was not visibly poisoned: {poison_error!r}")
        assert reset_memory is not None
        recovered = pipe.generate_ids(ids, max_new_tokens=1, eos_token_ids=())
        if len(recovered) != 1:
            raise AssertionError("pipeline did not recover after epoch reset")
        self.report["epoch_error_recovery"] = {
            "epoch_error": epoch_error.splitlines()[0],
            "poison_error": poison_error.splitlines()[0],
            "both_reset_memory": reset_memory,
            "recovered_token_ids": recovered,
            "post_recovery_reset_memory": self._worker_reset_snapshot(),
        }

    def _back_half_step_failure_and_recovery(self, ids: Sequence[int]) -> None:
        """Make real front decode succeed, then make real back reject that step.

        The test-only request wrapper changes one back-worker prefix field.
        Both CUDA workers and the public ``generate_ids`` cleanup path remain
        real: back poisons itself, and the controller must reset both halves
        automatically from ``generate_ids``'s ``finally`` block.
        """
        pipe = self._open()
        original_request = pipe.back.request
        injected = False
        front_half_step_ack: dict[str, Any] | None = None

        def reject_one_back_decode(header: dict[str, Any], payload: bytes = b""):
            nonlocal injected, front_half_step_ack
            if not injected and header.get("command") == "DECODE_HIDDEN":
                injected = True
                front_half_step_ack = dict(pipe.front.last_response or {})
                malformed = dict(header)
                malformed["expected_prefix_len"] = header["expected_prefix_len"] + 1
                return original_request(malformed, payload)
            return original_request(header, payload)

        pipe.back.request = reject_one_back_decode
        error_text = ""
        try:
            try:
                pipe.generate_ids(ids, max_new_tokens=2, eos_token_ids=())
            except q35.PipelineWorkerError as exc:
                error_text = str(exc)
            else:
                raise AssertionError("injected back half-step failure was not reported")
        finally:
            pipe.back.request = original_request
        if not injected or "prefix" not in error_text.lower():
            raise AssertionError(
                f"back half-step error was not reported: {error_text!r}"
            )
        if (
            front_half_step_ack is None
            or front_half_step_ack.get("kind") != "HIDDEN"
            or front_half_step_ack.get("step_id") != 1
            or front_half_step_ack.get("consumed_len") != len(ids) + 1
        ):
            raise AssertionError(
                f"front did not complete the injected half-step: {front_half_step_ack!r}"
            )
        automatic_reset = self._worker_reset_snapshot()
        recovered = pipe.generate_ids(ids, max_new_tokens=1, eos_token_ids=())
        if len(recovered) != 1:
            raise AssertionError(
                "pipeline did not recover after a back half-step failure"
            )
        self.report["back_half_step_failure_recovery"] = {
            "error": error_text.splitlines()[0],
            "front_half_step_ack": front_half_step_ack,
            "automatic_both_reset_memory": automatic_reset,
            "recovered_token_ids": recovered,
            "post_recovery_reset_memory": self._worker_reset_snapshot(),
        }

    def run(self) -> dict[str, Any]:
        pipe = self._open()
        raw_a = self._ids(self.args.raw_prompt_a)
        raw_b = self._ids(self.args.raw_prompt_b)
        chat = self._chat_ids(self.args.chat_prompt)
        if len(raw_a) + self.args.validation_steps > self.args.capacity:
            raise AssertionError("validation prompt plus steps exceeds capacity")
        if len(chat) + self.args.chat_max_new_tokens > self.args.capacity:
            raise AssertionError("chat prompt plus generation exceeds capacity")

        self._validate_stateless(raw_a)
        self._reset_determinism(raw_a, raw_b)
        self._chat_determinism(chat)
        self._full_prefill(raw_a)
        self.report["first_decode_peak"] = self._direct_prefill_decode(raw_a)
        self._reset_memory_stability(raw_a)
        self._bad_protocol_and_recovery(raw_a)
        self._bad_epoch_and_recovery(raw_a)
        self._back_half_step_failure_and_recovery(raw_a)
        self.report["final_worker_stats"] = {
            "front": pipe.worker_stats[0],
            "back": pipe.worker_stats[1],
        }
        return self.report

    def close(self) -> None:
        if self.pipe is None:
            return
        try:
            # RESET is permitted for inactive/poisoned workers and is the only
            # operation allowed after a failed half-step.  Always try it before
            # closing the child processes.
            self._reset()
        finally:
            self.pipe.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=q35.MODEL_DIR_DEFAULT)
    parser.add_argument("--front-uuid", default=q35.DEFAULT_FRONT_UUID)
    parser.add_argument("--back-uuid", default=q35.DEFAULT_BACK_UUID)
    parser.add_argument("--split-layer", type=int, default=q35.DEFAULT_SPLIT_LAYER)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--chat-max-new-tokens", type=int, default=8)
    parser.add_argument("--raw-prompt-a", default="Hello")
    parser.add_argument(
        "--raw-prompt-b", default="Explain cache reuse in one sentence."
    )
    parser.add_argument("--chat-prompt", default="请用一句话解释 KV 缓存。")
    parser.add_argument(
        "--output", type=Path, help="also write the one JSON report here"
    )
    args = parser.parse_args(argv)
    if args.capacity != 2048:
        parser.error("this acceptance contract requires --capacity 2048")
    if args.validation_steps < 8:
        parser.error("this acceptance contract requires at least 8 validation steps")
    if args.chat_max_new_tokens < 1:
        parser.error("--chat-max-new-tokens must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    acceptance: PipelineAcceptance | None = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "model_dir": str(Path(args.model_dir)),
        "capacity": args.capacity,
        "split_layer": args.split_layer,
        "front_uuid": args.front_uuid,
        "back_uuid": args.back_uuid,
        "validation_steps": args.validation_steps,
        "chat_max_new_tokens": args.chat_max_new_tokens,
    }
    status = 0
    try:
        acceptance = PipelineAcceptance(args)
        result = acceptance.run()
        result["ok"] = True
    except BaseException as exc:
        status = 1
        if acceptance is not None:
            result = dict(acceptance.report)
        result.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if acceptance is not None:
            try:
                acceptance.close()
            except BaseException as cleanup_exc:
                status = 1
                result["ok"] = False
                result["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                result["cleanup_traceback"] = traceback.format_exc()
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
