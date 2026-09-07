"""Two-process, batch-one Qwen3.5 pipeline inference.

The controller intentionally stays CPU-only.  Each worker starts in a fresh
Python process with exactly one GPU UUID visible, which is required by the
process-global target cache in the Triton version used by this compatibility
path.  Hidden states cross the process boundary as contiguous FP16 CPU bytes;
no CUDA IPC, P2P, NCCL, radix cache, or scheduler state is involved.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import socket
import struct
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

MODEL_DIR_DEFAULT = "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16"
V100_UUID = "GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"
SM89_UUID = "GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4"
TOKENIZER_VOCAB_SIZE = 248077
MODEL_VOCAB_SIZE = 248320
EOS_TOKEN_IDS = (248046, 248044)
PROTOCOL_VERSION = 1
ROUTER_TOP_K = 8
ROUTER_CAPTURE_KEY = "router_top8"
NUM_HIDDEN_LAYERS = 40
DEFAULT_SPLIT_LAYER = 17
DEFAULT_FRONT_UUID = SM89_UUID
DEFAULT_BACK_UUID = V100_UUID
SUPPORTED_CAPABILITIES = ((7, 0), (8, 9))
# Kept as public aliases for callers which inspect the production-default
# layout. Runtime validation uses each pipeline instance's layer IDs instead.
FRONT_LAYER_IDS = tuple(range(DEFAULT_SPLIT_LAYER))
BACK_LAYER_IDS = tuple(range(DEFAULT_SPLIT_LAYER, NUM_HIDDEN_LAYERS))

_MAGIC = b"Q35P"
_FRAME = struct.Struct("!4sIQ")
_MAX_HEADER_BYTES = 1 << 20
_MAX_PAYLOAD_BYTES = 32 << 20
_MODULE = "sglang.srt.layers.qwen3_5.pipeline"
_STARTUP_TIMEOUT_SECONDS = 1800.0
_REQUEST_TIMEOUT_SECONDS = 1800.0
_SHUTDOWN_TIMEOUT_SECONDS = 2.0


class PipelineProtocolError(RuntimeError):
    pass


class PipelineWorkerError(RuntimeError):
    pass


RouterWireCapture = dict[int, tuple[tuple[int, ...], tuple[float, ...]]]


def _require_exact_int(value: Any, label: str) -> int:
    """Return a JSON integer, rejecting bool and lossy numeric coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineProtocolError(f"{label} must be an integer")
    return value


def _require_exact_shape(value: Any, expected: Sequence[int], label: str) -> None:
    """Require a JSON integer list equal to ``expected`` without coercion."""
    if not isinstance(value, list) or len(value) != len(expected):
        raise PipelineProtocolError(f"{label} has an invalid shape")
    for index, (dimension, wanted) in enumerate(zip(value, expected)):
        dimension = _require_exact_int(dimension, f"{label}[{index}]")
        if dimension != wanted:
            raise PipelineProtocolError(f"{label} has an invalid shape")


def _parse_router_capture(
    header: dict[str, Any], expected_layer_ids: Sequence[int]
) -> RouterWireCapture:
    """Validate and unpack the compact top-8 routing response header.

    Routes are diagnostic data, but they identify exactly which experts were
    executed.  Treating this as a strict wire contract prevents a malformed
    worker response from being reported as a misleading cache discrepancy.
    """
    raw = header.get(ROUTER_CAPTURE_KEY)
    if not isinstance(raw, dict) or set(raw) != {
        "layer_ids",
        "expert_ids",
        "probabilities",
    }:
        raise PipelineProtocolError("worker returned malformed router capture")
    layer_ids = raw["layer_ids"]
    expert_ids = raw["expert_ids"]
    probabilities = raw["probabilities"]
    wanted = list(expected_layer_ids)
    if (
        not isinstance(layer_ids, list)
        or layer_ids != wanted
        or not isinstance(expert_ids, list)
        or not isinstance(probabilities, list)
        or len(expert_ids) != len(wanted)
        or len(probabilities) != len(wanted)
    ):
        raise PipelineProtocolError("worker router capture has invalid layer shape")
    parsed: RouterWireCapture = {}
    for layer_id, ids, probs in zip(layer_ids, expert_ids, probabilities):
        _require_exact_int(layer_id, "router layer ID")
        if (
            not isinstance(ids, list)
            or not isinstance(probs, list)
            or len(ids) != ROUTER_TOP_K
            or len(probs) != ROUTER_TOP_K
        ):
            raise PipelineProtocolError("worker router capture must contain Top-8 rows")
        parsed_ids: list[int] = []
        parsed_probs: list[float] = []
        for slot, (expert_id, probability) in enumerate(zip(ids, probs)):
            expert_id = _require_exact_int(expert_id, "router expert ID")
            if not 0 <= expert_id < 256:
                raise PipelineProtocolError("worker router expert ID is out of range")
            # JSON has only number, but bool is a Python int.  Require a real
            # floating point value so this remains an unambiguous FP32 route
            # diagnostic rather than accepting silently coerced data.
            if isinstance(probability, bool) or not isinstance(probability, float):
                raise PipelineProtocolError("worker router probability must be a float")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise PipelineProtocolError("worker router probability is invalid")
            parsed_ids.append(expert_id)
            parsed_probs.append(probability)
        if not math.isclose(sum(parsed_probs), 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise PipelineProtocolError(
                "worker router probabilities are not normalized"
            )
        parsed[layer_id] = (tuple(parsed_ids), tuple(parsed_probs))
    return parsed


def _router_capture_summary(
    cached: RouterWireCapture, stateless: RouterWireCapture
) -> dict[str, Any]:
    """Compare all layers while retaining enough detail to debug a route fork."""
    expected = tuple(range(40))
    if tuple(sorted(cached)) != expected or tuple(sorted(stateless)) != expected:
        raise PipelineProtocolError("router validation did not cover all 40 layers")
    layers: list[dict[str, Any]] = []
    mismatch_layers: list[int] = []
    mismatch_slots = 0
    maximum_probability_error = 0.0
    for layer_id in expected:
        cached_ids, cached_probs = cached[layer_id]
        stateless_ids, stateless_probs = stateless[layer_id]
        bad_slots = [
            slot
            for slot, (actual, expected_id) in enumerate(zip(cached_ids, stateless_ids))
            if actual != expected_id
        ]
        max_probability_error = max(
            abs(actual - expected_probability)
            for actual, expected_probability in zip(cached_probs, stateless_probs)
        )
        maximum_probability_error = max(
            maximum_probability_error, max_probability_error
        )
        mismatch_slots += len(bad_slots)
        if bad_slots:
            mismatch_layers.append(layer_id)
        layers.append(
            {
                "layer_id": layer_id,
                "cached_ids": list(cached_ids),
                "stateless_ids": list(stateless_ids),
                "id_mismatch_slots": bad_slots,
                "max_prob_abs": max_probability_error,
            }
        )
    return {
        "mismatch_layers": mismatch_layers,
        "id_mismatch_slot_count": mismatch_slots,
        "max_prob_abs": maximum_probability_error,
        "layers": layers,
    }


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("pipeline worker connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(
    sock: socket.socket, header: dict[str, Any], payload: bytes = b""
) -> None:
    """Send one versioned JSON header followed by an optional binary payload."""
    message = dict(header)
    message["version"] = PROTOCOL_VERSION
    encoded = json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > _MAX_HEADER_BYTES:
        raise PipelineProtocolError("pipeline frame header is too large")
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise PipelineProtocolError("pipeline frame payload is too large")
    sock.sendall(_FRAME.pack(_MAGIC, len(encoded), len(payload)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    """Receive and validate one pipeline protocol frame."""
    magic, header_size, payload_size = _FRAME.unpack(_recv_exact(sock, _FRAME.size))
    if magic != _MAGIC:
        raise PipelineProtocolError("invalid pipeline frame magic")
    if header_size > _MAX_HEADER_BYTES or payload_size > _MAX_PAYLOAD_BYTES:
        raise PipelineProtocolError("pipeline frame exceeds size limit")
    try:
        header = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineProtocolError("invalid pipeline JSON header") from exc
    if not isinstance(header, dict):
        raise PipelineProtocolError("unsupported pipeline protocol version")
    version = _require_exact_int(header.get("version"), "protocol version")
    if version != PROTOCOL_VERSION:
        raise PipelineProtocolError("unsupported pipeline protocol version")
    return header, _recv_exact(sock, payload_size)


def load_tokenizer_compat(model_dir: str | Path):
    """Load the checkpoint tokenizer under tokenizers 0.19 or newer.

    New checkpoints serialize BPE merges as two-element arrays, while the
    tokenizers version paired with Transformers 4.43 expects legacy strings.
    Conversion happens only in memory; checkpoint files are never rewritten.
    """
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    path = Path(model_dir)
    tokenizer_data = json.loads((path / "tokenizer.json").read_text("utf-8"))
    merges = tokenizer_data.get("model", {}).get("merges")
    if not isinstance(merges, list):
        raise ValueError("tokenizer.json is missing model.merges")
    tokenizer_data["model"]["merges"] = [
        " ".join(pair) if isinstance(pair, list) and len(pair) == 2 else pair
        for pair in merges
    ]
    if not all(isinstance(merge, str) for merge in tokenizer_data["model"]["merges"]):
        raise ValueError("tokenizer merges must be strings or two-token arrays")
    backend = Tokenizer.from_str(json.dumps(tokenizer_data, ensure_ascii=False))
    config = json.loads((path / "tokenizer_config.json").read_text("utf-8"))
    forwarded = {
        key: config[key]
        for key in (
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "model_max_length",
            "clean_up_tokenization_spaces",
        )
        if config.get(key) is not None
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **forwarded)
    tokenizer.chat_template = (path / "chat_template.jinja").read_text("utf-8")
    if len(tokenizer) != TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            f"expected tokenizer size {TOKENIZER_VOCAB_SIZE}, got {len(tokenizer)}"
        )
    return tokenizer


def _int32_payload(values: Sequence[int]) -> bytes:
    packed = array.array("i", values)
    if packed.itemsize != 4:
        raise RuntimeError("native signed int is not 32 bits")
    return packed.tobytes()


class _WorkerClient:
    def __init__(
        self,
        role: str,
        gpu_uuid: str,
        model_dir: str | Path,
        capacity: int,
        layer_start: int,
        layer_end: int,
    ) -> None:
        parent_sock, child_sock = socket.socketpair()
        self.role = role
        self.sock = parent_sock
        self.sock.settimeout(_STARTUP_TIMEOUT_SECONDS)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-m",
            _MODULE,
            "--worker-role",
            role,
            "--worker-fd",
            str(child_sock.fileno()),
            "--model-dir",
            str(model_dir),
            "--capacity",
            str(capacity),
            "--layer-start",
            str(layer_start),
            "--layer-end",
            str(layer_end),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                pass_fds=(child_sock.fileno(),),
                # Keep terminal SIGINT/SIGHUP on the controller.  Long-lived
                # API workers are shut down through the framed SHUTDOWN command
                # so they can close their CUDA context without a traceback.
                start_new_session=True,
            )
        except Exception:
            parent_sock.close()
            child_sock.close()
            raise
        child_sock.close()
        self.closed = False
        self.ready: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None

    def wait_ready(self) -> dict[str, Any]:
        try:
            header, payload = recv_frame(self.sock)
        except TimeoutError as exc:
            raise PipelineWorkerError(f"{self.role} worker startup timed out") from exc
        if payload or header.get("kind") != "READY":
            raise PipelineProtocolError(f"{self.role} sent an invalid READY frame")
        if header.get("role") != self.role:
            raise PipelineProtocolError(f"{self.role} worker reported wrong role")
        self.ready = header
        self.sock.settimeout(_REQUEST_TIMEOUT_SECONDS)
        return header

    def request(
        self, header: dict[str, Any], payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes]:
        if self.closed:
            raise PipelineWorkerError(f"{self.role} worker is closed")
        try:
            send_frame(self.sock, header, payload)
            response, response_payload = recv_frame(self.sock)
        except TimeoutError as exc:
            raise PipelineWorkerError(f"{self.role} worker request timed out") from exc
        if response.get("role") != self.role:
            raise PipelineProtocolError(
                f"{self.role} worker response has the wrong role"
            )
        if response.get("kind") == "ERROR":
            raise PipelineWorkerError(
                f"{self.role} worker failed: {response.get('error', 'unknown error')}\n"
                f"{response.get('traceback', '')}"
            )
        command = header.get("command")
        stateful_commands = (
            "BEGIN",
            "PREFILL_IDS",
            "PREFILL_HIDDEN",
            "DECODE_ID",
            "DECODE_HIDDEN",
            "VALIDATE_IDS",
            "VALIDATE_HIDDEN",
            "SAMPLE",
        )
        if command in stateful_commands:
            response_epoch = _require_exact_int(
                response.get("epoch"), f"{self.role} response epoch"
            )
            response_step = _require_exact_int(
                response.get("step_id"), f"{self.role} response step_id"
            )
            if response_epoch != header.get("epoch") or response_step != header.get(
                "step_id"
            ):
                raise PipelineProtocolError(
                    f"{self.role} worker response epoch/step does not match request"
                )
        if command == "RESET":
            response_step = _require_exact_int(
                response.get("step_id"), f"{self.role} reset step_id"
            )
            if response.get("epoch") is not None or response_step != -1:
                raise PipelineProtocolError(
                    f"{self.role} worker did not reset its request identity"
                )
        self.last_response = response
        return response, response_payload

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                self.sock.settimeout(_SHUTDOWN_TIMEOUT_SECONDS)
                try:
                    self.request({"command": "SHUTDOWN"})
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=10)
        finally:
            self.closed = True
            self.sock.close()


class Qwen35Pipeline:
    """Configurable two-GPU layer pipeline with one reusable request slot.

    The front worker always owns embedding/final norm/LM head and layers
    ``[0, split_layer)``. The back worker owns the remaining transformer
    layers. GPU architecture is deliberately independent from that role.
    """

    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR_DEFAULT,
        *,
        capacity: int = 2048,
        front_uuid: str | None = None,
        back_uuid: str | None = None,
        split_layer: int = DEFAULT_SPLIT_LAYER,
        # Deprecated physical-device aliases retained for existing launchers.
        # In the new default layout SM89 is front and V100 is back.
        v100_uuid: str | None = None,
        sm89_uuid: str | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be a Python int")
        if not 1 <= capacity <= 2048:
            raise ValueError("capacity must be in [1,2048]")
        if isinstance(split_layer, bool) or not isinstance(split_layer, int):
            raise TypeError("split_layer must be a Python int")
        if not 1 <= split_layer < NUM_HIDDEN_LAYERS:
            raise ValueError(f"split_layer must be in [1,{NUM_HIDDEN_LAYERS - 1}]")
        if front_uuid is not None and sm89_uuid is not None:
            raise ValueError("front_uuid and sm89_uuid aliases are mutually exclusive")
        if back_uuid is not None and v100_uuid is not None:
            raise ValueError("back_uuid and v100_uuid aliases are mutually exclusive")
        resolved_front_uuid = (
            front_uuid
            if front_uuid is not None
            else sm89_uuid if sm89_uuid is not None else DEFAULT_FRONT_UUID
        )
        resolved_back_uuid = (
            back_uuid
            if back_uuid is not None
            else v100_uuid if v100_uuid is not None else DEFAULT_BACK_UUID
        )
        for value, label in (
            (resolved_front_uuid, "front_uuid"),
            (resolved_back_uuid, "back_uuid"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{label} must be a nonempty string")
        if resolved_front_uuid == resolved_back_uuid:
            raise ValueError("front_uuid and back_uuid must identify different GPUs")
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        self.capacity = capacity
        self.split_layer = split_layer
        self.front_uuid = resolved_front_uuid
        self.back_uuid = resolved_back_uuid
        self.front_layer_ids = tuple(range(split_layer))
        self.back_layer_ids = tuple(range(split_layer, NUM_HIDDEN_LAYERS))
        self._epoch = 0
        self._closed = False
        self.last_validation: list[dict[str, Any]] = []
        self.front = _WorkerClient(
            "front",
            resolved_front_uuid,
            self.model_dir,
            capacity,
            0,
            split_layer,
        )
        try:
            self.back = _WorkerClient(
                "back",
                resolved_back_uuid,
                self.model_dir,
                capacity,
                split_layer,
                NUM_HIDDEN_LAYERS,
            )
            front_ready = self.front.wait_ready()
            back_ready = self.back.wait_ready()
            for ready, role, start, end in (
                (front_ready, "front", 0, split_layer),
                (back_ready, "back", split_layer, NUM_HIDDEN_LAYERS),
            ):
                capability = ready.get("capability")
                if not isinstance(capability, list) or len(capability) != 2:
                    raise PipelineProtocolError(
                        f"{role} worker returned malformed capability"
                    )
                parsed_capability = tuple(
                    _require_exact_int(value, f"{role} capability[{index}]")
                    for index, value in enumerate(capability)
                )
                if parsed_capability not in SUPPORTED_CAPABILITIES:
                    raise PipelineProtocolError(
                        f"{role} worker capability {parsed_capability} is not validated"
                    )
                if (
                    _require_exact_int(ready.get("layer_start"), f"{role} layer_start")
                    != start
                    or _require_exact_int(ready.get("layer_end"), f"{role} layer_end")
                    != end
                    or _require_exact_int(ready.get("capacity"), f"{role} capacity")
                    != capacity
                ):
                    raise PipelineProtocolError(
                        f"{role} worker READY does not match requested layout"
                    )
        except Exception:
            self.front.close()
            if hasattr(self, "back"):
                self.back.close()
            raise

    @property
    def worker_info(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.front.ready or {}, self.back.ready or {}

    @property
    def worker_stats(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return each worker's most recent ACK, including peak memory."""
        return self.front.last_response or {}, self.back.last_response or {}

    def __enter__(self) -> "Qwen35Pipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.back.close()
        self.front.close()

    @staticmethod
    def _expect_hidden(response: dict[str, Any], payload: bytes, rows: int) -> bytes:
        _require_exact_shape(response.get("shape"), [rows, 2048], "hidden shape")
        if (
            response.get("kind") != "HIDDEN"
            or response.get("dtype") != "float16"
            or len(payload) != rows * 2048 * 2
        ):
            raise PipelineProtocolError("worker returned malformed hidden states")
        return payload

    @staticmethod
    def _expect_router_capture(
        response: dict[str, Any], expected_layer_ids: Sequence[int], *, enabled: bool
    ) -> RouterWireCapture:
        """Require routes exactly for validation frames, and never otherwise."""
        has_capture = ROUTER_CAPTURE_KEY in response
        if not enabled:
            if has_capture:
                raise PipelineProtocolError(
                    "worker returned router capture outside stateless validation"
                )
            return {}
        if not has_capture:
            raise PipelineProtocolError("worker omitted requested router capture")
        return _parse_router_capture(response, expected_layer_ids)

    @staticmethod
    def _check_progress(
        first: dict[str, Any], second: dict[str, Any], expected_len: int
    ) -> None:
        first_len = _require_exact_int(first.get("consumed_len"), "front consumed_len")
        second_len = _require_exact_int(second.get("consumed_len"), "back consumed_len")
        if first_len != expected_len or second_len != expected_len:
            raise PipelineProtocolError("worker cache lengths diverged")

    def _begin(self, epoch: int) -> None:
        command = {
            "command": "BEGIN",
            "epoch": epoch,
            "step_id": -1,
            "expected_prefix_len": 0,
            "token_count": 0,
        }
        front, front_payload = self.front.request(command)
        back, back_payload = self.back.request(command)
        if (
            front.get("kind") != "OK"
            or back.get("kind") != "OK"
            or front_payload
            or back_payload
        ):
            raise PipelineProtocolError("BEGIN returned an unexpected payload")
        self._check_progress(front, back, 0)

    def _reset(self) -> None:
        errors: list[Exception] = []
        for worker in (self.front, self.back):
            try:
                response, payload = worker.request({"command": "RESET"})
                reset_len = _require_exact_int(
                    response.get("consumed_len"), f"{worker.role} reset consumed_len"
                )
                if response.get("kind") != "OK" or payload or reset_len != 0:
                    raise PipelineProtocolError("invalid RESET response")
            except Exception as exc:
                errors.append(exc)
        if errors:
            self.close()
            raise PipelineWorkerError(
                "failed to reset both pipeline workers"
            ) from errors[0]

    def _sample(
        self,
        hidden_payload: bytes,
        epoch: int,
        step_id: int,
        consumed_len: int,
        *,
        return_logits: bool = False,
    ) -> tuple[int, bytes | None]:
        response, payload = self.front.request(
            {
                "command": "SAMPLE",
                "epoch": epoch,
                "step_id": step_id,
                "expected_prefix_len": consumed_len,
                "token_count": 1,
                "shape": [1, 2048],
                "dtype": "float16",
                "return_logits": return_logits,
            },
            hidden_payload,
        )
        expected_logits_bytes = MODEL_VOCAB_SIZE * 2 if return_logits else 0
        if return_logits:
            _require_exact_shape(
                response.get("logits_shape"),
                [1, MODEL_VOCAB_SIZE],
                "logits shape",
            )
        if (
            response.get("kind") != "TOKEN"
            or len(payload) != expected_logits_bytes
            or (return_logits and response.get("logits_dtype") != "float16")
        ):
            raise PipelineProtocolError("front worker returned an invalid token")
        sampled_len = _require_exact_int(
            response.get("consumed_len"), "sample consumed_len"
        )
        if sampled_len != consumed_len:
            raise PipelineProtocolError("front worker sampled from the wrong prefix")
        token_id = _require_exact_int(response.get("token_id"), "sampled token ID")
        if not 0 <= token_id < TOKENIZER_VOCAB_SIZE:
            raise PipelineProtocolError("front worker sampled an invalid token ID")
        return token_id, payload if return_logits else None

    @staticmethod
    def _numeric_metrics(actual: bytes, expected: bytes) -> tuple[float, float]:
        import numpy as np

        actual_f32 = np.frombuffer(actual, dtype=np.float16).astype(np.float32)
        expected_f32 = np.frombuffer(expected, dtype=np.float16).astype(np.float32)
        if actual_f32.shape != expected_f32.shape:
            raise PipelineProtocolError("validation payload sizes differ")
        actual_finite = np.isfinite(actual_f32)
        expected_finite = np.isfinite(expected_f32)
        equal_nonfinite = (
            (~actual_finite) & (~expected_finite) & (actual_f32 == expected_f32)
        )
        if np.any(actual_finite != expected_finite) or np.any(
            (~actual_finite) & ~equal_nonfinite
        ):
            raise RuntimeError(
                "cached/stateless validation has mismatched non-finite values"
            )
        actual_f32 = actual_f32[actual_finite]
        expected_f32 = expected_f32[expected_finite]
        if not actual_f32.size:
            return 0.0, 0.0
        difference = actual_f32 - expected_f32
        denominator = max(float(np.sqrt(np.mean(expected_f32 * expected_f32))), 1e-8)
        return (
            float(np.sqrt(np.mean(difference * difference)) / denominator),
            float(np.max(np.abs(difference))),
        )

    def _validate_stateless_step(
        self,
        prefix_ids: Sequence[int],
        cached_hidden: bytes,
        cached_token: int,
        cached_logits: bytes,
        cached_routes: RouterWireCapture,
        *,
        epoch: int,
        step_id: int,
    ) -> None:
        """Use full-prefix recomputation strictly as an end-to-end oracle."""
        tokens = len(prefix_ids)
        front_header, front_payload = self.front.request(
            {
                "command": "VALIDATE_IDS",
                "epoch": epoch,
                "step_id": step_id,
                "expected_prefix_len": tokens,
                "token_count": tokens,
                "shape": [tokens],
                "dtype": "int32",
                "capture_router": True,
            },
            _int32_payload(prefix_ids),
        )
        self._expect_hidden(front_header, front_payload, tokens)
        stateless_front_routes = self._expect_router_capture(
            front_header, self.front_layer_ids, enabled=True
        )
        back_header, back_payload = self.back.request(
            {
                "command": "VALIDATE_HIDDEN",
                "epoch": epoch,
                "step_id": step_id,
                "expected_prefix_len": tokens,
                "token_count": tokens,
                "shape": [tokens, 2048],
                "dtype": "float16",
                "capture_router": True,
            },
            front_payload,
        )
        oracle_hidden = self._expect_hidden(back_header, back_payload, 1)
        stateless_back_routes = self._expect_router_capture(
            back_header, self.back_layer_ids, enabled=True
        )
        self._check_progress(front_header, back_header, tokens)
        oracle_token, oracle_logits = self._sample(
            oracle_hidden,
            epoch,
            step_id,
            tokens,
            return_logits=True,
        )
        assert oracle_logits is not None
        hidden_nrmse, hidden_max = self._numeric_metrics(cached_hidden, oracle_hidden)
        logits_nrmse, logits_max = self._numeric_metrics(cached_logits, oracle_logits)
        metrics = {
            "step_id": step_id,
            "prefix_len": tokens,
            "cached_token": cached_token,
            "stateless_token": oracle_token,
            "hidden_nrmse": hidden_nrmse,
            "hidden_max_abs": hidden_max,
            "logits_nrmse": logits_nrmse,
            "logits_max_abs": logits_max,
            "router": _router_capture_summary(
                {**cached_routes},
                {**stateless_front_routes, **stateless_back_routes},
            ),
        }
        self.last_validation.append(metrics)
        if oracle_token != cached_token:
            raise RuntimeError(
                "cached/stateless greedy token mismatch: "
                + json.dumps(metrics, sort_keys=True)
            )

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int] = EOS_TOKEN_IDS,
        validate_stateless: bool = False,
    ) -> list[int]:
        """Generate greedily, resetting the sole request slot on every exit."""
        if self._closed:
            raise RuntimeError("pipeline is closed")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be a Python int")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if not isinstance(validate_stateless, bool):
            raise TypeError("validate_stateless must be bool")
        if any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in prompt_ids
        ):
            raise TypeError("prompt_ids must contain Python ints")
        ids = list(prompt_ids)
        if not ids or any(token < 0 or token >= TOKENIZER_VOCAB_SIZE for token in ids):
            raise ValueError(
                "prompt_ids must be a nonempty sequence of valid token IDs"
            )
        # The last generated token is not inserted into KV/GDN state, but it
        # still belongs to the model's public 2048-token context contract.
        # Keep that total-context limit distinct from the R-1 decode count.
        if len(ids) + max_new_tokens > self.capacity:
            raise ValueError("prompt plus generated tokens exceeds cache capacity")
        if max_new_tokens == 0:
            return []
        if any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in eos_token_ids
        ):
            raise TypeError("eos_token_ids must contain Python ints")
        eos = frozenset(eos_token_ids)
        if any(token < 0 or token >= TOKENIZER_VOCAB_SIZE for token in eos):
            raise ValueError("eos_token_ids contains an invalid token")

        self._epoch += 1
        self.last_validation = []
        epoch = self._epoch
        reset_needed = False
        primary_error: BaseException | None = None
        try:
            # A one-sided BEGIN is already state mutation, so cleanup is
            # required even when the second worker rejects the command.
            reset_needed = True
            self._begin(epoch)
            step_id = 0
            front_header, front_payload = self.front.request(
                {
                    "command": "PREFILL_IDS",
                    "epoch": epoch,
                    "step_id": step_id,
                    "expected_prefix_len": 0,
                    "token_count": len(ids),
                    "shape": [len(ids)],
                    "dtype": "int32",
                    "capture_router": validate_stateless,
                },
                _int32_payload(ids),
            )
            self._expect_hidden(front_header, front_payload, len(ids))
            cached_front_routes = self._expect_router_capture(
                front_header, self.front_layer_ids, enabled=validate_stateless
            )
            back_header, back_payload = self.back.request(
                {
                    "command": "PREFILL_HIDDEN",
                    "epoch": epoch,
                    "step_id": step_id,
                    "expected_prefix_len": 0,
                    "token_count": len(ids),
                    "shape": [len(ids), 2048],
                    "dtype": "float16",
                    "capture_router": validate_stateless,
                },
                front_payload,
            )
            last_hidden = self._expect_hidden(back_header, back_payload, 1)
            cached_back_routes = self._expect_router_capture(
                back_header, self.back_layer_ids, enabled=validate_stateless
            )
            self._check_progress(front_header, back_header, len(ids))
            first_token, first_logits = self._sample(
                last_hidden,
                epoch,
                step_id,
                len(ids),
                return_logits=validate_stateless,
            )
            if validate_stateless:
                assert first_logits is not None
                self._validate_stateless_step(
                    ids,
                    last_hidden,
                    first_token,
                    first_logits,
                    {**cached_front_routes, **cached_back_routes},
                    epoch=epoch,
                    step_id=step_id,
                )
            generated = [first_token]

            while len(generated) < max_new_tokens and generated[-1] not in eos:
                step_id += 1
                prefix_len = len(ids) + len(generated) - 1
                front_header, front_payload = self.front.request(
                    {
                        "command": "DECODE_ID",
                        "epoch": epoch,
                        "step_id": step_id,
                        "expected_prefix_len": prefix_len,
                        "token_count": 1,
                        "shape": [1],
                        "dtype": "int32",
                        "capture_router": validate_stateless,
                    },
                    _int32_payload([generated[-1]]),
                )
                self._expect_hidden(front_header, front_payload, 1)
                cached_front_routes = self._expect_router_capture(
                    front_header, self.front_layer_ids, enabled=validate_stateless
                )
                back_header, back_payload = self.back.request(
                    {
                        "command": "DECODE_HIDDEN",
                        "epoch": epoch,
                        "step_id": step_id,
                        "expected_prefix_len": prefix_len,
                        "token_count": 1,
                        "shape": [1, 2048],
                        "dtype": "float16",
                        "capture_router": validate_stateless,
                    },
                    front_payload,
                )
                last_hidden = self._expect_hidden(back_header, back_payload, 1)
                cached_back_routes = self._expect_router_capture(
                    back_header, self.back_layer_ids, enabled=validate_stateless
                )
                self._check_progress(front_header, back_header, prefix_len + 1)
                token, logits = self._sample(
                    last_hidden,
                    epoch,
                    step_id,
                    prefix_len + 1,
                    return_logits=validate_stateless,
                )
                if validate_stateless:
                    assert logits is not None
                    self._validate_stateless_step(
                        [*ids, *generated],
                        last_hidden,
                        token,
                        logits,
                        {**cached_front_routes, **cached_back_routes},
                        epoch=epoch,
                        step_id=step_id,
                    )
                generated.append(token)
            return generated
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if reset_needed:
                try:
                    self._reset()
                except Exception:
                    if primary_error is None:
                        raise


def _memory_header(torch_module) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "reserved_bytes": int(torch_module.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    }


def _worker_tensor(torch_module, payload: bytes, dtype, shape: tuple[int, ...]):
    expected = 1
    for dimension in shape:
        expected *= dimension
    expected *= torch_module.empty((), dtype=dtype).element_size()
    if len(payload) != expected:
        raise PipelineProtocolError(
            f"payload has {len(payload)} bytes, expected {expected}"
        )
    host = torch_module.frombuffer(bytearray(payload), dtype=dtype).reshape(shape)
    return host.to(device="cuda", non_blocking=False).contiguous()


def _hidden_payload(tensor) -> bytes:
    return tensor.detach().to(device="cpu").contiguous().numpy().tobytes()


def _capture_router_requested(header: dict[str, Any]) -> bool:
    """Read the opt-in diagnostic flag without accepting truthy impostors."""
    capture = header.get("capture_router", False)
    if not isinstance(capture, bool):
        raise PipelineProtocolError("capture_router must be bool")
    return capture


def _router_capture_header(
    torch_module, capture, expected_layer_ids: Sequence[int]
) -> dict[str, Any]:
    """Make the sole, validation-only CUDA-to-CPU copy of route diagnostics."""
    expected = tuple(expected_layer_ids)
    if not isinstance(capture, dict) or tuple(sorted(capture)) != expected:
        raise RuntimeError("router capture did not contain every worker layer")
    ids_rows = []
    probability_rows = []
    for layer_id in expected:
        try:
            ids, probabilities = capture[layer_id]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("router capture entry is malformed") from exc
        if (
            not isinstance(ids, torch_module.Tensor)
            or not isinstance(probabilities, torch_module.Tensor)
            or ids.shape != (ROUTER_TOP_K,)
            or probabilities.shape != (ROUTER_TOP_K,)
            or ids.dtype != torch_module.int32
            or probabilities.dtype != torch_module.float32
            or not ids.is_cuda
            or not probabilities.is_cuda
        ):
            raise RuntimeError("router capture tensors must be CUDA Top-8 int32/FP32")
        ids_rows.append(ids)
        probability_rows.append(probabilities)
    # Stack first so every worker transfers a compact 20x8 block, rather than
    # synchronizing/copying one small tensor per original layer.
    ids_host = torch_module.stack(ids_rows).to(device="cpu").tolist()
    probabilities_host = torch_module.stack(probability_rows).to(device="cpu").tolist()
    for probabilities in probabilities_host:
        if (
            len(probabilities) != ROUTER_TOP_K
            or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities
            )
            or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-4)
        ):
            raise RuntimeError("router capture contains invalid probabilities")
    return {
        ROUTER_CAPTURE_KEY: {
            "layer_ids": list(expected),
            "expert_ids": ids_host,
            "probabilities": probabilities_host,
        }
    }


def _worker_main(
    role: str,
    fd: int,
    model_dir: str,
    capacity: int,
    layer_start: int,
    layer_end: int,
) -> int:
    # Imports happen only after CUDA_VISIBLE_DEVICES was fixed by Popen.
    import torch

    from .runner import Qwen35StatelessRunner

    if role not in ("front", "back"):
        raise ValueError("worker role must be front or back")
    if (
        isinstance(layer_start, bool)
        or not isinstance(layer_start, int)
        or isinstance(layer_end, bool)
        or not isinstance(layer_end, int)
        or not 0 <= layer_start < layer_end <= NUM_HIDDEN_LAYERS
    ):
        raise ValueError("worker layer range is invalid")
    if (role == "front" and layer_start != 0) or (
        role == "back" and layer_end != NUM_HIDDEN_LAYERS
    ):
        raise ValueError("worker layer range does not match its pipeline role")
    sock = socket.socket(fileno=fd)
    layer_ids = tuple(range(layer_start, layer_end))
    runner = Qwen35StatelessRunner(
        model_dir,
        layer_ids,
        device="cuda",
        load_globals=role == "front",
    )
    cache = runner.allocate_request_cache(capacity)
    torch.cuda.synchronize()
    capability = list(torch.cuda.get_device_capability())
    send_frame(
        sock,
        {
            "kind": "READY",
            "role": role,
            "device_name": torch.cuda.get_device_name(),
            "capability": capability,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "capacity": capacity,
            **_memory_header(torch),
        },
    )
    active = False
    poisoned = False
    epoch: int | None = None
    highest_epoch = 0
    last_step = -1

    def progress(kind: str, **extra: Any) -> dict[str, Any]:
        return {
            "kind": kind,
            "role": role,
            "epoch": epoch,
            "step_id": last_step,
            "consumed_len": cache.consumed_len,
            **_memory_header(torch),
            **extra,
        }

    def validate_stateful(header: dict[str, Any], *, next_step: bool) -> None:
        if not active or poisoned or cache.poisoned:
            raise RuntimeError("worker request is inactive or poisoned")
        received_epoch = header.get("epoch")
        received_step = header.get("step_id")
        received_prefix = header.get("expected_prefix_len")
        if (
            isinstance(received_epoch, bool)
            or not isinstance(received_epoch, int)
            or received_epoch != epoch
        ):
            raise RuntimeError("request epoch mismatch")
        wanted_step = last_step + 1 if next_step else last_step
        if (
            isinstance(received_step, bool)
            or not isinstance(received_step, int)
            or received_step != wanted_step
        ):
            raise RuntimeError("request step mismatch")
        if (
            isinstance(received_prefix, bool)
            or not isinstance(received_prefix, int)
            or received_prefix != cache.consumed_len
        ):
            raise RuntimeError("request prefix length mismatch")

    try:
        while True:
            header, payload = recv_frame(sock)
            command = header.get("command")
            try:
                if command == "SHUTDOWN":
                    send_frame(sock, {"kind": "OK", "role": role})
                    return 0
                if command == "RESET":
                    runner.reset_request_cache(cache)
                    active = False
                    poisoned = False
                    epoch = None
                    last_step = -1
                    send_frame(sock, progress("OK"))
                    continue
                if command == "BEGIN":
                    if payload:
                        raise PipelineProtocolError("BEGIN must not have a payload")
                    requested_epoch = header.get("epoch")
                    requested_step = header.get("step_id")
                    requested_prefix = header.get("expected_prefix_len")
                    requested_tokens = header.get("token_count")
                    if (
                        isinstance(requested_epoch, bool)
                        or not isinstance(requested_epoch, int)
                        or requested_epoch <= highest_epoch
                    ):
                        raise PipelineProtocolError(
                            "BEGIN requires a strictly increasing positive epoch"
                        )
                    if (
                        active
                        or poisoned
                        or cache.poisoned
                        or isinstance(requested_step, bool)
                        or not isinstance(requested_step, int)
                        or requested_step != -1
                        or isinstance(requested_prefix, bool)
                        or not isinstance(requested_prefix, int)
                        or requested_prefix != 0
                        or isinstance(requested_tokens, bool)
                        or not isinstance(requested_tokens, int)
                        or requested_tokens != 0
                    ):
                        raise RuntimeError(
                            "BEGIN requires an explicitly reset, inactive worker"
                        )
                    runner.reset_request_cache(cache)
                    torch.cuda.reset_peak_memory_stats()
                    active = True
                    poisoned = False
                    epoch = requested_epoch
                    highest_epoch = requested_epoch
                    last_step = -1
                    send_frame(sock, progress("OK"))
                    continue
                if command in ("PREFILL_IDS", "PREFILL_HIDDEN"):
                    validate_stateful(header, next_step=True)
                    if last_step != -1 or cache.consumed_len != 0:
                        raise RuntimeError("PREFILL is allowed only once after BEGIN")
                    tokens = header.get("token_count")
                    if (
                        isinstance(tokens, bool)
                        or not isinstance(tokens, int)
                        or not 1 <= tokens <= capacity
                    ):
                        raise PipelineProtocolError("invalid prefill token_count")
                    if role == "front" and command == "PREFILL_IDS":
                        _require_exact_shape(
                            header.get("shape"), [tokens], "PREFILL_IDS shape"
                        )
                        if header.get("dtype") != "int32":
                            raise PipelineProtocolError("invalid PREFILL_IDS metadata")
                        input_ids = _worker_tensor(
                            torch, payload, torch.int32, (tokens,)
                        )
                        hidden = runner.embed(input_ids)
                    elif role == "back" and command == "PREFILL_HIDDEN":
                        _require_exact_shape(
                            header.get("shape"),
                            [tokens, 2048],
                            "PREFILL_HIDDEN shape",
                        )
                        if header.get("dtype") != "float16":
                            raise PipelineProtocolError(
                                "invalid PREFILL_HIDDEN metadata"
                            )
                        hidden = _worker_tensor(
                            torch, payload, torch.float16, (tokens, 2048)
                        )
                    else:
                        raise PipelineProtocolError(
                            "prefill command does not match worker role"
                        )
                    capture_router = _capture_router_requested(header)
                    router_capture = {} if capture_router else None
                    with torch.inference_mode():
                        hidden = runner.prefill_hidden(
                            hidden, cache=cache, router_capture=router_capture
                        )
                    last_step = int(header["step_id"])
                    output = hidden if role == "front" else hidden[-1:]
                    router_header = (
                        _router_capture_header(
                            torch,
                            router_capture,
                            runner.layer_ids,
                        )
                        if router_capture is not None
                        else {}
                    )
                    send_frame(
                        sock,
                        progress(
                            "HIDDEN",
                            shape=list(output.shape),
                            dtype="float16",
                            **router_header,
                        ),
                        _hidden_payload(output),
                    )
                    continue
                if command in ("DECODE_ID", "DECODE_HIDDEN"):
                    validate_stateful(header, next_step=True)
                    if (
                        _require_exact_int(
                            header.get("token_count"), "decode token_count"
                        )
                        != 1
                    ):
                        raise PipelineProtocolError("decode token_count must equal one")
                    prefix_len = cache.consumed_len
                    if role == "front" and command == "DECODE_ID":
                        _require_exact_shape(
                            header.get("shape"), [1], "DECODE_ID shape"
                        )
                        if header.get("dtype") != "int32":
                            raise PipelineProtocolError("invalid DECODE_ID metadata")
                        input_ids = _worker_tensor(torch, payload, torch.int32, (1,))
                        hidden = runner.embed(input_ids)
                    elif role == "back" and command == "DECODE_HIDDEN":
                        _require_exact_shape(
                            header.get("shape"),
                            [1, 2048],
                            "DECODE_HIDDEN shape",
                        )
                        if header.get("dtype") != "float16":
                            raise PipelineProtocolError(
                                "invalid DECODE_HIDDEN metadata"
                            )
                        hidden = _worker_tensor(
                            torch, payload, torch.float16, (1, 2048)
                        )
                    else:
                        raise PipelineProtocolError(
                            "decode command does not match worker role"
                        )
                    capture_router = _capture_router_requested(header)
                    router_capture = {} if capture_router else None
                    with torch.inference_mode():
                        hidden = runner.decode_hidden(
                            hidden,
                            cache=cache,
                            expected_prefix_len=prefix_len,
                            router_capture=router_capture,
                        )
                    last_step = int(header["step_id"])
                    router_header = (
                        _router_capture_header(
                            torch,
                            router_capture,
                            runner.layer_ids,
                        )
                        if router_capture is not None
                        else {}
                    )
                    send_frame(
                        sock,
                        progress(
                            "HIDDEN",
                            shape=[1, 2048],
                            dtype="float16",
                            **router_header,
                        ),
                        _hidden_payload(hidden),
                    )
                    continue
                if command in ("VALIDATE_IDS", "VALIDATE_HIDDEN"):
                    validate_stateful(header, next_step=False)
                    tokens = header.get("token_count")
                    if (
                        isinstance(tokens, bool)
                        or not isinstance(tokens, int)
                        or tokens != cache.consumed_len
                    ):
                        raise PipelineProtocolError(
                            "validation token_count must equal the cached prefix"
                        )
                    if role == "front" and command == "VALIDATE_IDS":
                        _require_exact_shape(
                            header.get("shape"), [tokens], "VALIDATE_IDS shape"
                        )
                        if header.get("dtype") != "int32":
                            raise PipelineProtocolError("invalid VALIDATE_IDS metadata")
                        input_ids = _worker_tensor(
                            torch, payload, torch.int32, (tokens,)
                        )
                        hidden = runner.embed(input_ids)
                    elif role == "back" and command == "VALIDATE_HIDDEN":
                        _require_exact_shape(
                            header.get("shape"),
                            [tokens, 2048],
                            "VALIDATE_HIDDEN shape",
                        )
                        if header.get("dtype") != "float16":
                            raise PipelineProtocolError(
                                "invalid VALIDATE_HIDDEN metadata"
                            )
                        hidden = _worker_tensor(
                            torch, payload, torch.float16, (tokens, 2048)
                        )
                    else:
                        raise PipelineProtocolError(
                            "validation command does not match worker role"
                        )
                    capture_router = _capture_router_requested(header)
                    router_capture = {} if capture_router else None
                    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
                    cu_seqlens = torch.tensor(
                        [0, tokens], device="cuda", dtype=torch.int32
                    )
                    with torch.inference_mode():
                        for layer_id in runner.layer_ids:
                            try:
                                hidden = runner.forward_layer(
                                    hidden,
                                    layer_id,
                                    positions=positions,
                                    cu_seqlens=cu_seqlens,
                                    max_seqlen=tokens,
                                    router_capture=router_capture,
                                )
                            except Exception as exc:
                                raise RuntimeError(
                                    f"stateless validation failed at layer {layer_id}"
                                ) from exc
                    output = hidden if role == "front" else hidden[-1:]
                    router_header = (
                        _router_capture_header(
                            torch,
                            router_capture,
                            runner.layer_ids,
                        )
                        if router_capture is not None
                        else {}
                    )
                    send_frame(
                        sock,
                        progress(
                            "HIDDEN",
                            shape=list(output.shape),
                            dtype="float16",
                            **router_header,
                        ),
                        _hidden_payload(output),
                    )
                    continue
                if command == "SAMPLE":
                    if role != "front":
                        raise PipelineProtocolError(
                            "only the front worker owns the LM head"
                        )
                    validate_stateful(header, next_step=False)
                    if (
                        _require_exact_int(
                            header.get("token_count"), "sample token_count"
                        )
                        != 1
                    ):
                        raise PipelineProtocolError("invalid SAMPLE metadata")
                    _require_exact_shape(header.get("shape"), [1, 2048], "SAMPLE shape")
                    if header.get("dtype") != "float16":
                        raise PipelineProtocolError("invalid SAMPLE metadata")
                    hidden = _worker_tensor(torch, payload, torch.float16, (1, 2048))
                    with torch.inference_mode():
                        logits = runner.logits(runner.final_hidden(hidden))
                        logits[:, TOKENIZER_VOCAB_SIZE:MODEL_VOCAB_SIZE] = float("-inf")
                        token_id = int(torch.argmax(logits[0]).item())
                    return_logits = header.get("return_logits", False)
                    if not isinstance(return_logits, bool):
                        raise PipelineProtocolError("return_logits must be bool")
                    send_frame(
                        sock,
                        progress(
                            "TOKEN",
                            token_id=token_id,
                            **(
                                {
                                    "logits_shape": [1, MODEL_VOCAB_SIZE],
                                    "logits_dtype": "float16",
                                }
                                if return_logits
                                else {}
                            ),
                        ),
                        _hidden_payload(logits) if return_logits else b"",
                    )
                    continue
                raise PipelineProtocolError(f"unknown worker command {command!r}")
            except Exception as exc:
                poisoned = True
                cache.poisoned = True
                send_frame(
                    sock,
                    {
                        "kind": "ERROR",
                        "role": role,
                        "epoch": epoch,
                        "step_id": last_step,
                        "consumed_len": cache.consumed_len,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
    finally:
        sock.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    parser.add_argument("--prompt", help="user prompt; reads stdin when omitted")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="tokenize prompt literally instead of applying the chat template",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument(
        "--front-uuid",
        help=f"GPU UUID for embedding/front layers/head (default: {DEFAULT_FRONT_UUID})",
    )
    parser.add_argument(
        "--back-uuid",
        help=f"GPU UUID for the remaining layers (default: {DEFAULT_BACK_UUID})",
    )
    parser.add_argument("--split-layer", type=int, default=DEFAULT_SPLIT_LAYER)
    parser.add_argument("--v100-uuid", help=argparse.SUPPRESS)
    parser.add_argument("--sm89-uuid", help=argparse.SUPPRESS)
    parser.add_argument("--print-token-ids", action="store_true")
    parser.add_argument(
        "--validate-stateless",
        action="store_true",
        help="recompute each complete prefix as an end-to-end test oracle",
    )
    parser.add_argument(
        "--worker-role", choices=("front", "back"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--layer-start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--layer-end", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker_role is not None:
        if args.worker_fd is None or args.layer_start is None or args.layer_end is None:
            raise SystemExit(
                "--worker-fd, --layer-start and --layer-end are required with "
                "--worker-role"
            )
        return _worker_main(
            args.worker_role,
            args.worker_fd,
            args.model_dir,
            args.capacity,
            args.layer_start,
            args.layer_end,
        )
    if any(
        value is not None
        for value in (args.worker_fd, args.layer_start, args.layer_end)
    ):
        raise SystemExit(
            "worker process options are internal and require --worker-role"
        )
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
    with Qwen35Pipeline(
        args.model_dir,
        capacity=args.capacity,
        front_uuid=args.front_uuid,
        back_uuid=args.back_uuid,
        split_layer=args.split_layer,
        v100_uuid=args.v100_uuid,
        sm89_uuid=args.sm89_uuid,
    ) as pipeline:
        generated = pipeline.generate_ids(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            validate_stateless=args.validate_stateless,
        )
        front_info, back_info = pipeline.worker_info
        front_stats, back_stats = pipeline.worker_stats
        print(
            "QWEN35_PIPELINE="
            + json.dumps(
                {
                    "front": {"ready": front_info, "last": front_stats},
                    "back": {"ready": back_info, "last": back_stats},
                    "validation": pipeline.last_validation,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    if args.print_token_ids:
        print("TOKEN_IDS=" + json.dumps(generated), file=sys.stderr)
    visible = [token for token in generated if token not in EOS_TOKEN_IDS]
    print(tokenizer.decode(visible, skip_special_tokens=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
