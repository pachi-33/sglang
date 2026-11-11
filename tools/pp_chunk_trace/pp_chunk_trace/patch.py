"""Scoped monkeypatch for eager PP prefill; never replaces the scheduler loop."""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from collections import defaultdict

from .compat import verify_runtime
from .recorder import Recorder, run_directory

WIRE_KEY = "__pp_chunk_trace_v1__"
_PATCHED = {}
_MISSING = object()
logger = logging.getLogger(__name__)


def snapshot_chunks(batch):
    """Read existing CPU lists only; never call tensor.item()/cpu()/tolist()."""
    reqs, starts, sizes = batch.reqs, batch.prefix_lens, batch.extend_lens
    if not isinstance(starts, (list, tuple)) or not isinstance(sizes, (list, tuple)):
        raise ValueError("Expected CPU prefix_lens/extend_lens lists")
    if len(reqs) != len(starts) or len(reqs) != len(sizes):
        raise ValueError("Request and chunk metadata lengths disagree")
    result = []
    for req, start, size in zip(reqs, starts, sizes):
        if type(start) is not int or type(size) is not int or start < 0 or size < 0:
            raise ValueError("Invalid CPU chunk lengths")
        if size:
            result.append(dict(rid=str(req.rid), start=start, end=start + size))
    return result


def _semantic(chunks):
    return [(c["rid"], c["start"], c["end"]) for c in chunks]


class Identity:
    def __init__(self, run_id, dp_rank):
        self.run_id, self.dp_rank = run_id, dp_rank
        self.ordinals, self.attempts = {}, defaultdict(int)
        self.next_ordinal = defaultdict(int)

    def create(self, chunks, enabled):
        if not enabled:
            # No server-lifetime accumulation of warmup/post-capture requests.
            return dict(run_id=self.run_id, batch_id="inactive", chunks=chunks, enabled=False)
        frozen = []
        for chunk in chunks:
            key = (chunk["rid"], chunk["start"], chunk["end"])
            if key not in self.ordinals:
                self.next_ordinal[chunk["rid"]] += 1
                self.ordinals[key] = self.next_ordinal[chunk["rid"]]
            self.attempts[key] += 1
            frozen.append(dict(chunk, ordinal=self.ordinals[key], attempt=self.attempts[key]))
        # Deterministic across CP/TP members without another collective. Different
        # batches, DP replicas and replays get different identities.
        data = json.dumps([self.run_id, self.dp_rank, frozen], sort_keys=True, separators=(",", ":"))
        return dict(run_id=self.run_id, batch_id=hashlib.sha256(data.encode()).hexdigest()[:32],
                    chunks=frozen, enabled=bool(enabled))

    def receive(self, wire, chunks):
        if not isinstance(wire, dict) or wire.get("run_id") != self.run_id:
            raise ValueError("Missing/mismatched PP trace header; enable the same patch and run ID on all PP ranks")
        if _semantic(wire.get("chunks", [])) != _semantic(chunks):
            raise ValueError("Incoming PP chunk metadata disagrees with this stage's scheduled batch")
        if not isinstance(wire.get("batch_id"), str) or type(wire.get("enabled")) is not bool:
            raise ValueError("Malformed PP trace header")
        return wire


def _eligible(batch):
    # EXACT EXTEND only: graph capture/verify/decode/mixed/split modes are excluded.
    return getattr(batch.forward_mode, "name", "") == "EXTEND" and bool(batch.reqs)


def _metadata(module, scheduler):
    parallel = module.get_parallel()
    dp = (parallel.dp_rank or 0) * parallel.attn_dp_size + parallel.attn_dp_rank
    runner = scheduler.tp_worker.model_runner
    layers = getattr(runner, "layer_info", None)
    return dict(rank=module.torch.distributed.get_rank(), pp_rank=parallel.pp_rank,
                pp_size=parallel.pp_size, tp_rank=parallel.tp_rank,
                attn_tp_rank=parallel.attn_tp_rank, cp_rank=parallel.attn_cp_rank, dp_rank=dp,
                expected_stage_ranks=parallel.attn_tp_size * parallel.attn_cp_size,
                device_id=module.get_device().gpu_id,
                token_scope="logical attention-DP chunk before CP split; do not sum tokens across CP/TP ranks",
                layer_start=getattr(layers, "start_layer", None), layer_end=getattr(layers, "end_layer", None),
                sglang_source=module.__file__)


def install(module):
    cls = module.Scheduler
    if cls in _PATCHED:
        return
    run_id = os.environ.get("PP_CHUNK_TRACE_RUN_ID", "")
    output = os.environ.get("PP_CHUNK_TRACE_DIR", "")
    if not output:
        raise ValueError("Set PP_CHUNK_TRACE_DIR on every P worker")
    run_directory(output, run_id)
    verify_runtime(module)
    _patch_class(cls, module, output, run_id)


def _patch_class(cls, module, output, run_id, recorder_factory=Recorder):
    """Separated from compatibility checking for CPU-only contract tests."""
    names = ("init_pp_loop_state", "_pp_recv_proxy_tensors", "_pp_launch_batch",
             "run_batch", "_pp_send_proxy_to_next_stage")
    previous = {name: cls.__dict__.get(name, _MISSING) for name in names}
    originals = {name: getattr(cls, name) for name in names}
    instances = []

    @functools.wraps(originals["init_pp_loop_state"])
    def initialize(self, *args, **kwargs):
        result = originals["init_pp_loop_state"](self, *args, **kwargs)
        if hasattr(self, "_pp_trace_recorder"):
            return result
        if not str(self.device).startswith("npu"):
            raise RuntimeError("PP chunk trace v1 supports NPU only")
        if not self.spec_algorithm.is_none():
            raise RuntimeError("PP chunk trace v1 requires speculative decoding disabled")
        meta = _metadata(module, self)
        recorder = recorder_factory(self.device_module, module.get_device().gpu_id, meta,
                                    output, run_id, os.environ.get("PP_CHUNK_TRACE_CLOCK_ADDR") or None,
                                    int(os.environ.get("PP_CHUNK_TRACE_MAX_PENDING", "128")))
        self._pp_trace_recorder = recorder
        self._pp_trace_identity = Identity(run_id, meta["dp_rank"])
        self._pp_trace_incoming = None
        self._pp_trace_context = None
        self._pp_trace_seq = 0
        instances.append(self)
        return result

    @functools.wraps(originals["_pp_recv_proxy_tensors"])
    def receive(self, *args, **kwargs):
        result = originals["_pp_recv_proxy_tensors"](self, *args, **kwargs)
        if hasattr(self, "_pp_trace_recorder"):
            self._pp_trace_incoming = None if result is None else result.tensors.pop(WIRE_KEY, None)
        return result

    @functools.wraps(originals["_pp_launch_batch"])
    def launch(self, mb_id, cur_batch, pp_proxy_tensors, mb_metadata, last_rank_comm_queue):
        recorder = self._pp_trace_recorder
        incoming, self._pp_trace_incoming = self._pp_trace_incoming, None
        wire = None
        if _eligible(cur_batch):
            chunks = snapshot_chunks(cur_batch)
            if chunks:
                if self.pp_group.is_first_rank:
                    wire = self._pp_trace_identity.create(chunks, recorder.armed and not recorder.stop_requested
                                                         and not recorder.closed and not recorder.failure)
                else:
                    wire = self._pp_trace_identity.receive(incoming, chunks)
        self._pp_trace_seq += 1
        self._pp_trace_context = None if wire is None or not wire["enabled"] else dict(
            batch_id=wire["batch_id"], chunks=wire["chunks"], mb_id=mb_id,
            local_forward_seq=self._pp_trace_seq, mode="EXTEND")
        try:
            result, event = originals["_pp_launch_batch"](self, mb_id, cur_batch, pp_proxy_tensors,
                                                         mb_metadata, last_rank_comm_queue)
            # Result-local identity survives processing an older microbatch before
            # this result is sent. Never infer identity from the last loop slot.
            result._pp_chunk_trace_wire = wire
            return result, event
        finally:
            self._pp_trace_context = None

    @functools.wraps(originals["run_batch"])
    def run(self, batch, *args, **kwargs):
        context = getattr(self, "_pp_trace_context", None)
        if context is None:
            return originals["run_batch"](self, batch, *args, **kwargs)
        recorder = self._pp_trace_recorder
        try:
            token = recorder.begin(context, self.device_module.current_stream())
        except Exception as exc:
            recorder.failure = repr(exc)
            logger.exception("PP trace start failed; model execution continues")
            token = None
        failure = None
        try:
            result = originals["run_batch"](self, batch, *args, **kwargs)
            if getattr(result, "can_run_cuda_graph", False):
                failure = "Prefill graph replay unsupported by v1 boundary contract"
            return result
        except BaseException as exc:
            failure = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            try:
                recorder.end(token, failure)
            except Exception as exc:
                recorder.failure = repr(exc)
                logger.exception("PP trace end failed; preserving model result/exception")

    @functools.wraps(originals["_pp_send_proxy_to_next_stage"])
    def send(self, result):
        wire = getattr(result, "_pp_chunk_trace_wire", None)
        proxy = result.pp_hidden_states_proxy_tensors
        original = proxy.tensors
        if WIRE_KEY in original:
            raise RuntimeError("PP trace header collision")
        proxy.tensors = dict(original, **{WIRE_KEY: wire})
        try:
            return originals["_pp_send_proxy_to_next_stage"](self, result)
        finally:
            proxy.tensors = original

    wrappers = (initialize, receive, launch, run, send)
    for name, wrapper in zip(names, wrappers):
        setattr(cls, name, wrapper)
    _PATCHED[cls] = (previous, instances, dict(zip(names, wrappers)))


def uninstall(module):
    """Restore wrappers for embedding/tests. CLI uninstall takes effect on restart."""
    state = _PATCHED.get(module.Scheduler)
    if state is None:
        return
    previous, instances, wrappers = state
    for name in previous:
        if getattr(module.Scheduler, name) is not wrappers[name]:
            raise RuntimeError("Another patch replaced a traced method; refusing to overwrite it")
    for instance in instances:
        instance._pp_trace_recorder.request_finish()
    for name, original in previous.items():
        if original is _MISSING:
            delattr(module.Scheduler, name)
        else:
            setattr(module.Scheduler, name, original)
    del _PATCHED[module.Scheduler]
