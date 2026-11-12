"""Bounded, asynchronous NPU boundary recording; no model imports."""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from .clocks import host_monotonic_ns, sample_host_clock

logger = logging.getLogger(__name__)


def run_directory(output, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or ""):
        raise ValueError("PP_CHUNK_TRACE_RUN_ID must be a shared simple run name (letters/digits/._-)")
    return Path(output).expanduser().resolve() / run_id


class Recorder:
    """One recorder per scheduler process. Events are never re-used in flight.

    Only the startup/finish calibration waits for an event on a separate empty
    stream. Forward boundaries record events and enqueue references; the reader
    thread polls completion and writes the actual device timestamps, not poll time.
    """

    def __init__(self, device_module, device, metadata, output, run_id,
                 clock_address=None, capacity=128, poll_seconds=0.02):
        if not 1 <= capacity <= 4096:
            raise ValueError("PP_CHUNK_TRACE_MAX_PENDING must be in [1,4096]")
        self.dm, self.device = device_module, device
        self.clock_address, self.capacity = clock_address, capacity
        self.poll_seconds = poll_seconds
        self.directory = run_directory(output, run_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / ".finish").exists():
            raise ValueError("Capture already finished; use a new PP_CHUNK_TRACE_RUN_ID")
        self.worker_id = uuid.uuid4().hex
        host = re.sub(r"[^A-Za-z0-9._-]", "_", socket.gethostname())
        self.path = self.directory / f"{host}-rank{metadata['rank']}-pid{os.getpid()}-{self.worker_id[:8]}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.pending, self.free = deque(), []
        self.cpu_pending = deque()
        self.cpu_capacity = capacity * 8
        self.cpu_dropped = 0
        self.active = 0
        self.armed = False
        self.dropped = 0
        self.stop_requested = False
        self.closed = False
        self.failure = None
        self.had_record_errors = False
        self.orphans = []
        self.wake = threading.Event()
        self.calibration_stream = self.dm.Stream(device=device)
        self._write(dict(metadata, type="meta", schema_version=1, run_id=run_id,
                         worker_id=self.worker_id, hostname=socket.gethostname(), pid=os.getpid(),
                         boundary="Scheduler.run_batch on forward stream, after existing PP dependencies; includes internal waits and host submission gaps"))
        try:
            self.calibrate("start")
            # Initialize timing events outside the measurement window. This also
            # rejects unsupported Event timestamps before the scheduler loop starts.
            for _ in range(capacity):
                pair = (self.dm.Event(enable_timing=True), self.dm.Event(enable_timing=True))
                for event in pair:
                    event.record(self.calibration_stream)
                self.free.append(pair)
            self.calibration_stream.synchronize()
            self.free[0][0].recorded_time()
        except Exception:
            self.file.close()
            raise
        self.thread = threading.Thread(target=self._reader, name="pp-chunk-trace", daemon=True)
        self.thread.start()
        logger.info("PP chunk trace enabled: %s", self.path)

    def _write(self, record):
        self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self.file.flush()

    def calibrate(self, phase):
        """Bracket an event on an empty stream and retain calibration uncertainty."""
        reference = sample_host_clock(self.clock_address)
        candidates = []
        for _ in range(5):
            event = self.dm.Event(enable_timing=True)
            # First recording materializes the event, outside the timed bracket.
            event.record(self.calibration_stream)
            event.synchronize()
            before = host_monotonic_ns()
            event.record(self.calibration_stream)
            event.synchronize()
            after = host_monotonic_ns()
            candidates.append((after - before, (before + after) // 2, event.recorded_time()))
        width, host, device_us = min(candidates)
        record = dict(type="clock", phase=phase, device_us=device_us, host_ns=host,
                      reference_ns=reference["reference_ns"] + host - reference["host_ns"],
                      uncertainty_ns=reference["uncertainty_ns"] + (width + 1) // 2,
                      clock_id=reference["clock_id"], rtt_ns=reference["rtt_ns"],
                      device_bracket_ns=width)
        self._write(record)

    def begin(self, fields, stream):
        with self.lock:
            if not self.armed or self.stop_requested or self.closed or self.failure:
                return None
            if not self.free:
                self.dropped += 1
                return None
            pair = self.free.pop()
            self.active += 1
        token = dict(fields=fields, pair=pair, stream=stream, cpu_start_ns=host_monotonic_ns())
        try:
            pair[0].record(stream)
        except Exception as exc:
            self._record_failure(token, exc)
            return None
        return token

    def end(self, token, error=None):
        if token is None:
            return
        token["cpu_end_ns"] = host_monotonic_ns()
        token["error"] = error
        if error:
            self.had_record_errors = True
        try:
            token["pair"][1].record(token["stream"])
        except Exception as exc:
            self._record_failure(token, exc)
            return
        with self.lock:
            self.active -= 1
            self.pending.append(token)
        self.wake.set()

    def instant(self, event, rids):
        """Enqueue bounded Host completion markers; no I/O or device operation."""
        if event not in ("result_processed", "release_ready", "response_enqueued"):
            raise ValueError("Unknown PP completion marker")
        stamp = host_monotonic_ns()
        with self.lock:
            if not self.armed or self.stop_requested or self.closed or self.failure:
                return
            for rid in rids:
                if len(self.cpu_pending) >= self.cpu_capacity:
                    self.cpu_dropped += 1
                    continue
                self.cpu_pending.append(dict(type="completion", event=event,
                                             rid=str(rid), host_ns=stamp))
        self.wake.set()

    def _drain_cpu(self):
        # Independent of pending device events: ready Host notifications must
        # not wait for an unrelated forward's timing event to complete.
        with self.lock:
            records = list(self.cpu_pending)
            self.cpu_pending.clear()
        for record in records:
            self._write(record)

    def _record_failure(self, token, exc):
        # Keep events alive after a partial submission. Do not turn a tracing
        # failure into a model exception or recycle an event still in flight.
        with self.lock:
            self.active -= 1
            self.failure = repr(exc)
            self.pending.append(dict(token, broken=True))
        logger.error("PP chunk timing failed; capture will be marked incomplete: %s", exc)
        self.wake.set()

    def request_finish(self):
        with self.lock:
            self.stop_requested = True
        self.wake.set()

    def _drain(self):
        while True:
            with self.lock:
                token = self.pending[0] if self.pending else None
            if token is None:
                return
            if token.get("broken"):
                self._write(dict(type="error", message=self.failure, batch_id=token["fields"].get("batch_id")))
                # Broken events are retained to process exit, not destroyed.
                self.orphans.append(token)
            else:
                start, end = token["pair"]
                if not end.query():
                    return
                record = dict(token["fields"], type="forward", start_us=start.recorded_time(),
                              end_us=end.recorded_time(), cpu_start_ns=token["cpu_start_ns"],
                              cpu_end_ns=token["cpu_end_ns"], status="error" if token["error"] else "ok")
                if token["error"]:
                    record["error"] = token["error"]
                self._write(record)
            with self.lock:
                self.pending.popleft()
                if not token.get("broken"):
                    self.free.append(token["pair"])

    def _reader(self):
        try:
            self.dm.set_device(self.device)
            while True:
                self.wake.wait(self.poll_seconds)
                self.wake.clear()
                if not self.stop_requested and (self.directory / ".finish").exists():
                    self.request_finish()
                if not self.armed and not self.stop_requested and (self.directory / ".start").exists():
                    self.calibrate("capture_start")
                    self.armed = True
                    self._write(dict(type="capture_start", host_ns=host_monotonic_ns()))
                self._drain_cpu()
                self._drain()
                with self.lock:
                    finish = self.stop_requested and not self.pending and not self.cpu_pending and self.active == 0
                if finish:
                    self.calibrate("end")
                    if self.dropped:
                        self._write(dict(type="dropped", count=self.dropped, reason="event capacity exhausted"))
                    if self.cpu_dropped:
                        self._write(dict(type="dropped", count=self.cpu_dropped, reason="CPU completion capacity exhausted"))
                    self._write(dict(type="session_end", complete=not self.failure and not self.dropped and not self.cpu_dropped and not self.had_record_errors))
                    return
        except Exception as exc:
            self.failure = repr(exc)
            logger.exception("PP chunk trace collector failed; model execution continues")
            try:
                self._write(dict(type="error", message=self.failure, pending=len(self.pending)))
            except Exception:
                pass
        finally:
            with self.lock:
                self.closed = True
            self.file.close()
