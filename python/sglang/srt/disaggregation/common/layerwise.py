"""Run synchronous layerwise transfers outside the model/scheduler thread."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock


class LayerwiseTransferWorker:
    def __init__(self, *, transfer, is_active, on_failure, on_drained, initializer):
        self._transfer = transfer
        self._is_active = is_active
        self._on_failure = on_failure
        self._on_drained = on_drained
        self._lock = Lock()
        self._pending = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="layerwise-kv",
            initializer=initializer,
        )

    def submit(self, room, session_id, blocks, compute_event) -> Future:
        # Count before publishing work: an abort must not acknowledge reusable
        # decode pages while an accepted write is queued or running.
        with self._lock:
            self._pending[room] = self._pending.get(room, 0) + 1
        try:
            return self._executor.submit(
                self._run, room, session_id, tuple(blocks), compute_event
            )
        except BaseException:
            self._drained(room)
            raise

    def _run(self, room, session_id, blocks, compute_event):
        try:
            # Transport engines can use their own device streams. A wait on
            # the caller's stream does not order those reads; wait on the CPU
            # worker before giving the engine the source addresses.
            if compute_event is not None:
                compute_event.synchronize()
            if not self._is_active(room):
                return
            rc = self._transfer(session_id, blocks)
            if rc != 0:
                self._on_failure(room, f"Layerwise KV transfer returned {rc}")
        except Exception as exc:
            self._on_failure(room, f"Layerwise KV transfer failed: {exc}")
        finally:
            self._drained(room)

    def _drained(self, room):
        with self._lock:
            remaining = self._pending[room] - 1
            if remaining:
                self._pending[room] = remaining
            else:
                del self._pending[room]
        self._on_drained(room)

    def has_pending(self, room=None) -> bool:
        with self._lock:
            return (
                bool(self._pending) if room is None else self._pending.get(room, 0) > 0
            )

    def close(self):
        # Pool memory and transport registrations must outlive the worker.
        self._executor.shutdown(wait=True)
