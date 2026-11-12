"""Recorder lifecycle with fake device events; no NPU or model dependency."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pp_chunk_trace.clocks import host_monotonic_ns
from pp_chunk_trace.control import control
from pp_chunk_trace.merge import merge
from pp_chunk_trace.recorder import Recorder, run_directory


class FakeStream:
    def __init__(self, device_module):
        self.dm = device_module
        self.pending = []
        self.lock = threading.Lock()

    def append(self, event):
        with self.lock:
            self.pending.append(event)

    def complete(self):
        with self.lock:
            events, self.pending = self.pending, []
        for event in events:
            event.complete = True

    def synchronize(self):
        self.dm.stream_syncs += 1
        self.complete()


class FakeEvent:
    def __init__(self, dm, enable_timing):
        if not enable_timing:
            raise AssertionError("A timing event is required")
        self.dm = dm
        self.complete = False
        self.timestamp = None
        self.stream = None
        self.records = 0
        self.query_error = False
        self.record_error = False

    def record(self, stream):
        if self.record_error:
            raise RuntimeError("event record failed")
        if self.stream is not None and not self.complete:
            raise AssertionError("Event re-used before device completion")
        self.stream = stream
        self.complete = False
        self.timestamp = host_monotonic_ns() / 1000
        self.records += 1
        self.dm.records += 1
        stream.append(self)

    def query(self):
        if self.query_error:
            raise RuntimeError("event query failed")
        return self.complete

    def synchronize(self):
        self.dm.event_syncs += 1
        self.stream.complete()

    def recorded_time(self):
        if not self.complete:
            raise AssertionError("Read a timestamp before its event completed")
        return self.timestamp


class FakeDevice:
    def __init__(self):
        self.event_syncs = self.stream_syncs = self.records = 0
        self.events, self.streams, self.selected = [], [], []

    def Event(self, enable_timing=False):
        event = FakeEvent(self, enable_timing)
        self.events.append(event)
        return event

    def Stream(self, device):
        stream = FakeStream(self)
        self.streams.append(stream)
        return stream

    def set_device(self, device):
        self.selected.append(device)

    def complete(self):
        for stream in self.streams:
            stream.complete()


def sample_clock(address=None):
    stamp = host_monotonic_ns()
    return dict(host_ns=stamp, reference_ns=stamp, uncertainty_ns=0,
                clock_id="fake-host-clock", rtt_ns=0)


def metadata():
    return dict(rank=0, pp_rank=0, pp_size=1, tp_rank=0, cp_rank=0, dp_rank=0,
                expected_stage_ranks=1, layer_start=0, layer_end=4)


def fields(sequence=1):
    return dict(batch_id=f"run-b{sequence}", mb_id=(sequence - 1) % 2,
                local_forward_seq=sequence, mode="EXTEND",
                chunks=[dict(rid="request", start=(sequence - 1) * 16,
                             end=sequence * 16, ordinal=sequence, attempt=0)])


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dm = FakeDevice()
        self.recorders = []
        self.clock_patch = patch("pp_chunk_trace.recorder.sample_host_clock", side_effect=sample_clock)
        self.clock_patch.start()

    def tearDown(self):
        self.dm.complete()
        for recorder in self.recorders:
            recorder.request_finish()
            recorder.thread.join(timeout=1)
            self.assertFalse(recorder.thread.is_alive(), "collector thread did not terminate")
        self.clock_patch.stop()
        self.temp.cleanup()

    def recorder(self, capacity=2):
        recorder = Recorder(self.dm, "npu:0", metadata(), self.root, "run",
                            capacity=capacity, poll_seconds=0.002)
        self.recorders.append(recorder)
        return recorder

    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("collector condition timed out")
            threading.Event().wait(0.001)

    def records(self, recorder):
        records = []
        for line in recorder.path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass  # Readers can observe a partially written final line.
        return records

    def start(self, recorder):
        result = control("start", self.root, "run", timeout=1)
        self.assertEqual(result["pending"], [])
        self.assertTrue(recorder.armed)
        return result

    def finish(self, recorder):
        self.dm.complete()
        result = control("finish", self.root, "run", timeout=1)
        recorder.thread.join(timeout=1)
        self.assertTrue(recorder.closed)
        self.assertEqual(result["pending"], [])
        return result

    def test_unarmed_startup_calibrates_and_materializes_pool(self):
        recorder = self.recorder(capacity=3)
        self.assertFalse(recorder.armed)
        self.assertEqual(len(recorder.free), 3)
        self.assertIsNone(recorder.begin(fields(), self.dm.Stream("npu:0")))
        self.assertEqual(self.dm.event_syncs, 10)
        self.assertEqual(self.dm.stream_syncs, 1)
        records = self.records(recorder)
        self.assertEqual([r["type"] for r in records], ["meta", "clock"])
        self.assertEqual(records[1]["phase"], "start")
        self.start(recorder)
        self.assertEqual(self.dm.event_syncs, 20)
        self.assertTrue(any(r.get("phase") == "capture_start" for r in self.records(recorder)))

    def test_host_markers_drain_while_forward_is_pending_without_device_ops(self):
        recorder = self.recorder()
        recorder.instant("release_ready", ["startup"])
        self.start(recorder)
        stream = self.dm.Stream("npu:0")
        token = recorder.begin(fields(), stream)
        recorder.end(token)
        device_ops = self.dm.event_syncs, self.dm.stream_syncs, self.dm.records
        recorder.instant("release_ready", ["request"])
        self.wait_for(lambda: any(r["type"] == "completion" for r in self.records(recorder)))
        self.assertEqual((self.dm.event_syncs, self.dm.stream_syncs, self.dm.records), device_ops)
        records = self.records(recorder)
        self.assertEqual([r["rid"] for r in records if r["type"] == "completion"], ["request"])
        self.assertFalse(any(r["type"] == "forward" for r in records))
        self.finish(recorder)
        recorder.instant("response_enqueued", ["late"])
        result = merge([recorder.path], self.root / "merged")
        self.assertEqual(result["completion_markers"], 1)
        self.assertEqual([r["rid"] for r in self.records(recorder) if r["type"] == "completion"], ["request"])

    def test_host_queue_is_bounded_and_reports_drops_without_synchronous_write(self):
        recorder = self.recorder(capacity=1)
        self.start(recorder)
        # Hold the consumer at the start of its device drain. Producers continue
        # to enqueue Host records; no NPU event or file write is used by instant.
        entered, resume = threading.Event(), threading.Event()
        original_drain = recorder._drain
        def pause_drain():
            entered.set()
            resume.wait(1)
            original_drain()
        recorder._drain = pause_drain
        recorder.wake.set()
        self.assertTrue(entered.wait(1))
        try:
            device_ops = self.dm.event_syncs, self.dm.stream_syncs, self.dm.records
            recorder.instant("response_enqueued", (f"r{i}" for i in range(11)))
            self.assertEqual(len(recorder.cpu_pending), 8)
            self.assertEqual(recorder.cpu_dropped, 3)
            self.assertFalse(any(r["type"] == "completion" for r in self.records(recorder)))
            self.assertEqual((self.dm.event_syncs, self.dm.stream_syncs, self.dm.records), device_ops)
        finally:
            resume.set()
        result = self.finish(recorder)
        self.assertTrue(result["errors"])
        records = self.records(recorder)
        self.assertEqual(len([r for r in records if r["type"] == "completion"]), 8)
        self.assertEqual([r["count"] for r in records if r["type"] == "dropped"], [3])
        self.assertFalse(records[-1]["complete"])

    def test_forward_boundaries_never_synchronize_or_write_pending_early(self):
        recorder = self.recorder()
        self.start(recorder)
        stream = self.dm.Stream("npu:0")
        syncs = self.dm.event_syncs, self.dm.stream_syncs
        token = recorder.begin(fields(), stream)
        recorder.end(token)
        self.assertEqual((self.dm.event_syncs, self.dm.stream_syncs), syncs)
        # Drain explicitly too: incomplete end events must remain queued.
        recorder._drain()
        self.assertFalse(any(r["type"] == "forward" for r in self.records(recorder)))
        self.assertEqual(len(recorder.pending), 1)
        stream.complete()
        recorder.wake.set()
        self.wait_for(lambda: len(recorder.pending) == 0)
        records = [r for r in self.records(recorder) if r["type"] == "forward"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["start_us"], token["pair"][0].timestamp)
        self.assertEqual(records[0]["end_us"], token["pair"][1].timestamp)
        self.assertEqual(records[0]["status"], "ok")
        self.assertGreaterEqual(records[0]["cpu_end_ns"], records[0]["cpu_start_ns"])

    def test_capacity_drop_marked_and_event_not_reused_in_flight(self):
        recorder = self.recorder(capacity=1)
        self.start(recorder)
        stream = self.dm.Stream("npu:0")
        first = recorder.begin(fields(), stream)
        recorder.end(first)
        self.assertIsNone(recorder.begin(fields(2), stream))
        self.assertEqual(recorder.dropped, 1)
        self.assertEqual(len(recorder.free), 0)
        stream.complete()
        recorder.wake.set()
        self.wait_for(lambda: len(recorder.free) == 1)
        second = recorder.begin(fields(3), stream)
        self.assertIs(second["pair"], first["pair"])
        recorder.end(second)
        result = self.finish(recorder)
        self.assertTrue(result["errors"])
        records = self.records(recorder)
        self.assertEqual([r["count"] for r in records if r["type"] == "dropped"], [1])
        self.assertFalse(records[-1]["complete"])
        self.assertEqual(len([r for r in records if r["type"] == "forward"]), 2)

    def test_finish_drains_and_adds_end_clock_and_real_schema_merges(self):
        recorder = self.recorder()
        self.start(recorder)
        stream = self.dm.Stream("npu:0")
        token = recorder.begin(fields(), stream)
        recorder.end(token)
        self.finish(recorder)
        records = self.records(recorder)
        self.assertEqual(records[-2]["type"], "clock")
        self.assertEqual(records[-2]["phase"], "end")
        self.assertEqual(records[-1], {"type": "session_end", "complete": True})
        self.assertIsNone(recorder.begin(fields(2), stream))
        result = merge([recorder.path], self.root / "merged")
        self.assertEqual(result["forward_records"], 1)
        trace = json.loads((self.root / "merged" / "trace.json").read_text())
        self.assertEqual(trace["metadata"]["clock_id"], "fake-host-clock")
        self.assertEqual(result["warnings"], [])

    def test_model_error_is_saved_and_not_counted_as_success(self):
        recorder = self.recorder()
        self.start(recorder)
        stream = self.dm.Stream("npu:0")
        first = recorder.begin(fields(), stream)
        recorder.end(first)
        second = recorder.begin(fields(2), stream)
        recorder.end(second, error="model execution failed")
        finish_result = self.finish(recorder)
        self.assertTrue(finish_result["errors"])
        self.assertFalse(self.records(recorder)[-1]["complete"])
        records = [r for r in self.records(recorder) if r["type"] == "forward"]
        self.assertEqual(records[1]["status"], "error")
        self.assertEqual(records[1]["error"], "model execution failed")
        result = merge([recorder.path], self.root / "merged")
        self.assertEqual(result["forward_records"], 1)
        self.assertTrue(any("unfinished/failed" in w for w in result["warnings"]))

    def test_query_failure_marks_incomplete_and_does_not_recycle(self):
        recorder = self.recorder(capacity=1)
        self.start(recorder)
        stream = self.dm.Stream("npu:0")
        token = recorder.begin(fields(), stream)
        token["pair"][1].query_error = True
        with self.assertLogs("pp_chunk_trace.recorder", level="ERROR"):
            recorder.end(token)
            recorder.thread.join(timeout=1)
        self.assertTrue(recorder.closed)
        self.assertIn("query failed", recorder.failure)
        self.assertEqual(len(recorder.free), 0)
        self.assertEqual(len(recorder.pending), 1)
        records = self.records(recorder)
        self.assertTrue(any(r["type"] == "error" and r["pending"] == 1 for r in records))
        self.assertFalse(any(r["type"] == "session_end" for r in records))
        self.assertIsNone(recorder.begin(fields(2), stream))
        status = control("finish", self.root, "run", timeout=0)
        self.assertTrue(status["pending"])
        self.assertTrue(status["errors"])

    def test_record_failure_retains_broken_events_and_model_continues(self):
        recorder = self.recorder(capacity=1)
        self.start(recorder)
        recorder.free[-1][0].record_error = True
        with self.assertLogs("pp_chunk_trace.recorder", level="ERROR"):
            token = recorder.begin(fields(), self.dm.Stream("npu:0"))
        self.assertIsNone(token)
        self.wait_for(lambda: len(recorder.orphans) == 1)
        self.assertEqual(len(recorder.free), 0)
        result = self.finish(recorder)
        self.assertTrue(result["errors"])
        self.assertFalse(self.records(recorder)[-1]["complete"])

    def test_control_requires_existing_workers_and_new_run_after_finish(self):
        with self.assertRaisesRegex(ValueError, "No rank traces"):
            control("start", self.root, "missing", timeout=0)
        recorder = self.recorder()
        self.start(recorder)
        self.finish(recorder)
        with self.assertRaisesRegex(ValueError, "Capture is finished"):
            control("start", self.root, "run", timeout=0)
        with self.assertRaisesRegex(ValueError, "already finished"):
            Recorder(self.dm, "npu:0", metadata(), self.root, "run", capacity=1)

    def test_invalid_paths_and_pool_boundaries_rejected(self):
        for run_id in ("../escape", "", "a/b", ".hidden", "a" * 129):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                run_directory(self.root, run_id)
        for capacity in (0, -1, 4097):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                Recorder(self.dm, "npu:0", metadata(), self.root, "run", capacity=capacity)


if __name__ == "__main__":
    unittest.main()
