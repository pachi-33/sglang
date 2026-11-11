import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pp_chunk_trace.merge import ClockMap, TraceError, merge


def meta(rank=0, pp=0, expected=1, run="run", dp=0):
    return dict(type="meta", schema_version=1, run_id=run, worker_id=f"worker-{dp}-{rank}",
                hostname="host", rank=rank, pp_rank=pp, pp_size=2, tp_rank=0, cp_rank=0,
                dp_rank=dp, expected_stage_ranks=expected, layer_start=pp * 2, layer_end=pp * 2 + 2)


def clock(device=0, reference=10_000_000_000, clock_id="test-clock", uncertainty=100):
    return dict(type="clock", device_us=device, host_ns=reference, reference_ns=reference,
                uncertainty_ns=uncertainty, clock_id=clock_id, phase="start" if not device else "end")


def chunk(rid="request", start=0, end=16, ordinal=1):
    return dict(rid=rid, start=start, end=end, ordinal=ordinal, attempt=0)


def forward(batch="b1", start=100, end=200, sequence=1, chunks=None):
    return dict(type="forward", batch_id=batch, chunks=chunks or [chunk()], mb_id=0,
                local_forward_seq=sequence, mode="EXTEND", start_us=start, end_us=end,
                cpu_start_ns=0, cpu_end_ns=1, status="ok")


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs, self.output = self.root / "inputs", self.root / "out"
        self.inputs.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, metadata, records, end=True, anchors=None):
        lines = [metadata, *(anchors if anchors is not None else [clock()]), *records]
        if end:
            lines.append(dict(type="session_end"))
        path = self.inputs / name
        path.write_text("".join(json.dumps(r) + "\n" for r in lines))
        return path

    def run_merge(self, **kwargs):
        return merge([self.inputs], self.output, **kwargs)

    def read_csv(self, name):
        with (self.output / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_pair_handoff_and_fill_drain(self):
        self.write("rank0.jsonl", meta(), [forward()])
        self.write("rank1.jsonl", meta(1, 1), [forward(start=250, end=400)])
        result = self.run_merge()
        self.assertEqual(result["forward_records"], 2)
        handoff = self.read_csv("handoffs.csv")[0]
        self.assertAlmostEqual(float(handoff["handoff_ms"]), .05)
        self.assertEqual(handoff["status"], "ok")
        summary = self.read_csv("summary.csv")
        self.assertAlmostEqual(float(summary[0]["forward_fraction"]), 1 / 3)
        self.assertAlmostEqual(float(summary[0]["drain_ms"]), .2)
        self.assertAlmostEqual(float(summary[1]["fill_ms"]), .15)
        trace = json.loads((self.output / "trace.json").read_text())
        self.assertEqual(len([e for e in trace["traceEvents"] if e["ph"] in ("s", "f")]), 2)
        self.assertTrue(any("C1 (16 tokens)" in e.get("name", "") for e in trace["traceEvents"]))

    def test_duplicate_file_is_not_double_counted(self):
        path = self.write("rank0.jsonl", meta(), [forward()])
        shutil.copy(path, self.inputs / "copy.jsonl")
        result = self.run_merge()
        self.assertEqual(result["forward_records"], 1)
        self.assertEqual(self.read_csv("summary.csv")[0]["forward_count"], "1")

    def test_conflicting_duplicate_sequence_fails(self):
        self.write("rank0.jsonl", meta(), [forward(), forward(batch="other", start=300)])
        with self.assertRaisesRegex(TraceError, "conflicting local_forward_seq"):
            self.run_merge()

    def test_different_runs_require_selection(self):
        self.write("a.jsonl", meta(), [forward()])
        self.write("b.jsonl", meta(1, 1, run="other"), [forward(start=300, end=400)])
        with self.assertRaisesRegex(TraceError, "Multiple run IDs"):
            self.run_merge()
        result = self.run_merge(run_id="run")
        self.assertEqual(result["forward_records"], 1)

    def test_different_clock_domains_refused(self):
        self.write("a.jsonl", meta(), [forward()])
        self.write("b.jsonl", meta(1, 1), [forward()], anchors=[clock(clock_id="other")])
        with self.assertRaisesRegex(TraceError, "different clock_id"):
            self.run_merge()

    def test_two_anchor_mapping_uses_drift(self):
        warnings = []
        mapping = ClockMap([clock(reference=0), clock(1000, 1_001_000)], "w", warnings)
        self.assertEqual(mapping.map(500)[0], 500_500)
        self.assertFalse(warnings)
        mapping.map(1200)
        self.assertTrue(any("extrapolated" in w for w in warnings))

    def test_capture_start_anchor_is_not_discarded(self):
        mapping = ClockMap([clock(reference=0), clock(1000, 1_001_000), clock(2000, 2_000_000)], "w", [])
        self.assertEqual(mapping.map(1000)[0], 1_001_000)
        self.assertEqual(mapping.map(1500)[0], 1_500_500)
        self.assertEqual(mapping.map(500)[0], 500_500)

    def test_negative_handoff_preserved(self):
        self.write("a.jsonl", meta(), [forward(start=100, end=400)])
        self.write("b.jsonl", meta(1, 1), [forward(start=300, end=500)])
        result = self.run_merge()
        handoff = self.read_csv("handoffs.csv")[0]
        self.assertEqual(float(handoff["handoff_ms"]), -.1)
        self.assertEqual(handoff["status"], "negative")
        self.assertTrue(any("negative envelope" in w for w in result["warnings"]))
        events = json.loads((self.output / "trace.json").read_text())["traceEvents"]
        self.assertFalse(any(e["ph"] in ("s", "f") for e in events))

    def test_handoff_under_calibration_error_is_uncertain(self):
        self.write("a.jsonl", meta(), [forward()], anchors=[clock(uncertainty=10_000)])
        self.write("b.jsonl", meta(1, 1), [forward(start=210, end=400)], anchors=[clock(uncertainty=10_000)])
        self.run_merge()
        self.assertEqual(self.read_csv("handoffs.csv")[0]["status"], "uncertain")

    def test_missing_rank_and_stage_marked(self):
        self.write("a.jsonl", meta(expected=2), [forward()])
        result = self.run_merge()
        self.assertTrue(any("incomplete stage" in w for w in result["warnings"]))
        self.assertTrue(any("missing PP stages" in w for w in result["warnings"]))
        self.assertEqual(self.read_csv("summary.csv")[0]["complete"], "False")
        self.assertEqual(self.read_csv("handoffs.csv"), [])

    def test_stage_envelopes_union_overlap(self):
        self.write("a.jsonl", meta(expected=2), [forward(end=300), forward("b2", 400, 500, 2, [chunk(start=16, end=32, ordinal=2)])])
        self.write("b.jsonl", meta(1, 0, expected=2), [forward(start=120, end=450), forward("b2", 460, 600, 2, [chunk(start=16, end=32, ordinal=2)])])
        self.run_merge()
        row = self.read_csv("summary.csv")[0]
        self.assertAlmostEqual(float(row["forward_union_ms"]), .5)
        self.assertEqual(float(row["internal_gap_ms"]), 0)
        self.assertEqual(float(row["forward_fraction"]), 1)

    def test_multiple_request_batch_counted_once(self):
        pieces = [chunk("a"), chunk("b", end=8)]
        self.write("a.jsonl", meta(), [forward(chunks=pieces)])
        self.run_merge()
        summary = self.read_csv("summary.csv")[0]
        rows = self.read_csv("chunks.csv")
        self.assertEqual(summary["forward_count"], "1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["token_count"], "24")
        self.assertEqual(len(json.loads(rows[0]["request_chunks"])), 2)

    def test_token_identity_mismatch_refused(self):
        self.write("a.jsonl", meta(), [forward()])
        self.write("b.jsonl", meta(1, 1), [forward(chunks=[chunk(end=17)])])
        with self.assertRaisesRegex(TraceError, "chunks disagree"):
            self.run_merge()

    def test_incomplete_tail_is_flagged_not_fabricated(self):
        path = self.write("a.jsonl", meta(), [forward()], end=False)
        with path.open("a") as handle:
            handle.write('{"type":"forward"')
        result = self.run_merge()
        self.assertEqual(result["forward_records"], 1)
        self.assertTrue(any("truncated final" in w for w in result["warnings"]))
        self.assertTrue(any("no session_end" in w for w in result["warnings"]))

    def test_corrupt_middle_record_is_error(self):
        path = self.write("a.jsonl", meta(), [forward()])
        path.write_text(path.read_text().replace('"type": "forward"', 'broken!!!'))
        with self.assertRaisesRegex(TraceError, "invalid JSON"):
            self.run_merge()

    def test_error_and_missing_end_excluded(self):
        incomplete = forward("b2", 300, 400, 2)
        del incomplete["end_us"]
        self.write("a.jsonl", meta(), [forward(), incomplete, {"type": "error", "message": "test failure"}])
        result = self.run_merge()
        self.assertEqual(result["forward_records"], 1)
        self.assertTrue(any("unfinished/failed" in w for w in result["warnings"]))
        self.assertTrue(any("test failure" in w for w in result["warnings"]))

    def test_dp_lanes_separate(self):
        self.write("a.jsonl", meta(), [forward()])
        self.write("b.jsonl", meta(dp=1), [forward(chunks=[chunk("different")])])
        self.run_merge()
        rows = self.read_csv("summary.csv")
        self.assertEqual({r["dp_rank"] for r in rows}, {"0", "1"})

    def test_incomplete_session_disables_gap_statistics(self):
        rows = [forward(), forward("b2", 300, 400, 2),
                {"type": "session_end", "complete": False}]
        self.write("a.jsonl", meta(), rows, end=False)
        result = self.run_merge()
        summary = self.read_csv("summary.csv")[0]
        self.assertEqual(summary["complete"], "False")
        self.assertEqual(summary["internal_gap_ms"], "")
        self.assertEqual(summary["outside_forward_fraction"], "")
        self.assertTrue(any("session_end marks" in w for w in result["warnings"]))
        events = json.loads((self.output / "trace.json").read_text())["traceEvents"]
        self.assertFalse(any(e.get("cat") == "gap" for e in events))

    def test_missing_whole_batch_is_not_a_pipeline_gap(self):
        self.write("a.jsonl", meta(), [forward(), forward("b2", 300, 400, 2), forward("b3", 500, 600, 3)])
        self.write("b.jsonl", meta(1, 1), [forward(start=200, end=300), forward("b3", 600, 700, 3)])
        result = self.run_merge()
        summary = self.read_csv("summary.csv")[1]
        self.assertEqual(summary["complete"], "False")
        self.assertEqual(summary["missing_batch_count"], "1")
        self.assertEqual(summary["internal_gap_ms"], "")
        events = json.loads((self.output / "trace.json").read_text())["traceEvents"]
        self.assertFalse(any(e.get("cat") == "gap" and e.get("pid") == 2 for e in events))
        self.assertTrue(any("missing PP stages" in w for w in result["warnings"]))

    def test_dropped_records_disable_gap_even_with_complete_footer(self):
        self.write("a.jsonl", meta(), [forward(), forward("b2", 300, 400, 2), {"type": "dropped", "count": 1}])
        self.run_merge()
        summary = self.read_csv("summary.csv")[0]
        self.assertEqual(summary["complete"], "False")
        self.assertEqual(summary["forward_fraction"], "")
        self.assertAlmostEqual(float(summary["forward_union_ms"]), .2)

    def test_missing_footer_does_not_claim_valid_occupancy(self):
        self.write("a.jsonl", meta(), [forward(), forward("b2", 300, 400, 2)], end=False)
        self.run_merge()
        summary = self.read_csv("summary.csv")[0]
        self.assertEqual(summary["complete"], "False")
        self.assertEqual(summary["outside_forward_ms"], "")

    def test_missing_clock_fails(self):
        self.write("a.jsonl", meta(), [forward()], anchors=[])
        with self.assertRaisesRegex(TraceError, "missing clock"):
            self.run_merge()

    def test_clock_reset_rejected(self):
        with self.assertRaisesRegex(TraceError, "non-monotonic"):
            ClockMap([clock(reference=100), clock(1000, 0)], "w", [])

    def test_internal_gap_distinct_from_fill_drain(self):
        self.write("a.jsonl", meta(), [forward(), forward("b2", 300, 400, 2, [chunk(start=16, end=32, ordinal=2)])])
        self.write("b.jsonl", meta(1, 1), [forward(start=200, end=300), forward("b2", 500, 600, 2, [chunk(start=16, end=32, ordinal=2)])])
        self.run_merge()
        stage0, stage1 = self.read_csv("summary.csv")
        self.assertAlmostEqual(float(stage0["internal_gap_ms"]), .1)
        self.assertAlmostEqual(float(stage0["drain_ms"]), .2)
        self.assertAlmostEqual(float(stage1["fill_ms"]), .1)
        self.assertAlmostEqual(float(stage1["internal_gap_ms"]), .2)


if __name__ == "__main__":
    unittest.main()
