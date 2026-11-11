"""Merge local PP chunk JSONL files without importing torch or SGLang.

Stage spans are envelopes, not hardware utilization. Handoffs include scheduling
and queueing; they are not measurements of pure HCCL transfer time.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class TraceError(ValueError):
    """Input cannot support a coherent global timeline."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _warn(warnings, message):
    if message not in warnings:
        warnings.append(message)


def _load(paths, run_id, warnings):
    workers = {}
    runs = set()
    for path in paths:
        data = path.read_text(encoding="utf-8").splitlines()
        records = []
        truncated = False
        for number, line in enumerate(data, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if number == len(data):
                    _warn(warnings, f"{path}: truncated final JSONL record ignored")
                    truncated = True
                    break
                raise TraceError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TraceError(f"{path}:{number}: expected an object")
            records.append(record)
        if not records or records[0].get("type") != "meta":
            raise TraceError(f"{path}: first record must be meta")
        meta = records[0]
        runs.add(meta.get("run_id"))
        if run_id is not None and meta.get("run_id") != run_id:
            continue
        if meta.get("schema_version") != 1:
            raise TraceError(f"{path}: unsupported schema_version")
        required = ("run_id", "worker_id", "rank", "pp_rank", "pp_size", "dp_rank", "expected_stage_ranks")
        if any(k not in meta for k in required):
            raise TraceError(f"{path}: missing required metadata")
        key = str(meta["worker_id"])
        if key not in workers:
            workers[key] = {"meta": meta, "records": [], "seen": set(), "files": [], "truncated": False}
        worker = workers[key]
        if worker["meta"] != meta:
            raise TraceError(f"{path}: conflicting metadata for worker {key}")
        worker["files"].append(str(path))
        worker["truncated"] |= truncated
        for record in records[1:]:
            signature = _canonical(record)
            if signature not in worker["seen"]:
                worker["seen"].add(signature)
                worker["records"].append(record)
    if run_id is None and len(runs) > 1:
        raise TraceError("Multiple run IDs found; select one with --run-id")
    if not workers:
        raise TraceError("No input files match the requested run")
    return workers


class ClockMap:
    def __init__(self, anchors, worker_id, warnings):
        if not anchors:
            raise TraceError(f"{worker_id}: missing clock calibration")
        self.clock_id = anchors[0]["clock_id"]
        if any(a["clock_id"] != self.clock_id for a in anchors):
            raise TraceError(f"{worker_id}: clock reference changed during capture")
        by_device = {}
        for anchor in anchors:
            stamp = float(anchor["device_us"])
            if stamp in by_device and by_device[stamp] != anchor:
                raise TraceError(f"{worker_id}: conflicting clock anchors")
            by_device[stamp] = anchor
            if int(anchor["uncertainty_ns"]) < 0:
                raise TraceError(f"{worker_id}: negative calibration uncertainty")
        self.anchors = sorted(by_device.values(), key=lambda a: float(a["device_us"]))
        self.base = self.anchors[0]
        self.stamps = [float(a["device_us"]) for a in self.anchors]
        self.scales = []
        for left, right in zip(self.anchors, self.anchors[1:]):
            delta = float(right["device_us"]) - float(left["device_us"])
            scale = (int(right["reference_ns"]) - int(left["reference_ns"])) / delta
            if scale <= 0:
                raise TraceError(f"{worker_id}: non-monotonic clock mapping")
            if abs(scale / 1000.0 - 1) > 0.01:
                raise TraceError(f"{worker_id}: clock scale differs by over 1%; check units/reset")
            self.scales.append(scale)
        if len(self.anchors) == 1:
            _warn(warnings, f"{worker_id}: single clock anchor; clock drift is unmeasured")
        self.worker_id, self.warnings = worker_id, warnings

    def map(self, device_us):
        stamp = float(device_us)
        if len(self.anchors) == 1:
            result = int(self.base["reference_ns"]) + round((stamp - self.stamps[0]) * 1000)
            return result, int(self.base["uncertainty_ns"])
        if not self.stamps[0] <= stamp <= self.stamps[-1]:
            _warn(self.warnings, f"{self.worker_id}: events extend outside clock anchors; drift extrapolated")
        # Startup, explicit capture start, and finish may each supply anchors.
        # Retain all of them; do not discard the fresh capture-start offset.
        index = min(max(bisect.bisect_right(self.stamps, stamp) - 1, 0), len(self.anchors) - 2)
        left, right = self.anchors[index:index + 2]
        first, last = self.stamps[index:index + 2]
        fraction = (stamp - first) / (last - first)
        result = int(left["reference_ns"]) + round((stamp - first) * self.scales[index])
        error = round(abs(1 - fraction) * int(left["uncertainty_ns"]) + abs(fraction) * int(right["uncertainty_ns"]))
        return result, error


def _chunks(value):
    if not isinstance(value, list) or not value:
        raise TraceError("forward must have at least one request chunk")
    result = []
    for chunk in value:
        if not isinstance(chunk, dict) or not all(k in chunk for k in ("rid", "start", "end", "ordinal", "attempt")):
            raise TraceError("incomplete request chunk metadata")
        if any(type(chunk[k]) is not int for k in ("start", "end", "ordinal", "attempt")):
            raise TraceError("chunk positions/ordinal/attempt must be integers")
        if chunk["start"] < 0 or chunk["end"] <= chunk["start"]:
            raise TraceError("chunk token range must be positive and nonempty")
        result.append({k: chunk[k] for k in ("rid", "start", "end", "ordinal", "attempt")})
    return result


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(end, merged[-1][1])
    return merged


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge(inputs, output, run_id=None):
    """Return output paths and warnings. Accept directories or individual files."""
    paths = set()
    for item in inputs:
        path = Path(item)
        paths.update(path.rglob("*.jsonl") if path.is_dir() else [path])
    if not paths:
        raise TraceError("No JSONL inputs found")
    warnings = []
    workers = _load(sorted(paths), run_id, warnings)
    rows, clock_ids, pp_sizes = [], set(), set()
    stage_workers = defaultdict(set)
    expected, healthy = {}, {}
    for worker_id, worker in workers.items():
        meta = worker["meta"]
        pp_size, pp_rank, dp_rank = int(meta["pp_size"]), int(meta["pp_rank"]), int(meta["dp_rank"])
        if not 0 <= pp_rank < pp_size or int(meta["expected_stage_ranks"]) <= 0:
            raise TraceError(f"{worker_id}: invalid stage metadata")
        pp_sizes.add(pp_size)
        stage = (dp_rank, pp_rank)
        if stage in expected and expected[stage] != int(meta["expected_stage_ranks"]):
            raise TraceError(f"{stage}: inconsistent expected_stage_ranks")
        expected[stage] = int(meta["expected_stage_ranks"])
        stage_workers[stage].add(int(meta["rank"]))
        records = worker["records"]
        clock = ClockMap([r for r in records if r.get("type") == "clock"], worker_id, warnings)
        clock_ids.add(clock.clock_id)
        endings = [r for r in records if r.get("type") == "session_end"]
        healthy[worker_id] = bool(endings) and not worker["truncated"]
        if not endings:
            _warn(warnings, f"{worker_id}: no session_end; capture may be live or interrupted")
        if any(r.get("complete") is False for r in endings):
            healthy[worker_id] = False
            _warn(warnings, f"{worker_id}: session_end marks capture incomplete")
        seen_seq, seen_batches = {}, set()
        for record in records:
            kind = record.get("type")
            if kind in ("error", "dropped", "truncated"):
                healthy[worker_id] = False
                _warn(warnings, f"{worker_id}: {kind}: {_canonical(record)}")
            if kind != "forward":
                continue
            if record.get("status") != "ok" or "start_us" not in record or "end_us" not in record:
                healthy[worker_id] = False
                _warn(warnings, f"{worker_id}: unfinished/failed forward excluded: {record.get('batch_id')}")
                continue
            sequence = record.get("local_forward_seq")
            if sequence in seen_seq:
                raise TraceError(f"{worker_id}: conflicting local_forward_seq={sequence}")
            seen_seq[sequence] = record
            if not record.get("batch_id") or record["batch_id"] in seen_batches:
                raise TraceError(f"{worker_id}: missing or repeated batch_id")
            seen_batches.add(record["batch_id"])
            if float(record["end_us"]) < float(record["start_us"]):
                raise TraceError(f"{worker_id}: forward ends before it starts")
            start, start_error = clock.map(record["start_us"])
            end, end_error = clock.map(record["end_us"])
            chunks = _chunks(record.get("chunks"))
            rows.append(dict(record, chunks=chunks, meta=meta, start_ns=start, end_ns=end,
                             start_error_ns=start_error, end_error_ns=end_error,
                             worker_id=worker_id, rank=int(meta["rank"]), dp_rank=dp_rank, pp_rank=pp_rank))
    if len(clock_ids) != 1:
        raise TraceError("Inputs use different clock_id values; calibrate to the same clock server")
    if len(pp_sizes) != 1:
        raise TraceError("Inputs have inconsistent pp_size")
    if not rows:
        raise TraceError("No completed forwards to merge")
    pp_size = pp_sizes.pop()
    groups = defaultdict(list)
    identities = {}
    for row in rows:
        identity = (row["dp_rank"], row["batch_id"])
        signature = _canonical(row["chunks"])
        if identity in identities and identities[identity] != signature:
            raise TraceError(f"{identity}: request chunks disagree across stages/ranks")
        identities[identity] = signature
        groups[(row["dp_rank"], row["pp_rank"], row["batch_id"])].append(row)
    spans = []
    for (dp, pp, batch), members in groups.items():
        ranks = [m["rank"] for m in members]
        if len(ranks) != len(set(ranks)):
            raise TraceError(f"DP{dp} PP{pp} {batch}: multiple workers recorded the same rank")
        complete = len(ranks) == expected[(dp, pp)]
        if len(ranks) > expected[(dp, pp)]:
            raise TraceError(f"DP{dp} PP{pp}: more ranks than expected")
        if not complete:
            _warn(warnings, f"DP{dp} PP{pp} {batch}: incomplete stage {len(ranks)}/{expected[(dp, pp)]} ranks")
        first = min(members, key=lambda m: m["start_ns"])
        last = max(members, key=lambda m: m["end_ns"])
        spans.append(dict(dp_rank=dp, pp_rank=pp, batch_id=batch, chunks=first["chunks"],
                          start_ns=first["start_ns"], end_ns=last["end_ns"],
                          start_error_ns=max(m["start_error_ns"] for m in members),
                          end_error_ns=max(m["end_error_ns"] for m in members),
                          complete=complete and all(healthy[m["worker_id"]] for m in members), rank_count=len(ranks)))
    lane_batches = defaultdict(set)
    for dp, _, batch in groups:
        lane_batches[dp].add(batch)
    stage_health = defaultdict(lambda: True)
    for worker_id, worker in workers.items():
        meta = worker["meta"]
        stage_health[(int(meta["dp_rank"]), int(meta["pp_rank"]))] &= healthy[worker_id]
    origin = min(r["start_ns"] for r in rows)
    finish = max(r["end_ns"] for r in rows)
    total = finish - origin
    if total <= 0:
        raise TraceError("Timeline has zero duration")
    rids = {}
    for row in sorted(rows, key=lambda r: (r["start_ns"], r["pp_rank"], r["rank"])):
        for chunk in row["chunks"]:
            rids.setdefault(str(chunk["rid"]), f"R{len(rids) + 1}")
    colors = ("thread_state_running", "rail_response", "rail_animation", "rail_idle", "rail_load", "good", "bad", "cq_build_running")
    events = []

    def pid(dp, pp):
        return dp * pp_size + pp + 1

    def label(chunks):
        return "; ".join(f"{rids[str(c['rid'])]} C{c['ordinal']} ({c['end'] - c['start']} tokens)" for c in chunks)

    def event(span, tid, category):
        digest = hashlib.sha256(_canonical([c["rid"] for c in span["chunks"]]).encode()).digest()
        args = {k: span[k] for k in ("batch_id", "chunks", "dp_rank", "pp_rank", "start_error_ns", "end_error_ns")}
        args.update(complete=span.get("complete", True), measured="forward stream span; includes internal waits")
        events.append({"name": label(span["chunks"]), "cat": category, "ph": "X",
                       "pid": pid(span["dp_rank"], span["pp_rank"]), "tid": tid,
                       "ts": (span["start_ns"] - origin) / 1000, "dur": (span["end_ns"] - span["start_ns"]) / 1000,
                       "cname": colors[digest[0] % len(colors)], "args": args})

    summary, handoffs, chunk_rows = [], [], []
    active_lanes = sorted({r["dp_rank"] for r in rows})
    for dp in active_lanes:
        for pp in range(pp_size):
            stage = [s for s in spans if (s["dp_rank"], s["pp_rank"]) == (dp, pp)]
            process = pid(dp, pp)
            events.extend([
                {"ph": "M", "name": "process_name", "pid": process, "args": {"name": f"DP{dp} PP{pp}"}},
                {"ph": "M", "name": "process_sort_index", "pid": process, "args": {"sort_index": process}},
                {"ph": "M", "name": "thread_name", "pid": process, "tid": 0, "args": {"name": "stage envelope"}},
            ])
            if not stage:
                _warn(warnings, f"DP{dp} PP{pp}: missing stage")
                summary.append({"dp_rank": dp, "pp_rank": pp, "complete": False, "forward_count": 0})
                continue
            missing_batches = lane_batches[dp] - {s["batch_id"] for s in stage}
            valid = all(s["complete"] for s in stage) and not missing_batches and stage_health[(dp, pp)]
            if not valid:
                _warn(warnings, f"DP{dp} PP{pp}: gap/occupancy statistics unavailable because capture is incomplete ({len(missing_batches)} missing batches)")
                for span in stage:
                    span["complete"] = False
            intervals = _union((s["start_ns"], s["end_ns"]) for s in stage)
            occupied = sum(b - a for a, b in intervals)
            fill, drain = intervals[0][0] - origin, finish - intervals[-1][1]
            gaps = [(intervals[i][1], intervals[i + 1][0]) for i in range(len(intervals) - 1)]
            for start, end in gaps if valid else []:
                events.append({"ph": "X", "name": "outside forward", "cat": "gap", "pid": process, "tid": 0,
                               "ts": (start - origin) / 1000, "dur": (end - start) / 1000,
                               "cname": "thread_state_sleeping", "args": {"note": "May include communication, CPU work or KV transfer; not measured hardware idle"}})
            summary.append(dict(dp_rank=dp, pp_rank=pp, complete=valid, missing_batch_count=len(missing_batches), forward_count=len(stage),
                                global_window_ms=total / 1e6, forward_union_ms=occupied / 1e6,
                                forward_fraction=occupied / total if valid else None,
                                outside_forward_ms=(total - occupied) / 1e6 if valid else None,
                                outside_forward_fraction=(total - occupied) / total if valid else None,
                                fill_ms=fill / 1e6 if valid else None,
                                internal_gap_ms=sum(b - a for a, b in gaps) / 1e6 if valid else None,
                                drain_ms=drain / 1e6 if valid else None))
            for span in stage:
                event(span, 0, "stage_forward_envelope")
    for row in rows:
        tid = row["rank"] + 1
        process = pid(row["dp_rank"], row["pp_rank"])
        meta = row["meta"]
        events.append({"ph": "M", "name": "thread_name", "pid": process, "tid": tid,
                       "args": {"name": f"rank {row['rank']} / TP{meta.get('tp_rank', '?')} CP{meta.get('cp_rank', '?')}"}})
        event(row, tid, "rank_forward")
        chunk_rows.append(dict(worker_id=row["worker_id"], rank=row["rank"], pp_rank=row["pp_rank"], dp_rank=row["dp_rank"],
                               batch_id=row["batch_id"], mb_id=row.get("mb_id"), local_forward_seq=row.get("local_forward_seq"),
                               start_ms=(row["start_ns"] - origin) / 1e6, end_ms=(row["end_ns"] - origin) / 1e6,
                               duration_ms=(row["end_ns"] - row["start_ns"]) / 1e6,
                               token_count=sum(c["end"] - c["start"] for c in row["chunks"]),
                               request_chunks=_canonical(row["chunks"])))
    by_batch = defaultdict(dict)
    for span in spans:
        by_batch[(span["dp_rank"], span["batch_id"])][span["pp_rank"]] = span
    for (dp, batch), stages in sorted(by_batch.items()):
        missing = sorted(set(range(pp_size)) - set(stages))
        if missing:
            _warn(warnings, f"DP{dp} {batch}: missing PP stages {missing}; no endpoints invented")
        for pp in range(pp_size - 1):
            if pp not in stages or pp + 1 not in stages:
                continue
            source, dest = stages[pp], stages[pp + 1]
            duration = dest["start_ns"] - source["end_ns"]
            uncertainty = source["end_error_ns"] + dest["start_error_ns"]
            status = "incomplete" if not (source["complete"] and dest["complete"]) else (
                "uncertain" if abs(duration) <= uncertainty else "negative" if duration < 0 else "ok")
            if status == "negative":
                _warn(warnings, f"DP{dp} {batch} PP{pp}->{pp + 1}: negative envelope handoff; inspect rank skew/clock calibration")
            handoffs.append(dict(dp_rank=dp, batch_id=batch, source_pp=pp, dest_pp=pp + 1,
                                 handoff_ms=duration / 1e6, uncertainty_ms=uncertainty / 1e6, status=status))
            # Do not draw backward flows or claim incomplete/uncertain data proves a dependency duration.
            if duration >= 0:
                flow_id = hashlib.sha256(f"{dp}:{batch}:{pp}".encode()).hexdigest()[:16]
                for phase, span, stamp in (("s", source, source["end_ns"]), ("f", dest, dest["start_ns"])):
                    events.append({"name": "PP handoff (includes scheduling)", "cat": "handoff", "ph": phase,
                                   "id": flow_id, "pid": pid(dp, span["pp_rank"]), "tid": 0,
                                   "ts": (stamp - origin) / 1000, "bp": "e", "args": {"batch_id": batch, "status": status}})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace = {"traceEvents": events, "displayTimeUnit": "ms", "metadata": {
        "schema_version": 1, "run_id": next(iter(workers.values()))["meta"]["run_id"],
        "clock_id": next(iter(clock_ids)), "reference_origin_ns": origin, "warnings": warnings,
        "note": "Stage envelope occupancy is not hardware utilization. Handoff includes scheduling and queueing. Calibration error excludes unmeasured clock drift.",
        "requests": rids,
    }}
    (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    (output / "warnings.json").write_text(json.dumps(warnings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _csv(output / "summary.csv", summary, ["dp_rank", "pp_rank", "complete", "missing_batch_count", "forward_count", "global_window_ms", "forward_union_ms", "forward_fraction", "outside_forward_ms", "outside_forward_fraction", "fill_ms", "internal_gap_ms", "drain_ms"])
    _csv(output / "chunks.csv", chunk_rows, ["worker_id", "rank", "pp_rank", "dp_rank", "batch_id", "mb_id", "local_forward_seq", "start_ms", "end_ms", "duration_ms", "token_count", "request_chunks"])
    _csv(output / "handoffs.csv", handoffs, ["dp_rank", "batch_id", "source_pp", "dest_pp", "handoff_ms", "uncertainty_ms", "status"])
    return {"output": str(output), "forward_records": len(rows), "stage_spans": len(spans), "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        result = merge(args.inputs, args.output, args.run_id)
    except (TraceError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Cannot merge trace: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
