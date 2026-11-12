"""CPU-only regression for the PP prefill completion scheduling order.

Run directly with Python; no torch, SGLang installation or NPU is needed.
The tested methods are extracted from the production AST, not copied here.
Set SGLANG_TEST_REPO to test a checkout other than the containing repository.
"""

from __future__ import annotations

import ast
import logging
import os
import queue
import threading
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace


def _source_path():
    root = os.environ.get("SGLANG_TEST_REPO")
    if root:
        return Path(root) / "python/sglang/srt/managers/scheduler_pp_mixin.py"
    for parent in Path(__file__).resolve().parents:
        source = parent / "python/sglang/srt/managers/scheduler_pp_mixin.py"
        if source.is_file():
            return source
    raise RuntimeError("Set SGLANG_TEST_REPO to the checkout under test")


class StopLoop(Exception):
    pass


class _Flag:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _load_methods(early, parallel):
    tree = ast.parse(_source_path().read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SchedulerPPMixin"
    )
    names = {
        "event_loop_pp_disagg_prefill",
        "_pp_prefill_exchange_release",
        "_pp_prefill_send_control",
        "_pp_pd_send_consensus_release_ids",
        "_pp_pd_get_prefill_transferred_ids",
        "_pp_prefill_retire_transfers",
    }
    functions = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    missing = names - {n.name for n in functions}
    if missing:
        raise AssertionError(f"Production methods not found: {sorted(missing)}")
    for function in functions:
        function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    namespace = {
        "envs": SimpleNamespace(SGLANG_PP_PREFILL_EARLY_RELEASE=_Flag(early)),
        "get_parallel": lambda: parallel,
        "KVPoll": SimpleNamespace(Success="success", Failed="failed"),
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.fix_missing_locations(module), str(_source_path()), "exec"),
        namespace,
    )
    return {name: namespace[name] for name in names}


class _Work:
    def __init__(self, kind, ack=None):
        self.kind = kind
        self.ack = ack


class FakeStage:
    def __init__(
        self, early, pp_size, depth, rank=0, steps=12, wire=None, sparse=False
    ):
        self.parallel = SimpleNamespace(pp_size=pp_size, pp_async_batch_depth=depth)
        self.pp_group = SimpleNamespace(
            is_first_rank=rank == 0, is_last_rank=rank == pp_size - 1
        )
        self.rank, self.steps, self.wire = rank, steps, wire
        self.sparse = sparse
        self.step = -1
        self.ops = []
        self.enable_staging = False
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )
        self.disagg_prefill_inflight_queue = []
        self._release_send = False
        self.received_releases = []
        methods = _load_methods(early, self.parallel)
        for name, method in methods.items():
            if name != "_pp_pd_get_prefill_transferred_ids":
                setattr(self, name, MethodType(method, self))
        original_release = self._pp_pd_send_consensus_release_ids

        def release(*args):
            self.log("exchange_release")
            self._release_send = True
            try:
                return original_release(*args)
            finally:
                self._release_send = False

        self._pp_pd_send_consensus_release_ids = release

    def log(self, name):
        self.ops.append((self.step, name))

    def init_pp_loop_state(self):
        self.pp_loop_size = self.parallel.pp_size + self.parallel.pp_async_batch_depth
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [SimpleNamespace(batch_is_full=False) for _ in self.mbs]
        self.mb_metadata = [None] * self.pp_loop_size
        self.last_rank_comm_queue = []
        self.send_req_work = []

    def ingest_requests(self):
        self.step += 1
        if self.wire:
            if self.wire.stop.is_set():
                raise StopLoop
        elif self.step >= self.steps:
            raise StopLoop
        self.log("ingest")
        if self.wire and not self.pp_group.is_first_rank:
            return self.wire.recv(self.rank, "requests")
        return [f"request-{self.step}"]

    def process_bootstrapped_queue(self):
        self.log("bootstrap")

    def _pp_pd_get_prefill_transferred_ids(self):
        self.log("snapshot_transferred")
        if self.wire and not self.pp_group.is_first_rank:
            return self.wire.recv(self.rank, "transferred")
        # Alternating nonempty and empty snapshots exercise the two-pass ring.
        return {f"done-{self.step}": bool(self.step % 3)} if self.step % 2 else {}

    def _pp_commit_comm_work(self, works):
        for work in works:
            self.log(f"commit_{work.kind}")
            if self.wire and work.ack is not None:
                self.wire.wait_ack(work.ack)
        works.clear()

    def _pp_recv_pyobj_from_prev_stage(self):
        self.log("recv_release")
        value = self.wire.recv(self.rank, "release") if self.wire else {}
        self.received_releases.append(value)
        return value

    def _pp_send_pyobj_to_next_stage(self, value, async_send):
        assert async_send
        kind = (
            "release"
            if self._release_send
            else "requests" if isinstance(value, list) else "transferred"
        )
        self.log(f"send_{kind}")
        ack = self.wire.send(self.rank, kind, value) if self.wire else None
        return [_Work(kind, ack)]

    def process_disagg_prefill_inflight_queue(self, release):
        assert release is not None, "None would bypass PP consensus"
        self.log("retire_and_respond")
        return []

    def _pp_prefill_completion_event(self, event, reqs):
        self.log(event)

    def process_prefill_chunk(self, **kwargs):
        self.log("prepare_chunk")

    def _process_hicache_events(self):
        pass

    def get_new_batch_prefill(self, running_batch):
        self.log("schedule")
        return SimpleNamespace(
            batch_to_run=(
                None
                if self.sparse and self.step % 3 != 1
                else SimpleNamespace(index=self.step, reqs=[])
            ),
            running_batch=running_batch,
        )

    def _pp_recv_proxy_tensors(self):
        self.log("recv_proxy")
        return object()

    def _pp_commit_proxy_send_work(self):
        self.log("commit_proxy")

    def _pp_launch_batch(self, *args):
        self.log("launch")
        return object(), object()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(self, *args):
        self.log("output_wait")
        return None, object(), SimpleNamespace(synchronize=lambda: self.log("d2h_wait"))

    def _pp_process_batch_result(self, *args):
        self.log("process_result")

    def _pp_send_proxy_to_next_stage(self, result):
        self.log("send_proxy")

    def on_idle(self):
        pass

    def run(self):
        try:
            self.event_loop_pp_disagg_prefill()
        except StopLoop:
            pass


class Wire:
    """Strict FIFO CPU P2P channels; send commits require a matching receive.

    This tests the scheduler's control-message order and startup gates. It does
    not model HCCL/device streams; those still need a multi-NPU smoke test.
    """

    def __init__(self, pp_size):
        self.queues = [queue.Queue() for _ in range(pp_size)]
        self.stop = threading.Event()
        self.pp_size = pp_size

    def send(self, rank, kind, value):
        ack = threading.Event()
        self.queues[(rank + 1) % self.pp_size].put((kind, value, ack))
        return ack

    def recv(self, rank, expected):
        while not self.stop.is_set():
            try:
                kind, value, ack = self.queues[rank].get(timeout=0.01)
            except queue.Empty:
                continue
            if kind != expected:
                raise AssertionError(
                    f"rank {rank}: expected {expected}, received {kind}"
                )
            ack.set()
            return value
        raise StopLoop

    def wait_ack(self, ack):
        while not ack.wait(0.01):
            if self.stop.is_set():
                raise StopLoop


class TestPPPrefillEarlyRelease(unittest.TestCase):
    def test_unsupported_speculative_path_fails_before_ingesting_requests(self):
        stage = FakeStage(True, 4, 0)
        stage.spec_algorithm = SimpleNamespace(is_none=lambda: False)
        with self.assertRaises(ValueError):
            stage.run()
        self.assertEqual(stage.ops, [])

    def test_retirement_preserves_none_empty_and_failed_consensus(self):
        method = _load_methods(True, SimpleNamespace())["_pp_prefill_retire_transfers"]
        a, b, c = [SimpleNamespace(rid=rid) for rid in ("a", "b", "c")]
        events = []
        calls = []

        def retire(consensus):
            calls.append(consensus)
            return [a, b] if consensus else []

        stage = SimpleNamespace(
            disagg_prefill_inflight_queue=[a, b, c],
            _pp_prefill_completion_event=lambda name, reqs: events.append(
                (name, [r.rid for r in reqs])
            ),
            process_disagg_prefill_inflight_queue=retire,
        )
        method(stage, None)
        self.assertEqual(calls, [])
        self.assertEqual(events, [])
        method(stage, {})
        self.assertEqual(calls, [{}])
        self.assertEqual(events, [("release_ready", []), ("response_enqueued", [])])
        consensus = {"a": False, "b": True}
        method(stage, consensus)
        self.assertIs(calls[-1], consensus)
        self.assertEqual(
            events[-2:],
            [("release_ready", ["a", "b"]), ("response_enqueued", ["a", "b"])],
        )

    def test_release_still_requires_every_stage_and_propagates_failure(self):
        method = _load_methods(True, SimpleNamespace())[
            "_pp_pd_get_prefill_transferred_ids"
        ]
        stage = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False),
            disagg_prefill_inflight_queue=[],
            get_rids=lambda *args: (
                ["local-only", "both", "failed-remotely"],
                ["failed-locally"],
            ),
            _pp_recv_pyobj_from_prev_stage=lambda: {
                "both": False,
                "remote-only": False,
                "failed-remotely": True,
                "failed-locally": False,
            },
        )
        self.assertEqual(
            method(stage),
            {"both": False, "failed-remotely": True, "failed-locally": True},
        )

    def test_ready_responses_precede_unrelated_d2h_waits(self):
        for pp_size in (2, 4):
            for depth in (0, 1):
                for rank in range(pp_size):
                    with self.subTest(pp=pp_size, depth=depth, rank=rank):
                        stage = FakeStage(True, pp_size, depth, rank)
                        stage.run()
                        for step in range(stage.steps):
                            ops = [name for i, name in stage.ops if i == step]
                            self.assertLess(
                                ops.index("recv_proxy"), ops.index("exchange_release")
                            )
                            self.assertLess(
                                ops.index("output_wait"), ops.index("exchange_release")
                            )
                            if "retire_and_respond" in ops:
                                self.assertLess(
                                    ops.index("exchange_release"),
                                    ops.index("retire_and_respond"),
                                )
                                self.assertLess(
                                    ops.index("retire_and_respond"),
                                    ops.index("d2h_wait"),
                                )
                                self.assertLess(
                                    ops.index("retire_and_respond"),
                                    ops.index("process_result"),
                                )
                            if rank < pp_size - 1:
                                self.assertLess(
                                    ops.index("send_requests"),
                                    ops.index("send_transferred"),
                                )
                                self.assertLess(
                                    ops.index("send_transferred"),
                                    ops.index("send_proxy"),
                                )

    def test_early_release_preserves_all_communication_order(self):
        # The flag may move retirement, never a CPU or device communication.
        comm_ops = {
            "ingest",
            "snapshot_transferred",
            "recv_proxy",
            "commit_proxy",
            "output_wait",
            "exchange_release",
            "send_release",
            "recv_release",
            "commit_release",
            "send_requests",
            "send_transferred",
            "send_proxy",
            "commit_requests",
            "commit_transferred",
        }
        for pp_size in (2, 4):
            for depth in (0, 1, 2):
                for rank in range(pp_size):
                    with self.subTest(pp=pp_size, depth=depth, rank=rank):
                        sequences = []
                        for early in (False, True):
                            stage = FakeStage(early, pp_size, depth, rank)
                            stage.run()
                            sequences.append(
                                [op for op in stage.ops if op[1] in comm_ops]
                            )
                        self.assertEqual(*sequences)

    def _retirement_lifecycle_stage(self, early, depth=0, steps=8):
        """Use the real retirement method with CPU-only resource/output spies."""
        source = _source_path().parents[1] / "disaggregation/prefill.py"
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "SchedulerDisaggregationPrefillMixin"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "process_disagg_prefill_inflight_queue"
        )
        method.decorator_list = []
        stage = FakeStage(early, 4, depth, steps=steps)
        stage.released, stage.responded, stage.cleared, stage.metadata_freed = (
            [],
            [],
            [],
            [],
        )
        stage.finished_handles = []
        polls = SimpleNamespace(Success=4, Failed=0, WaitingForInput=2, Transferring=3)

        def release_kv(req, cache, checkpoint):
            self.assertTrue(checkpoint)
            self.assertNotIn(req.rid, stage.released, "KV released twice")
            stage.released.append(req.rid)

        def release_metadata(req, allocator):
            self.assertGreaterEqual(req.metadata_buffer_index, 0)
            stage.metadata_freed.append(req.rid)
            req.metadata_buffer_index = -1

        namespace = {
            "KVPoll": polls,
            "FINISH_ABORT": type("AbortReason", (), {}),
            "FINISH_LENGTH": lambda length: SimpleNamespace(length=length),
            "CacheRequestOutcome": SimpleNamespace(SUCCESS="success"),
            "FAKE_BOOTSTRAP_HOST": "fake",
            "release_kv_cache": release_kv,
            "maybe_release_metadata_buffer": release_metadata,
            "poll_and_all_reduce_attn_cp_tp_group": lambda senders, *groups: [
                sender.poll() for sender in senders
            ],
            "rank_consensus_checker": SimpleNamespace(assert_same=lambda *args: None),
        }
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        stage.process_disagg_prefill_inflight_queue = MethodType(
            namespace[method.name], stage
        )
        stage.attn_cp_cpu_group = stage.attn_tp_cpu_group = None
        stage.tree_cache = SimpleNamespace(
            finish=lambda handle, outcome: stage.finished_handles.append(
                (handle, outcome)
            )
        )
        stage.req_to_metadata_buffer_idx_allocator = object()
        stage.output_streamer = SimpleNamespace(
            stream_output=lambda reqs, *args: stage.responded.extend(
                r.rid for r in reqs
            )
        )

        def request(rid):
            req = SimpleNamespace(
                rid=rid,
                pending_bootstrap=False,
                return_routed_experts=False,
                return_logprob=False,
                finished_reason=None,
                bootstrap_host="fake",
                cache_request_handle=rid,
                metadata_buffer_index=1,
                time_stats=SimpleNamespace(
                    set_prefill_kv_transfer_finish_time=lambda: None,
                    set_completion_time=lambda: None,
                ),
                disagg_kv_sender=SimpleNamespace(
                    poll=lambda: polls.Success,
                    clear=lambda: stage.cleared.append(rid),
                ),
            )
            req.finish_outside_step = lambda reason: setattr(
                req, "finished_reason", reason
            )
            return req

        stage.old_req, stage.new_req = request("old"), request("new")
        stage.disagg_prefill_inflight_queue = [stage.old_req]
        # Repeat the old release, including after it has already been retired.
        stage._pp_recv_pyobj_from_prev_stage = lambda: {"old": False}
        return stage

    def test_ready_transfer_responds_before_a_blocked_unrelated_d2h(self):
        # The D2H barrier stops the loop before the next result can be processed.
        # Early mode must have already sent the old request's response there.
        for early in (False, True):
            for depth in (0, 1):
                with self.subTest(early=early, depth=depth):
                    stage = self._retirement_lifecycle_stage(early, depth)
                    reached_d2h = []

                    def blocked_d2h():
                        reached_d2h.append(True)
                        raise StopLoop

                    stage._pp_commit_send_output_work_and_preprocess_output_tensors = (
                        lambda *args: (
                            None,
                            object(),
                            SimpleNamespace(synchronize=blocked_d2h),
                        )
                    )
                    stage._pp_process_batch_result = lambda *args: self.fail(
                        "A blocked D2H must prevent result processing"
                    )
                    stage.run()
                    self.assertEqual(reached_d2h, [True])
                    expected = ["old"] if early else []
                    self.assertEqual(stage.responded, expected)
                    self.assertEqual(stage.released, expected)
                    self.assertEqual(stage.metadata_freed, expected)

    def test_old_consensus_does_not_retire_new_result_or_reply_twice(self):
        for early in (False, True):
            for depth in (0, 1):
                with self.subTest(early=early, depth=depth):
                    stage = self._retirement_lifecycle_stage(early, depth, steps=10)
                    appended = []

                    def process_result(*args):
                        if not appended:
                            stage.disagg_prefill_inflight_queue.append(stage.new_req)
                            appended.append(stage.step)

                    stage._pp_process_batch_result = process_result
                    stage.run()
                    self.assertEqual(len(appended), 1)
                    self.assertEqual(stage.responded, ["old"])
                    self.assertEqual(stage.released, ["old"])
                    self.assertEqual(stage.cleared, ["old"])
                    self.assertEqual(stage.metadata_freed, ["old"])
                    self.assertEqual(stage.finished_handles, [("old", "success")])
                    self.assertEqual(
                        stage.disagg_prefill_inflight_queue, [stage.new_req]
                    )
                    self.assertIsNone(stage.new_req.finished_reason)
                    self.assertEqual(stage.new_req.metadata_buffer_index, 1)

    def test_disabled_retains_legacy_completion_order(self):
        for pp_size in (2, 4):
            for depth in (0, 1):
                with self.subTest(pp=pp_size, depth=depth):
                    stage = FakeStage(False, pp_size, depth)
                    stage.run()
                    for step in range(stage.steps):
                        ops = [name for i, name in stage.ops if i == step]
                        self.assertLess(
                            ops.index("output_wait"), ops.index("exchange_release")
                        )
                        if "retire_and_respond" in ops:
                            self.assertLess(
                                ops.index("process_result"),
                                ops.index("retire_and_respond"),
                            )
                            self.assertLess(
                                ops.index("retire_and_respond"),
                                ops.index("send_requests"),
                            )

    def test_no_release_during_startup_but_empty_consensus_is_processed(self):
        for early in (False, True):
            for depth in (0, 1):
                stage = FakeStage(early, 4, depth)
                stage.run()
                retired = [
                    step for step, name in stage.ops if name == "retire_and_respond"
                ]
                self.assertEqual(
                    retired, list(range(stage.pp_loop_size - 1, stage.steps))
                )
                self.assertEqual(stage.received_releases, [{}] * len(retired))

    def test_cpu_ring_progress_and_fifo_order(self):
        for early in (False, True):
            for pp_size in (2, 3, 4):
                for depth in (0, 1, 2):
                    for sparse in (False, True):
                        with self.subTest(
                            early=early, pp=pp_size, depth=depth, sparse=sparse
                        ):
                            wire = Wire(pp_size)
                            stages = [
                                FakeStage(
                                    early,
                                    pp_size,
                                    depth,
                                    rank,
                                    wire=wire,
                                    sparse=sparse,
                                )
                                for rank in range(pp_size)
                            ]
                            errors = []

                            def run(stage):
                                try:
                                    stage.run()
                                except BaseException as exc:
                                    errors.append(exc)
                                    wire.stop.set()

                            threads = [
                                threading.Thread(target=run, args=(stage,), daemon=True)
                                for stage in stages
                            ]
                            for thread in threads:
                                thread.start()
                            deadline = time.monotonic() + 3
                            target = 3 * (pp_size + depth)
                            while (
                                not errors
                                and min(stage.step for stage in stages) < target
                                and time.monotonic() < deadline
                            ):
                                time.sleep(0.002)
                            progressed = [stage.step for stage in stages]
                            wire.stop.set()
                            for thread in threads:
                                thread.join(timeout=1)
                            self.assertFalse(errors, repr(errors))
                            self.assertTrue(
                                all(not thread.is_alive() for thread in threads)
                            )
                            self.assertGreaterEqual(
                                min(progressed),
                                target,
                                f"Control ring stalled at {progressed}",
                            )


if __name__ == "__main__":
    unittest.main()
