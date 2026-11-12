"""CPU-only PP prefill regression with real production tensor/output ordering.

Place beside test_pp_prefill_early_release.py. No torch or NPU is required.
Set SGLANG_TEST_REPO to test a different checkout. Production AST methods cover
output parity, typed proxy/output inboxes, PP2 atomic send/recv and skip markers.
The transport models blocking HCCL setup and strict matching Work completion.
It does not model device execution time, memory lifetime or HCCL internals.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import queue
import threading
import time
import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from types import MethodType, SimpleNamespace

repo = os.environ.get("SGLANG_TEST_REPO")
helpers_path = (
    Path(repo) / "test/registered/unit/managers/test_pp_prefill_early_release.py"
    if repo
    else Path(__file__).with_name("test_pp_prefill_early_release.py")
)
spec = importlib.util.spec_from_file_location(
    "pp_early_release_test_helpers", helpers_path
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Proxy:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


class Event:
    def record(self, *args):
        pass

    def synchronize(self):
        pass

    def wait_event(self, *args):
        pass

    def wait_stream(self, *args):
        pass


EXTRA_METHODS = {
    "_pp_commit_send_output_work_and_preprocess_output_tensors",
    "_pp_send_recv_and_preprocess_output_tensors",
    "_pp_send_output_to_next_stage",
    "_pp2_only_send_recv_output_tensors_npu",
    "_pp_recv_typed_dict",
    "_pp_recv_proxy_tensors",
}


@lru_cache(maxsize=1)
def compile_extras():
    tree = ast.parse(base._source_path().read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SchedulerPPMixin"
    )
    funcs = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in EXTRA_METHODS
    ]
    assert {n.name for n in funcs} == EXTRA_METHODS
    for f in funcs:
        f.decorator_list = []
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *funcs,
        ],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(mod), str(base._source_path()), "exec")


def load_extras(parallel):
    namespace = {
        "get_parallel": lambda: parallel,
        "_is_npu": True,
        "_pp_can_skip_output_comm": lambda batch: batch.skip_output,
        "PPProxyTensors": Proxy,
        "torch": SimpleNamespace(
            profiler=SimpleNamespace(record_function=lambda *a: nullcontext())
        ),
        "envs": SimpleNamespace(
            SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD=base._Flag(False)
        ),
        "logger": logging.getLogger(__name__),
    }
    exec(compile_extras(), namespace)
    return {n: namespace[n] for n in EXTRA_METHODS}


class TensorWire:
    """A single FIFO typed tensor stream per directed adjacent stage pair.

    Initial send waits for the matching low-level recv, modeling blocking
    communicator setup observed in HCCL. Later sends return Work handles;
    commit waits until matching recv. batch_send_recv posts send before recv.
    """

    def __init__(self, control, pp_size, always_rendezvous=False):
        self.control = control
        self.always_rendezvous = always_rendezvous
        self.pp_size = pp_size
        self.queues = [queue.Queue() for _ in range(pp_size)]
        self.initialized = [False] * pp_size
        self.states = {}
        self.counts = defaultdict(int)
        self.count_lock = threading.Lock()

    def count(self, key):
        with self.count_lock:
            self.counts[key] += 1

    def post(self, rank, data):
        ack = threading.Event()
        packet = (dict(data), ack)
        self.queues[(rank + 1) % self.pp_size].put(packet)
        self.count("send_" + data["__msg_type__"])
        return base._Work(data["__msg_type__"], ack)

    def send(self, rank, data):
        work = self.post(rank, data)
        if self.always_rendezvous or not self.initialized[rank]:
            self.states[rank] = (
                "device_send_wait_for_peer_recv:%s" % data["__msg_type__"]
            )
            self.control.wait_ack(work.ack)
            self.initialized[rank] = True
            self.states.pop(rank, None)
        return [work]

    def recv(self, rank):
        self.states[rank] = "device_recv"
        while not self.control.stop.is_set():
            try:
                data, ack = self.queues[rank].get(timeout=0.005)
            except queue.Empty:
                continue
            self.initialized[(rank - 1) % self.pp_size] = True
            ack.set()
            self.count("recv_" + data["__msg_type__"])
            self.states.pop(rank, None)
            return data
        raise base.StopLoop

    def batch_send_recv(self, rank, data):
        self.count("atomic_send_recv")
        work = self.post(rank, data)
        received = self.recv(rank)
        self.control.wait_ack(work.ack)
        return received


class Stage(base.FakeStage):
    def __init__(
        self,
        early,
        pp_size,
        depth,
        rank,
        *,
        wire,
        tensor_wire,
        active,
        delayed_rank,
        delay_step,
        delay_ms,
        skip_outputs=False,
    ):
        super().__init__(early, pp_size, depth, rank, wire=wire)
        self.parallel.pp_rank = rank
        self.tensor_wire = tensor_wire
        self.active = set(active)
        self.delayed_rank = delayed_rank
        self.delay_step = delay_step
        self.delay_ms = delay_ms
        self.skip_outputs = skip_outputs
        self.completed = []
        self.launched = []
        self.control_state = ""
        for name, method in load_extras(self.parallel).items():
            setattr(self, name, MethodType(method, self))
        self.pp_group.recv_tensor_dict = lambda **kw: self.tensor_wire.recv(self.rank)
        self.pp_group.send_recv_tensor_dict = (
            lambda send_tensor_dict, **kw: self.tensor_wire.batch_send_recv(
                self.rank, send_tensor_dict
            )
        )
        self.pp_comm_stream_ctx = nullcontext()
        self.copy_stream_ctx = nullcontext()
        self.copy_stream = Event()
        self.schedule_stream = Event()
        self.attn_tp_group = None
        self.device_module = SimpleNamespace(Event=Event, current_stream=Event)
        self._pp_tensor_dict_inbox = defaultdict(deque)
        self._pp_spec_relay = False
        self.launch_event = None

    def init_pp_loop_state(self):
        super().init_pp_loop_state()
        self.last_rank_comm_queue = deque()
        self.pp_outputs = None
        self.send_output_work = []
        self.send_proxy_work = []

    def ingest_requests(self):
        if self.rank == self.delayed_rank and self.step + 1 == self.delay_step:
            self.control_state = "delayed_peer"
            time.sleep(self.delay_ms / 1000)
        self.control_state = "ingest_requests"
        result = super().ingest_requests()
        self.control_state = ""
        return result

    def _pp_pd_get_prefill_transferred_ids(self):
        self.log("snapshot_transferred")
        if not self.pp_group.is_first_rank:
            self.control_state = "recv_transferred"
            value = self.wire.recv(self.rank, "transferred")
            self.control_state = ""
            return value
        return {}  # Empty snapshots still initialize tmbs and activate release ring.

    def _pp_recv_pyobj_from_prev_stage(self):
        self.control_state = "recv_release"
        result = super()._pp_recv_pyobj_from_prev_stage()
        self.control_state = ""
        return result

    def _pp_commit_comm_work(self, works=None, *, work=None, **kwargs):
        works = work if works is None else works
        self.control_state = "commit:" + ",".join(w.kind for w in works)
        super()._pp_commit_comm_work(works)
        self.control_state = ""

    def get_new_batch_prefill(self, running_batch):
        self.log("schedule")
        batch = None
        if self.step in self.active:
            batch = SimpleNamespace(
                index=self.step,
                reqs=[],
                skip_output=self.skip_outputs and self.step % 2 == 0,
                forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
            )
        return SimpleNamespace(batch_to_run=batch, running_batch=running_batch)

    def _pp_commit_proxy_send_work(self):
        self.log("commit_proxy")
        self._pp_commit_comm_work(self.send_proxy_work)

    def _pp_launch_batch(self, mb_id, batch, proxy, metadata, result_queue):
        self.log("launch")
        if not self.pp_group.is_first_rank:
            assert proxy.tensors["index"] == batch.index, (
                self.rank,
                batch.index,
                proxy.tensors,
            )
        self.launched.append(batch.index)
        if self.pp_group.is_last_rank:
            result_queue.append((Event(), Proxy({"index": batch.index})))
        return SimpleNamespace(index=batch.index), Event()

    def _pp_send_dict_to_next_stage(
        self, tensor_dict, async_send=True, msg_type="default", ready_event=None
    ):
        assert async_send
        self.log("send_tensor_" + msg_type)
        return self.tensor_wire.send(
            self.rank, {**tensor_dict, "__msg_type__": msg_type}
        )

    def _pp_send_proxy_to_next_stage(self, result):
        self.send_proxy_work = self._pp_send_dict_to_next_stage(
            {"index": result.index}, msg_type="proxy"
        )

    def _pp_record_comm_event(self):
        return Event()

    def _pp_recv_dict_from_prev_stage(self):
        return self._pp_recv_typed_dict(expected_kind="output")

    def _pp_prep_batch_result(self, batch, metadata, proxy):
        assert proxy.tensors["index"] == batch.index, (
            "output_identity",
            self.rank,
            batch.index,
            proxy.tensors,
        )
        return SimpleNamespace(index=batch.index, logits_output=None)

    def _pp_make_skip_output_result(self, batch, metadata):
        return None, SimpleNamespace(index=batch.index, logits_output=None), Event()

    def _pp_process_batch_result(self, batch, result):
        assert batch.index == result.index
        assert batch.index not in self.completed, (
            "duplicate_result",
            self.rank,
            batch.index,
        )
        self.completed.append(batch.index)
        self.log("process_result")


def pattern(name, pp_size, depth):
    width = pp_size + depth
    if name == "cold":
        return [0]
    if name == "idle_active":
        return [3 * width]
    if name == "continuous":
        return list(range(2 * width + 3))
    if name == "sparse":
        return list(range(1, 5 * width, 3))
    if name == "idle_resume":
        return [0, 1] + list(range(4 * width, 5 * width))
    raise ValueError(name)


def run_case(
    early,
    pp_size,
    depth,
    workload,
    delay_ms=0,
    skip_outputs=False,
    timeout=2.0,
    always_rendezvous=False,
):
    control = base.Wire(pp_size)
    tensor = TensorWire(control, pp_size, always_rendezvous)
    active = pattern(workload, pp_size, depth)
    stages = [
        Stage(
            early,
            pp_size,
            depth,
            rank,
            wire=control,
            tensor_wire=tensor,
            active=active,
            delayed_rank=1,
            delay_step=active[0],
            delay_ms=delay_ms,
            skip_outputs=skip_outputs,
        )
        for rank in range(pp_size)
    ]
    errors = []

    def run(stage):
        try:
            stage.run()
        except BaseException as exc:
            errors.append(repr(exc))
            control.stop.set()

    threads = [threading.Thread(target=run, args=(s,), daemon=True) for s in stages]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + timeout
        while (
            not errors
            and not all(s.completed == active for s in stages)
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        result = dict(
            always_rendezvous=always_rendezvous,
            expected=active,
            launched=[list(s.launched) for s in stages],
            early=early,
            pp=pp_size,
            depth=depth,
            workload=workload,
            delay_ms=delay_ms,
            skip_outputs=skip_outputs,
            passed=all(s.completed == active for s in stages),
            steps=[s.step for s in stages],
            completed=[list(s.completed) for s in stages],
            tensor_wait=dict(tensor.states),
            control_wait=[s.control_state for s in stages],
            operations=[s.ops[-8:] for s in stages],
            counts=dict(tensor.counts),
            errors=errors,
        )
    finally:
        control.stop.set()
        for thread in threads:
            thread.join(timeout=1)
    assert all(not t.is_alive() for t in threads), "Worker leak"
    return result


class TestPPPrefillTensorProtocol(unittest.TestCase):
    def assert_case(
        self, early, pp, depth, workload, delay_ms, skip, *, always_rendezvous=False
    ):
        result = run_case(
            early,
            pp,
            depth,
            workload,
            delay_ms,
            skip,
            always_rendezvous=always_rendezvous,
        )
        detail = json.dumps(result, indent=2)
        # Do not use subTest here: a deadlocked configuration fails immediately
        # after its 2-second timeout, instead of running hundreds more timeouts.
        self.assertFalse(result["errors"], detail)
        self.assertTrue(result["passed"], detail)
        self.assertTrue(
            all(x == result["expected"] for x in result["launched"]), detail
        )
        proxies = len(result["expected"]) * (pp - 1)
        outputs = pp * sum(
            not skip or pp == 2 or i % 2 != 0 for i in result["expected"]
        )
        for kind, expected in [("proxy", proxies), ("output", outputs)]:
            for action in ("send", "recv"):
                self.assertEqual(
                    result["counts"].get(action + "_" + kind, 0), expected, detail
                )

    def test_first_send_rendezvous_with_real_output_ring(self):
        for pp in (2, 4, 8):
            for depth in (0, 1, 2):
                for workload in (
                    "cold",
                    "idle_active",
                    "continuous",
                    "sparse",
                    "idle_resume",
                ):
                    for early in (False, True):
                        for delay_ms in (0, 15):
                            for skip in (False, True):
                                self.assert_case(
                                    early, pp, depth, workload, delay_ms, skip
                                )

    def test_every_device_send_may_wait_for_peer(self):
        # Stronger than cold communicator setup: bounded transport resources can
        # also make later sends wait. Exercise both output payloads and skips.
        for pp in (2, 4, 8):
            for depth in (0, 1, 2):
                for early in (False, True):
                    for skip in (False, True):
                        self.assert_case(
                            early,
                            pp,
                            depth,
                            "continuous",
                            15,
                            skip,
                            always_rendezvous=True,
                        )


if __name__ == "__main__":
    unittest.main()
