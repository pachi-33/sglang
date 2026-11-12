"""CPU contract tests: actual wrappers over a small scheduler/stream double."""

import copy
from types import SimpleNamespace
import unittest

from pp_chunk_trace import patch


class FakeRecorder:
    def __init__(self, device_module, device, metadata, output, run_id,
                 clock_address, capacity):
        self.dm = device_module
        self.metadata = metadata
        self.armed = True
        self.stop_requested = False
        self.closed = False
        self.failure = None
        self.begins = []
        self.ends = []
        self.finished = False
        self.instants = []

    def begin(self, fields, stream):
        if not self.armed or self.stop_requested or self.closed or self.failure:
            return None
        fields = copy.deepcopy(fields)
        self.begins.append(fields)
        self.dm.trace.append("trace_start")
        return fields

    def end(self, token, failure=None):
        if token is not None:
            self.dm.trace.append("trace_end")
            self.ends.append((token, failure))

    def instant(self, event, rids):
        self.instants.append((event, list(rids)))

    def request_finish(self):
        self.finished = True


class FakeMixin:
    def init_pp_loop_state(self):
        self.initialized += 1
        return "initialized"

    def _pp_recv_proxy_tensors(self):
        if self.pp_group.is_first_rank:
            return None
        return self.incoming.pop(0)

    def _pp_wait_forward_dependencies(self):
        self.device_module.trace.append("dependencies")

    def _pp_launch_batch(self, mb_id, cur_batch, pp_proxy_tensors,
                         mb_metadata, last_rank_comm_queue):
        self._pp_wait_forward_dependencies()
        result = self.run_batch(cur_batch, pp_proxy_tensors)
        self.device_module.trace.append("existing_end_event")
        return result, "existing_event"

    def run_batch(self, batch, pp_proxy_tensors=None):
        self.device_module.trace.append("model")
        self.model_inputs.append(pp_proxy_tensors)
        if self.model_error is not None:
            raise self.model_error
        if self.mutate_batch:
            batch.prefix_lens[:] = [999] * len(batch.reqs)
            batch.extend_lens[:] = [1] * len(batch.reqs)
        return SimpleNamespace(
            pp_hidden_states_proxy_tensors=SimpleNamespace(tensors={"hidden": object()}),
            can_run_cuda_graph=self.graph,
        )

    def _pp_send_proxy_to_next_stage(self, result):
        # Real send_tensor_dict consumes the dictionary at submission time.
        # Also emulate the normal __msg_type__ mutation, which must remain local.
        tensor_dict = result.pp_hidden_states_proxy_tensors.tensors
        tensor_dict["__msg_type__"] = "proxy"
        self.sent.append(dict(tensor_dict))
        if self.send_error is not None:
            raise self.send_error
        return "submitted"


def batch(*chunks, mode="EXTEND"):
    return SimpleNamespace(
        reqs=[SimpleNamespace(rid=c[0]) for c in chunks],
        prefix_lens=[c[1] for c in chunks],
        extend_lens=[c[2] for c in chunks],
        forward_mode=SimpleNamespace(name=mode),
    )


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.modules = []

    def tearDown(self):
        for module in reversed(self.modules):
            patch.uninstall(module)

    def scheduler(self, pp_rank=0, dp_rank=0, attn_dp_rank=0,
                  attn_dp_size=1, attn_tp_size=1, cp_size=4, completion_hook=False):
        # Distinct classes model distinct worker processes without importing
        # real torch or mutating production classes in this Python process.
        cls = type(f"SchedulerPP{pp_rank}_{len(self.modules)}", (FakeMixin,), {})
        if completion_hook:
            cls._pp_prefill_completion_event = lambda self, event, reqs: "original-result"
        parallel = SimpleNamespace(
            pp_rank=pp_rank, pp_size=2, tp_rank=0, dp_rank=dp_rank,
            attn_dp_rank=attn_dp_rank, attn_dp_size=attn_dp_size,
            attn_tp_rank=0, attn_tp_size=attn_tp_size, attn_cp_rank=0,
            attn_cp_size=cp_size,
        )
        module = SimpleNamespace(
            Scheduler=cls,
            get_parallel=lambda: parallel,
            get_device=lambda: SimpleNamespace(gpu_id=0),
            torch=SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda: pp_rank * 4)),
            __file__="/fake/sglang/srt/managers/scheduler.py",
        )
        patch._patch_class(cls, module, "/unused/test", "run-a", recorder_factory=FakeRecorder)
        self.modules.append(module)
        scheduler = cls()
        scheduler.initialized = 0
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.device = "npu:0"
        scheduler.device_module = SimpleNamespace(trace=[], current_stream=lambda: "forward")
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        scheduler.pp_group = SimpleNamespace(is_first_rank=pp_rank == 0)
        scheduler.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(
            layer_info=SimpleNamespace(start_layer=pp_rank * 20, end_layer=(pp_rank + 1) * 20)))
        scheduler.incoming, scheduler.sent, scheduler.model_inputs = [], [], []
        scheduler.model_error = scheduler.send_error = None
        scheduler.mutate_batch = scheduler.graph = False
        scheduler.init_pp_loop_state()
        return scheduler, module

    def launch(self, scheduler, inputs, mb_id=0):
        proxy = scheduler._pp_recv_proxy_tensors()
        result, event = scheduler._pp_launch_batch(mb_id, inputs, proxy, [], [])
        self.assertEqual(event, "existing_event")
        return result

    def transfer(self, source, destination, result):
        original = result.pp_hidden_states_proxy_tensors.tensors
        source._pp_send_proxy_to_next_stage(result)
        self.assertIs(result.pp_hidden_states_proxy_tensors.tensors, original)
        self.assertNotIn(patch.WIRE_KEY, original)
        self.assertNotIn("__msg_type__", original)
        destination.incoming.append(SimpleNamespace(tensors=dict(source.sent[-1])))

    def test_optional_completion_hook_filters_intermediate_chunks_and_uninstalls(self):
        s, module = self.scheduler(completion_hook=True)
        active = SimpleNamespace(rid="finished")
        intermediate = SimpleNamespace(rid="middle")
        s.disagg_prefill_inflight_queue = [active]
        answer = s._pp_prefill_completion_event("result_processed", [active, intermediate])
        self.assertEqual(answer, "original-result")
        s._pp_prefill_completion_event("release_ready", [active])
        s._pp_prefill_completion_event("response_enqueued", [active])
        self.assertEqual(s._pp_trace_recorder.instants, [
            ("result_processed", ["finished"]), ("release_ready", ["finished"]),
            ("response_enqueued", ["finished"])])
        patch.uninstall(module)
        self.assertEqual(s._pp_prefill_completion_event("release_ready", [active]), "original-result")
        self.assertEqual(len(s._pp_trace_recorder.instants), 3)

    def test_old_scheduler_has_no_completion_hook_added(self):
        s, module = self.scheduler()
        self.assertFalse(hasattr(s, "_pp_prefill_completion_event"))
        self.launch(s, batch(("r", 0, 128)))
        self.assertEqual(len(s._pp_trace_recorder.begins), 1)

    def test_completion_hook_skips_inactive_capture_without_reading_queue(self):
        for attr, value in (("armed", False), ("stop_requested", True),
                            ("closed", True), ("failure", "broken")):
            s, _ = self.scheduler(completion_hook=True)
            setattr(s._pp_trace_recorder, attr, value)
            del s.disagg_prefill_inflight_queue
            self.assertEqual(s._pp_prefill_completion_event("result_processed", None), "original-result")
            self.assertEqual(s._pp_trace_recorder.instants, [])

    def test_completion_record_failure_preserves_original_result(self):
        s, _ = self.scheduler(completion_hook=True)
        def fail(*args):
            raise RuntimeError("queue broken")
        s._pp_trace_recorder.instant = fail
        with self.assertLogs(patch.logger, level="ERROR"):
            self.assertEqual(s._pp_prefill_completion_event("release_ready", []), "original-result")
        self.assertIn("queue broken", s._pp_trace_recorder.failure)

    def test_events_follow_input_dependency_and_wrap_model_only(self):
        s, _ = self.scheduler()
        self.launch(s, batch(("r", 0, 16384)))
        self.assertEqual(s.device_module.trace, [
            "dependencies", "trace_start", "model", "trace_end", "existing_end_event"])
        self.assertEqual(s._pp_trace_recorder.begins[0]["chunks"][0]["end"], 16384)
        self.assertIsNone(s._pp_trace_context)

    def test_two_stage_multi_request_identity_and_header_stripping(self):
        upstream, _ = self.scheduler()
        downstream, _ = self.scheduler(pp_rank=1)
        chunks = (("r1", 0, 16384), ("r2", 400, 123))
        result = self.launch(upstream, batch(*chunks))
        self.transfer(upstream, downstream, result)
        received = self.launch(downstream, batch(*chunks))
        self.assertEqual(result._pp_chunk_trace_wire, received._pp_chunk_trace_wire)
        self.assertNotIn(patch.WIRE_KEY, downstream.model_inputs[-1].tensors)
        self.assertEqual(
            downstream._pp_trace_recorder.begins[0]["chunks"],
            upstream._pp_trace_recorder.begins[0]["chunks"],
        )

    def test_reused_microbatch_slot_and_result_local_wire(self):
        s, _ = self.scheduler()
        first = self.launch(s, batch(("r", 0, 4096)), mb_id=0)
        second = self.launch(s, batch(("r", 4096, 257)), mb_id=0)
        s._pp_send_proxy_to_next_stage(first)
        s._pp_send_proxy_to_next_stage(second)
        a, b = [record[patch.WIRE_KEY] for record in s.sent]
        self.assertNotEqual(a["batch_id"], b["batch_id"])
        self.assertEqual([a["chunks"][0]["ordinal"], b["chunks"][0]["ordinal"]], [1, 2])
        self.assertEqual([x["local_forward_seq"] for x in s._pp_trace_recorder.begins], [1, 2])
        self.assertEqual(b["chunks"][0]["end"] - b["chunks"][0]["start"], 257)

    def test_retry_preserves_ordinal_and_changes_attempt(self):
        s, _ = self.scheduler()
        a = self.launch(s, batch(("r", 0, 4096)))._pp_chunk_trace_wire
        b = self.launch(s, batch(("r", 0, 4096)))._pp_chunk_trace_wire
        self.assertNotEqual(a["batch_id"], b["batch_id"])
        self.assertEqual(a["chunks"][0]["ordinal"], b["chunks"][0]["ordinal"])
        self.assertEqual([a["chunks"][0]["attempt"], b["chunks"][0]["attempt"]], [1, 2])

    def test_cpu_chunk_lists_are_frozen_before_forward_mutation(self):
        s, _ = self.scheduler()
        s.mutate_batch = True
        result = self.launch(s, batch(("r", 2048, 64)))
        self.assertEqual(result._pp_chunk_trace_wire["chunks"][0]["start"], 2048)
        self.assertEqual(s._pp_trace_recorder.begins[0]["chunks"][0]["end"], 2112)

    def test_disabled_capture_propagates_disabled_wire(self):
        upstream, _ = self.scheduler()
        downstream, _ = self.scheduler(pp_rank=1)
        upstream._pp_trace_recorder.armed = False
        result = self.launch(upstream, batch(("r", 0, 128)))
        self.assertFalse(result._pp_chunk_trace_wire["enabled"])
        self.transfer(upstream, downstream, result)
        self.launch(downstream, batch(("r", 0, 128)))
        self.assertEqual(upstream._pp_trace_recorder.begins, [])
        self.assertEqual(downstream._pp_trace_recorder.begins, [])

    def test_stop_requested_disables_new_chunk_capture(self):
        s, _ = self.scheduler()
        s._pp_trace_recorder.stop_requested = True
        result = self.launch(s, batch(("r", 0, 128)))
        self.assertFalse(result._pp_chunk_trace_wire["enabled"])
        self.assertEqual(s._pp_trace_recorder.begins, [])

    def test_closed_or_failed_collector_disables_new_wire_capture(self):
        for attr, value in (("closed", True), ("failure", "collector failed")):
            with self.subTest(attr=attr):
                s, _ = self.scheduler()
                setattr(s._pp_trace_recorder, attr, value)
                result = self.launch(s, batch(("r", 0, 128)))
                self.assertFalse(result._pp_chunk_trace_wire["enabled"])
                self.assertEqual(s._pp_trace_recorder.begins, [])
                self.assertIn("model", s.device_module.trace)

    def test_inactive_identity_does_not_accumulate_request_state(self):
        identity = patch.Identity("run-a", 0)
        for number in range(500):
            wire = identity.create([dict(rid=f"warmup-{number}", start=0, end=128)], False)
            self.assertFalse(wire["enabled"])
        self.assertEqual(identity.ordinals, {})
        self.assertEqual(dict(identity.attempts), {})
        self.assertEqual(dict(identity.next_ordinal), {})
        wire = identity.create([dict(rid="measured", start=0, end=128)], True)
        self.assertEqual((wire["chunks"][0]["ordinal"], wire["chunks"][0]["attempt"]), (1, 1))
        saved = (dict(identity.ordinals), dict(identity.attempts), dict(identity.next_ordinal))
        identity.create([dict(rid="after-capture", start=0, end=128)], False)
        self.assertEqual(saved, (dict(identity.ordinals), dict(identity.attempts),
                                 dict(identity.next_ordinal)))

    def test_decode_mixed_split_idle_and_direct_calls_not_captured(self):
        s, _ = self.scheduler()
        for mode in ("DECODE", "MIXED", "SPLIT_PREFILL", "TARGET_VERIFY", "IDLE"):
            result = self.launch(s, batch(("r", 0, 128), mode=mode))
            self.assertIsNone(result._pp_chunk_trace_wire)
        s.run_batch(batch(("r", 0, 128)))
        self.assertEqual(s._pp_trace_recorder.begins, [])

    def test_missing_header_is_explicit_and_does_not_run_model(self):
        s, _ = self.scheduler(pp_rank=1)
        s.incoming.append(SimpleNamespace(tensors={"hidden": object()}))
        with self.assertRaisesRegex(ValueError, "Missing/mismatched"):
            self.launch(s, batch(("r", 0, 128)))
        self.assertEqual(s.device_module.trace, [])

    def test_mismatched_semantics_and_run_id_are_rejected(self):
        upstream, _ = self.scheduler()
        for variant in ("run", "range"):
            with self.subTest(variant=variant):
                downstream, _ = self.scheduler(pp_rank=1)
                result = self.launch(upstream, batch(("r", 0, 128)))
                self.transfer(upstream, downstream, result)
                if variant == "run":
                    downstream.incoming[0].tensors[patch.WIRE_KEY] = dict(
                        result._pp_chunk_trace_wire, run_id="other")
                    inputs = batch(("r", 0, 128))
                else:
                    inputs = batch(("r", 128, 128))
                with self.assertRaises(ValueError):
                    self.launch(downstream, inputs)
                self.assertEqual(downstream.device_module.trace, [])

    def test_original_exception_is_rethrown_and_context_cleared(self):
        s, _ = self.scheduler()
        error = ValueError("model failed")
        s.model_error = error
        with self.assertRaises(ValueError) as caught:
            self.launch(s, batch(("r", 0, 128)))
        self.assertIs(caught.exception, error)
        self.assertEqual(s._pp_trace_recorder.ends[-1][1], "ValueError: model failed")
        self.assertIsNone(s._pp_trace_context)

    def test_collector_start_exception_preserves_model_result(self):
        s, _ = self.scheduler()

        def broken_begin(*args):
            raise RuntimeError("collector begin failed")

        s._pp_trace_recorder.begin = broken_begin
        with self.assertLogs(patch.logger, level="ERROR"):
            result = self.launch(s, batch(("r", 0, 128)))
        self.assertFalse(result.can_run_cuda_graph)
        self.assertIn("model", s.device_module.trace)
        self.assertIn("collector begin failed", s._pp_trace_recorder.failure)
        self.assertIsNone(s._pp_trace_context)

    def test_collector_end_exception_preserves_model_result_or_exception(self):
        for model_fails in (False, True):
            with self.subTest(model_fails=model_fails):
                s, _ = self.scheduler()

                def broken_end(*args):
                    raise RuntimeError("collector end failed")

                s._pp_trace_recorder.end = broken_end
                model_error = ValueError("original model failure")
                if model_fails:
                    s.model_error = model_error
                with self.assertLogs(patch.logger, level="ERROR"):
                    if model_fails:
                        with self.assertRaises(ValueError) as caught:
                            self.launch(s, batch(("r", 0, 128)))
                        self.assertIs(caught.exception, model_error)
                    else:
                        result = self.launch(s, batch(("r", 0, 128)))
                        self.assertFalse(result.can_run_cuda_graph)
                self.assertIn("collector end failed", s._pp_trace_recorder.failure)
                self.assertIsNone(s._pp_trace_context)

    def test_send_exception_preserves_original_proxy(self):
        s, _ = self.scheduler()
        result = self.launch(s, batch(("r", 0, 128)))
        original = result.pp_hidden_states_proxy_tensors.tensors
        error = RuntimeError("send failed")
        s.send_error = error
        with self.assertRaises(RuntimeError) as caught:
            s._pp_send_proxy_to_next_stage(result)
        self.assertIs(caught.exception, error)
        self.assertIs(result.pp_hidden_states_proxy_tensors.tensors, original)
        self.assertNotIn(patch.WIRE_KEY, original)

    def test_existing_wire_collision_rejected_without_mutating_input(self):
        s, _ = self.scheduler()
        result = self.launch(s, batch(("r", 0, 128)))
        original = result.pp_hidden_states_proxy_tensors.tensors
        original[patch.WIRE_KEY] = "owned elsewhere"
        with self.assertRaisesRegex(RuntimeError, "collision"):
            s._pp_send_proxy_to_next_stage(result)
        self.assertIs(result.pp_hidden_states_proxy_tensors.tensors, original)

    def test_graph_replay_marked_as_error(self):
        s, _ = self.scheduler()
        s.graph = True
        self.launch(s, batch(("r", 0, 128)))
        self.assertIn("graph replay unsupported", s._pp_trace_recorder.ends[0][1])

    def test_metadata_counts_only_ranks_in_attention_dp_lane(self):
        s, _ = self.scheduler(dp_rank=1, attn_dp_rank=2, attn_dp_size=4,
                              attn_tp_size=2, cp_size=4)
        self.assertEqual(s._pp_trace_recorder.metadata["dp_rank"], 6)
        self.assertEqual(s._pp_trace_recorder.metadata["expected_stage_ranks"], 8)

    def test_uninstall_restores_inherited_methods(self):
        s, module = self.scheduler()
        self.assertIn("run_batch", module.Scheduler.__dict__)
        patch.uninstall(module)
        self.assertNotIn("run_batch", module.Scheduler.__dict__)
        self.assertIs(module.Scheduler.run_batch, FakeMixin.run_batch)
        self.assertNotIn(module.Scheduler, patch._PATCHED)
        self.assertTrue(s._pp_trace_recorder.finished)
        s.run_batch(batch(("r", 0, 128)))
        self.assertEqual(s._pp_trace_recorder.begins, [])

    def test_uninstall_conflict_is_atomic_and_keeps_state(self):
        s, module = self.scheduler()
        state = patch._PATCHED[module.Scheduler]
        wrappers = dict(state[2])
        foreign = lambda self, result: None
        # Conflict at the last method catches implementations that start
        # restoring earlier methods before validating the entire patch set.
        module.Scheduler._pp_send_proxy_to_next_stage = foreign
        try:
            with self.assertRaisesRegex(RuntimeError, "Another patch"):
                patch.uninstall(module)
            self.assertIs(patch._PATCHED[module.Scheduler], state)
            self.assertFalse(s._pp_trace_recorder.finished)
            for name, wrapper in wrappers.items():
                expected = foreign if name == "_pp_send_proxy_to_next_stage" else wrapper
                self.assertIs(getattr(module.Scheduler, name), expected)
        finally:
            module.Scheduler._pp_send_proxy_to_next_stage = wrappers["_pp_send_proxy_to_next_stage"]

    def test_invalid_cpu_metadata_never_uses_tensor_readback(self):
        s, _ = self.scheduler()
        inputs = batch(("r", 0, 128))
        inputs.prefix_lens = object()
        with self.assertRaisesRegex(ValueError, "CPU"):
            self.launch(s, inputs)
        self.assertEqual(s.device_module.trace, [])


if __name__ == "__main__":
    unittest.main()
