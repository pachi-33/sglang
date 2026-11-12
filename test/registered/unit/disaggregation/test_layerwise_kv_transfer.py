"""Layerwise wire-buffer coverage and scheduler/transfer lifetime regressions."""

import threading
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.common.layerwise import LayerwiseTransferWorker
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    disagg_transfers_in_flight,
)
from sglang.srt.model_executor.forward_batch_info import _prepare_layerwise_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _ComputeEvent:
    def __init__(self):
        self.waiting = threading.Event()
        self.ready = threading.Event()

    def synchronize(self):
        self.waiting.set()
        if not self.ready.wait(5):
            raise TimeoutError("Test did not complete the attention write")


class TestLayerwiseKVTransfer(CustomTestCase):
    def setUp(self):
        super().setUp()
        topology = patch(
            "sglang.srt.disaggregation.ascend.conn.get_parallel",
            return_value=SimpleNamespace(pp_group=SimpleNamespace(is_last_rank=False)),
        )
        topology.start()
        self.addCleanup(topology.stop)

    def make_manager(self, *, start=0, end=2, groups=3, decode_layers=2):
        mgr = object.__new__(AscendKVManager)
        layers = end - start
        mgr.pp_size = 1 if (start, end) == (0, decode_layers) else 4
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=[100_000 * (i + 1) for i in range(groups * layers)],
            kv_item_lens=[16 * (g + 1) for g in range(groups) for _ in range(layers)],
            kv_buf_groups=groups,
            prefill_start_layer=start,
            prefill_end_layer=end,
            hidden_kv_layers=decode_layers,
            draft_kv_layers=0,
            num_draft_entries=0,
        )
        mgr.peer = SimpleNamespace(
            dst_kv_ptrs=[1_000_000 * (i + 1) for i in range(groups * decode_layers)]
        )
        mgr.decode_kv_args_table = {"decode": mgr.peer}
        mgr.transfer_infos = {}
        mgr.request_status = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.deferred_bootstrap = None
        mgr.enable_deferred_decode_kv_release = True
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._staging_outstanding = {}
        mgr.notifications, mgr.acks, mgr.writes = [], [], []
        # Only network IO is substituted; failure and abort bookkeeping is real.
        mgr.send_kv_status_message = lambda **kw: mgr.notifications.append(kw)
        mgr._send_abort_ack = lambda room, target: mgr.acks.append((room, target))
        mgr._transfer_data = lambda session, blocks: mgr.writes.extend(blocks) or 0
        # No device allocation is needed: tests supply the attention event.
        mgr._layerwise_initialized = True
        mgr._layerwise_worker = LayerwiseTransferWorker(
            transfer=lambda session, blocks: mgr._transfer_data(session, blocks),
            is_active=mgr._layerwise_room_active,
            on_failure=mgr._layerwise_transfer_failed,
            on_drained=mgr._maybe_ack_drained_abort,
            initializer=lambda: None,
        )
        self.addCleanup(mgr.shutdown_layerwise)
        return mgr

    def make_sender(self, mgr, room=1, event=None):
        mgr.request_status[room] = KVPoll.WaitingForInput
        mgr.transfer_infos[room] = {
            "decode": SimpleNamespace(
                is_dummy=False,
                mooncake_session_id="decode",
                endpoint="decode",
                dst_port=1,
            )
        }
        sender = object.__new__(CommonKVSender)
        sender.kv_mgr, sender.bootstrap_room = mgr, room
        sender._layerwise_enabled = True
        sender._layerwise_futures = []
        sender._layerwise_send_waited = True
        sender.conclude_state = None
        sender.trace_ctx = SimpleNamespace(trace_req_finish=lambda: None)
        sender._record_compute_event = lambda: event
        sender.set_layerwise_indices(np.array([2, 3]), np.array([5, 6]))
        sender.start_layerwise_send(
            mgr.kv_args.prefill_end_layer - mgr.kv_args.prefill_start_layer
        )
        return sender

    def test_every_ascend_buffer_reaches_its_decode_layer(self):
        """All buffer groups must arrive, including PP-local -> PP1 and scales."""
        for groups in (2, 3, 4):
            for start, end, decode_layers in ((0, 2, 2), (2, 5, 8), (2, 5, 3)):
                with self.subTest(
                    groups=groups, start=start, decode_layers=decode_layers
                ):
                    mgr = self.make_manager(
                        start=start, end=end, groups=groups, decode_layers=decode_layers
                    )
                    if groups == 4:
                        # Explicit registrations can include an extra index
                        # scale buffer; repeated layer ids pair each component.
                        mgr.kv_args.kv_layer_ids = list(range(start, end)) * groups
                        dst_ids = (
                            range(start, end)
                            if decode_layers == end - start
                            else range(decode_layers)
                        )
                        mgr.peer.dst_kv_layer_ids = list(dst_ids) * groups
                    # Equal-sized peers have the same PP-local layout.
                    sender = self.make_sender(mgr)
                    for layer in range(start, end):
                        sender.save_kv_layer(layer)
                    sender.wait_layerwise_send_done()
                    expected = []
                    for group in range(groups):
                        for local_layer in range(end - start):
                            src_index = group * (end - start) + local_layer
                            dst_layer = (
                                local_layer
                                if decode_layers == end - start
                                else start + local_layer
                            )
                            dst_index = group * decode_layers + dst_layer
                            page_bytes = 16 * (group + 1)
                            expected.append(
                                (
                                    mgr.kv_args.kv_data_ptrs[src_index]
                                    + 2 * page_bytes,
                                    mgr.peer.dst_kv_ptrs[dst_index] + 5 * page_bytes,
                                    2 * page_bytes,
                                )
                            )
                    self.assertCountEqual(mgr.writes, expected)

    def make_indexed_request(
        self, sender, page_size, input_pages, decode_prefix_pages=0
    ):
        token_count = input_pages * page_size
        req = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=0),
            origin_input_ids=[0] * token_count,
            prefix_len=0,
            extend_end=0,
            start_send_idx=decode_prefix_pages * page_size,
            disagg_decode_prefix_len=decode_prefix_pages * page_size,
            disagg_kv_sender=sender,
        )
        # Unallocated positions deliberately contain invalid indices, so reading
        # future chunks would produce visibly incorrect wire addresses.
        runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.full((1, token_count), -100, dtype=torch.int64)
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=page_size,
                translate_kv_indices_for_transfer=lambda indices: (
                    indices + 3 * page_size
                ),
            ),
        )
        sender.kv_mgr.transfer_infos[sender.bootstrap_room][
            "decode"
        ].dst_kv_indices = np.arange(
            50, 50 + input_pages - decode_prefix_pages, dtype=np.int32
        )
        return req, runner

    def test_chunked_forward_sends_only_allocated_pages(self):
        """An unfinished prompt must never send indices from its future chunks."""
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                mgr = self.make_manager()
                sender = self.make_sender(mgr)
                req, runner = self.make_indexed_request(sender, page_size, 10)
                for end_page in (4, 8, 10):
                    req.prefix_len = req.extend_end
                    req.extend_end = end_page * page_size
                    runner.req_to_token_pool.req_to_token[0, : req.extend_end] = (
                        torch.arange(2 * page_size, (2 + end_page) * page_size)
                    )
                    _prepare_layerwise_indices(
                        None, SimpleNamespace(reqs=[req]), runner
                    )
                    sender.start_layerwise_send(2)
                    mgr.writes.clear()
                    for layer in range(2):
                        sender.save_kv_layer(layer)
                    sender.wait_layerwise_send_done()
                    expected = [
                        (
                            src + 5 * item_len,
                            dst + 50 * item_len,
                            end_page * item_len,
                        )
                        for src, dst, item_len in zip(
                            mgr.kv_args.kv_data_ptrs,
                            mgr.peer.dst_kv_ptrs,
                            mgr.kv_args.kv_item_lens,
                        )
                    ]
                    self.assertCountEqual(mgr.writes, expected)

    def test_layerwise_destination_is_relative_to_decode_prefix(self):
        """Cached decode pages must not shift or truncate the suffix transfer."""
        for start_page in (4, 8):
            with self.subTest(start_page=start_page):
                mgr = self.make_manager()
                sender = self.make_sender(mgr)
                req, runner = self.make_indexed_request(sender, 4, 12, 4)
                # Prefill has a longer cache hit than decode. Its cached KV
                # before prefix_len must still be sent if start_send_idx < it.
                req.prefix_len = 8 * 4
                req.start_send_idx = start_page * 4
                req.extend_end = len(req.origin_input_ids) + 1
                runner.req_to_token_pool.req_to_token[0] = torch.arange(2 * 4, 14 * 4)
                _prepare_layerwise_indices(None, SimpleNamespace(reqs=[req]), runner)
                sender.start_layerwise_send(2)
                for layer in range(2):
                    sender.save_kv_layer(layer)
                sender.wait_layerwise_send_done()
                expected = [
                    (
                        src + (5 + start_page) * item_len,
                        dst + (50 + start_page - 4) * item_len,
                        (12 - start_page) * item_len,
                    )
                    for src, dst, item_len in zip(
                        mgr.kv_args.kv_data_ptrs,
                        mgr.peer.dst_kv_ptrs,
                        mgr.kv_args.kv_item_lens,
                    )
                ]
                self.assertCountEqual(mgr.writes, expected)

    def test_forward_returns_before_compute_and_transport_finish(self):
        """A blocked attention event or transport must not hold up forward."""
        mgr = self.make_manager()
        event = _ComputeEvent()
        transport_started, release_transport = threading.Event(), threading.Event()
        self.addCleanup(event.ready.set)
        self.addCleanup(release_transport.set)

        def transfer(session, blocks):
            transport_started.set()
            if not release_transport.wait(5):
                raise TimeoutError("Test did not release transport")
            mgr.writes.extend(blocks)
            return 0

        mgr._transfer_data = transfer
        sender = self.make_sender(mgr, event=event)
        returned, retired = threading.Event(), threading.Event()

        def forward():
            sender.save_kv_layer(0)
            returned.set()

        caller = threading.Thread(target=forward)
        caller.start()
        waiter = None
        try:
            self.assertTrue(returned.wait(1), "forward blocked on KV transfer")
            self.assertTrue(event.waiting.wait(1))
            self.assertFalse(transport_started.is_set())
            event.ready.set()
            self.assertTrue(transport_started.wait(1))
            waiter = threading.Thread(
                target=lambda: (sender.wait_layerwise_send_done(), retired.set())
            )
            waiter.start()
            self.assertFalse(
                retired.wait(0.05), "KV was released while a write was running"
            )
            release_transport.set()
            waiter.join(2)
            self.assertTrue(retired.is_set())
            self.assertEqual(len(mgr.writes), 3)
        finally:
            event.ready.set()
            release_transport.set()
            caller.join(2)
            if waiter is not None:
                waiter.join(2)

    def test_later_microbatch_does_not_delay_earlier_request(self):
        """Completing B must not overwrite the completion being awaited for A."""
        mgr = self.make_manager()
        first = self.make_sender(mgr, room=1)
        later_event = _ComputeEvent()
        self.addCleanup(later_event.ready.set)
        later = self.make_sender(mgr, room=2, event=later_event)
        first.save_kv_layer(0)
        later.save_kv_layer(0)
        self.assertTrue(later_event.waiting.wait(1))
        done = threading.Event()
        thread = threading.Thread(
            target=lambda: (first.wait_layerwise_send_done(), done.set())
        )
        thread.start()
        try:
            self.assertTrue(done.wait(1))
            self.assertTrue(mgr.has_pending_layerwise(2))
        finally:
            later_event.ready.set()
            thread.join(2)
        later.wait_layerwise_send_done()

    def test_new_chunk_retains_previous_pending_writes(self):
        """Reusing a request for a new chunk cannot discard its old writes."""
        mgr = self.make_manager()
        sender = self.make_sender(mgr)
        pending = Future()
        sender._layerwise_futures.append(pending)
        sender.start_layerwise_send(2)
        done = threading.Event()
        thread = threading.Thread(
            target=lambda: (sender.wait_layerwise_send_done(), done.set())
        )
        thread.start()
        try:
            self.assertFalse(done.wait(0.05))
            pending.set_result(None)
            self.assertTrue(done.wait(1))
        finally:
            if not pending.done():
                pending.set_result(None)
            thread.join(2)
        sender.wait_layerwise_send_done()

    def test_abort_ack_waits_for_the_running_write(self):
        """Decode pages are not reusable while a background write targets them."""
        mgr = self.make_manager()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def transfer(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release transport")
            return 0

        mgr._transfer_data = transfer
        sender = self.make_sender(mgr)
        sender.save_kv_layer(0)
        self.assertTrue(entered.wait(1))
        mgr.update_status(1, KVPoll.Failed)
        mgr._deferred_ack_targets[1] = {("decode", 1): "generation-1"}
        mgr._maybe_ack_drained_abort(1)
        self.assertEqual(mgr.acks, [])
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        release.set()
        sender.wait_layerwise_send_done()
        self.assertEqual(mgr.acks, [(1, "generation-1")])
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_failed_write_cannot_complete_successfully(self):
        """A synchronous engine error must survive the background handoff."""
        mgr = self.make_manager()
        mgr._transfer_data = lambda *args: -1
        sender = self.make_sender(mgr)
        sender.save_kv_layer(0)
        sender.wait_layerwise_send_done()
        self.assertEqual(mgr.request_status[1], KVPoll.Failed)
        self.assertIn("returned -1", mgr.failure_records[1])
        mgr.conclude_transfer(bootstrap_room=1, status=KVPoll.Success)
        self.assertEqual(mgr.request_status[1], KVPoll.Failed)
        self.assertIn(1, mgr._deferred_ack_poisoned_rooms)

    def test_compaction_is_blocked_before_request_enters_transfer_queue(self):
        """Pending forward-time writes pin source pages outside scheduler queues."""
        mgr = self.make_manager()
        event = _ComputeEvent()
        self.addCleanup(event.ready.set)
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=mgr),
            disagg_prefill_inflight_queue=[],
            disagg_prefill_pending_chunk_rids=set(),
        )
        busy = disagg_transfers_in_flight(scheduler)
        self.assertFalse(busy())
        sender = self.make_sender(mgr, event=event)
        sender.save_kv_layer(0)
        self.assertTrue(event.waiting.wait(1))
        self.assertTrue(busy())
        event.ready.set()
        sender.wait_layerwise_send_done()
        self.assertFalse(busy())

    def test_shutdown_drains_writes_before_transport_can_be_destroyed(self):
        """Role switching cannot deregister buffers while the worker uses them."""
        mgr = self.make_manager()
        event = _ComputeEvent()
        self.addCleanup(event.ready.set)
        sender = self.make_sender(mgr, event=event)
        sender.save_kv_layer(0)
        self.assertTrue(event.waiting.wait(1))
        closed = threading.Event()
        thread = threading.Thread(
            target=lambda: (mgr.shutdown_layerwise(), closed.set())
        )
        thread.start()
        try:
            self.assertFalse(closed.wait(0.05))
            event.ready.set()
            self.assertTrue(closed.wait(1))
            self.assertEqual(len(mgr.writes), 3)
        finally:
            event.ready.set()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
