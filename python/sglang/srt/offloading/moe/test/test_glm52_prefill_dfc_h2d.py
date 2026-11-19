r"""Single-node EP16 GLM-5.2 routed-layer DFC under next-layer H2D.

From the checkout (in the Ascend deployment environment):
    torchrun --standalone --nproc-per-node=16 \
        python/sglang/srt/offloading/moe/test/test_glm52_prefill_dfc_h2d.py \
        --native-extension-dir /path/to/native_acl --output /tmp/glm52-dfc \
        --tokens-per-rank 32 128 512 --ep-load balanced 3to1 7to1

Add --profile to capture one baseline and one overlap trace per selected case
and rank. For example, --tokens-per-rank 128 --ep-load 3to1 --profile
selects only that workload. Timing and bank/output checks run in both modes.

The fixture models routed layer 3 with W8A8/NZ and an EP16 group. Bank 0
holds its cold experts. Bank 1 receives layer 4; the complete type-2 pool
has 470 slots per rank (about 16.6 GiB). Router outputs are synthetic.
Token counts are input rows on each of the 16 source ranks, before top-8
dispatch. EP load describes the routed rows received by ranks 0-7 vs 8-15:
balanced assigns equal traffic, 3to1 sends 75% to the first half, and 7to1
sends 87.5%. Real per-expert receive counts are checked against the router.
Baseline and overlap use identical inputs. Baseline has no weight prefetch.
When profiling is enabled, any uncovered DFC time on any rank fails the test
after all cases are reported. Event timing alone gives the copy-stream window.
CPU-only checks: python test_glm52_prefill_dfc_h2d.py --self-test
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import unittest
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from decimal import Decimal
from itertools import product
from pathlib import Path
from statistics import median
from types import SimpleNamespace

EP = 16
EXPERTS = 256
LOCAL = EXPERTS // EP
TOP_K = 8
HIDDEN = 6144
INTERMEDIATE = 2048
ROUTED_LAYERS = 78 - 3
RESIDENT_RATIO = 0.4
MODES = ("baseline", "overlap")
BACKENDS = ("deepep", "ascend_ops")
TOKEN_SIZES = (32, 128, 512)
# Number of this token's eight routes sent to EP ranks 0 through 7.
HOT_ROUTES = {"balanced": 4, "3to1": 6, "7to1": 7}


def route_ids(row, lane, source_rank, ep_load="balanced"):
    """Works on integer coordinates or broadcastable CPU tensors."""
    if ep_load == "balanced":
        destination = (source_rank + row + 2 * lane) % EP
    else:
        half = EP // 2
        # Rotate within each half so the eight destinations are distinct and
        # the requested hot:cold receive ratio holds even for one input row.
        within_half = (source_rank + row + lane) % half
        destination = within_half + (lane >= HOT_ROUTES[ep_load]) * half
    local_expert = (source_rank + 2 * row + 4 * lane) % LOCAL
    return destination * LOCAL + local_expert


def covered_ns(start, end, intervals):
    """Length of the union intersected with [start, end], without filling gaps."""
    if end <= start:
        raise ValueError("Empty DFC device interval")
    cursor, covered = start, 0
    for lo, hi in sorted(intervals):
        lo, hi = max(lo, cursor), min(hi, end)
        if hi > lo:
            covered += hi - lo
            cursor = hi
    return covered


def _compact(label):
    return re.sub("[^a-z0-9]", "", str(label).lower())


def _us_to_ns(value):
    return int(1000 * Decimal(str(value).strip().removesuffix("us").strip()))


def trace_coverage(events, overlap):
    """Use only device activities. Chrome trace timestamps and durations are us."""
    hardware = {
        e["pid"]
        for e in events
        if e.get("ph") == "M"
        and e.get("name") == "process_name"
        and "ascendhardware" in _compact(e.get("args", {}).get("name", ""))
    }
    tasks = []
    markers = {name: [] for name in ("sin", "cos", "atan")}
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        args = {_compact(k): v for k, v in (e.get("args") or {}).items()}
        # Some CANN versions omit the process label but retain stream/task fields.
        if hardware:
            if e.get("pid") not in hardware:
                continue
        elif not {"streamid", "tasktype"} <= args.keys() or "acl" in e.get("cat", "").lower():
            continue
        # The hardware track's tid is the logical stream. An event's "Stream
        # Id" may be its physical stream, which can vary within that track.
        stream = (e.get("pid"), str(e.get("tid", args.get("streamid"))))
        start = _us_to_ns(e["ts"])
        end = start + _us_to_ns(e["dur"])
        name = e.get("name", "")
        labels = (name, str(args.get("opname", "")), str(args.get("optype", "")))
        for marker in markers:
            if any(re.search(rf"(?:^|[^a-zA-Z]|aclnn){marker}(?:$|[^a-zA-Z]|AiCore)",
                             label, re.I) for label in labels):
                markers[marker].append((stream, start, end))
        description = _compact(" ".join(map(str, (
            name, args.get("tasktype", ""), args.get("memcpykind", ""),
            args.get("direction", ""), args.get("operation", ""),
        ))))
        tasks.append((stream, start, end, description, args))

    def marker_stream(name):
        streams = {task[0] for task in markers[name]}
        if len(streams) != 1:
            raise ValueError(f"Cannot identify a unique hardware {name} marker: {streams}")
        return streams.pop()

    compute = marker_stream("sin")
    copy = marker_stream("atan")
    if compute != marker_stream("cos") or compute == copy:
        raise ValueError("Expected Sin/Cos on compute and Atan on a separate copy stream")
    start = max(end for _, _, end in markers["sin"])
    end = min(begin for _, begin, _ in markers["cos"])
    # This isolated stream executes only Atan, the two known H2D batches and
    # their events. CANN may label their SDMA tasks simply MEMCPY_ASYNC.
    intervals = []
    for lane, lo, hi, label, args in tasks:
        if lane != copy or hi <= lo:
            continue
        if not any(k in label for k in ("memcpy", "memcopy", "hosttodevice", "h2d", "sdma")):
            continue
        if any(k in label for k in ("devicetohost", "devicetodevice", "d2h", "d2d")):
            continue
        # CANN may expose the DMA timestamps inside the outer scheduler task.
        if "starttime" in args and "endtime" in args:
            actual_start = _us_to_ns(args["starttime"])
            actual_end = _us_to_ns(args["endtime"])
            if not lo <= actual_start <= actual_end <= hi:
                raise ValueError("Inconsistent DMA and device-task timestamps")
            lo, hi = actual_start, actual_end
        intervals.append((lo, hi))
    if overlap != bool(intervals):
        raise ValueError(f"Expected overlap={overlap}, found {len(intervals)} copy tasks")
    covered = covered_ns(start, end, intervals)
    return {
        "dfc_us": (end - start) / 1000,
        "covered_us": covered / 1000,
        "uncovered_us": (end - start - covered) / 1000,
        "coverage_pct": 100 * covered / (end - start),
        "copy_tasks": len(intervals),
        "compute_stream": compute[1],
        "copy_stream": copy[1],
    }


class TestFixture(unittest.TestCase):
    def test_ep_receive_counts_at_each_token_size(self):
        # Expected rows per destination rank for one input token on each rank.
        rates = {"balanced": (8, 8), "3to1": (12, 4), "7to1": (14, 2)}
        for tokens, load in product((1, *TOKEN_SIZES), rates):
            with self.subTest(tokens=tokens, load=load):
                counts = [[0] * LOCAL for _ in range(EP)]
                for source in range(EP):
                    for row in range(tokens):
                        ids = [route_ids(row, lane, source, load) for lane in range(TOP_K)]
                        self.assertEqual(len({index // LOCAL for index in ids}), TOP_K)
                        for index in ids:
                            rank, expert = divmod(index, LOCAL)
                            counts[rank][expert] += 1
                for rank, experts in enumerate(counts):
                    self.assertEqual(sum(experts), tokens * rates[load][rank // (EP // 2)])
                    if tokens >= LOCAL:
                        self.assertTrue(all(experts))

    def test_matrix_defaults_and_profile_switch(self):
        from unittest.mock import patch

        default = parse_options([])
        self.assertEqual(default.tokens_per_rank, (32, 128, 512))
        self.assertEqual(default.ep_load, ("balanced", "3to1", "7to1"))
        self.assertFalse(default.profile)
        with patch.dict(sys.modules, {"torch_npu": None}), profile_context(None):
            pass
        chosen = parse_options(["--tokens-per-rank", "64", "256", "--ep-load", "7to1", "--profile"])
        self.assertEqual(chosen.tokens_per_rank, (64, 256))
        self.assertEqual(chosen.ep_load, ("7to1",))
        self.assertTrue(chosen.profile)

    def test_union_clips_and_does_not_double_count(self):
        self.assertEqual(covered_ns(10, 30, [(16, 25), (5, 18), (27, 40)]), 18)

    def test_hardware_coverage_and_baseline(self):
        def task(lane, name, ts, dur, pid=7):
            return dict(ph="X", pid=pid, tid=lane, name=name, ts=ts, dur=dur)

        markers = [
            dict(ph="M", pid=7, name="process_name", args=dict(name="Ascend Hardware")),
            task(2, "aclnnAtan_AtanAiCore_Atan", 0, 1),
            task(1, "Sin", 10, 1),
            task(1, "Cos", 21, 1),
            task(1, "MemcpyAsync_H2D", 10, 13),  # dispatch's own small copy
            task(2, "aclrtMemcpyBatchAsync", 1, 50, pid=99),  # host API
        ]
        self.assertEqual(trace_coverage(markers, False)["copy_tasks"], 0)
        with self.assertRaises(ValueError):
            trace_coverage(markers, True)
        copy = [task(2, "MemcpyAsync", 9, 7), task(2, "MemcpyAsync", 17, 12)]
        self.assertEqual(trace_coverage(markers + copy, True)["uncovered_us"], 1)
        copy.append(task(2, "MemcpyAsync", 16, 1))
        self.assertEqual(trace_coverage(markers + copy, True)["coverage_pct"], 100)
        copy_task = task(2, "MemcopyAsync", 4, 30)
        copy_task["args"] = {"Start Time": 13, "End Time": 19, "operation": "host to device"}
        self.assertEqual(trace_coverage(markers + [copy_task], True)["covered_us"], 6)
        copy_task["args"]["operation"] = "device to host"
        self.assertEqual(trace_coverage(markers + [copy_task], False)["copy_tasks"], 0)


class ExpertWeights:
    def __init__(self, device, native_dir, rank, cleanup):
        import torch
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps, get_native_acl
        from sglang.srt.offloading.moe.npu.options import prefill_weight_slots
        from sglang.srt.offloading.moe.npu.prefill import _projection_copy_plans
        from sglang.srt.offloading.moe.npu.w8a8_format import NpuV4W8A8Format
        from sglang.srt.offloading.moe.resources import ResourceKind
        from sglang.srt.offloading.moe.storage import HostMemoryCapability

        self.device = device
        self.native = get_native_acl(native_dir)
        fmt = NpuV4W8A8Format(HIDDEN, INTERMEDIATE, device, native_acl=self.native)
        self.resident, slots = prefill_weight_slots(LOCAL, ROUTED_LAYERS, RESIDENT_RATIO, None)
        cold = LOCAL - self.resident
        self.cold_base = ROUTED_LAYERS * self.resident
        pool = fmt.allocate_bank(slots, ResourceKind.EXECUTION_WEIGHTS, "glm52_dfc")
        self.quant = pool.quant_info()
        self.next_bank = pool.view(self.cold_base + cold, cold)
        self.groups = torch.zeros((slots, 2), device=device, dtype=torch.int64)
        self.counts = torch.empty(LOCAL, device=device, dtype=torch.int64)
        self.error = torch.zeros((), device=device, dtype=torch.int32)
        self.stream = torch.npu.Stream(device=device)
        self.keepalive = []
        self.ops = AclHostTransferOps(self.native, self.keepalive.append)
        self.copy_bytes = cold * sum(layout.nbytes for layout in fmt.layouts.values())

        # Initialize only the current layer's six resident and ten bank-0 slots.
        # The transfer is outside all runs. Random INT8 bytes fill physical NZ
        # storage directly; scales and offsets have their real typed ND layout.
        rng = torch.Generator(device="cpu").manual_seed(1000 + rank)
        slots_used = list(range(self.resident)) + list(range(self.cold_base, self.cold_base + cold))
        caps = {}
        self.signatures = {}
        for index, (name, layout) in enumerate(fmt.layouts.items()):
            element_size = torch.empty((), dtype=layout.dtype).element_size()
            data = torch.empty(layout.nbytes // element_size, dtype=layout.dtype, device="cpu")
            for slot in slots_used:
                if name.endswith("_weight"):
                    data.random_(-4, 5, generator=rng)
                elif name.endswith("_scale"):
                    data.fill_(1 / 128)
                else:
                    data.zero_()
                dest = pool.component_regions[name][slot]
                self.native.memcpy_sync(dest.address, dest.nbytes, data.data_ptr(),
                                        layout.nbytes, "host_to_device", device.index)

            # Only next-layer cold experts have an H2D source. aclrtMallocHost
            # supplies page-locked storage; preserve real host strides for scales.
            ptr = self.native.malloc_host(cold * layout.host_nbytes, device.index)
            cleanup.callback(self.native.free_host, ptr, device.index)
            allocation = HostMemoryCapability(ptr, cold * layout.host_nbytes, True, None, cleanup)
            regions = []
            for expert in range(cold):
                address = ptr + expert * layout.host_nbytes
                raw = (ctypes.c_ubyte * layout.nbytes).from_address(address)
                payload = torch.frombuffer(raw, dtype=layout.dtype)
                if name.endswith("_weight"):
                    payload.fill_(1 + index + expert)
                elif name.endswith("_scale"):
                    payload.fill_((1 + expert) / 128)
                else:
                    payload.zero_()
                self.signatures[name, expert] = ctypes.string_at(address, 16)
                regions.append(HostMemoryCapability(address, layout.nbytes, True, None, allocation))
            caps[name] = tuple(regions)
        self.plans = _projection_copy_plans(SimpleNamespace(component_capabilities=caps),
                                             self.next_bank, fmt.layouts)
        for plan in self.plans:
            self.ops.prepare(plan, device)
        # Last-in, first-out: drain queued copies even on exceptions, then free host memory.
        cleanup.callback(torch.npu.synchronize, device)

    def submit(self, scope):
        for name, plan in zip(("W13", "W2"), self.plans, strict=True):
            with scope("prefetch/" + name):
                self.ops.materialize_into(plan, self.next_bank, self.stream)

    def check_bank(self, *, poison=False, prefetched=False):
        # Poison both ends after timing so an old warm-up copy cannot pass the check.
        sentinel = bytes([0xA5]) * 16
        scratch = (ctypes.c_ubyte * 16).from_buffer_copy(sentinel)
        for name, regions in self.next_bank.component_regions.items():
            for expert, region in enumerate(regions):
                for offset in (0, region.nbytes - len(scratch)):
                    if poison:
                        self.native.memcpy_sync(region.address + offset, len(scratch),
                                                ctypes.addressof(scratch), len(scratch),
                                                "host_to_device", self.device.index)
                        continue
                    self.native.memcpy_sync(ctypes.addressof(scratch), len(scratch),
                                            region.address + offset, len(scratch),
                                            "device_to_host", self.device.index)
                    expected = self.signatures[name, expert] if prefetched else sentinel
                    if bytes(scratch) != expected:
                        raise AssertionError(f"Wrong bank-1 payload in {name}, expert {expert}")

    def ffn(self, core, inputs):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerOutput
        from sglang.srt.offloading.moe.npu.small_batch_kernels import prepare_prefill_counts

        if inputs.group_list_type not in (0, 1) or inputs.expert_tokens.shape != (LOCAL,):
            raise AssertionError("Unexpected prefill dispatch group list")
        prepare_prefill_counts[(1,)](
            inputs.expert_tokens, self.error, None, None, self.counts,
            EXPERTS=LOCAL, CUMULATIVE=inputs.group_list_type == 0, BLOCK=LOCAL,
            TRACK_USAGE=False, Groups=self.groups, ResidentBase=0, ColdBase=self.cold_base,
            RESIDENT_EXPERTS=self.resident,
        )
        inputs.expert_tokens, inputs.group_list_type = self.groups, 2
        return AscendRunnerOutput(hidden_states=core._run_ffn(
            inputs, self.quant, activation_tokens=self.counts, activation_group_type=1,
        ))


def backend_context(stack, backend, rank, device):
    import torch.distributed as dist
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import SpawnRanks, create_context, get_parallel, use_context
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy",
        device="npu",
        tp_size=EP,
        ep_size=EP,
        moe_a2a_backend=backend,
        moe_runner_backend="ascend",
        attention_backend="ascend",
        deepep_mode="normal" if backend == "deepep" else "auto",
        deepep_dispatcher_output_dtype="int8",
    )
    placement = SpawnRanks(world_rank=rank, gpu_id=device.index)
    ctx = create_context(args, role="test", ranks=placement)
    stack.enter_context(use_context(ctx))
    group = SimpleNamespace(device_group=dist.group.WORLD, world_size=EP,
                            rank_in_group=rank, barrier=dist.barrier)
    stack.enter_context(get_parallel().override(tp_group=group, moe_ep_group=group))
    initialize_moe_config()
    set_is_extend_in_batch(True)


class RoutedLayer:
    def __init__(self, backend, weights, rank, *, profile):
        import torch
        import torch.distributed as dist
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
        from sglang.srt.layers.moe.moe_runner import ascend
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.utils import AscendOpsConfig, DeepEPMode, MoeRunnerBackend

        self.weights = weights
        layer = SimpleNamespace(
            w13_kernel=NPUW8A8Int8MoEMethod(),
            w2_kernel=NPUW8A8Int8MoEMethod(),
            layer_id=3,
            moe_ep_size=EP,
            moe_ep_rank=rank,
            moe_tp_size=1,
            moe_tp_rank=0,
        )
        self.config = MoeRunnerConfig(
            layer=layer,
            layer_id=layer.layer_id,
            num_experts=EXPERTS,
            num_local_experts=LOCAL,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            top_k=TOP_K,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            activation="silu",
            runner_backend=MoeRunnerBackend.ASCEND,
        )
        self.core = ascend.AscendRunnerCore(self.config)
        if backend == "deepep":
            from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher

            self.dispatcher = DeepEPDispatcher(
                group=dist.group.WORLD,
                router_topk=TOP_K,
                permute_fusion=True,
                num_experts=EXPERTS,
                num_local_experts=LOCAL,
                hidden_size=HIDDEN,
                params_dtype=torch.bfloat16,
                deepep_mode=DeepEPMode.NORMAL,
                async_finish=True,
                return_recv_hook=True,
                runner_backend=MoeRunnerBackend.ASCEND,
            )
            self.pre = ascend.pre_permute_deepep_normal_to_ascend
            self.post = ascend.post_permute_ascend_to_deepep_normal
        else:
            from sglang.srt.layers.moe.token_dispatcher.ascend_ops import AscendOpsDispatcher

            self.dispatcher = AscendOpsDispatcher(self.config, AscendOpsConfig(prefill="alltoallv"))
            self.pre = ascend.pre_permute_ascend_ops_to_ascend
            self.post = ascend.post_permute_ascend_to_ascend_ops
        self.dispatcher.set_quant_config(
            {"dispatcher_output_dtype": "int8", "normal_dispatcher_output_dtype": "int8"}
        )
        self.compute = torch.npu.current_stream(weights.device)
        if profile:
            self.marker_input = torch.ones(1024, device=weights.device, dtype=torch.float32)
            self.copy_marker = torch.empty_like(self.marker_input)
            self.compute_marker = torch.empty_like(self.marker_input)
            for op in (torch.sin, torch.cos, torch.atan):
                op(self.marker_input, out=self.compute_marker)
        self.events = [torch.npu.Event(enable_timing=True) for _ in range(7)]
        for event in self.events:
            event.record(self.compute)

    def set_workload(self, tokens, ep_load, rank, control):
        import torch
        import torch.distributed as dist
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        device = self.weights.device
        gen = torch.Generator(device="cpu").manual_seed(8000 + rank)
        hidden = torch.randn((tokens, HIDDEN), generator=gen, device="cpu")
        self.hidden = hidden.to(device, torch.bfloat16)
        row = torch.arange(tokens, device="cpu")[:, None]
        lane = torch.arange(TOP_K, device="cpu")[None, :]
        ids = route_ids(row, lane, rank, ep_load)
        probs = torch.rand((tokens, TOP_K), generator=gen, device="cpu").softmax(dim=1) * 2.5
        self.topk = StandardTopKOutput(probs.to(device), ids.to(device, torch.int32), None)

        # Count the actual top-k IDs from all sources on Gloo, before timing.
        # Row rank of this histogram is the receiving EP rank; columns are its experts.
        histogram = torch.bincount(ids.flatten(), minlength=EXPERTS)
        dist.all_reduce(histogram, group=control)
        self.expected_counts = histogram.reshape(EP, LOCAL)[rank].tolist()
        # The offloader latches a bad group list. Each workload has its own verdict.
        self.weights.error.zero_()

    def run(self, overlap, profile=False):
        import torch

        w = self.weights
        scope = torch.profiler.record_function if profile else nullcontext
        origin, copy_start, copy_end, begin, dispatched, calculated, combined = self.events
        if profile:
            with torch.npu.stream(w.stream):
                torch.atan(self.marker_input, out=self.copy_marker)
        origin.record(self.compute)
        if overlap:
            w.stream.wait_event(origin)
            copy_start.record(w.stream)
            w.submit(scope)
            copy_end.record(w.stream)
        # The current layer is already ready. There is deliberately no wait on
        # copy_end on the compute stream, including after combine.
        if profile:
            torch.sin(self.marker_input, out=self.compute_marker)
        begin.record(self.compute)
        state = {}
        with scope("DFC/dispatch"):
            routing = self.dispatcher.dispatch(self.hidden, self.topk)
            inputs = self.pre(routing, w.quant, self.config, state)
        dispatched.record(self.compute)
        with scope("DFC/ffn"):
            result = w.ffn(self.core, inputs)
        calculated.record(self.compute)
        with scope("DFC/combine"):
            combine_input = self.post(result, w.quant, self.config, state)
            output = self.dispatcher.combine(combine_input)
        combined.record(self.compute)
        if profile:
            torch.cos(self.marker_input, out=self.compute_marker)
        torch.npu.synchronize(w.device)
        w.keepalive.clear()
        dfc_ms = begin.elapsed_time(combined)
        makespan = dfc_ms
        outside_copy_window = dfc_ms
        if overlap:
            dfc_begin, dfc_end = origin.elapsed_time(begin), origin.elapsed_time(combined)
            copy_begin, copy_finish = origin.elapsed_time(copy_start), origin.elapsed_time(copy_end)
            makespan = max(dfc_end, copy_finish) - min(dfc_begin, copy_begin)
            intersection = max(0.0, min(dfc_end, copy_finish) - max(dfc_begin, copy_begin))
            outside_copy_window = max(0.0, dfc_end - dfc_begin - intersection)
        stats = {
            "dispatch_ms": begin.elapsed_time(dispatched),
            "ffn_ms": dispatched.elapsed_time(calculated),
            "combine_ms": calculated.elapsed_time(combined),
            "dfc_ms": dfc_ms,
            "copy_stream_ms": copy_start.elapsed_time(copy_end) if overlap else 0,
            "dfc_outside_copy_window_ms": outside_copy_window,
            "makespan_ms": makespan,
        }
        return output, stats


def profile_context(folder):
    if folder is None:
        return nullcontext()

    from torch_npu import profiler as prof

    folder.mkdir(parents=True, exist_ok=False)
    return prof.profile(
        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
        on_trace_ready=prof.tensorboard_trace_handler(str(folder)),
        experimental_config=prof._ExperimentalConfig(
            export_type=prof.ExportType.Text, profiler_level=prof.ProfilerLevel.Level1,
        ),
    )


def read_profile(folder, overlap):
    paths = list(folder.rglob("trace_view.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one trace_view.json under {folder}, found {len(paths)}")
    with paths[0].open(encoding="utf-8") as file:
        data = json.load(file, parse_float=Decimal)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return {"path": str(paths[0]), **trace_coverage(events, overlap)}


def measure_case(layer, options, case_dir, control, rank):
    import torch
    import torch.distributed as dist

    ref = None
    samples = {mode: [] for mode in MODES}
    checks = {}
    for _ in range(options.warmup):
        for mode in MODES:
            dist.barrier(group=control)
            layer.run(mode == "overlap")
    # Alternate case order; all ranks perform exactly the same collectives.
    for i in range(options.iterations):
        mode_order = MODES if i % 2 == 0 else reversed(MODES)
        for mode in mode_order:
            dist.barrier(group=control)
            out, stats = layer.run(mode == "overlap")
            samples[mode].append(stats)
            if ref is None:
                ref = out.detach().clone()
                torch.npu.synchronize(layer.weights.device)

    rank_dir = case_dir / f"rank{rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    profiles = {}
    received = {}
    errors = []
    layer.weights.check_bank(poison=True)
    for mode in MODES:
        overlap = mode == "overlap"
        trace_dir = rank_dir / mode if options.profile else None
        with profile_context(trace_dir):
            dist.barrier(group=control)
            out, _ = layer.run(overlap, profile=options.profile)
        # Preserve results for both modes and all ranks even if a trace cannot
        # establish full coverage. These checks enqueue no measured device work.
        received[mode] = layer.weights.counts.cpu().tolist()
        try:
            if out.shape != layer.hidden.shape or not bool(torch.isfinite(out).all()) or not bool(out.any()):
                raise AssertionError("Invalid combined output")
            torch.testing.assert_close(out, ref, rtol=0.02, atol=1e-4)
            if layer.weights.error.item() or received[mode] != layer.expected_counts:
                raise AssertionError(
                    f"Dispatch returned {received[mode]}, expected {layer.expected_counts}"
                )
            layer.weights.check_bank(prefetched=overlap)
            checks[mode] = True
        except AssertionError as exc:
            errors.append(f"{mode} verification: {exc}")
        if trace_dir is None:
            continue
        try:
            profiles[mode] = read_profile(trace_dir, overlap)
            if overlap and profiles[mode]["uncovered_us"] > 0:
                errors.append(f"overlap left {profiles[mode]['uncovered_us']:.3f} us of DFC uncovered")
        except (ArithmeticError, KeyError, ValueError, OSError) as exc:
            errors.append(f"{mode} timeline: {exc}")

    report = dict(
        rank=rank, samples=samples, profiles=profiles, output_matches=checks, errors=errors,
        expected_expert_counts=layer.expected_counts, received_expert_counts=received,
    )
    (rank_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def aggregate(reports):
    modes = {}
    for mode in MODES:
        iterations = list(zip(*(r["samples"][mode] for r in reports), strict=True))
        modes[mode] = {
            key: median(max(r[key] for r in iteration) for iteration in iterations)
            for key in iterations[0][0]
        }
    coverage = [r["profiles"]["overlap"]["coverage_pct"] for r in reports if "overlap" in r["profiles"]]
    return {
        "median_of_per_iteration_slowest_rank_ms": modes,
        "additional_dfc_ms": modes["overlap"]["dfc_ms"] - modes["baseline"]["dfc_ms"],
        "min_dfc_coverage_pct": min(coverage) if len(coverage) == len(reports) else None,
        "received_expert_tokens_by_rank": {
            mode: [sum(r["received_expert_counts"][mode]) for r in reports] for mode in MODES
        },
        "errors_by_rank": {r["rank"]: r["errors"] for r in reports if r["errors"]},
    }


def run_ep16(options):
    if int(os.environ.get("WORLD_SIZE", 0)) != EP or int(os.environ.get("LOCAL_WORLD_SIZE", 0)) != EP:
        raise ValueError("Run on one NPU node with torchrun --standalone --nproc-per-node=16")
    import torch
    import torch.distributed as dist
    import torch_npu
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    local = int(os.environ["LOCAL_RANK"])
    if torch.npu.device_count() < EP:
        raise RuntimeError("Expose 16 local NPUs to every torchrun worker")
    torch.set_num_threads(1)
    device = torch.device("npu", local)
    torch.npu.set_device(device)
    init_npu_backend()
    dist.init_process_group("hccl", timeout=timedelta(minutes=20))
    try:
        from tempfile import mkdtemp

        rank = dist.get_rank()
        # The control group keeps barriers and result collection off NPU streams.
        control = dist.new_group(backend="gloo", timeout=timedelta(minutes=20))
        location = [None]
        if rank == 0:
            options.output.mkdir(parents=True, exist_ok=True)
            location[0] = mkdtemp(prefix="glm52-dfc-", dir=options.output.resolve())
            print(f"Output: {location[0]}", flush=True)
        dist.broadcast_object_list(location, src=0, group=control)
        root = Path(location[0])
        results = []
        with ExitStack() as cleanup, torch.inference_mode():
            w = ExpertWeights(device, options.native_extension_dir, rank, cleanup)
            backends = BACKENDS if options.backend == "both" else (options.backend,)
            for backend in backends:
                with ExitStack() as stack:
                    backend_context(stack, backend, rank, device)
                    layer = RoutedLayer(backend, w, rank, profile=options.profile)
                    for tokens, load in product(options.tokens_per_rank, options.ep_load):
                        layer.set_workload(tokens, load, rank, control)
                        torch.npu.synchronize(device)
                        case_dir = root / backend / f"tokens-{tokens}" / load
                        local_report = measure_case(layer, options, case_dir, control, rank)
                        reports = [None] * EP
                        dist.all_gather_object(reports, local_report, group=control)
                        result = dict(
                            backend=backend, tokens_per_rank=tokens, global_input_tokens=EP * tokens,
                            ep_load=load, **aggregate(reports),
                        )
                        results.append(result)
                        if rank == 0:
                            print(json.dumps(result), flush=True)
                    del layer
            if rank == 0:
                summary = dict(
                    torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                    ep=EP, hidden=HIDDEN, intermediate=INTERMEDIATE, top_k=TOP_K, experts=EXPERTS,
                    tokens_per_rank=options.tokens_per_rank, ep_loads=options.ep_load,
                    profile_enabled=options.profile, resident_per_rank=w.resident,
                    cold_per_rank=LOCAL - w.resident, pool_slots=len(w.groups),
                    bytes_prefetched_per_rank=w.copy_bytes, native_acl=dict(w.native.api_info()),
                    results=results,
                )
                (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if any(result["errors_by_rank"] for result in results):
            raise AssertionError(f"EP16 DFC checks failed; see {root / 'summary.json'}")
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("LOCAL_WORLD_SIZE") == str(EP), "requires single-node EP16 torchrun")
class TestGLM52PrefillDFC(unittest.TestCase):
    def test_prefill_dfc_h2d(self):
        run_ep16(parse_options([]))


def parse_options(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("both", *BACKENDS), default="both")
    parser.add_argument(
        "--tokens-per-rank", nargs="+", type=int, default=TOKEN_SIZES, metavar="N",
        help="input tokens per source rank before routing (default: 32 128 512)",
    )
    parser.add_argument(
        "--ep-load", nargs="+", choices=tuple(HOT_ROUTES), default=tuple(HOT_ROUTES),
        help="receive load on ranks 0-7 vs 8-15 (default: balanced 3to1 7to1)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="capture torch_npu traces and require full DMA coverage for every selected case (default: off)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("/tmp/glm52-dfc"))
    parser.add_argument("--native-extension-dir", default=os.environ.get("SGLANG_MOE_NATIVE_EXTENSION_DIR"),
                        help="directory containing the deployed _sglang_moe_acl extension")
    parser.add_argument("--self-test", action="store_true", help="run CPU fixture and timeline tests")
    opts = parser.parse_args(argv)
    if min(*opts.tokens_per_rank, opts.warmup, opts.iterations) < 1:
        parser.error("--tokens-per-rank, --warmup and --iterations must be positive")
    opts.tokens_per_rank = tuple(dict.fromkeys(opts.tokens_per_rank))
    opts.ep_load = tuple(dict.fromkeys(opts.ep_load))
    return opts


if __name__ == "__main__":
    opts = parse_options(sys.argv[1:])
    if opts.self_test:
        unittest.main(argv=[sys.argv[0]], defaultTest="TestFixture")
    else:
        # Prefer the adjacent checkout when invoked without an editable installation.
        for parent in Path(__file__).resolve().parents:
            if (parent / "sglang/srt/offloading/moe").is_dir():
                sys.path.insert(0, str(parent))
                break
        run_ep16(opts)
