r"""Simulate a GLM-5.2 W8A8 prefill dispatch, expert FFN, and combine on 16 NPUs.

From the checkout (in the Ascend deployment environment):
    torchrun --standalone --nproc-per-node=16 \
        python/sglang/srt/offloading/moe/test/test_glm52_prefill_dfc_h2d.py \
        --native-extension-dir /path/to/native_acl --output /tmp/glm52-dfc \
        --backend ascend_ops --mode overlap --tokens-per-rank 128 --ep-load balanced --profile

Each rank generates its input activations, top-8 routes, and expert weights.
All ranks meet at a Gloo barrier at the start of every forward. Both modes run
routed layer 3. Overlap also prefetches layer 4's cold experts on a separate stream.
Use --mode both to compare them with identical inputs.
Token counts are input rows per rank before dispatch. EP loads balanced,
3to1, and 7to1 send 50%, 75%, and 87.5% of routed rows to ranks 0–7.

By default each rank runs one warmup and one measured forward. With --profile,
all ranks finish warming up every selected backend, workload, and mode before
recording the measured iterations in one continuous CPU and NPU trace per rank.
For each workload, --mode both records the baseline forwards first, then the
overlap forwards. The trace paths are listed in rank order in summary.json.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from contextlib import ExitStack
from datetime import timedelta
from itertools import product
from pathlib import Path
from types import SimpleNamespace

EP = 16
EXPERTS = 256
LOCAL = EXPERTS // EP
TOP_K = 8
HIDDEN = 6144
INTERMEDIATE = 2048
ROUTED_LAYERS = 78 - 3
RESIDENT_RATIO = 0
MODES = ("baseline", "overlap")
BACKENDS = ("deepep", "ascend_ops")
# Number of this token's eight routes sent to EP ranks 0 through 7.
HOT_ROUTES = {"balanced": 4, "3to1": 6, "7to1": 7}


def route_ids(row, lane, source_rank, ep_load="balanced"):
    """Select a global expert for each input row and top-k lane."""
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


class ExpertWeights:
    def __init__(self, device, native_dir, rank, cleanup, *, prefetch, control_group):
        import torch
        from sglang.srt.offloading.moe.npu.native_acl import AclHostTransferOps, get_native_acl
        from sglang.srt.offloading.moe.npu.options import prefill_weight_slots
        from sglang.srt.offloading.moe.npu.prefill import _projection_copy_plans
        from sglang.srt.offloading.moe.npu.w8a8_format import NpuV4W8A8Format
        from sglang.srt.offloading.moe.resources import ResourceKind
        from sglang.srt.offloading.moe.storage import HostMemoryCapability

        self.device = device
        self.control_group = control_group
        self.native = get_native_acl(native_dir)
        fmt = NpuV4W8A8Format(HIDDEN, INTERMEDIATE, device, native_acl=self.native)
        self.resident, slots = prefill_weight_slots(LOCAL, ROUTED_LAYERS, RESIDENT_RATIO, None)
        cold = LOCAL - self.resident
        self.cold_base = ROUTED_LAYERS * self.resident # 预取buffer的起始地址
        pool = fmt.allocate_bank(slots, ResourceKind.EXECUTION_WEIGHTS, "glm52_dfc")
        self.quant = pool.quant_info()
        self.groups = torch.zeros((slots, 2), device=device, dtype=torch.int64)
        self.counts = torch.empty(LOCAL, device=device, dtype=torch.int64)
        self.error = torch.zeros((), device=device, dtype=torch.int32)
        self.keepalive = []
        self.copy_bytes = 0

        # Initialize the current layer's resident weights and cold bank before profiling.
        # Random INT8 bytes fill NZ storage. Scales and offsets use their ND layout.
        rng = torch.Generator(device="cpu").manual_seed(1000 + rank)
        slots_used = list(range(self.resident)) + list(range(self.cold_base, self.cold_base + cold))
        caps = {}
        for index, (name, layout) in enumerate(fmt.layouts.items()):
            element_size = torch.empty((), dtype=layout.dtype).element_size()
            data = torch.empty(layout.nbytes // element_size, dtype=layout.dtype, device="cpu")
            for slot in slots_used:
                if name.endswith("_weight"):
                    data.random_(-128, 128, generator=rng)
                elif name.endswith("_scale"):
                    data.fill_(1 / 128)
                else:
                    data.zero_()
                dest = pool.component_regions[name][slot]
                self.native.memcpy_sync(dest.address, dest.nbytes, data.data_ptr(),
                                        layout.nbytes, "host_to_device", device.index)

            if not prefetch:
                continue
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
                regions.append(HostMemoryCapability(address, layout.nbytes, True, None, allocation))
            caps[name] = tuple(regions)
        if prefetch:
            self.next_bank = pool.view(self.cold_base + cold, cold)
            self.stream = torch.npu.Stream(device=device)
            self.ops = AclHostTransferOps(self.native, self.keepalive.append)
            self.plans = _projection_copy_plans(
                SimpleNamespace(component_capabilities=caps), self.next_bank, fmt.layouts
            )
            self.copy_bytes = cold * sum(layout.nbytes for layout in fmt.layouts.values())
            for plan in self.plans:
                self.ops.prepare(plan, device)
        # Last-in, first-out: drain queued copies even on exceptions, then free host memory.
        cleanup.callback(torch.npu.synchronize, device)

    def submit(self):
        for plan in self.plans:
            self.ops.materialize_into(plan, self.next_bank, self.stream)

    def ffn(self, core, inputs):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerOutput
        from sglang.srt.offloading.moe.npu.small_batch_kernels import prepare_prefill_counts

        if inputs.group_list_type not in (0, 1) or inputs.expert_tokens.shape != (LOCAL,):
            raise ValueError("Unexpected prefill dispatch group list")
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


def prepare_backend_layer(stack, backend, weights, rank):
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
    placement = SpawnRanks(world_rank=rank, gpu_id=weights.device.index)
    ctx = create_context(args, role="test", ranks=placement)
    group = SimpleNamespace(device_group=dist.group.WORLD, world_size=EP,
                            rank_in_group=rank, barrier=dist.barrier)
    with use_context(ctx):
        stack.enter_context(get_parallel().override(tp_group=group, moe_ep_group=group))
        initialize_moe_config()
        set_is_extend_in_batch(True)
        return ctx, RoutedLayer(backend, weights, rank)


class RoutedLayer:
    def __init__(self, backend, weights, rank):
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
        self.compute_ready = torch.npu.Event()

    def set_workload(self, tokens, ep_load, rank):
        import torch
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

    def run(self, overlap):
        import torch.distributed as dist

        w = self.weights
        dist.barrier(group=w.control_group)
        if overlap:
            self.compute_ready.record(self.compute)
            w.stream.wait_event(self.compute_ready)
            w.submit()
        state = {}
        routing = self.dispatcher.dispatch(self.hidden, self.topk)
        inputs = self.pre(routing, w.quant, self.config, state)
        result = w.ffn(self.core, inputs)
        combine_input = self.post(result, w.quant, self.config, state)
        return self.dispatcher.combine(combine_input)


def profile_context(folder):
    from torch_npu import profiler as prof

    folder.mkdir(parents=True, exist_ok=False)
    return prof.profile(
        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
        on_trace_ready=prof.tensorboard_trace_handler(str(folder)),
        experimental_config=prof._ExperimentalConfig(
            export_type=prof.ExportType.Text, profiler_level=prof.ProfilerLevel.Level1,
        ),
    )


def profile_trace_path(folder):
    paths = list(folder.rglob("trace_view.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one trace_view.json under {folder}, found {len(paths)}")
    return str(paths[0])


def run_iterations(layers, options, rank, iterations):
    import torch
    from sglang.srt.runtime_context import use_context

    for ctx, layer in layers:
        with use_context(ctx):
            for tokens, load in product(options.tokens_per_rank, options.ep_load):
                layer.set_workload(tokens, load, rank)
                for mode in options.modes:
                    for _ in range(iterations):
                        layer.run(mode == "overlap")
                # Drain compute and H2D before releasing tickets or switching workloads.
                torch.npu.synchronize(layer.weights.device)
                layer.weights.keepalive.clear()


def run_workloads(backends, weights, options, rank, trace_dir, control):
    import torch.distributed as dist

    with ExitStack() as stack:
        layers = [prepare_backend_layer(stack, backend, weights, rank) for backend in backends]
        if trace_dir is None:
            run_iterations(layers, options, rank, options.warmup + options.iterations)
            return

        if options.warmup:
            run_iterations(layers, options, rank, options.warmup)
            dist.barrier(group=control)
        with profile_context(trace_dir):
            run_iterations(layers, options, rank, options.iterations)


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
        # Coordinate forward starts, profiling, and output paths on the host.
        control = dist.new_group(backend="gloo", timeout=timedelta(minutes=20))
        location = [None]
        if rank == 0:
            options.output.mkdir(parents=True, exist_ok=True)
            location[0] = mkdtemp(prefix="glm52-dfc-", dir=options.output.resolve())
            print(f"Output: {location[0]}", flush=True)
        dist.broadcast_object_list(location, src=0, group=control)
        root = Path(location[0])
        trace_dir = root / f"rank{rank:02d}" if options.profile else None
        with ExitStack() as cleanup, torch.inference_mode():
            w = ExpertWeights(
                device, options.native_extension_dir, rank, cleanup,
                prefetch="overlap" in options.modes, control_group=control,
            )
            backends = BACKENDS if options.backend == "both" else (options.backend,)
            run_workloads(backends, w, options, rank, trace_dir, control)
            if w.error.item():
                raise RuntimeError(f"Dispatch returned invalid expert token counts on rank {rank}")

            trace_paths = [None] * EP if rank == 0 and options.profile else None
            if options.profile:
                dist.gather_object(profile_trace_path(trace_dir), trace_paths, dst=0, group=control)
            if rank == 0:
                summary = dict(
                    torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                    ep=EP, hidden=HIDDEN, intermediate=INTERMEDIATE, top_k=TOP_K, experts=EXPERTS,
                    backends=backends, tokens_per_rank=options.tokens_per_rank, ep_loads=options.ep_load,
                    modes=options.modes, warmup=options.warmup, iterations=options.iterations,
                    profile_enabled=options.profile, resident_per_rank=w.resident,
                    cold_per_rank=LOCAL - w.resident, pool_slots=len(w.groups),
                    bytes_prefetched_per_rank=w.copy_bytes, native_acl=dict(w.native.api_info()),
                    trace_paths=trace_paths,
                )
                (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backend", choices=(*BACKENDS, "both"), default="ascend_ops",
        help="dispatch and combine backend (default: ascend_ops)",
    )
    parser.add_argument(
        "--mode", choices=(*MODES, "both"), default="overlap",
        help="run DFC alone, with next-layer H2D, or both (default: overlap)",
    )
    parser.add_argument(
        "--tokens-per-rank", nargs="+", type=int, default=(128,), metavar="N",
        help="input tokens per source rank before routing (default: 128)",
    )
    parser.add_argument(
        "--ep-load", nargs="+", choices=tuple(HOT_ROUTES), default=("balanced",),
        help="receive load on ranks 0-7 vs 8-15 (default: balanced)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="record measured iterations across all workloads in one CPU and NPU trace per rank",
    )
    parser.add_argument("--warmup", type=int, default=1, help="forwards per mode before profiling (default: 1)")
    parser.add_argument("--iterations", type=int, default=1, help="forwards per mode after warmup (default: 1)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/glm52-dfc"))
    parser.add_argument("--native-extension-dir", default=os.environ.get("SGLANG_MOE_NATIVE_EXTENSION_DIR"),
                        help="directory containing the deployed _sglang_moe_acl extension")
    opts = parser.parse_args(argv)
    if min(opts.tokens_per_rank) < 1 or opts.iterations < 1:
        parser.error("--tokens-per-rank and --iterations must be positive")
    if opts.warmup < 0:
        parser.error("--warmup must be nonnegative")
    opts.modes = MODES if opts.mode == "both" else (opts.mode,)
    opts.tokens_per_rank = tuple(dict.fromkeys(opts.tokens_per_rank))
    opts.ep_load = tuple(dict.fromkeys(opts.ep_load))
    return opts


if __name__ == "__main__":
    # Prefer the adjacent checkout when invoked without an editable installation.
    for parent in Path(__file__).resolve().parents:
        if (parent / "sglang/srt/offloading/moe").is_dir():
            sys.path.insert(0, str(parent))
            break
    run_ep16(parse_options())
