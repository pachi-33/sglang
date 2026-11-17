"""Single-NPU GLM-5.2 W8A8 GMM benchmark with a default 84 GB device budget.

Run this script directly in the Ascend torch_npu environment. --plan lists the
cases and memory estimates without allocating NPU tensors. --case accepts
repeatable shell patterns (quote them). The Markdown table, CSV, and raw JSON
are saved beside the script, including when cases fail or are skipped.

This is a per-projection benchmark with synthetic values at model dimensions.
Timing covers the GMM calls with weights and routing metadata already on NPU.
Split type1 measures the resident and cold calls for one projection; it
concatenates outputs only for correctness. Setup copies, dispatch, SwiGLU,
prefetch, and the rest of the FFN are outside the timed interval.
"""

import argparse
import csv
import ctypes
import gc
import importlib.util
import itertools
import json
import math
import random
import re
import statistics
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch

from test_gmm_group_list_types import grouped_matmul

GB = 10**9
MAX_DEVICE_GB = 84
RUNTIME_RESERVE_BYTES = 8 * GB
CASE_WORKSPACE_BYTES = 4 * GB
# Keep in sync with offloading/moe/npu/options.py.
MAX_PHYSICAL_WEIGHT_GROUPS = 1024
FRACTAL_NZ = 29
MODES = ("type1", "type2", "split_type1")
RTOL, ATOL = 0.008, 0.001


@dataclass(frozen=True)
class Projection:
    name: str
    k: int
    n: int

    @property
    def slot_bytes(self):
        return self.k * self.n


# GLM-5.2: hidden_size=6144, moe_intermediate_size=2048, MoE TP=1.
PROJECTIONS = (Projection("w13", 6144, 4096), Projection("w2", 2048, 6144))


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    capacity: int
    slots: tuple[int, ...]
    counts: tuple[int, ...]
    routing: str
    resident: int = 0

    @property
    def tokens(self):
        return sum(self.counts)

    @property
    def active(self):
        return sum(count > 0 for count in self.counts)

    @property
    def modes(self):
        modes = MODES[:2] if self.capacity <= MAX_PHYSICAL_WEIGHT_GROUPS else ()
        return (*modes, "split_type1") if self.resident else modes


def balanced_counts(experts, tokens):
    count, remainder = divmod(tokens, experts)
    return tuple(count + (index < remainder) for index in range(experts))


def scenario(name, family, capacity, slots, tokens, routing="uniform", resident=0):
    slots = tuple(slots)
    experts = len(slots)
    if routing == "uniform":
        counts = balanced_counts(experts, tokens)
    elif routing == "skew_80_20":
        hot = max(1, round(experts / 5))
        # Reserve one token per expert so only the distribution changes.
        remaining = tokens - experts
        more = remaining * 4 // 5
        counts = tuple(
            count + 1
            for count in (
                *balanced_counts(hot, more),
                *balanced_counts(experts - hot, remaining - more),
            )
        )
    elif routing == "resident_only":
        counts = balanced_counts(resident, tokens) + (0,) * (experts - resident)
    elif routing == "cold_only":
        counts = (0,) * resident + balanced_counts(experts - resident, tokens)
    else:
        raise ValueError(f"Unknown routing: {routing}")
    result = Scenario(name, family, capacity, slots, counts, routing, resident)
    if (
        sorted(set(slots)) != list(slots)
        or slots[0] < 0
        or slots[-1] >= capacity
        or min(counts) < 0
        or result.tokens != tokens
        or tokens <= 0
        or not result.modes
    ):
        raise ValueError(f"Invalid scenario: {name}")
    return result


def prefill_slots(layers, experts, resident, layer):
    """Use the same two spans as AscendPrefillWeights.weights()."""
    cold = experts - resident
    capacity = layers * resident + 2 * cold
    base = layer * resident
    cold_base = layers * resident + (layer % 2) * cold
    return capacity, (
        *range(base, base + resident),
        *range(cold_base, cold_base + cold),
    )


def build_suite(seed):
    # Fix capacity, token rows and active experts while changing slot layout.
    two_blocks = (*range(4), *range(60, 64))
    layouts = {
        "contiguous_head": range(8),
        "contiguous_tail": range(56, 64),
        "stride2": range(0, 16, 2),
        "stride4": range(0, 32, 4),
        "stride8": range(0, 64, 8),
        "random_a": sorted(random.Random(seed).sample(range(64), 8)),
        "random_b": sorted(random.Random(seed + 1).sample(range(64), 8)),
        "two_blocks": two_blocks,
    }
    cases = [
        scenario(name, "layout", 64, slots, 1024) for name, slots in layouts.items()
    ]
    cases.append(scenario("dense_all_active", "layout", 8, range(8), 1024))
    cases.append(scenario("single_token_last_slot", "edge", 64, (63,), 1))
    cases.extend(
        scenario(f"rows_{rows}", "tokens", 64, two_blocks, rows)
        for rows in (128, 4096, 16384, 32768)
    )
    cases.append(scenario("skew_80_20", "routing", 64, two_blocks, 1024, "skew_80_20"))
    cases.extend(
        scenario(f"active_{experts}", "experts", 128, range(0, experts * 2, 2), 1024)
        for experts in (1, 8, 16, 32, 64)
    )
    # Match stride2 above while changing just the allocated pool capacity.
    cases.extend(
        scenario(f"pool_{capacity}", "pool", capacity, range(0, 16, 2), 1024)
        for capacity in (16, 128, 256, 512, 1024)
    )

    # GLM-5.2 has 256 routed experts, 75 routed layers across the main model.
    # These stage geometries match the EP4, EP8, and EP16 prefill launchers.
    for ep, layers, resident in ((4, 24, 38), (8, 24, 8), (16, 75, 4)):
        experts = 256 // ep
        # Same workload, different layer offsets and cold bank parity.
        for layer in (0, layers - 1 if layers % 2 == 0 else layers - 2):
            capacity, slots = prefill_slots(layers, experts, resident, layer)
            cases.append(
                scenario(
                    f"prefill_ep{ep}_layer{layer}",
                    "prefill",
                    capacity,
                    slots,
                    4096,
                    resident=resident,
                )
            )
    capacity, slots = prefill_slots(24, 64, 38, 0)
    cases.append(
        scenario(
            "prefill_ep4_large_skew",
            "prefill",
            capacity,
            slots,
            32768,
            "skew_80_20",
            38,
        )
    )
    cases.extend(
        scenario(
            f"prefill_ep4_{route}", "routing", capacity, slots, 4096, route, resident=38
        )
        for route in ("resident_only", "cold_only")
    )

    for capacity in (1023, 1024, 1025):
        # Place the cold span at the end to hit the highest physical slot.
        slots = (*range(32), *range(capacity - 32, capacity))
        cases.append(
            scenario(
                f"boundary_{capacity}", "boundary", capacity, slots, 4096, resident=32
            )
        )
    cases.extend(
        scenario(f"split_{route}", "routing", 1025, slots, 4096, route, resident=32)
        for route in ("resident_only", "cold_only")
    )
    return cases


def group_lists(case):
    counts = torch.zeros(case.capacity, dtype=torch.int64)
    counts[list(case.slots)] = torch.tensor(case.counts, dtype=torch.int64)
    pairs = torch.zeros((case.capacity, 2), dtype=torch.int64)
    pairs[: case.active] = torch.tensor(
        [(slot, count) for slot, count in zip(case.slots, case.counts) if count],
        dtype=torch.int64,
    )
    return counts, pairs


def estimate_case_bytes(projection, case, timing, iterations):
    # Reserve space for all captured outputs even when the graph pool reuses them.
    output_copies = 3
    if timing == "graph":
        output_copies += len(case.modes) * (iterations + 1)
    weights_and_metadata = case.capacity * (
        projection.slot_bytes + 2 * projection.n + 64
    )
    activations = case.tokens * (projection.k + 4 + 2 * projection.n * output_copies)
    return weights_and_metadata + activations + CASE_WORKSPACE_BYTES


class InsufficientMemory(Exception):
    """A case was not started because its estimated allocation cannot fit."""


class MemoryLimitExceeded(RuntimeError):
    """The observed usage exceeded the configured limit."""


class DeviceBudget:
    def __init__(self, npu, device, cap_bytes):
        self.npu = npu
        self.device = device
        free, self.total = npu.mem_get_info(device)
        self.limit = min(cap_bytes, self.total)
        reserved = npu.memory_reserved(device)
        outside_allocator = max(0, self.total - free - reserved)
        self.allocator_limit = self.limit - outside_allocator - RUNTIME_RESERVE_BYTES
        if self.allocator_limit <= reserved:
            raise InsufficientMemory(
                "Not enough free NPU memory to keep the 8 GB reserve"
            )
        npu.set_per_process_memory_fraction(self.allocator_limit / self.total, device)

    def admit(self, nbytes):
        free, total = self.npu.mem_get_info(self.device)
        reserved = self.npu.memory_reserved(self.device)
        room = min(
            self.allocator_limit - reserved,
            self.limit - (total - free) - RUNTIME_RESERVE_BYTES,
        )
        if nbytes > room:
            raise InsufficientMemory(
                f"Case needs {nbytes / GB:.2f} GB including workspace, "
                f"available budget {max(0, room) / GB:.2f} GB"
            )

    def observe(self, metrics):
        self.npu.synchronize()
        free, total = self.npu.mem_get_info(self.device)
        current = {
            "peak_allocated_bytes": self.npu.max_memory_allocated(self.device),
            "peak_reserved_bytes": self.npu.max_memory_reserved(self.device),
            "sampled_device_used_bytes": total - free,
        }
        for key, value in current.items():
            metrics[key] = max(metrics.get(key, 0), int(value))
        if (
            current["peak_reserved_bytes"] > self.allocator_limit
            or current["sampled_device_used_bytes"] > self.limit
        ):
            raise MemoryLimitExceeded("NPU memory cap exceeded; stopping the suite")


class NzStorage:
    """Use bounded on-device copies of identical expert extents for pool setup."""

    def __init__(self, torch_npu, device):
        self.torch_npu = torch_npu
        self.device = torch.device(device)
        acl = ctypes.CDLL("libascendcl.so")
        self.memcpy = acl.aclrtMemcpy
        self.memcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.memcpy.restype = ctypes.c_int
        self.memset = acl.aclrtMemset
        self.memset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_size_t,
        ]
        self.memset.restype = ctypes.c_int

    def validate(self, tensor, projection, slots):
        if (
            tuple(tensor.shape) != (slots, projection.k, projection.n)
            or tensor.dtype != torch.int8
            or tensor.device != self.device
            or int(self.torch_npu.get_npu_format(tensor)) != FRACTAL_NZ
            or tensor.storage_offset() != 0
            or tensor.untyped_storage().nbytes() != slots * projection.slot_bytes
        ):
            raise RuntimeError("Expected a fixed-stride, aligned INT8 FRACTAL_NZ pool")

    def allocate(self, projection, slots):
        weight = torch.ops.npu.empty_with_format(
            [slots, projection.k, projection.n],
            dtype=torch.int8,
            device=self.device,
            acl_format=FRACTAL_NZ,
        )
        self.validate(weight, projection, slots)
        return weight

    def zero(self, weight):
        storage = weight.untyped_storage()
        size, base = storage.nbytes(), storage.data_ptr()
        chunk = 1 << 30
        # ACL memset is synchronous. Bound each call to avoid SDK size limits.
        for start in range(0, size, chunk):
            count = min(chunk, size - start)
            ret = self.memset(base + start, count, 0, count)
            if ret != 0:
                raise RuntimeError(f"aclrtMemset failed with status {ret}")

    def copy_slot(self, dst, dst_slot, src, src_slot, slot_bytes):
        addresses = []
        for tensor, slot in ((dst, dst_slot), (src, src_slot)):
            storage = tensor.untyped_storage()
            if (
                tensor.storage_offset() != 0
                or not 0 <= slot < tensor.shape[0]
                or (slot + 1) * slot_bytes > storage.nbytes()
            ):
                raise RuntimeError("NZ expert copy would exceed the allocated tensor")
            addresses.append(storage.data_ptr() + slot * slot_bytes)
        acl_memcpy_device_to_device = 3
        ret = self.memcpy(
            addresses[0],
            slot_bytes,
            addresses[1],
            slot_bytes,
            acl_memcpy_device_to_device,
        )
        if ret != 0:
            raise RuntimeError(f"aclrtMemcpy failed with status {ret}")

    def view(self, pool, start, count, slot_bytes):
        view = pool.narrow(0, start, count)
        if (
            view.untyped_storage().data_ptr() != pool.untyped_storage().data_ptr()
            or view.data_ptr() != pool.data_ptr() + start * slot_bytes
            or int(self.torch_npu.get_npu_format(view)) != FRACTAL_NZ
        ):
            raise RuntimeError(
                "Split type1 NPU view does not preserve the NZ expert slots"
            )
        return view


@dataclass
class SourceExperts:
    weight: torch.Tensor
    scale: torch.Tensor
    sampled_weight: torch.Tensor
    columns: torch.Tensor


def build_sources(storage, projection, experts, seed):
    """Keep one compact source bank, converting one expert at a time."""
    source = storage.allocate(projection, experts)
    columns = torch.linspace(0, projection.n - 1, steps=16).to(torch.int64)
    scale_generator = torch.Generator().manual_seed(seed)
    weight_generator = torch.Generator().manual_seed(seed + 2)
    scales = torch.pow(
        2.0, torch.randint(-8, -4, (experts, projection.n), generator=scale_generator)
    ).to(torch.bfloat16)
    samples = []
    for expert in range(experts):
        host = torch.randint(
            -8,
            8,
            (1, projection.k, projection.n),
            dtype=torch.int8,
            generator=weight_generator,
        )
        samples.append(host[0].index_select(1, columns).float())
        nd = host.to(storage.device)
        nz = storage.torch_npu.npu_format_cast(nd, FRACTAL_NZ)
        storage.validate(nz, projection, 1)
        # Synchronous raw copies do not wait for queued torch_npu operations.
        torch.npu.synchronize()
        storage.copy_slot(source, expert, nz, 0, projection.slot_bytes)
        del nz, nd, host
    return SourceExperts(source, scales, torch.stack(samples), columns)


def create_inputs(projection, case, sources, seed):
    # Shapes with the same row count use the same inputs across all layouts.
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randint(
        -8, 8, (case.tokens, projection.k), dtype=torch.int8, generator=generator
    )
    token_scale = torch.pow(
        2.0, torch.randint(-8, -4, (case.tokens,), generator=generator)
    ).to(torch.float32)
    rows, results = [], []
    start = 0
    for expert, count in enumerate(case.counts):
        if count:
            selected = [start + offset for offset in sorted({0, count // 2, count - 1})]
            ref = x[selected].float() @ sources.sampled_weight[expert]
            ref *= sources.scale[expert, sources.columns].float()
            ref *= token_scale[selected, None]
            rows.extend(selected)
            results.append(ref.to(torch.bfloat16))
        start += count
    return x, token_scale, torch.tensor(rows, dtype=torch.int64), torch.cat(results)


def make_calls(mode, case, projection, storage, x, token_scale, weight, scale, groups):
    if mode == "type1":
        return (partial(grouped_matmul, x, weight, scale, token_scale, groups[0], 1),)
    if mode == "type2":
        return (partial(grouped_matmul, x, weight, scale, token_scale, groups[1], 2),)

    calls = []
    row = 0
    for start, stop in ((0, case.resident), (case.resident, len(case.slots))):
        begin = case.slots[start]
        count = stop - start
        if case.slots[start:stop] != tuple(range(begin, begin + count)):
            raise ValueError("Each split type1 span must be contiguous")
        rows = sum(case.counts[start:stop])
        if rows:
            calls.append(
                partial(
                    grouped_matmul,
                    x[row : row + rows],
                    storage.view(weight, begin, count, projection.slot_bytes),
                    scale.narrow(0, begin, count),
                    token_scale[row : row + rows],
                    groups[0].narrow(0, begin, count),
                    1,
                )
            )
        row += rows
    return tuple(calls)


def invoke(calls):
    return tuple(call() for call in calls)


def to_host(outputs):
    return (
        outputs[0].cpu() if len(outputs) == 1 else torch.cat([t.cpu() for t in outputs])
    )


def assert_outputs_close(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError(f"Unexpected output: {actual.shape}, {actual.dtype}")
    # Bound temporary memory for the full CPU comparison.
    for index in range(0, actual.shape[0], 256):
        torch.testing.assert_close(
            actual[index : index + 256],
            reference[index : index + 256],
            rtol=RTOL,
            atol=ATOL,
        )


def capture(calls, iterations, reference):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        for _ in range(iterations):
            outputs = invoke(calls)
    graph.replay()
    torch.npu.synchronize()
    assert_outputs_close(to_host(outputs), reference)
    return graph, outputs  # Retain graph outputs through the final replay.


def timing_stats(samples, launches, projection, tokens):
    ordered = sorted(samples)
    median = statistics.median(ordered)
    return {
        "status": "passed",
        "median_us": median,
        "p95_round_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "gmm_launches": launches,
        "routed_rows_per_second": tokens * 1e6 / median,
        "effective_tops": 2 * tokens * projection.k * projection.n / (median * 1e6),
        "round_mean_us": samples,
    }


def measure(operations, graphs, settings):
    samples = {mode: [] for mode in operations}
    orders = tuple(itertools.permutations(operations))
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    for index in range(settings.rounds):
        for mode in orders[index % len(orders)]:
            torch.npu.synchronize()
            start.record()
            if settings.timing == "graph":
                graphs[mode][0].replay()
            else:
                for _ in range(settings.iterations):
                    invoke(operations[mode])
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / settings.iterations
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError(f"Invalid NPU event interval for {mode}: {us}")
            samples[mode].append(us)
    return samples


def run_case(projection, case, sources, storage, budget, settings, row):
    memory = row["memory"]
    budget.admit(
        estimate_case_bytes(projection, case, settings.timing, settings.iterations)
    )
    torch.npu.reset_peak_memory_stats(storage.device)
    device = storage.device
    weight = storage.allocate(projection, case.capacity)
    torch.npu.synchronize()
    storage.zero(weight)
    for logical, slot in enumerate(case.slots):
        storage.copy_slot(weight, slot, sources.weight, logical, projection.slot_bytes)
    scale = torch.zeros((case.capacity, projection.n), dtype=torch.bfloat16)
    scale[list(case.slots)] = sources.scale[: len(case.slots)]
    scale = scale.to(device)
    x, token_scale, rows, cpu_reference = create_inputs(
        projection, case, sources, settings.seed
    )
    x, token_scale = x.to(device), token_scale.to(device)
    groups = tuple(value.to(device) for value in group_lists(case))
    budget.observe(memory)

    operations, graphs = {}, {}
    reference = None
    for mode in case.modes:
        try:
            calls = make_calls(
                mode, case, projection, storage, x, token_scale, weight, scale, groups
            )
            actual = to_host(invoke(calls))
            if (
                actual.shape != (case.tokens, projection.n)
                or actual.dtype != torch.bfloat16
            ):
                raise AssertionError(
                    f"Unexpected output: {actual.shape}, {actual.dtype}"
                )
            assert_outputs_close(actual[rows[:, None], sources.columns], cpu_reference)
            if reference is not None:
                assert_outputs_close(actual, reference)
            for _ in range(settings.warmup):
                invoke(calls)
            torch.npu.synchronize()
            if settings.timing == "graph":
                graphs[mode] = capture(calls, settings.iterations, actual)
            operations[mode] = calls
            if reference is None:
                reference = actual
        except (AssertionError, RuntimeError) as error:
            row["modes"][mode] = {"status": "error", "detail": str(error)}
        budget.observe(memory)

    if operations:
        samples = measure(operations, graphs, settings)
        for mode, values in samples.items():
            row["modes"][mode] = timing_stats(
                values, len(operations[mode]), projection, case.tokens
            )
    budget.observe(memory)
    row["status"] = "passed" if len(operations) == len(case.modes) else "error"


def cleanup():
    # Release caches between cases; keep only the small source bank alive.
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()


def setup_npu(settings, environment):
    if importlib.util.find_spec("torch_npu") is None:
        return None
    import torch_npu

    if not torch.npu.is_available():
        return None
    device = torch.device(settings.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    cleanup()
    budget = DeviceBudget(torch.npu, device, int(settings.memory_gb * GB))
    environment.update(
        npu=torch.npu.get_device_name(),
        torch_npu=torch_npu.__version__,
        device_total_bytes=budget.total,
        device_limit_bytes=budget.limit,
        allocator_limit_bytes=budget.allocator_limit,
    )
    return NzStorage(torch_npu, device), budget


def execute(groups, settings, report):
    runtime = setup_npu(settings, report["environment"])
    if runtime is None:
        report.update(
            status="skipped",
            detail="torch_npu and an available Ascend NPU are required",
        )
        for row in report["cases"]:
            row["status"] = "skipped"
        return 2
    storage, budget = runtime
    with torch.inference_mode():
        for projection, cases_and_rows in groups:
            source_count = max(len(case.slots) for case, _ in cases_and_rows)
            try:
                budget.admit((source_count + 2) * projection.slot_bytes)
            except InsufficientMemory as error:
                for _, row in cases_and_rows:
                    row.update(status="skipped_memory", detail=str(error))
                continue

            torch.npu.reset_peak_memory_stats(storage.device)
            sources = build_sources(storage, projection, source_count, settings.seed)
            budget.observe(report["source_setup_memory"])
            cleanup()
            for case, row in cases_and_rows:
                print(
                    f"{projection.name} {case.name} ({case.capacity} slots)", flush=True
                )
                try:
                    run_case(projection, case, sources, storage, budget, settings, row)
                except InsufficientMemory as error:
                    row.update(status="skipped_memory", detail=str(error))
                except MemoryLimitExceeded as error:
                    row.update(status="error", detail=str(error))
                    raise
                except (AssertionError, RuntimeError) as error:
                    row.update(status="error", detail=str(error))
                    budget.observe(row["memory"])
                cleanup()
                print(f"  {row['status']}", flush=True)
            del sources
            cleanup()
    statuses = {row["status"] for row in report["cases"]}
    report["status"] = "passed" if statuses == {"passed"} else "incomplete"
    return 0 if report["status"] == "passed" else 1


def plan_cases(projections, cases, settings):
    grouped, records = [], []
    for projection in projections:
        source_bytes = max(len(case.slots) for case in cases) * projection.slot_bytes
        work = []
        for case in cases:
            estimate = source_bytes + estimate_case_bytes(
                projection, case, settings.timing, settings.iterations
            )
            row = {
                "projection": projection.name,
                "k": projection.k,
                "n": projection.n,
                "case": case.name,
                "family": case.family,
                "routing": case.routing,
                "pool_slots": case.capacity,
                "active_experts": case.active,
                "tokens": case.tokens,
                "resident_experts": case.resident,
                "slots": case.slots,
                "counts": case.counts,
                "estimated_process_bytes": estimate,
                "status": "pending",
                "memory": {},
                "modes": {mode: {"status": "pending"} for mode in case.modes},
            }
            work.append((case, row))
            records.append(row)
        grouped.append((projection, work))
    return grouped, records


def summarize(row):
    fields = (
        "projection",
        "case",
        "family",
        "routing",
        "k",
        "n",
        "pool_slots",
        "active_experts",
        "tokens",
        "resident_experts",
        "status",
    )
    summary = {key: row[key] for key in fields}
    for mode in MODES:
        result = row["modes"].get(mode, {})
        summary[f"{mode}_status"] = result.get("status")
        for key in ("median_us", "p95_round_us", "gmm_launches"):
            summary[f"{mode}_{key}"] = result.get(key)
    first, second = summary["type1_median_us"], summary["type2_median_us"]
    split = summary["split_type1_median_us"]
    summary["type2_over_type1"] = second / first if first and second else None
    summary["split_over_type2"] = split / second if split and second else None
    summary["estimated_process_GB"] = row["estimated_process_bytes"] / GB
    for source, dest in (
        ("peak_allocated_bytes", "peak_allocated_GB"),
        ("peak_reserved_bytes", "peak_reserved_GB"),
        ("sampled_device_used_bytes", "sampled_device_used_GB"),
    ):
        value = row["memory"].get(source)
        summary[dest] = value / GB if value is not None else None
    details = [row.get("detail", "")]
    details.extend(
        f"{mode}: {result['detail']}"
        for mode, result in row["modes"].items()
        if "detail" in result
    )
    summary["detail"] = "\n".join(detail for detail in details if detail)
    return summary


def format_number(value, decimals=3):
    return "—" if value is None else f"{value:.{decimals}f}"


def render_report(report, summaries, csv_name, json_name):
    settings, environment = report["settings"], report["environment"]
    lines = [
        "# GLM-5.2 W8A8 GMM 实验汇总",
        "",
        f"- 运行状态：{report['status']}",
        f"- 开始：{report['started_at']}，结束：{report['finished_at']}",
        f"- 设备：{settings['device']} {environment.get('npu', '')}",
        f"- 软件：PyTorch {environment['torch']}，torch_npu {environment.get('torch_npu', '—')}",
        f"- 计时：{settings['timing']}，预热 {settings['warmup']} 次，"
        f"每轮 {settings['iterations']} 次，共 {settings['rounds']} 轮",
        f"- 设备预算：{settings['memory_gb']:g} GB，"
        f"给运行时预留 {RUNTIME_RESERVE_BYTES / GB:g} GB",
        f"- [CSV 汇总]({csv_name}) · [JSON 原始数据]({json_name})",
        "",
        "type1 和 type2 在同一行复用同一个权重池和输入。各布局使用同一套合成专家权重。",
        "耗时按每轮均值统计，单位为 μs。split type1 一次计时包含两个专家块实际发出的 GMM，"
        "空块跳过。图捕获、准备数据、预取与激活不计时。",
        "显存峰值是 torch_npu 分配器的 reserved 峰值。"
        "JSON 另列采样到的整卡占用，包含采样时的 CANN 和其他进程占用。",
        "",
        "| 投影 | 场景 | 槽位 | 活跃专家 | Token 行数 | type1 中位 μs | "
        "type2 中位 μs | type2/type1 | split type1 中位 μs | 预估 GB | 峰值 GB | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['projection']} | {row['case']} | {row['pool_slots']} | "
            f"{row['active_experts']} | {row['tokens']} | {format_number(row['type1_median_us'])} | "
            f"{format_number(row['type2_median_us'])} | {format_number(row['type2_over_type1'], 4)} | "
            f"{format_number(row['split_type1_median_us'])} | "
            f"{format_number(row['estimated_process_GB'], 2)} | "
            f"{format_number(row['peak_reserved_GB'], 2)} | {row['status']} |"
        )
    details = [row for row in summaries if row["detail"]]
    if details or report.get("detail"):
        lines.extend(("", "## 异常与跳过原因", ""))
        if report.get("detail"):
            lines.append(f"- {report['detail'].splitlines()[-1]}")
        for row in details:
            lines.append(
                f"- {row['projection']} / {row['case']}。"
                f"{' '.join(row['detail'].splitlines())[:500]}"
            )
    return "\n".join(lines) + "\n"


def save_report(report):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"gmm_glm52_suite_{stamp}_{uuid4().hex[:8]}"
    base = Path(__file__).resolve().with_name(stem)
    csv_path, json_path, md_path = (
        base.with_suffix(ext) for ext in (".csv", ".json", ".md")
    )
    summaries = [summarize(row) for row in report["cases"]]
    with csv_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with json_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    with md_path.open("x", encoding="utf-8") as output:
        output.write(render_report(report, summaries, csv_path.name, json_path.name))
    for path in (md_path, csv_path, json_path):
        print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--memory-gb", type=float, default=MAX_DEVICE_GB)
    parser.add_argument("--projection", choices=("both", "w13", "w2"), default="both")
    parser.add_argument("--case", action="append", default=[], help="Case name or glob")
    parser.add_argument("--timing", choices=("eager", "graph"), default="eager")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--plan", action="store_true", help="List cases without using NPU"
    )
    settings = parser.parse_args()
    if not re.fullmatch(r"npu:\d+", settings.device):
        parser.error("--device must select exactly one NPU, for example npu:0")
    if (
        not math.isfinite(settings.memory_gb)
        or not 0 < settings.memory_gb <= MAX_DEVICE_GB
    ):
        parser.error("--memory-gb must be greater than 0 and at most 84 (decimal GB)")
    if min(settings.warmup, settings.iterations, settings.rounds) < 1:
        parser.error("warmup, iterations and rounds must be positive")
    if not 0 <= settings.seed <= 2**63 - 3:
        parser.error("--seed must be between 0 and 2**63 - 3")
    cases = [
        case
        for case in build_suite(settings.seed)
        if not settings.case
        or any(fnmatchcase(case.name, pattern) for pattern in settings.case)
    ]
    if not cases:
        parser.error("No cases match --case")
    projections = [p for p in PROJECTIONS if settings.projection in ("both", p.name)]
    groups, records = plan_cases(projections, cases, settings)
    if settings.plan:
        print("proj case                        E     S      M  est.GB  modes")
        for row in records:
            fits = (
                row["estimated_process_bytes"] + RUNTIME_RESERVE_BYTES
                <= settings.memory_gb * GB
            )
            suffix = "" if fits else " [over budget]"
            print(
                f"{row['projection']:<4} {row['case']:<25} {row['active_experts']:>3} "
                f"{row['pool_slots']:>5} {row['tokens']:>6} "
                f"{row['estimated_process_bytes'] / GB:>7.2f}  {', '.join(row['modes'])}{suffix}"
            )
        largest = max(row["estimated_process_bytes"] for row in records)
        print(
            f"{len(records)} scenarios; largest estimate {largest / GB:.2f} GB "
            f"+ {RUNTIME_RESERVE_BYTES / GB:g} GB runtime reserve "
            f"(device budget {settings.memory_gb:g} GB)"
        )
        return 0

    report = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "error",
        "settings": vars(settings),
        "environment": {"torch": str(torch.__version__)},
        "source_setup_memory": {},
        "cases": records,
    }
    try:
        return execute(groups, settings, report)
    except InsufficientMemory as error:
        report.update(status="skipped_memory", detail=str(error))
        for row in records:
            if row["status"] == "pending":
                row["status"] = "skipped_memory"
        return 2
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
            detail=traceback.format_exc(),
        )
        print(report["detail"], file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save_report(report)


if __name__ == "__main__":
    sys.exit(main())
