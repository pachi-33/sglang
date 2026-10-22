"""One-command CPU contracts and real NPU LFU/statistics graph replay.

Run without arguments in an NPU environment with PyTorch and torch_npu. Reports
go to ./moe-cache-validation. Use --cpu-only explicitly for a CPU-only check.
This does not validate native H2D/D2D, FP4 kernels, or multi-rank serving.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPU_MODULES = (
    "test_cache_integration",
    "test_cache_metadata",
    "test_cache_model_binding",
    "test_cache_runtime",
    "test_moe_diagnostics",
    "test_moe_framework",
    "test_moe_graph_leases",
    "test_native_periodic_cache",
    "test_npu_backend",
    "test_npu_moe_graph_lifetime",
    "test_periodic_lfu",
    "test_tensor_residency",
)


def bind_source_tree():
    # Import this checkout without initializing unrelated serving dependencies.
    package = types.ModuleType("sglang")
    package.__path__ = [str(ROOT / "python/sglang")]
    sys.modules["sglang"] = package
    sys.path.insert(0, str(ROOT / "test/registered/unit/offloading"))


class NpuLFUReplay(unittest.TestCase):
    def runTest(self):
        import torch
        import torch_npu  # noqa: F401

        from sglang.srt.offloading.moe.cache_metadata import CacheRouteStats
        from sglang.srt.offloading.moe.config import ExpertCatalog, ExpertLayerSpec
        from sglang.srt.offloading.moe.periodic_lfu import (
            PeriodicLFUConfig,
            PeriodicLFUPolicy,
        )

        self.assertTrue(torch.npu.is_available(), "--npu requires an Ascend device")
        index = int(os.environ.get("MOE_TEST_NPU_DEVICE", "0"))
        torch.npu.set_device(index)
        policy = PeriodicLFUPolicy(
            PeriodicLFUConfig(None),
            ExpertCatalog(tuple(ExpertLayerSpec(i, 4, 1, 0) for i in (3, 4))),
        )
        initial_map = [0, -1, 1, -1, 2, -1, -1, 3]

        def allocate(device):
            return types.SimpleNamespace(
                state=policy.allocate_state(8, device),
                slots=torch.tensor(initial_map, device=device),
                counts=torch.zeros(8, dtype=torch.int64, device=device),
                enabled=torch.tensor(False, device=device),
                target=torch.empty(4, dtype=torch.int64, device=device),
                stats=CacheRouteStats.allocate(device),
            )

        def step(cache):
            cache.stats.clear()
            cache.stats.record_counts(cache.counts, cache.slots >= 0, cache.enabled)
            policy.update_statistics(cache.state, cache.counts, cache.enabled)
            policy.select_hot_set(cache.state, cache.slots, cache.target)
            policy.finish_step(cache.state)

        cpu, npu = allocate("cpu"), allocate(f"npu:{index}")

        def addresses():
            return [
                tensor.data_ptr()
                for tensor in (
                    npu.target,
                    *vars(npu.state).values(),
                    *vars(npu.stats).values(),
                )
            ]

        initial_addresses = addresses()
        stream = torch.npu.Stream()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            for _ in range(2):
                step(npu)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            step(npu)
        torch.npu.synchronize()
        self.assertEqual(npu.state.step_mod.item(), 0)
        self.assertEqual(npu.state.usage.sum().item(), 0)

        # Change live inputs between replays. The disabled fifth call models
        # the statistics gate; it must neither advance nor decay the cycle.
        for i in range(9):
            counts = torch.tensor(
                [0, 3, 0, 0, 0, 2, 0, 0] if i < 4 else [20, 0, 0, 0, 20, 0, 0, 0]
            )
            cpu.counts.copy_(counts)
            cpu.enabled.fill_(i != 4)
            npu.counts.copy_(cpu.counts)
            npu.enabled.copy_(cpu.enabled)
            npu.slots.copy_(cpu.slots)
            graph.replay()
            torch.npu.synchronize()
            step(cpu)
            for name in vars(cpu.state):
                torch.testing.assert_close(
                    getattr(npu.state, name).cpu(), getattr(cpu.state, name)
                )
            for name in vars(cpu.stats):
                torch.testing.assert_close(
                    getattr(npu.stats, name).cpu(), getattr(cpu.stats, name)
                )
            torch.testing.assert_close(npu.target.cpu(), cpu.target)
            expected = [0, 2, 4, 7] if i < 3 else [1, 2, 5, 7]
            self.assertEqual(
                npu.target.cpu().tolist(), [0, 1, 4, 5] if i == 8 else expected
            )
            # Simulate the published map for the next replay. This test does
            # not perform weight copies or exercise the residency publisher.
            cpu.slots.fill_(-1)
            cpu.slots[cpu.target] = torch.arange(4)

        # Exercise the new K-entry construction with holes/permuted slots too.
        cpu.slots.copy_(torch.tensor([3, -1, 0, -1, -1, -1, -1, 1]))
        npu.slots.copy_(cpu.slots)
        npu.enabled.fill_(False)
        graph.replay()
        torch.npu.synchronize()
        self.assertEqual(npu.target.cpu().tolist(), [0, 2, 7, -1])
        self.assertEqual(initial_addresses, addresses())


def worker(stage, output):
    bind_source_tree()
    report = {"stage": stage, "status": "failed"}
    try:
        suite = (
            unittest.defaultTestLoader.loadTestsFromNames(CPU_MODULES)
            if stage == "cpu"
            else unittest.TestSuite([NpuLFUReplay()])
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
        report.update(
            status="passed" if passed else "failed",
            tests=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            skipped=len(result.skipped),
        )
    except Exception:
        traceback.print_exc()
    (output / f"{stage}.json").write_text(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--npu", dest="npu", action="store_true", help="Require NPU replay (default)"
    )
    mode.add_argument(
        "--cpu-only", dest="npu", action="store_false", help="Run CPU contracts only"
    )
    parser.set_defaults(npu=True)
    parser.add_argument("--output", type=Path, default=Path("moe-cache-validation"))
    parser.add_argument(
        "--stage", choices=("cpu", "npu_lfu_graph"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage:
        return worker(args.stage, output)
    report = {
        "status": "passed_within_scope",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "stages": [],
        "not_covered": [
            "native H2D/D2D",
            "FP4 numerics",
            "multi-rank serving",
            "POSIX launcher",
        ],
    }
    if not args.npu:
        report["not_covered"].append("real NPU graph replay (use --npu)")
    for stage in ("cpu", "npu_lfu_graph") if args.npu else ("cpu",):
        stage_json = output / f"{stage}.json"
        stage_json.unlink(missing_ok=True)
        started = time.monotonic()
        with (output / f"{stage}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--stage",
                    stage,
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        detail = (
            json.loads(stage_json.read_text())
            if stage_json.exists()
            else {"stage": stage, "status": "failed"}
        )
        detail.update(
            exit_code=result.returncode, seconds=round(time.monotonic() - started, 2)
        )
        if result.returncode or detail["status"] != "passed":
            detail["status"] = report["status"] = "failed"
        report["stages"].append(detail)
        print(
            f"{stage}: {detail['status']} ({detail.get('tests', 0)} tests)", flush=True
        )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Report: {output / 'report.json'}")
    return int(report["status"] != "passed_within_scope")


if __name__ == "__main__":
    raise SystemExit(main())
