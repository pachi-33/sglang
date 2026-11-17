r"""Correctness and performance comparison of two Ascend W8A8 GMM layouts.

Run this file directly on an idle NPU; only torch and torch_npu are required::

    python test_gmm_group_list_types.py --experts 8 --tokens-per-expert 128 \
        --k 1024 --n 2048 --slot-stride 2

There are exactly two unittest cases:
  * type1_contiguous: active experts occupy consecutive physical slots.
  * type2_sparse_slots: the SAME experts occupy slots 0, stride, 2*stride, ...

"Sparse" describes expert-slot placement in a dense NZ tensor, not COO/CSR.
Both weights have shape [experts * slot_stride, K, N], identical dtype/format,
and the same data up to a permutation of the expert axis. The input, token
counts, scales, active expert count and arithmetic work are identical. Type 1
uses zero counts for its unused tail; type 2 uses (slot, count) pairs followed
by zero pairs, matching the offloader's fixed-pool metadata convention.

Direct execution runs correctness tests, then a paired benchmark. Default
graph timing captures --iterations GMM calls per graph and reports replay time
per GMM, reducing Python launch gaps. --timing eager measures a synchronized
event interval around the same number of ordinary calls, including any device
idle gaps caused by host dispatch. Neither mode times data preparation, H2D,
format conversion, routing, prefetch or graph capture. Weights are reused, so
these are steady-state measurements, not cold-cache or whole-FFN timings.

The result compares layout + group_list_type together, not the flag alone.
Every direct run saves a Markdown experiment report and a JSON data file next
to this script, including failed/skipped runs. Timestamped unique filenames
preserve earlier experiments, regardless of the current working directory.
Importing this file through unittest/pytest runs only the correctness cases.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import traceback
import unittest
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch


@dataclass(frozen=True)
class Config:
    experts: int = 8
    tokens_per_expert: int = 128
    k: int = 1024
    n: int = 2048
    slot_stride: int = 2
    device: str = "npu:0"
    seed: int = 1234
    warmup: int = 20
    iterations: int = 100
    rounds: int = 10
    timing: str = "graph"

    @property
    def pool_slots(self):
        return self.experts * self.slot_stride

    def validate(self):
        if self.experts < 2 or self.slot_stride < 2:
            raise ValueError("experts and slot-stride must be >= 2")
        if self.pool_slots > 1024:
            raise ValueError("experts * slot-stride must be <= 1024 (type 2 limit)")
        if min(self.k, self.n) <= 0 or self.k % 64 or self.n % 64:
            raise ValueError("K and N must be positive multiples of 64 (NZ alignment)")
        if min(self.tokens_per_expert, self.warmup, self.iterations, self.rounds) < 1:
            raise ValueError("tokens, warmup, iterations and rounds must be positive")
        if self.timing not in ("graph", "eager"):
            raise ValueError("timing must be graph or eager")


def build_host_inputs(config):
    """Build both layouts from one seeded data set; no NPU dependency here."""
    config.validate()
    e, s, m = config.experts, config.pool_slots, config.tokens_per_expert
    generator = torch.Generator().manual_seed(config.seed)
    x = torch.randint(-8, 8, (e * m, config.k), dtype=torch.int8, generator=generator)
    weight = torch.randint(
        -8, 8, (s, config.k, config.n), dtype=torch.int8, generator=generator
    )
    # Nonuniform, exactly representable scales exercise both scale lookups.
    scale = torch.pow(
        2.0, torch.randint(-8, -4, (s, config.n), generator=generator)
    ).to(torch.bfloat16)
    token_scale = torch.pow(
        2.0, torch.randint(-8, -4, (e * m,), generator=generator)
    ).to(torch.float32)

    sparse_slots = torch.arange(e, dtype=torch.int64) * config.slot_stride
    unused = torch.ones(s, dtype=torch.bool)
    unused[sparse_slots] = False
    permutation = torch.empty(s, dtype=torch.int64)
    permutation[sparse_slots] = torch.arange(e)
    permutation[unused] = torch.arange(e, s)
    # Permute unused experts too: both pools contain the same multiset of data.
    sparse_weight = weight.index_select(0, permutation)
    sparse_scale = scale.index_select(0, permutation)
    counts = torch.zeros(s, dtype=torch.int64)
    counts[:e] = m
    pairs = torch.zeros((s, 2), dtype=torch.int64)
    pairs[:e, 0] = sparse_slots
    pairs[:e, 1] = m

    # A CPU reference for first/middle/last rows of EVERY expert bounds the
    # CPU cost at large prefill sizes. The two NPU outputs are compared in full.
    local_rows = torch.tensor(sorted({0, m // 2, m - 1}), dtype=torch.int64)
    reference_rows, reference = [], []
    for expert in range(e):
        rows = local_rows + expert * m
        result = x[rows].float() @ weight[expert].float()
        result *= scale[expert].float()
        result *= token_scale[rows, None]
        reference_rows.append(rows)
        reference.append(result.to(torch.bfloat16))

    return {
        "x": x,
        "token_scale": token_scale,
        "cases": {
            1: (weight, scale, counts),
            2: (sparse_weight, sparse_scale, pairs),
        },
        "reference_rows": torch.cat(reference_rows),
        "reference": torch.cat(reference),
    }


class GmmComparison:
    def __init__(self, config):
        import torch_npu

        self.config = config
        self.torch_npu_version = torch_npu.__version__
        torch.npu.set_device(config.device)
        host = build_host_inputs(config)
        self.x = host["x"].to(config.device)
        self.token_scale = host["token_scale"].to(config.device)
        self.reference_rows = host["reference_rows"]
        self.reference = host["reference"]
        self.cases = {}
        self.outputs = {}
        for kind, (weight, scale, groups) in host["cases"].items():
            weight = torch_npu.npu_format_cast(weight.to(config.device), 29)
            if int(torch_npu.get_npu_format(weight)) != 29:
                raise RuntimeError("Both cases require FRACTAL_NZ (format 29)")
            self.cases[kind] = (
                weight,
                scale.to(config.device),
                groups.to(config.device),
            )
        torch.npu.synchronize()

    def run(self, kind):
        weight, scale, groups = self.cases[kind]
        return torch.ops.npu.npu_grouped_matmul(
            x=[self.x],
            weight=[weight],
            scale=[scale],
            per_token_scale=[self.token_scale],
            split_item=2,
            group_type=0,
            group_list_type=kind,
            group_list=groups,
            output_dtype=torch.bfloat16,
        )[0]

    def output(self, kind):
        if kind not in self.outputs:
            self.outputs[kind] = self.run(kind).cpu()
        return self.outputs[kind]

    def check(self, kind):
        actual = self.output(kind)
        expected_shape = (
            self.config.experts * self.config.tokens_per_expert,
            self.config.n,
        )
        if actual.shape != expected_shape or actual.dtype != torch.bfloat16:
            raise AssertionError(f"Unexpected output: {actual.shape}, {actual.dtype}")
        torch.testing.assert_close(
            actual[self.reference_rows], self.reference, rtol=0.008, atol=0.001
        )

    @torch.inference_mode()
    def benchmark(self):
        config = self.config
        names = {1: "type1_contiguous", 2: "type2_sparse_slots"}
        metadata = {
            "torch": torch.__version__,
            "torch_npu": self.torch_npu_version,
            "npu": torch.npu.get_device_name(),
            "config": asdict(config),
            "weight_shape_both": list(self.cases[1][0].shape),
            "input_shape_both": list(self.x.shape),
            "type1_active_slots": list(range(config.experts)),
            "type2_active_slots": [
                e * config.slot_stride for e in range(config.experts)
            ],
        }
        print(json.dumps(metadata), flush=True)
        for _ in range(config.warmup):
            for kind in (1, 2):
                self.run(kind)
        torch.npu.synchronize()

        graphs, graph_outputs = {}, {}
        if config.timing == "graph":
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            for kind in (1, 2):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
                    for _ in range(config.iterations):
                        output = self.run(kind)
                graphs[kind], graph_outputs[kind] = graph, output
                for _ in range(3):
                    graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(
                    output.cpu(), self.output(kind), rtol=0.008, atol=0.001
                )

        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
        samples = {1: [], 2: []}
        for round_index in range(config.rounds):
            # AB / BA ordering balances drift; default rounds is even.
            for kind in ((1, 2) if round_index % 2 == 0 else (2, 1)):
                torch.npu.synchronize()
                start.record()
                if config.timing == "graph":
                    graphs[kind].replay()
                else:
                    for _ in range(config.iterations):
                        self.run(kind)
                end.record()
                end.synchronize()
                us = start.elapsed_time(end) * 1000 / config.iterations
                if us <= 0:
                    raise RuntimeError("Nonpositive timing; increase --iterations")
                samples[kind].append(us)

        print(
            "case                    median_us     min_us     max_us    effective_TOPS"
        )
        operations = 2 * self.x.shape[0] * config.k * config.n
        case_results = {}
        for kind in (1, 2):
            times = samples[kind]
            median = statistics.median(times)
            case_results[names[kind]] = {
                "group_list_type": kind,
                "median_us": median,
                "min_us": min(times),
                "max_us": max(times),
                "effective_tops": operations / (median * 1e6),
                "round_mean_us": times,
            }
            print(
                f"{names[kind]:<23} {median:10.3f} {min(times):10.3f} "
                f"{max(times):10.3f} {operations / (median * 1e6):17.3f}"
            )
        ratio = statistics.median(samples[2]) / statistics.median(samples[1])
        print(f"type2/type1 latency ratio: {ratio:.4f} (> 1 means type2 is slower)")
        print(json.dumps({"round_mean_us": {names[k]: v for k, v in samples.items()}}))
        return {
            "metadata": metadata,
            "cases": case_results,
            "latency_ratio_type2_over_type1": ratio,
        }


class TestGmmGroupListTypes(unittest.TestCase):
    config = Config()
    comparison = None

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_npu") is None:
            raise unittest.SkipTest(
                "torch_npu is not installed; an Ascend NPU is required"
            )
        import torch_npu  # noqa: F401 -- registers torch.npu and the custom ops

        if not torch.npu.is_available():
            raise unittest.SkipTest("No Ascend NPU is available")
        cls.comparison = GmmComparison(cls.config)

    @torch.inference_mode()
    def test_type1_contiguous(self):
        self.comparison.check(1)

    @torch.inference_mode()
    def test_type2_sparse_slots(self):
        self.comparison.check(2)
        torch.testing.assert_close(
            self.comparison.output(2),
            self.comparison.output(1),
            rtol=0.008,
            atol=0.001,
        )


def render_report(report):
    """Render measured values only; skipped/failed runs have no speed conclusion."""
    config = report["config"]
    benchmark = report["benchmark"]
    status = {
        "passed": "完成",
        "failed": "正确性校验失败",
        "skipped": "已跳过",
        "error": "执行异常",
        "interrupted": "被中断",
    }[report["status"]]
    lines = [
        "# GMM type1 / type2 性能对照实验报告",
        "",
        f"- 实验状态：{status}",
        f"- 开始时间：{report['started_at']}",
        f"- 结束时间：{report['finished_at']}",
        f"- 原始数据：[{report['artifacts']['json']}]({report['artifacts']['json']})",
        "",
        "## 实验目的与控制变量",
        "",
        "比较相同计算规格下，连续专家槽位的 type1 与间隔专家槽位的 type2。"
        "两组使用同一批输入和逻辑专家权重，并保持 token 分布、权重池大小、"
        "量化参数、INT8 输入/权重、NZ 权重格式和 BF16 输出一致。",
        "",
        "| Case | 活跃专家物理槽位 | group_list |",
        "| --- | --- | --- |",
        "| type1_contiguous | 0, 1, …, E−1 | S 个 count，未使用尾部为 0 |",
        "| type2_sparse_slots | 0, stride, …, (E−1)×stride | S×2 的 (slot, count)，尾部为 0 |",
        "",
        "E 为活跃专家数，S = E × slot_stride。这里的稀疏指槽位间隔。"
        "实验同时改变槽位布局和 group_list_type，结果不能单独归因于 type 字段。",
        "",
        "## 环境与实验配置",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| Python | {report['python']} |",
        f"| PyTorch | {report['torch']} |",
    ]
    if benchmark:
        for key in ("torch_npu", "npu"):
            lines.append(f"| {key} | {benchmark['metadata'][key]} |")
    for key, value in config.items():
        lines.append(f"| {key} | {value} |")
    slots = config["experts"] * config["slot_stride"]
    tokens = config["experts"] * config["tokens_per_expert"]
    lines += [
        f"| 两组权重形状 [S, K, N] | [{slots}, {config['k']}, {config['n']}] |",
        f"| 两组输入形状 [T, K] | [{tokens}, {config['k']}] |",
        "",
        "## 正确性校验",
        "",
        "每个专家的首、中、尾行与 CPU FP32 参考结果对比；"
        "两种 NPU 布局的输出进行完整对比。容差为 rtol=0.008、atol=0.001。",
        "",
    ]
    correctness = report["correctness"]
    if correctness is None:
        lines.append("正确性测试未完成，详见执行异常。")
    else:
        lines.append(
            f"计划 2 个 case，运行 {correctness['tests_run']} 个；"
            f"断言失败 {len(correctness['failures'])} 项，"
            f"异常 {len(correctness['errors'])} 项，"
            f"跳过 {len(correctness['skipped'])} 项（包含测试类级别的记录）。"
        )
        for category in ("failures", "errors", "skipped"):
            for item in correctness[category]:
                lines += [
                    "",
                    f"**{category}: {item['test']}**",
                    "",
                    "```text",
                    item["detail"].rstrip(),
                    "```",
                ]
    lines += ["", "## 性能结果", ""]
    if benchmark:
        cases = benchmark["cases"]
        lines += [
            "| Case | 中位耗时 (μs/GMM) | 最小耗时 | 最大耗时 | 有效 TOPS |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, values in cases.items():
            lines.append(
                f"| {name} | {values['median_us']:.3f} | {values['min_us']:.3f} | "
                f"{values['max_us']:.3f} | {values['effective_tops']:.3f} |"
            )
        lines += [
            "",
            "### 各轮原始测量",
            "",
            "| 轮次 | 测量顺序 | type1 (μs/GMM) | type2 (μs/GMM) |",
            "| --- | --- | ---: | ---: |",
        ]
        for index, (first, second) in enumerate(
            zip(
                cases["type1_contiguous"]["round_mean_us"],
                cases["type2_sparse_slots"]["round_mean_us"],
            )
        ):
            order = "type1 → type2" if index % 2 == 0 else "type2 → type1"
            lines.append(f"| {index + 1} | {order} | {first:.3f} | {second:.3f} |")
        ratio = benchmark["latency_ratio_type2_over_type1"]
        direction = "高" if ratio >= 1 else "低"
        lines += [
            "",
            "## 本次测量结论",
            "",
            f"type2/type1 中位耗时比为 **{ratio:.4f}**。"
            f"在本次配置下，type2 的中位耗时比 type1 {direction} "
            f"**{abs(ratio - 1) * 100:.2f}%**。该结果仅描述本次测量。",
        ]
    else:
        lines.append("本次未产生完整、有效的性能数据，不生成性能优劣结论。")
    lines += [
        "",
        "## 计时口径与范围",
        "",
        "预热后按 AB/BA 顺序交替测量；每轮耗时除以 iterations，得到每个 GMM 的平均耗时。"
        "表中的中位值、最小值、最大值来自这些轮平均值，不是单次调用的延迟分位数。",
        "",
        "graph 模式计时已捕获的多次 GMM 回放；eager 模式计时普通调用，可能包含下发不足导致的设备空闲。"
        "数据准备、H2D、格式转换、路由、预取和图捕获不计时。"
        "权重重复使用，因此结果反映稳态 GMM，不代表冷缓存或完整 FFN/推理性能。",
    ]
    if report.get("error"):
        lines += ["", "## 执行异常", "", "```text", report["error"].rstrip(), "```"]
    return "\n".join(lines) + "\n"


def save_report(report):
    directory = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"gmm_experiment_{stamp}_{uuid4().hex[:8]}"
    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"
    report["artifacts"] = {"markdown": markdown_path.name, "json": json_path.name}
    markdown = render_report(report)
    with json_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    with markdown_path.open("x", encoding="utf-8") as output:
        output.write(markdown)
    print(f"Experiment report: {markdown_path}", flush=True)
    print(f"Raw results:       {json_path}", flush=True)


def run_experiment(config):
    report = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "error",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "config": asdict(config),
        "correctness": None,
        "benchmark": None,
    }
    try:
        TestGmmGroupListTypes.config = config
        TestGmmGroupListTypes.comparison = None
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGmmGroupListTypes)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report["correctness"] = {
            "tests_run": result.testsRun,
            **{
                name: [
                    {"test": str(test), "detail": detail}
                    for test, detail in getattr(result, name)
                ]
                for name in ("failures", "errors", "skipped")
            },
        }
        if not result.wasSuccessful():
            report["status"] = "failed"
            return 1
        if result.skipped:
            report["status"] = "skipped"
            print("Benchmark not run: Ascend NPU tests were skipped.")
            return 0
        report["benchmark"] = TestGmmGroupListTypes.comparison.benchmark()
        report["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        )
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save_report(report)


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    integer_options = {
        "experts": "Number of active local experts (default: 8).",
        "tokens_per_expert": "Routed rows per expert; total rows = experts * this value.",
        "k": "Input width of one local GMM projection; multiple of 64.",
        "n": "Output width of one local GMM projection; multiple of 64.",
        "slot_stride": "Sparse slot spacing; both pools have experts * stride slots.",
        "seed": "CPU data seed shared by both layouts.",
        "warmup": "Warmup calls per case, excluded from timing.",
        "iterations": "GMM calls per timing round (or per captured graph).",
        "rounds": "Alternating AB/BA rounds; an even count balances order.",
    }
    for name, help_text in integer_options.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(defaults, name),
            help=help_text,
        )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--timing", choices=("graph", "eager"), default=defaults.timing)
    args = parser.parse_args()
    config = Config(**vars(args))
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    return run_experiment(config)


if __name__ == "__main__":
    sys.exit(main())
