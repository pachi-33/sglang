"""Compare layer 1 resident and ExpertPack execution on one V100."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from sglang.srt.layers.qwen3_5.expert_pack.store import ExpertOffloadConfig
from sglang.srt.layers.qwen3_5.runner import Qwen35StatelessRunner

MODEL_DIR = Path("/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16")
MANIFEST = Path(
    "/home/yaozhenyang/huggingface/"
    "Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json"
)
TOKEN_COUNTS = (1, 32, 2048)


def _forward(
    runner: Qwen35StatelessRunner, hidden_cpu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = hidden_cpu.shape[0]
    hidden = hidden_cpu.to(runner.device)
    positions = torch.arange(tokens, device=runner.device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, tokens], device=runner.device, dtype=torch.int32)
    router_capture: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        output = runner.forward_no_cache(
            hidden,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=tokens,
            router_capture=router_capture,
        )
    torch.cuda.synchronize(runner.device)
    route_ids, route_weights = router_capture[1]
    return output.cpu(), route_ids.cpu(), route_weights.cpu()


def run_acceptance(
    *, model_dir: Path, manifest: Path, output_path: Path | None
) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("layer comparison requires exactly one visible CUDA GPU")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("layer comparison requires an SM70 V100")

    torch.manual_seed(3501)
    inputs = {
        tokens: (torch.randn((tokens, 2048), dtype=torch.float16) * 0.05).contiguous()
        for tokens in TOKEN_COUNTS
    }
    resident_results: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    resident = Qwen35StatelessRunner(model_dir, [1], device="cuda", load_globals=False)
    try:
        for tokens in TOKEN_COUNTS:
            resident_results[tokens] = _forward(resident, inputs[tokens])
    finally:
        resident.close()
        del resident
        gc.collect()
        torch.cuda.empty_cache()

    offloaded = Qwen35StatelessRunner(
        model_dir,
        [1],
        device="cuda",
        load_globals=False,
        expert_offload=ExpertOffloadConfig(
            manifest, cache_mib=433, stage_slots=16, io_workers=2
        ),
    )
    cases: dict[str, dict[str, bool | float]] = {}
    try:
        for tokens in TOKEN_COUNTS:
            actual = _forward(offloaded, inputs[tokens])
            expected = resident_results[tokens]
            cases[str(tokens)] = {
                "output_exact": bool(torch.equal(actual[0], expected[0])),
                "router_ids_exact": bool(torch.equal(actual[1], expected[1])),
                "router_weights_exact": bool(torch.equal(actual[2], expected[2])),
                "max_abs_output_diff": float(
                    (actual[0].float() - expected[0].float()).abs().max().item()
                ),
            }

        before = offloaded.expert_stats
        hot_output = _forward(offloaded, inputs[2048])[0]
        after = offloaded.expert_stats
        assert before is not None and after is not None
        hot_hit = {
            "output_exact": bool(torch.equal(hot_output, resident_results[2048][0])),
            "new_pack_reads": int(after["pack_reads"]) - int(before["pack_reads"]),
            "new_h2d_bytes": int(after["h2d_bytes"]) - int(before["h2d_bytes"]),
        }
        store = {
            key: after[key]
            for key in (
                "state",
                "cache_capacity_experts",
                "cache_capacity_bytes",
                "resident_experts",
                "pack_reads",
                "pack_read_bytes",
                "h2d_bytes",
                "cache_hits",
                "cache_misses",
                "cache_evictions",
            )
        }
    finally:
        offloaded.close()

    passed = all(
        case["output_exact"]
        and case["router_ids_exact"]
        and case["router_weights_exact"]
        and case["max_abs_output_diff"] == 0.0
        for case in cases.values()
    )
    passed = passed and all(
        (
            hot_hit["output_exact"],
            hot_hit["new_pack_reads"] == 0,
            hot_hit["new_h2d_bytes"] == 0,
            store["state"] == "READY",
            store["resident_experts"] == 256,
        )
    )
    report = {
        "status": "passed" if passed else "failed",
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "visible_count": torch.cuda.device_count(),
        },
        "cases": cases,
        "hot_hit": hot_hit,
        "store": store,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output_path)
    print(encoded, end="")
    if not passed:
        raise AssertionError("layer 1 resident/ExpertPack comparison failed")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_acceptance(
        model_dir=args.model_dir, manifest=args.manifest, output_path=args.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
