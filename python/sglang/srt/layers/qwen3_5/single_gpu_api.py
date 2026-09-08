"""Serve the single-GPU Qwen3.5 ExpertPack backend over HTTP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .pipeline import MODEL_DIR_DEFAULT, load_tokenizer_compat
from .pipeline_api import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    Qwen35PipelineAPIEngine,
    create_app,
)
from .single_gpu import EXPERT_PACK_MANIFEST_DEFAULT, Qwen35SingleGPU

# A descriptive alias for new callers; the original class name remains public
# for the existing dual-GPU entry point.
Qwen35SingleGPUAPIEngine = Qwen35PipelineAPIEngine


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    parser.add_argument("--served-model-name")
    parser.add_argument("--expert-pack-manifest", default=EXPERT_PACK_MANIFEST_DEFAULT)
    parser.add_argument("--expert-cache-mib", type=int, default=7168)
    parser.add_argument("--expert-stage-slots", type=int, default=16)
    parser.add_argument("--expert-io-workers", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--stats-path")
    parser.add_argument(
        "--expert-trace-dir",
        help="server-local directory for explicitly requested expert traces",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY"))
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1,65535]")
    if args.api_key == "":
        raise SystemExit("--api-key must not be empty")
    model_dir = Path(args.model_dir)
    model_id = args.served_model_name or model_dir.name
    tokenizer = load_tokenizer_compat(model_dir)
    backend = Qwen35SingleGPU(
        model_dir,
        expert_pack_manifest=args.expert_pack_manifest,
        expert_cache_mib=args.expert_cache_mib,
        expert_stage_slots=args.expert_stage_slots,
        expert_io_workers=args.expert_io_workers,
        capacity=args.capacity,
        stats_path=args.stats_path,
        expert_trace_dir=args.expert_trace_dir,
    )
    engine = None
    try:
        engine = Qwen35SingleGPUAPIEngine(backend, tokenizer, model_id=model_id)
        app = create_app(engine, api_key=args.api_key)
        import uvicorn

        # The backend owns one CUDA context and one request cache.  Forking an
        # additional server worker would duplicate both and exceed GPU memory.
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            workers=1,
        )
    finally:
        if engine is None:
            backend.close()
        else:
            engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
