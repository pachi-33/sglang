from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from typing import Any, ContextManager, Optional

import torch

logger = logging.getLogger(__name__)

STARTUP_WARMUP_RID_PREFIX = "__sglang_startup_warmup__"


def profile_startup_warmup_batch(func):
    """Add a profiler range around an internally tagged startup warmup batch."""

    @wraps(func)
    def wrapped(scheduler, batch, *args, **kwargs):
        with scheduler.startup_memory_profiler.profile_batch(batch):
            return func(scheduler, batch, *args, **kwargs)

    return wrapped


class StartupMemoryProfiler:
    """Profile startup allocations through the first server warmup request.

    This deliberately uses PyTorch's public, backend-neutral entry points:
    ``torch.profiler`` and ``torch.get_device_module``. On Ascend,
    ``profiler_manager`` installs the existing torch_npu PrivateUse1 patches
    before a Scheduler is constructed, so the same code records NPU activity.
    """

    def __init__(
        self,
        *,
        output_dir: Optional[str],
        device_type: str,
        device_index: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        attn_tp_rank: int,
        attn_cp_rank: int,
    ) -> None:
        self.output_dir = (
            None if not output_dir else Path(output_dir).expanduser().resolve()
        )
        self.device_type = device_type
        self.device_index = device_index
        self.device = f"{device_type}:{device_index}"
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.dp_rank = 0 if dp_rank is None else dp_rank
        self.attn_tp_rank = attn_tp_rank
        self.attn_cp_rank = attn_cp_rank

        self.device_module: Any = None
        self.profiler: Any = None
        self.started_wall_time: Optional[float] = None
        self.started_monotonic: Optional[float] = None
        self.checkpoints: list[dict[str, Any]] = []
        self.export_errors: list[str] = []

    @classmethod
    def maybe_start(cls, **kwargs) -> StartupMemoryProfiler:
        profiler = cls(**kwargs)
        profiler.start()
        return profiler

    @property
    def active(self) -> bool:
        return self.profiler is not None

    def _activities(self) -> list[Any]:
        activities = [torch.profiler.ProfilerActivity.CPU]
        activity_name = {
            "npu": "NPU",
            "xpu": "XPU",
            "hpu": "HPU",
            "mtia": "MTIA",
        }.get(self.device_type, "CUDA")
        device_activity = getattr(torch.profiler.ProfilerActivity, activity_name, None)
        if device_activity is None and self.device_type == "npu":
            # Older torch_npu releases patch CUDA to mean NPU.
            device_activity = getattr(torch.profiler.ProfilerActivity, "CUDA", None)
        if device_activity is not None:
            activities.append(device_activity)
        return activities

    def start(self) -> None:
        if self.output_dir is None:
            return
        if self.device_type == "cpu":
            logger.warning(
                "SGLANG_STARTUP_MEMORY_PROFILE_DIR is ignored for the CPU backend"
            )
            return

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.device_module = torch.get_device_module(self.device_type)
            self.device_module.set_device(self.device_index)
            self.device_module.synchronize()
            reset_peak = getattr(self.device_module, "reset_peak_memory_stats", None)
            if reset_peak is not None:
                reset_peak(self.device_index)

            self.profiler = torch.profiler.profile(
                activities=self._activities(),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
            )
            self.profiler.start()
            self.started_wall_time = time.time()
            self.started_monotonic = time.perf_counter()
            self.checkpoint("profile_started")
            logger.warning(
                "Startup memory profiling enabled for %s; profiling overhead is "
                "expected and results will be written to %s",
                self.device,
                self.output_dir,
            )
        except Exception:
            self.profiler = None
            logger.exception(
                "Failed to start startup memory profiling for %s", self.device
            )

    def record_phase(self, name: str) -> ContextManager[Any]:
        if not self.active:
            return nullcontext()
        return torch.profiler.record_function(f"sglang.startup.{name}")

    @staticmethod
    def is_startup_warmup_rid(rid: Any) -> bool:
        return isinstance(rid, str) and rid.startswith(STARTUP_WARMUP_RID_PREFIX)

    def is_startup_warmup_batch(self, batch: Any) -> bool:
        return any(
            self.is_startup_warmup_rid(getattr(req, "rid", None))
            for req in getattr(batch, "reqs", ())
        )

    def profile_batch(self, batch: Any) -> ContextManager[Any]:
        if self.active and self.is_startup_warmup_batch(batch):
            return self.record_phase("first_server_warmup")
        return nullcontext()

    def checkpoint(self, name: str, *, synchronize: bool = False) -> None:
        if not self.active:
            return
        if synchronize:
            try:
                self.device_module.synchronize()
            except Exception as exc:
                self.export_errors.append(f"checkpoint {name} synchronize: {exc!r}")

        memory: dict[str, int] = {}
        for api_name in (
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        ):
            api = getattr(self.device_module, api_name, None)
            if api is None:
                continue
            try:
                memory[f"{api_name}_bytes"] = int(api(self.device_index))
            except Exception as exc:
                self.export_errors.append(
                    f"checkpoint {name} {api_name}: {exc!r}"
                )

        mem_get_info = getattr(self.device_module, "mem_get_info", None)
        if mem_get_info is not None:
            try:
                free_bytes, total_bytes = mem_get_info(self.device_index)
                memory["device_free_bytes"] = int(free_bytes)
                memory["device_total_bytes"] = int(total_bytes)
                memory["device_used_bytes"] = int(total_bytes - free_bytes)
            except Exception as exc:
                self.export_errors.append(
                    f"checkpoint {name} mem_get_info: {exc!r}"
                )

        elapsed = (
            None
            if self.started_monotonic is None
            else time.perf_counter() - self.started_monotonic
        )
        self.checkpoints.append({"name": name, "elapsed_s": elapsed, **memory})

    def maybe_stop_after_batch(self, batch: Any) -> bool:
        if not self.active:
            return False
        warmup_reqs = [
            req
            for req in getattr(batch, "reqs", ())
            if self.is_startup_warmup_rid(getattr(req, "rid", None))
        ]
        if not warmup_reqs or not all(req.finished() for req in warmup_reqs):
            return False
        self.stop(reason="first_server_warmup_completed")
        return True

    def stop(self, *, reason: str) -> bool:
        if not self.active:
            return False

        self.checkpoint(reason, synchronize=True)
        profiler = self.profiler
        self.profiler = None
        try:
            profiler.stop()
        except Exception as exc:
            self.export_errors.append(f"profiler stop: {exc!r}")
            logger.exception("Failed to stop startup memory profiler")

        self._export(profiler, reason=reason)
        return True

    def _stem(self) -> str:
        return (
            f"startup-memory-{self.device_type}{self.device_index}"
            f"-tp{self.tp_rank}-pp{self.pp_rank}-dp{self.dp_rank}"
            f"-attp{self.attn_tp_rank}-attcp{self.attn_cp_rank}-pid{os.getpid()}"
        )

    def _export(self, profiler: Any, *, reason: str) -> None:
        assert self.output_dir is not None
        stem = self._stem()
        exported: dict[str, str] = {}

        timeline_outputs = {
            "memory_timeline_html": self.output_dir / f"{stem}.memory.html",
            "memory_timeline_json": self.output_dir / f"{stem}.memory.json.gz",
            "memory_events_raw": self.output_dir / f"{stem}.memory.raw.json.gz",
        }
        for label, path in timeline_outputs.items():
            try:
                profiler.export_memory_timeline(str(path), device=self.device)
                exported[label] = str(path)
            except Exception as exc:
                self.export_errors.append(f"{label}: {exc!r}")
                logger.exception("Failed to export %s to %s", label, path)

        trace_path = self.output_dir / f"{stem}.trace.json"
        try:
            profiler.export_chrome_trace(str(trace_path))
            exported["chrome_trace"] = str(trace_path)
        except Exception as exc:
            self.export_errors.append(f"chrome_trace: {exc!r}")
            logger.exception("Failed to export Chrome trace to %s", trace_path)

        operator_path = self.output_dir / f"{stem}.operators.txt"
        try:
            operator_table = profiler.key_averages(
                group_by_input_shape=True, group_by_stack_n=5
            ).table(row_limit=500)
            operator_path.write_text(operator_table, encoding="utf-8")
            exported["operator_table"] = str(operator_path)
        except Exception as exc:
            self.export_errors.append(f"operator_table: {exc!r}")
            logger.exception("Failed to export operator table to %s", operator_path)

        summary_path = self.output_dir / f"{stem}.summary.json"
        summary = {
            "device": self.device,
            "ranks": {
                "tp": self.tp_rank,
                "pp": self.pp_rank,
                "dp": self.dp_rank,
                "attention_tp": self.attn_tp_rank,
                "attention_cp": self.attn_cp_rank,
            },
            "pid": os.getpid(),
            "started_at_unix_s": self.started_wall_time,
            "duration_s": (
                None
                if self.started_monotonic is None
                else time.perf_counter() - self.started_monotonic
            ),
            "stop_reason": reason,
            "profiler_options": {
                "profile_memory": True,
                "record_shapes": True,
                "with_stack": True,
                "with_modules": True,
            },
            "checkpoints": self.checkpoints,
            "exports": exported,
            "errors": self.export_errors,
        }
        try:
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            logger.exception(
                "Failed to export startup memory summary to %s", summary_path
            )
            return

        logger.info(
            "Startup memory profile stopped (%s). Summary: %s; timeline: %s",
            reason,
            summary_path,
            timeline_outputs["memory_timeline_html"],
        )
