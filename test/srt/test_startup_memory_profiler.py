import json
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.startup_memory_profiler import (
    STARTUP_WARMUP_RID_PREFIX,
    StartupMemoryProfiler,
)


class _FakeDeviceModule:
    def __init__(self):
        self.device_index = None
        self.synchronize_calls = 0
        self.reset_calls = []

    def set_device(self, device_index):
        self.device_index = device_index

    def synchronize(self):
        self.synchronize_calls += 1

    def reset_peak_memory_stats(self, device_index):
        self.reset_calls.append(device_index)

    def memory_allocated(self, device_index):
        return 100 + device_index

    def memory_reserved(self, device_index):
        return 200 + device_index

    def max_memory_allocated(self, device_index):
        return 300 + device_index

    def max_memory_reserved(self, device_index):
        return 400 + device_index

    def mem_get_info(self, device_index):
        return 500 + device_index, 1000 + device_index


class _FakeAverages:
    def table(self, row_limit):
        return f"operators row_limit={row_limit}"


class _FakeProfiler:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_memory_timeline(self, path, device):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"timeline for {device}")

    def export_chrome_trace(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("trace")

    def key_averages(self, **kwargs):
        return _FakeAverages()


def _make_profiler(tmp_path, monkeypatch):
    device_module = _FakeDeviceModule()
    created = []

    def fake_profile(**kwargs):
        profiler = _FakeProfiler(kwargs)
        created.append(profiler)
        return profiler

    monkeypatch.setattr(torch, "get_device_module", lambda _device: device_module)
    monkeypatch.setattr(torch.profiler, "profile", fake_profile)
    monkeypatch.setattr(torch.profiler, "record_function", lambda _name: nullcontext())

    profiler = StartupMemoryProfiler.maybe_start(
        output_dir=str(tmp_path),
        device_type="npu",
        device_index=2,
        tp_rank=1,
        pp_rank=0,
        dp_rank=None,
        attn_tp_rank=1,
        attn_cp_rank=0,
    )
    return profiler, device_module, created[0]


def test_startup_memory_profiler_exports_after_finished_warmup(
    tmp_path, monkeypatch
):
    profiler, device_module, torch_profiler = _make_profiler(tmp_path, monkeypatch)

    assert profiler.active
    assert torch_profiler.started
    assert torch_profiler.kwargs["profile_memory"] is True
    assert torch_profiler.kwargs["record_shapes"] is True
    assert torch_profiler.kwargs["with_stack"] is True
    assert device_module.device_index == 2
    assert device_module.reset_calls == [2]

    unfinished = SimpleNamespace(
        rid=f"{STARTUP_WARMUP_RID_PREFIX}-0", finished=lambda: False
    )
    batch = SimpleNamespace(reqs=[unfinished])
    with profiler.profile_batch(batch):
        pass
    assert profiler.maybe_stop_after_batch(batch) is False
    assert profiler.active

    finished = SimpleNamespace(
        rid=f"{STARTUP_WARMUP_RID_PREFIX}-0", finished=lambda: True
    )
    assert profiler.maybe_stop_after_batch(SimpleNamespace(reqs=[finished])) is True
    assert not profiler.active
    assert torch_profiler.stopped

    summary_path = next(tmp_path.glob("*.summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["device"] == "npu:2"
    assert summary["stop_reason"] == "first_server_warmup_completed"
    assert summary["checkpoints"][-1]["memory_allocated_bytes"] == 102
    assert summary["checkpoints"][-1]["device_used_bytes"] == 500
    assert not summary["errors"]

    assert len(list(tmp_path.glob("*.memory.html"))) == 1
    assert len(list(tmp_path.glob("*.memory.json.gz"))) == 1
    assert len(list(tmp_path.glob("*.memory.raw.json.gz"))) == 1
    assert len(list(tmp_path.glob("*.trace.json"))) == 1
    assert len(list(tmp_path.glob("*.operators.txt"))) == 1


def test_startup_memory_profiler_is_noop_without_output_dir(monkeypatch):
    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )
    profiler = StartupMemoryProfiler.maybe_start(
        output_dir=None,
        device_type="npu",
        device_index=0,
        tp_rank=0,
        pp_rank=0,
        dp_rank=0,
        attn_tp_rank=0,
        attn_cp_rank=0,
    )
    assert not profiler.active
    assert profiler.maybe_stop_after_batch(SimpleNamespace(reqs=[])) is False
