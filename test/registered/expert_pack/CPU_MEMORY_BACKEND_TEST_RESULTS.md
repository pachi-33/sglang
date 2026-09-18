# CPU-memory expert backend test results

This file records manual validation of the uncommitted CPU-memory expert
offloader implementation. It is a test record, not a claim of end-to-end model
or performance validation.

## 2026-09-18 targeted validation

- Start time: 2026-09-18 13:43:19 +0800 (Asia/Singapore)
- Repository: `/home/yaozhenyang/dev/sglang`
- Branch: `feat/expertpack/cpu-memory-backend`
- Base commit: `4b186cfea59371cc8ec38597f750a5597f7147a0`
- Worktree state: uncommitted implementation and tests
- SHA-256 over the 14 implementation/test files under review, in the order
  listed by the command below:
  `4f622b3b872882e9fc773c901e5d8be70001535b457b791ecf1da0fa095b8060`

### Environment

- Conda environment: `sgl-main`
- Python: 3.12.0
- PyTorch: 2.13.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- GPU: NVIDIA GeForce RTX 4070 SUPER, compute capability 8.9
- GPU UUID: `GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4`
- Driver: 615.71.09
- GPU memory: 12282 MiB
- `pytest` was not installed in this environment. The files use
  `unittest`, so they were executed directly.
- `SGLANG_CACHE_DIR` was redirected to writable `/tmp/codex-sglang-cache`
  to allow FlashInfer import-time logging in the restricted runner.

### Commands

```bash
env SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  test/registered/unit/layers/moe/test_expert_offload.py -v

env SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  test/registered/unit/layers/moe/test_cpu_memory_expert_backend.py -v

env SGLANG_CACHE_DIR=/tmp/codex-sglang-cache \
  PYTHONPATH=/home/yaozhenyang/dev/sglang/python \
  /home/yaozhenyang/downloads/yes/envs/sgl-main/bin/python \
  test/registered/expert_pack/test_cpu_memory_backend.py -v
```

Code fingerprint command:

```bash
sha256sum \
  python/sglang/srt/arg_groups/expert_pack_hook.py \
  python/sglang/srt/layers/attention/linear/gdn_backend.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py \
  python/sglang/srt/layers/moe/cpu_memory_expert_backend.py \
  python/sglang/srt/layers/moe/expert_offload.py \
  python/sglang/srt/layers/quantization/awq/awq.py \
  python/sglang/srt/layers/quantization/base_config.py \
  python/sglang/srt/layers/quantization/unquant.py \
  python/sglang/srt/model_loader/expert_pack_loader.py \
  python/sglang/srt/model_loader/loader.py \
  python/sglang/srt/server_args.py \
  test/registered/expert_pack/test_cpu_memory_backend.py \
  test/registered/unit/layers/moe/test_cpu_memory_expert_backend.py \
  test/registered/unit/layers/moe/test_expert_offload.py \
  | sha256sum
```

### Results

| Test file | Passed | Failed | Skipped | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `test_expert_offload.py` | 26 | 0 | 0 | 0.106 s |
| `test_cpu_memory_expert_backend.py` | 11 | 0 | 0 | 0.158 s |
| `expert_pack/test_cpu_memory_backend.py` | 9 | 0 | 0 | 0.003 s |
| **Total** | **46** | **0** | **0** | **0.267 s** |

The GPU-dependent tests that passed were:

- `test_cuda_cache_copy_and_remap_smoke`
- `test_cuda_staging_ring_uses_pinned_memory`
- `test_cuda_unquantized_final_layout_cache_matches_baseline_across_microbatches`
- `test_cuda_awq_final_layout_payload_is_bit_exact_in_compact_slots`

The test GPU had no remaining compute process after the run.

### What this run establishes

- Capability declarations, coverage tracking, wrapper delegation, runtime
  layer views, logical-to-physical expert remapping, and lease release paths
  pass their targeted tests.
- Fake-event tests cover staging reuse, cache admission/eviction, active lease
  protection, rollback, pool poisoning, budget allocation, timing counters,
  and stats flushing.
- Real CUDA tests cover pinned staging, asynchronous cache copies, compact slot
  remapping, unquantized numerical parity for the toy case, and bit-exact AWQ
  final-layout payload copies.
- Public loader/config guard tests pass.

### Not covered by this run

- Loading and serving an actual Qwen3.5 AWQ checkpoint end to end.
- Numerical parity for a complete model over prefill and multi-token decode.
- Cache hit-rate, H2D bandwidth, TTFT, ITL, throughput, or peak-memory
  measurements.
- Multi-request, TP, EP, CUDA graph, EPLB, LoRA, shared-expert, hot weight
  reload, and shutdown-lifecycle behavior.
- The repository's full regression suite.
