# Qwen3.5 MoE on V100

This suite implements and verifies the text-only, stateless computation for
`Qwen-AgentWorld-35B-A3B-NVFP4_fp16`. It does not register cache-backed serving as
supported. Integration uses original layers 0-3 with all 256 experts, embedding,
final normalization and the separate LM head. Precision is checked for every
original layer independently with only one layer resident at a time.

## Model structure

- Hidden size 2048; vocabulary 248320; 40 layers `[GDN,GDN,GDN,Full] x 10`.
- GDN: Q/K 16x128, V/Z 32x128, depthwise Conv kernel 4 over 8192 QKV channels,
  FP32 delta-rule state 32x128x128 per sequence, ordinary gated RMSNorm.
- Full Attention: Q 16x256, K/V 2x256, per-head Q/gate interleaving,
  Gemma Q/K RMSNorm, partial NeoX RoPE over 64 of 256 dimensions, sigmoid output
  gate. Text-only RoPE uses one position per token.
- Every layer has Gemma pre-attention and pre-MoE RMSNorm, 256 routed experts,
  normalized Top-8 routing, and a gated shared expert. Expert intermediate size
  is 512. Decoder/final Gemma normalization uses `1 + weight`; GDN output
  normalization uses `weight`.
- Attention projections: 130 serialized W8A8 FP8 matrices, weight blocks 128x128,
  dynamic per-token activation groups of 128. B/A projections remain FP16.
- Routed layers 1-38: 29,184 NVFP4 W4A4 matrices, groups of 16, FP8 local scales,
  static FP32 global multipliers. Layers 0/39 and shared experts remain FP16.

The full rendered Mermaid graph is in [DESIGN.md](DESIGN.md), with source in
[model_design.mmd](model_design.mmd). See [CHECKLIST.md](CHECKLIST.md) for the
operator/source/test mapping.

## Numerical contract

FP8 and FP4 weights stay encoded in device memory. Triton decodes only the tile
being computed and executes FP16 Tensor Core operations with FP32 accumulation.
This is software FP8/FP4 support on SM70, which has no native FP8/FP4 MMA.

W8A8 computes each K=128 partial in FP32, multiplies that partial by its
activation scale, then by its weight scale, and accumulates K blocks in FP32.
Checkpoint weight scales are FP16; dynamic activation scales are FP32.

For NVFP4, `u=x*G`, local scale is E4M3FN RNE-saturated
`RN32(amax(u_group16) * RN32(1/6))`, and
FP4 codes are E2M1 RNE-saturated `u/decoded_local_scale`. Zero local scales produce
zero codes. A8 scale uses `RN32(amax * RN32(1/448))`; normalized
payload quotients use round-to-nearest FP32 division. Packing places even K in the low nibble. Reconstruction is
`decode4(code)*decode8(local_scale)/G`, without per-row global renormalization.
The product of a finite FP4 code and its FP8 scale is exactly representable in
FP16 (maximum magnitude 2688); only this local product is folded into the HMMA
operand. Global reciprocals remain FP32.

GEMM1 fusion must retain both FP16 gate/up output rounding and the FP16 SwiGLU
output boundary before quantizing down inputs. Gate/up activation global scales
are shared by all experts in each routed layer. Down activation globals are
expert specific. Route weights are applied after the down output boundary;
combine uses a fixed Top-8 order.

## Running

Every Python invocation must use the `sglang-v100` environment and the V100.
GPU unit tests acquire a common lock in `V100TestCase`; do not wrap those tests
in a second shell `flock`, which would deadlock. From the repository root:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m unittest discover -s test/Qwen3_5MoECompat -p 'test_*.py' -v
```

The complete real-weight scan is a separate command; the small scan smoke in
unittest discovery is opt-in:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python -u -m \
  test.Qwen3_5MoECompat.integration.layer_scan --layers 0-39 \
  --output-dir test/Qwen3_5MoECompat/reports/layer_scan
```

The scan CLI and benchmark modules own the same lock internally. **Do not add
an outer shell `flock`** to those commands or to `V100TestCase` tests.
Only standalone ad hoc GPU probes that do not own a lock need shell `flock`.

Benchmarks run with the same interpreter, UUID and `PYTHONPATH` prefix:

- `-m test.Qwen3_5MoECompat.bench.benchmark_moe`: real layer-1 MoE,
  including router, shared expert and residual; explicit unfused comparison.
- `-m test.Qwen3_5MoECompat.bench.benchmark_projection_packing`: real
  layer-0/3 merged versus separate projections, with exact output comparisons.
- `-m test.Qwen3_5MoECompat.bench.backend_audit`: complete original layer-0–3
  `forward_no_cache`, including embedding, final norm and selected-row head.

Each benchmark covers T=1/4/32/128/512/2048. Compilation and loading are excluded
from timings. The backend audit compares CUDA compute events to entry points
from its own freshly compiled Triton PTX; memory operations are reported
separately. Performance reports describe this V100 and this stateless scope.

Checkpoint default:
`/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16`.
No model tensors are committed to the repository. References must not
materialize all experts as FP32 simultaneously. The four-layer compressed
parameters total approximately 4.808 GiB; the integration peak-allocated budget
is 12 GiB at total tokens <=2048.

[VALIDATION.md](VALIDATION.md) records the accepted 40-layer scan and final
93-test discovery (92 passed, one opt-in scan smoke skipped).
[PERFORMANCE.md](PERFORMANCE.md) records fused comparisons and the final full
four-layer Triton profiler audit. [MILESTONES.md](MILESTONES.md) preserves
implementation milestones, verification evidence and scope limits.
