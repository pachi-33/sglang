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

The Mermaid source in `model_design.mmd` documents the full computation.

## Numerical contract

FP8 and FP4 weights stay encoded in device memory. Triton decodes only the tile
being computed and executes FP16 Tensor Core operations with FP32 accumulation.
This is software FP8/FP4 support on SM70, which has no native FP8/FP4 MMA.

W8A8 applies `activation_scale * weight_scale` to each K=128 partial in FP32.
Checkpoint weight scales are FP16; dynamic activation scales are FP32.

For NVFP4, `u=x*G`, local scale is E4M3FN RNE-saturated `amax(u_group16)/6`, and
FP4 codes are E2M1 RNE-saturated `u/decoded_local_scale`. Zero local scales produce
zero codes. Packing places even K in the low nibble. Reconstruction is
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
PYTHONPATH=python \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m unittest discover -s test/Qwen3_5MoECompat -p 'test_*.py' -v
```

Standalone GPU probes without `V100TestCase` must instead use
`flock /tmp/qwen35-v100-gpu.lock` around their Python command.

Checkpoint default:
`/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16`.
No model tensors are committed to the repository. References must not
materialize all experts as FP32 simultaneously. The four-layer compressed
parameters total approximately 4.808 GiB; the integration peak-allocated budget
is 12 GiB at total tokens <=2048.

See `MILESTONES.md` for verification evidence, accepted limits and remaining work.
