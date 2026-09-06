# Qwen3.5 MoE V100 implementation milestones

Branch: `feat/qwen3/compat`.

The implementation preserves the checkpoint's 130 FP8 W8A8 attention matrices
and 29,184 NVFP4 W4A4 routed-expert matrices. All layer numbers are zero based.
Production compute uses Triton on SM70. No persistent inference cache is part of
this implementation. Temporary Conv, attention and FP32 GDN state are local to a
single full-sequence call.

## Environment and review contract

- Python: `/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python`.
- Initial packages: Python 3.10, torch 2.3.1+cu121, Triton 2.3.1,
  transformers 4.43.2, safetensors 0.8.0, numpy 1.26.4.
- Device: Tesla V100-SXM2-16GB, SM70,
  `GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96`.
- All GPU tests and benchmarks take `/tmp/qwen35-v100-gpu.lock`.
- Astra ultra owns numerical/interface review. Terra agents implement code.
  Only the root coordinator stages explicit paths and commits milestones.
- Reference checkout: local SGLang `4349538c02e1566a1424510d5ac3ae853f49feef`.
- Starting implementation commit: `0d41cd2a569c87b348496f9c6481d81764897e6c`.

## Acceptance budgets

Normalized errors use `||actual-reference||_2 / ||reference||_2`; zero-reference
cases additionally require an absolute error check. Budgets are not to be
relaxed by an optimization patch.

| Measurement | Budget |
|---|---:|
| Single projection NRMSE | 2e-3 |
| Local attention / MoE NRMSE | 5e-3 |
| GDN FP32 recurrent internal output / state relative L2 | 1e-4 |
| GDN chunk output NRMSE at length 2048 | 5e-3 |
| Four-layer CUDA peak allocated | 12 GiB |

Codec bytes, shape, scale mapping, route mapping and identical-logit Top-8 are
exact checks. Native FlashInfer fast-math byte parity has not been established;
the portable contract uses RNE finite saturation and zero codes when local SF
rounds to zero. Report max/P99 errors and non-finite values as well as NRMSE.

## Milestone status

| ID | Deliverable | Status |
|---|---|---|
| M0 | Environment, SM70 baseline, manifest, interfaces | M0a passed; model/config interface pending |
| M1 | Independent references and codec tests | Codec / activation contract passed |
| M2a | Common layers, W8A8 and Full Attention | Common/W8A8/attention core passed; producer fusion pending |
| M2b | Fused NVFP4 / FP16 routed MoE | Pending |
| M2c | Stateless GDN recurrent and chunk paths | Pending |
| M3 | All 40 real layers independently checked | Pending |
| M4 | Real layers 0-3 integration | Pending |
| M5 | Performance / backend audit and documentation | Pending |

## Pre-implementation feasibility evidence

Read-only tests of Triton's existing matmul kernel on the selected V100 found
that NT tile 32x32x32, four warps and one stage passes CPU FP32 comparison for
M=1/17/128, K=2048, N=512 (NRMSE about 2.1e-4). NN tile 16x16x16, four warps and
one stage also passes the tested attention-PV shapes. Several larger tiles and
multi-stage configurations produce large errors or all-zero output. Each new
kernel, including decoded quantized operands and fused attention, must therefore
be tested independently; HMMA in generated PTX is necessary evidence for GEMM
but is not a correctness test. Autotuning may only use validated configurations.

Implementation results and exact verification commands will be appended as
each milestone passes. No milestone is complete merely because its files exist.

## M0a — compressed-weight contracts and SM70 foundations

Implemented typed Weight/QuantActivation storage, one-layer checkpoint loading,
full quantized-header audit, independent CPU format references, FP16 NT GEMM,
embedding, normalization and basic elementwise kernels. The default embedding
gather has no GPU-to-host validation synchronization; invalid IDs are masked to
zero, with explicit `validate_ids=True` available for input diagnostics. Matrix
layouts outside the verified inner-contiguous NT contract are rejected.

Verification in the required environment on the selected V100:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python -m unittest \
  test.Qwen3_5MoECompat.unit.test_environment \
  test.Qwen3_5MoECompat.unit.test_checkpoint \
  test.Qwen3_5MoECompat.unit.test_reference_codec \
  test.Qwen3_5MoECompat.unit.test_dense_ops -v
```

Result: **8 tests passed**, including the 130 / 29184 / 1536 matrix audit,
reference signed-zero/NaN/zero-scale cases, norm/gather tests, and a multi-CTA,
tail and empty-input GEMM sweep. An independent CPU FP32 comparison measured:

| M,N,K | NRMSE |
|---|---:|
| 1,512,2048 | 2.17825499e-4 |
| 17,512,2048 | 2.05125096e-4 |
| 128,512,2048 | 2.06699527e-4 |
| 33,37,257 | 2.07458090e-4 |
| 0,64,128 | 0 (empty output) |

Astra review identified and the implementation corrected default embedding
synchronization, unsupported GEMM layouts, codec negative-zero handling and
malformed FP8 NaN decoding. Initial nested shell/test `flock` acquisition was
also corrected: unittest classes own the lock; standalone probes use shell
`flock`. Quantized production kernels, attention, GDN, model wiring and full
coverage remain in later milestones.

## M1 / M2a-W8 — actual Triton codecs and fused W8A8

Implemented GPU E4M3FN/E2M1 encode/decode, A8 group128 and static-global A4
group16 activation quantization, and W8A8 GEMM with tile decoding fused into
Volta HMMA. Four K32 dot products accumulate a K128 partial before FP32
activation/weight scale application. Unverified noncontiguous payloads are
rejected. There is no full dequantized FP8 weight intermediate.

The quantization test launches the actual Triton codec helpers. It covers all
256 E4M3FN decodings (including both NaN codes), all finite round trips, all
adjacent-value midpoints and their nextafter neighbors with both signs, E2M1
codewords/ties, packed nibble order, signed zero, zero local scales, independent
A8/A4 payload comparison and multiple real GEMM sizes.

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python -m unittest \
  test.Qwen3_5MoECompat.unit.test_quantization -v
```

Result: **5 tests passed** on the V100. Each GEMM reference reads the actual
stored FP16 weight scales; the NRMSE gate is 2e-3, not an elementwise loose
tolerance against a different scale tensor.

| M,N,K | W8A8 NRMSE |
|---|---:|
| 1,256,2048 | 1.8405264e-4 |
| 17,256,4096 | 2.0427280e-4 |
| 128,256,2048 | 2.0639937e-4 |
| 0,256,2048 | empty output / payload validated |

An additional constant-matrix probe returns exactly 128 for a sum of 128 ones;
the compiled quantized GEMM PTX contains `.target sm_70` and
`mma.sync.aligned.m8n8k4`.

FP32 scale formation uses reciprocal-multiply semantics
`RN32(amax * RN32(1/448))` (or `1/6` for the NVFP4 local scale). The normalized
payload quotient uses `div_rn`; replacing it with approximate division caused
midpoint code differences during implementation and is not an allowed tuning
change. NVFP4 MoE fusion, Full Attention and GDN remain unaccepted at this point.

## M0b — selective loading without transient GPU weight copies

Expert packing now runs on CPU before transferring each final compressed tensor
to the V100. Loading a layer no longer allocates its raw, stacked and merged
expert weights on the GPU simultaneously. FP8 payload/FP16 block-scale storage
and NVFP4 packed bytes/local scales/FP32 global multipliers retain their dtypes.
Astra reviewed the loader diff; the coordinator loaded real layers 0, 1 and 3
individually on the required V100 and measured:

| Original layer | Resident GiB | Peak allocated GiB |
|---|---:|---:|
| 0, FP16 routed experts / GDN | 1.538408 | 1.538408 |
| 1, NVFP4 routed experts / GDN | 0.460290 | 0.460290 |
| 3, NVFP4 routed experts / Full Attention | 0.454125 | 0.454125 |

These measurements include each layer's complete 256 experts. They verify the
loader's device allocation behavior; the four-layer integration peak is still
a separate acceptance gate. Model/config registration remains pending.

## M2a-attention-core — stateless causal GQA and partial NeoX RoPE

Implemented SM70 Tensor Core QK/PV with K=512 online-softmax slabs. Each merge
CTA owns the full D=256 row, including its scalar max/sum state. Splitting those
scalar updates among dimension tiles caused a race in a discarded prototype.
Both accepted QK/PV JIT kernels were inspected in the current V100 process:
`.target sm_70` and `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` are present.

The coordinator reran `test.Qwen3_5MoECompat.unit.test_attention`: **10 tests
passed** in 2.993 s. Tests cover lengths 1/17/65/129/2048, ragged total 2048,
nonempty int64 sequence offsets, empty output identity, Q/K normalization,
RoPE overlap rejection, supported layouts and metadata dtypes. GPU metadata
contents are trusted: offsets start at zero, end at T, are nondecreasing and
each sequence fits the CPU-supplied maximum length. Both total T and maximum
sequence length are limited to 2048 in this delivery.

| Attention case | NRMSE |
|---|---:|
| One sequence, T=2048 | 2.540e-4 |
| Ragged lengths 1/17/65/129/1836 | 2.375e-4 |

Additional RoPE comparison against a CPU FP64 mathematical oracle measured
NRMSE 1.84e-5 / 3.18e-5 / 7.27e-4 at positions 2047 / 8192 / 262143.
At position 262143 the maximum absolute error was 0.0177; large-position
elementwise or bitwise parity is not claimed.

Warm CUDA-event measurements from the implementation agent, excluding compile:

| T | Triton slab ms / peak MiB | Test FP32 reference ms / peak MiB |
|---|---:|---:|
| 128 | 0.233 / 12.4 | 0.968 / 10.8 |
| 512 | 0.825 / 57.8 | 1.339 / 20.4 |
| 2048 | 12.240 / 206.6 | 9.825 / 96.1 |

The 2048-token path is slower than this reference; performance optimization
remains M5 work. This milestone accepts the attention core only. Projection
packing, fused Q/gate split + Q/K norm/RoPE, gate+A8 producer fusion and real
layer 3 end-to-end validation remain required before full M2a acceptance.
