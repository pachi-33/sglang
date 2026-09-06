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
the portable contract uses RNE finite saturation. NVFP4 uses zero codes when
local SF rounds to zero; A8 retains signed-zero codes, including all-zero K128
groups. Report max/P99 errors and non-finite values as well as NRMSE.

## Milestone status

| ID | Deliverable | Status |
|---|---|---|
| M0 | Environment, SM70 baseline, manifest, interfaces | Baseline, exact header manifest, config and stateless interface passed |
| M1 | Independent references and codec tests | Codec / activation contract passed |
| M2a | Common layers, W8A8 and Full Attention | Common / W8A8 / attention / producers passed; projection packing and tuning remain M5 |
| M2b | Fused NVFP4 / FP16 routed MoE | Core fusion and real-layer precision passed; M5 tuning pending |
| M2c | Stateless GDN recurrent and chunk paths | FP32 recurrent and bounded streamed BT16 WY passed |
| M3 | All 40 real layers independently checked | Pending |
| M4 | Real layers 0-3 integration | T1 / T3 / T65 smoke passed; complete integration acceptance pending |
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

## M2b — fused NVFP4 routed experts and Triton FP16/shared MoE

The routed input is A4-quantized once per token; GEMM1 reads it directly through
the dispatch token map. Each persistent GEMM1 CTA computes matching gate/up
output tiles, applies the global multipliers in FP32, rounds both projections
to FP16, computes SwiGLU in FP32, rounds to FP16, and emits the down input's
packed A4 bytes and E4M3FN scales. The down activation multiplier is selected
per expert. No full expanded weight or gate/up activation is materialized.

To cross Triton 2.3's MMA/reduction layout boundary, each CTA owns one 32x32
FP16 spill tile. A maximum of 80 CTAs bounds this scratch at **160 KiB**. Every
tile is fully stored, synchronized, reloaded into an independent blocked
layout, and synchronized again before reuse. Debug capture of the full SwiGLU
tensor exists only for tests. Astra approved the scratch ownership, barriers,
FP16 boundaries, low/high-nibble dot decomposition and global-scale mapping.

GEMM2 rounds the expert result to FP16 before multiplying its routing weight
in FP32. The final kernel sums routes in the fixed Top-8 order in FP32, then
rounds to FP16. FP16 experts in layers 0/39 use grouped Triton GEMMs and SwiGLU.
Shared projection, SwiGLU, scalar gate and output addition also use Triton;
the shared sigmoid is rounded to FP16 before multiplication, matching the
reference checkout's CUDA path. Routed output is rounded before adding shared
output. CPU checkpoint packing verifies the layer-wide gate/up input scale.

Independent reference tests stream one expert at a time. The final reference
contract includes down-output FP16 rounding and FP32 Top-8 accumulation; early
diagnostic references that omitted these boundaries were corrected without
changing the acceptance budgets. Coordinator measurements with the corrected
reference:

| Real checkpoint case | NRMSE |
|---|---:|
| Layer 0, FP16, T=32 balanced routes covering all 256 experts | 1.786774e-4 |
| Layer 1, NVFP4, T=32 balanced routes covering all 256 experts | 1.247981e-5 |
| Layer 1, T=1 complete router + routed + shared composition | 2.078950e-4 |

Layer 1 balanced maximum absolute error is 5.960464e-8 and P99 is zero; all
outputs are finite. The synthetic captured-SwiGLU fixture also checks exact
A4 payload/scale bytes against an independent CPU codec, including expert
specific down globals and persistent scratch reuse.

Additional coverage includes stable tied-score Top-8, dispatch padding/empty
experts/hotspots/tails, repeated execution, malformed dtype/device/shape/stride
rejection before launch, and an independent unweighted raw-GEMM baseline. The
unweighted specialization uses a compile-time branch so it never evaluates
route-placeholder loads. Run with the environment/UUID command above and
`-m unittest test.Qwen3_5MoECompat.unit.test_moe -v`.

Final coordinator rerun: **11 tests passed** in 12.478 s. Astra approved the
numerical paths and references; the coordinator verified the final symmetric
FP16/NVFP4 metadata validation and unweighted compile-time branch corrections.

Performance tuning is still pending: the current stable scatter is quadratic
in route count, and the fixed persistent grid is a correctness baseline.
Kernel-count/latency comparisons, longer-token MoE sweeps, backend profiling,
all 40 layers and four-layer integration remain later acceptance gates.

## M2c / M2a producers / stateless assembly — 2026-09-07

Implemented the true BT16 WY pipeline: isolated Gram, one FP32 register solve
per A column, FP16 A/U/W/H/R boundaries, FP32 R and state update, and separated
QK/decay/prior/local/output kernels. Public `chunk_gdn` streams one BT16 through
all stages before reusing `[B,1,...]` scratch. `token_offset` is a runtime value
and is separate from the fixed scratch slot. H16 snapshots the old state before
its update. There is no recurrence or Torch-matmul substitution for WY.
Rectangular diagnostic stage helpers are used only by component tests.

The streamed design removes `B * max_chunks` state-history amplification. The
suite includes the skewed packed lengths `[1]*127 + [1921]`, long slow decay
`g=-1e-4` at T2048, lengths 16/128/512, ragged tails, int64 offsets, empty
segments, repeated calls, 8192-channel Conv with no bias, stable softplus
negative tails and independent FP32 recurrence/state relative-L2 checks.
Nonempty inputs with `max_seqlen=0` are rejected before launch. Offset contents
remain documented, trusted device metadata.

The model now has nested config registration before AutoConfig, selective
original-layer assembly, global embedding/final-norm/head loading, `EntryClass`,
`forward_layer`, and `forward_no_cache`. Full Attention fuses the per-head
Q/gate split with Gemma Q/K normalization and partial NeoX RoPE. Both normalized
Q and K round to FP16 before rotation, and each K output has one writing CTA.
Full sigmoid-multiply and GDN ordinary-norm/SiLU producers directly emit A8;
residual add + post-attention Gemma norm retains the FP16 sum boundary.

Astra ultra reviewed the equations, casts, streamed addressing, scratch reuse,
state ordering, padding masks and model interfaces, and approved this
incremental milestone. Coordinator verification on the required V100:

| Verification | Result |
|---|---|
| `unit.test_gdn` | 15 passed, 6.561 s |
| `unit.test_reference_codec`, `unit.test_quantization`, `unit.test_model_ops` | 13 passed, 20.656 s |
| `unit.test_moe` after reference scale correction | 11 passed, 13.938 s |
| Earlier config + producer + T1/T3 four-layer suite | 8 passed, 7.768 s |
| Coordinator real layer 0, T2, complete GDN + MoE | finite FP16 `[2,2048]`, repeat bitwise equal; peak 1.593444 GiB |
| Coordinator real layers 0–3, T65, embedding/final norm/head | hidden `[65,2048]`, logits `[1,248320]`, finite, repeat bitwise equal; peak 4.878265 GiB |
| Same four-layer model, empty input | hidden `[0,2048]`, logits `[0,248320]` |

Agent warm CUDA measurement for isolated streamed GDN T2048, excluding
compilation: **43.890 ms**, peak allocated **72,453,632 bytes**. This is not a
full-layer latency or the final benchmark sweep. Earlier 136-second GDN test
time included compilation and repeated per-chunk specializations; it must not
be used as the denominator of a kernel speedup claim.

Review also caught a fused-A8 scale inconsistency: direct RN division by 448
differs from the frozen FP32 reciprocal multiply. On the coordinator's exact
half-boundary case (zero sigmoid gate, T17, seed 179), the rejected version had
319/544 differing scales and 13 differing payload bytes. Both producers now
use reciprocal multiplication; CPU A8/A4 references explicitly construct FP32
reciprocals of 448/6, so their scale rule is independent of CPU/CUDA constant
division lowering. The added regression passes exact scale and payload checks.
The real layer-3 attention oracle now uses independent A8 encoding throughout.

The complete M0 exact-name/shape manifest, all-40-layer scan, full M4 sequence
isolation and long-input memory acceptance, M5 projection packing, performance
sweep and profiler audit are still required. The four-layer smoke results do
not establish complete-model quality or cache-backed serving support.

`DESIGN.md` and `model_design.mmd` now document the quantized stateless design;
the earlier external architecture reports point to these maintained sources.

## M0 exact checkpoint manifest — 2026-09-07

The generated manifest describes **119,015** exact text/global tensor names,
shapes and dtypes. It includes all 130 attention FP8 matrices and their FP16
block scales, all 29,184 NVFP4 matrices and their three scale companions, all
1,536 FP16 routed matrices in layers 0/39, and the FP16 GDN, router, shared,
normalization, embedding and head components. Unrelated vision tensors are
allowed outside the text namespace.

The full audit compares exact name sets before validating each header.
`load_layer` checks the complete selected layer name set and headers before
reading tensor payloads; global loading also validates its manifest entries.
NVFP4 global multipliers are checked for positivity and finiteness on CPU
before the compact GPU transfer. Fixed config validation now covers GDN head
dimensions and Conv width, default partial RoPE, output gates, activation,
attention bias and untied embedding/head semantics.

Coordinator rerun of `unit.test_checkpoint` and `unit.test_model_config`:
**6 tests passed in 0.474 s**, including the real complete-header audit,
synthetic wrong-name/shape/dtype/scale cases and incompatible config rejection.
The real matrix counts remain **130 / 29,184 / 1,536**. This is structural
checkpoint validation; execution of every real expert remains the M3 gate.

## M1 signed-zero reference correction — 2026-09-07

The real layer-17 scan found 65 A8 payload differences, all `0x80` versus
`0x00` in a zero-scale K128 group. Captured FP16 producer values, activation
scales, the fused producer and the standalone Triton quantizer agreed. The
independent CPU A8 reference had incorrectly replaced negative zero by
positive zero for that group. It now divides by a safe scale of one and
preserves the sign. The separate NVFP4 zero-scale canonicalization remains
unchanged. No production math or error budget was changed.

Coordinator rerun of `unit.test_reference_codec` and `unit.test_quantization`
on the required V100: **9 tests passed in 1.690 s**, including new CPU/GPU
zero-group signed-zero fixtures. The three real-size W8A8 projection checks
remain between **1.84e-4 and 2.07e-4 NRMSE**. The all-layer scan will be rerun
with this corrected independent reference and fixed Top-8 accumulation order.
