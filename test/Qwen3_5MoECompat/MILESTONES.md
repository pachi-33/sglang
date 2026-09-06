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
| M1 | Independent references and codec tests | Pending |
| M2a | Common layers, W8A8 and Full Attention | Pending |
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
