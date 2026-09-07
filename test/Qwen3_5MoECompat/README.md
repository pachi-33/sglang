# Qwen3.5 MoE on V100 and RTX 4070 SUPER

This suite implements text-only inference for
`Qwen-AgentWorld-35B-A3B-NVFP4_fp16`, including a configurable two-worker pipeline
and a single-request cache. Production generation runs one complete prefill,
then one-token decode calls. By default RTX 4070 SUPER owns embedding, layers
0–16, final norm and the LM head; V100 owns layers 17–39. Each GPU has its own
Python process, and hidden states travel through CPU memory.

The selected-layer `forward_no_cache` compatibility API, original layers 0–3
integration, and independent 40-layer precision scan remain available. The
pipeline has a batch-one greedy text CLI and a persistent single-request HTTP
API; neither is integrated into SGLang's `ModelRunner`, scheduler, radix
attention or general sstate management.

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
The same software-decode path runs on validated SM70 and SM89; SM70 has no
native FP8/FP4 MMA. Existing performance reports remain V100-only measurements.

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

## Single-request state, text CLI and HTTP API

The cache holds at most 2048 consumed tokens, with one fresh contiguous prompt
and no subsequent multi-turn append. For a generation limit R, the public
context check is `prompt_tokens + R <= 2048`. The final sampled token is not
cached, but it still counts toward this public context limit.
The default front worker owns 13 GDN Conv tails/recurrent states and four Full
Attention K/V pairs; the back owns 17 GDN states and six Full Attention pairs.
At capacity 2048 these caches occupy approximately 42.61 MiB and 58.80 MiB,
respectively. GDN prefill retains its final recurrent/WY state and raw
QKV tail; Full Attention retains normalized, RoPE-applied K and projected V.
Decode updates these buffers in place. KV uses a valid length, without paging
or concatenation.

The runner API makes cache ownership explicit:

```python
cache = runner.allocate_request_cache(capacity=2048)
hidden = runner.prefill_hidden(prompt_hidden, cache=cache)
hidden = runner.decode_hidden(token_hidden, cache=cache, expected_prefix_len=L)
runner.reset_request_cache(cache)
```

Reset clears GDN/Conv state and invalidates KV length. An execution failure
poisons the cache until reset. The controller validates both workers' epoch,
step and consumed length before advancing; a partial failure resets both.
The first generated token comes from prompt logits, so R>0 generated tokens
require R−1 decode calls. The final sampled token is not inserted into cache,
and every completed or failed request resets both workers.

From the repository root, run a chat prompt with the cached checkpoint:

```bash
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.pipeline \
  --prompt '请用一句话解释 KV 缓存。' --max-new-tokens 8 --print-token-ids
```

Omit `--prompt` to read stdin, or add `--raw-prompt` to bypass the checkpoint's
chat template. The controller sets each worker's GPU UUID before process
startup; do not expose both GPUs inside one worker, because Triton 2.3.1 caches
its compilation target per process. The tokenizer's two-dimensional merges
are converted in memory for the legacy Transformers environment. Sampling
masks head rows `[248077,248320)` and stops on EOS IDs `{248046,248044}`.
The production defaults are `--front-uuid <4070 UUID>`,
`--back-uuid <V100 UUID>`, and `--split-layer 17`; these options can also select
the legacy V100-front 20/20 layout explicitly.

For a full-prefix test oracle, add `--validate-stateless`. This reports
cached/stateless greedy, hidden/logits error and router diagnostics; full-prefix
recomputation is only enabled by this validation option.

To keep both GPU workers loaded and return results over HTTP, start the
single-process API server:

```bash
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.pipeline_api \
  --host 127.0.0.1 --port 30000 --served-model-name agent-world
```

Set `--api-key` or `SGLANG_API_KEY` to require a Bearer token. The server
provides `/health`, `/generate`, `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`. For example:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"agent-world","messages":[{"role":"user","content":"解释 KV cache。"}],"max_completion_tokens":8,"temperature":0}'
```

The API is deliberately non-streaming, greedy (`temperature=0`, `top_p=1`),
and `n=1`. A concurrent request receives HTTP 429 instead of being batch
scheduled. Responses include OpenAI-style text/usage plus exact prompt and
completion token IDs under `sglang`; native `/generate` returns the same data
under `meta_info`. Every response or error still ends with the two-worker reset.

SM70 and SM89 execute the same Qwen3.5 Python/Triton operator sources, compiled
independently in their worker processes for each architecture; they do not
share a compiled binary. GPU numeric indices may appear in either order because
the controller selects physical devices by UUID. Physical roles and the split
are configurable. The validated production layout gives the smaller 4070 front
17 layers and its global weights; a trial with 18 front layers passed short
decode but OOMed during 2048-token prefill, so 17 is the safe default. The old
V100-front/4070-back 20/20 layout remains selectable.

## Running tests

Every Python invocation uses the `sglang-v100` environment. GPU unit tests
require exactly one validated UUID and its matching capability. The
`Qwen35GPUCompatTestCase` base (also exported as the legacy `V100TestCase` alias)
acquires `/tmp/qwen35-gpu-<UUID>-sm<capability>.lock`; do not wrap those tests
in another shell `flock`. Run the same `unit/` suite in separate processes:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m unittest discover -s test/Qwen3_5MoECompat/unit -p 'test_*.py' -v
```

```bash
CUDA_VISIBLE_DEVICES=GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m unittest discover -s test/Qwen3_5MoECompat/unit -p 'test_*.py' -v
```

Use either UUID prefix with
`-m unittest test.Qwen3_5MoECompat.integration.test_stateful_runner -v`
for real-layer cache checks. Full discovery from `test/Qwen3_5MoECompat`
additionally includes the historical four-layer integration and opt-in scan
smoke; its test count is distinct from the `unit/` count.

The complete real-weight scan is a separate command; the small scan smoke in
unittest discovery is opt-in:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python -u -m \
  test.Qwen3_5MoECompat.integration.layer_scan --layers 0-39 \
  --output-dir test/Qwen3_5MoECompat/reports/layer_scan
```

The scan CLI and benchmark modules remain V100-only and share its UUID lock
internally. **Do not add
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

[VALIDATION_SM70_SM89_PIPELINE.md](VALIDATION_SM70_SM89_PIPELINE.md) records the
dual-GPU cache contract, branch verification evidence and repeatable pipeline
acceptance command. Final-code validation passed 126/126 unit tests and 5/5
real-layer cache tests on each GPU, with no unit skips/errors. The saved
[pipeline acceptance JSON](reports/pipeline_acceptance.json) has `ok=true`,
including eight-step greedy agreement, reset/chat determinism and 2048-token
prefill memory checks. Full-model hidden/logits and router differences are
reported explicitly in the validation document.
[VALIDATION.md](VALIDATION.md) preserves the historical
independent 40-layer scan and four-layer stateless evidence, with source hashes.
[PERFORMANCE.md](PERFORMANCE.md) records fused comparisons and the final full
four-layer Triton profiler audit. [MILESTONES.md](MILESTONES.md) preserves
implementation milestones, verification evidence and scope limits.
