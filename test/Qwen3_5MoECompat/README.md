# Qwen3.5 MoE ExpertPack on one V100

The current delivery target is complete text-only inference for
`Qwen-AgentWorld-35B-A3B-NVFP4_fp16` in one process on one 16 GB SM70 V100.
Layers 1–38 read routed NVFP4 experts on demand from an immutable ExpertPack;
layers 0/39 routed FP16 experts and all dense/router/shared/global weights stay
resident. The public path is batch one, one active request, greedy generation,
and a maximum context of 2048. It does not use the RTX 4070 SUPER, the legacy
two-worker pipeline, SGLang's old scheduler, radix cache, TP or PP.

The earlier dual-GPU pipeline, selected-layer `forward_no_cache` API, four-layer
integration and independent 40-layer scan remain as regression oracles. Their
commands and measurements are historical evidence rather than the deployment
path for ExpertPack.

## ExpertPack build and single-V100 entry points

Build the byte-preserving pack once:

```bash
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.expert_pack.build \
  --model-dir /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16 \
  --output-dir /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1
```

The builder uses a process lock plus PID/UUID-specific partial files and publishes
`experts.pack` before atomically publishing `complete=true` `manifest.json`.
Validate all 9,728 records, padding, whole-pack SHA and source checkpoint bytes:

```bash
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.expert_pack.validate \
  --manifest /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json \
  --model-dir /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16
```

Run the complete 40-layer CLI with only the V100 visible:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.single_gpu \
  --raw-prompt --prompt Hello --max-new-tokens 8 --print-token-ids
```

The HTTP module is `sglang.srt.layers.qwen3_5.single_gpu_api` and exposes the
existing `/generate`, `/v1/completions` and `/v1/chat/completions` routes through
one backend and one uvicorn worker. Live V100 evidence covers HTTP 200 for all
three routes, a concurrent-request 429, and checksum/short-read/H2D-triggered
FAILED latches whose first request, `/health`, and later request all return 503.
The OpenAI completion and chat routes support `stream=true`; token frames carry
the exact sampled token ID, a terminal frame carries usage and finish reason,
and `[DONE]` is published only after request-cache reset. Incomplete byte-level
text is buffered until its UTF-8 sequence is stable. The completion route also
accepts SGLang's `ignore_eos` extension for fixed-length greedy benchmarks.
Start the benchmark-compatible server and run the saved streaming profile:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.single_gpu_api \
  --model-dir /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16 \
  --served-model-name /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16 \
  --expert-pack-manifest /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json \
  --host 127.0.0.1 --port 8818

PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.bench_serving --dataset-name random --backend sglang \
  --model /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16 \
  --dataset-path /home/yaozhenyang/dev/sglang-v100/ShareGPT_V3_unfiltered_cleaned_split.json \
  --host 127.0.0.1 --port 8818 --max-concurrency 1 \
  --random-input-len 1024 --random-output-len 128 --num-prompts 8 \
  --random-range-ratio 1 --request-rate inf
```

The recorded streaming run completed 8/8 requests in 220.50 s at 4.64 output
token/s. Mean/median TTFT were 7020.97/7442.25 ms; mean/median ITL were
161.73/157.26 ms over 1,016 decode intervals. Every terminal usage value was
128 and the concatenated stream text retokenized to all 1,024 generated tokens.
See [the streaming serving report](reports/bench_serving_single_gpu_v100_stream_1024_128.txt)
and [the timed eight-token smoke](reports/single_gpu_api_stream_smoke_v100.txt).

## Current ExpertPack evidence

The v1 pack is 17,253,269,504 bytes with SHA-256
`5d53114a227ed9d7a86e656a557b5c6ffbb9d10f46ddb5ca27671397ccdbe1f5`.
Independent full validation passed every payload SHA, every padding region,
the whole-pack digest and byte-for-byte comparison with all source tensors.

Root-reviewed V100 evidence is saved in
[expert_offload_acceptance_v100_7168.json](reports/expert_offload_acceptance_v100_7168.json):

- raw `Hello` produced `[11,271,40,1044,4313,310,958,279]` exactly;
- A/B/A reset and generation after a capacity error were exact;
- a 2048-token prefill was finite; the next decode failed explicitly without
  corrupting the following request;
- the typed cache used 7,515,015,536 bytes (4,247 experts); peak reserved was
  14,971,568,128 bytes and the 16 GB card retained 1,956,773,888 bytes;
- cold `Hello` TTFT/mean ITL were 3372.505/276.320 ms, warm `Hello` values were
  68.491/66.792 ms. These are one-run characterization values, not an SLA.

Additional root-reviewed V100 evidence is saved in:

- [expert_offload_layer1_v100.json](reports/expert_offload_layer1_v100.json):
  T=1/32/2048 resident/offload output, router IDs and router weights are exact;
  a hot acquire adds zero pack reads and zero H2D bytes;
- [single_gpu_api_smoke_v100.json](reports/single_gpu_api_smoke_v100.json):
  `/generate`, `/v1/completions` and `/v1/chat/completions` all return 200;
- [single_gpu_api_stream_smoke_v100.txt](reports/single_gpu_api_stream_smoke_v100.txt):
  eight `Hello` tokens arrive as separate timed SSE frames, followed by usage,
  finish reason and `[DONE]`;
- [bench_serving_single_gpu_v100_stream_1024_128.txt](reports/bench_serving_single_gpu_v100_stream_1024_128.txt):
  the fixed 1024-to-128 profile passes 8/8 with real TTFT/ITL measurements,
  exact terminal token counts and a post-run READY store;
- [single_gpu_api_concurrency_v100.json](reports/single_gpu_api_concurrency_v100.json):
  a live contender returns 429 while the active request completes normally;
- [single_gpu_api_checksum_failure_v100.json](reports/single_gpu_api_checksum_failure_v100.json):
  checksum failure is latched once as FAILED and the first request, health probe
  and subsequent request all return 503.
- [single_gpu_api_short_read_failure_v100.json](reports/single_gpu_api_short_read_failure_v100.json):
  truncating a startup-valid pack before the first routed read yields a 503,
  `FAILED` health with one I/O/fatal error, and persistent 503 rejection.
- [single_gpu_api_h2d_failure_v100.json](reports/single_gpu_api_h2d_failure_v100.json):
  an injected H2D RuntimeError poisons the request cache, publishes no resident
  expert, records one CUDA/fatal error, and returns 503 for first/health/later.

CPU API contracts cover disconnect-triggered cancellation, reset-before-unlock,
and shutdown waiting for the generation worker. A real V100 disconnect with an
in-flight lease, a fresh-process restart after fatal failure, long-running stress,
and clean-tree release regression remain pending.

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

## Legacy two-worker state, CLI and HTTP reference

The remainder of this section documents the retained 4070/V100 pipeline. It is
not the ExpertPack deployment command and is not part of current single-V100
acceptance.

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
The legacy pipeline defaults are `--front-uuid <4070 UUID>`,
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

This legacy two-worker backend remains non-streaming, greedy (`temperature=0`,
`top_p=1`), and `n=1`. A concurrent request receives HTTP 429 instead of being
batch scheduled. Responses include OpenAI-style text/usage plus exact prompt
and completion token IDs under `sglang`; native `/generate` returns the same
data under `meta_info`. Every response or error still ends with the two-worker
reset.

SM70 and SM89 execute the same Qwen3.5 Python/Triton operator sources, compiled
independently in their worker processes for each architecture; they do not
share a compiled binary. GPU numeric indices may appear in either order because
the controller selects physical devices by UUID. Physical roles and the split
are configurable. The previously validated dual-GPU layout gives the smaller 4070 front
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
