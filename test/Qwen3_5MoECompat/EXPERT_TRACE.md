# Qwen3.5 Expert Activation Trace

The single-GPU ExpertPack backend can record the logical Top-8 expert IDs that
directly produce each sampled output token. Tracing is disabled unless a
request opts in.

Start the API with a server-owned trace directory:

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.single_gpu_api \
  --model-dir /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16 \
  --expert-pack-manifest /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json \
  --expert-cache-mib 7168 \
  --expert-trace-dir /tmp/qwen35-expert-traces \
  --host 127.0.0.1 --port 8818
```

Set `expert_trace` on exactly the request to record:

```bash
curl http://127.0.0.1:8818/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","max_new_tokens":8,"expert_trace":true}'
```

If the response ID is `gen-abc`, the committed files are:

```text
/tmp/qwen35-expert-traces/gen-abc.trace.json
/tmp/qwen35-expert-traces/gen-abc.trace.npz
```

The API does not return a filesystem path. The request cannot choose one.
`expert_trace=true` is accepted by `/generate`, `/v1/completions`, and
`/v1/chat/completions`, including streaming requests. A server without
`--expert-trace-dir` rejects an opted-in request before inference.

For a one-shot CLI request, pass a basename:

```bash
python -m sglang.srt.layers.qwen3_5.single_gpu \
  --raw-prompt --prompt Hello --max-new-tokens 8 \
  --expert-trace-output /tmp/hello
```

This writes `/tmp/hello.trace.json` and `/tmp/hello.trace.npz`.

Load numeric arrays without pickle:

```python
import numpy as np

with np.load("/tmp/hello.trace.npz", allow_pickle=False) as trace:
    expert_ids = trace["expert_ids"]              # uint8 [N, 40, 8]
    sampled = trace["sampled_token_ids"]          # int32 [N]
    model_inputs = trace["model_input_token_ids"] # int32 [N]
    positions = trace["model_input_positions"]    # int32 [N]
    phase = trace["phase"]                        # uint8 [N]
```

Row zero contains the routes at the final prompt position that produced the
first sampled token. Each later row contains the routes of the preceding
sampled token during decode. The last sampled token is therefore represented
without running an extra decode. `expert_ids[row, layer, rank]` is always a
global expert ID in `0..255`; it is never a GPU cache slot.

The JSON file is the commit marker and includes model, ExpertPack, GPU, shape,
and SHA-256 identity. A traced inference failure can publish a `status=failed`
artifact containing only fully sampled rows. Failure to create, transfer,
validate, or atomically publish the trace poisons the single-request cache and
places the backend in `FAILED`; `/health` and later generation calls return
503 until the process is restarted.
