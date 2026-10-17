# Decode-time MoE tracing

This module records Mixture-of-Experts routing decisions during decode and,
in IDs-only mode, target speculative verification. On CUDA, it can also record the hidden states immediately before each
MoE gate projection. Router inputs are quantized and packed as symmetric
groupwise int4 before they leave the device.

Tracing is disabled by default. Enabling route and activation traces is
independent, so collect only the data needed for an experiment.

## Capability matrix

| Device | Expert IDs and weights | Router inputs | Parallelism |
| --- | --- | --- | --- |
| CUDA | Yes | Yes, groupwise int4 | TP and current CUDA DP modes; no PP |
| Ascend NPU | Yes | No | TP only |

`--moe-trace-ids-only` adds forward-aligned expert-ID traces for concurrent
requests on CUDA and NPU. It supports ordinary decode and the EAGLE, EAGLE3,
NEXTN (MTP), and STANDALONE verify paths with TP-only execution. It excludes
weights and activations. DP/EP/CP and two-batch overlap (TBO) are unsupported;
normal overlap scheduling and continuous batching are supported.

NPU support has been code-reviewed and covered by CPU structural tests, but has
not yet been validated on NPU hardware.

The route trace contains logical expert IDs and final routing weights. Capture
happens after routing normalization and scaling, before any EPLB
logical-to-physical remapping. Fused shared-expert slots are not included.

## Requirements

- Use an MoE model whose active backend exposes materialized expert IDs and
  weights. Startup fails instead of silently producing incomplete traces when
  a routed MoE site or backend cannot be traced.
- Install `safetensors` in the server environment. Trace chunks use the
  safetensors format and do not use pickle.
- Choose an output directory with enough free space. Router-input traces can be
  large even after int4 packing.

## Start a traced server

Replace `MODEL_PATH` and the output directory in these examples.

### CUDA routes and router inputs

```bash
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --device cuda \
  --moe-trace-output-dir /tmp/sglang-moe-trace \
  --moe-trace-expert-routes \
  --moe-trace-router-inputs \
  --moe-trace-max-decode-tokens 8
```

To collect routes without activations, omit
`--moe-trace-router-inputs`. To collect activations without routes, omit
`--moe-trace-expert-routes`.

### Ascend NPU routes with tensor parallelism

```bash
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --device npu \
  --tp-size 2 \
  --moe-trace-output-dir /tmp/sglang-moe-trace \
  --moe-trace-expert-routes \
  --moe-trace-max-decode-tokens 8
```

NPU rejects `--moe-trace-router-inputs`, DP attention, data parallelism,
MoE data parallelism, and DWDP while tracing is enabled.

### Concurrent target IDs with speculative decoding

For an EAGLE/NEXTN-compatible model, add the appropriate speculative options
for that model, for example:

```bash
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --device npu \
  --tp-size 2 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --moe-trace-output-dir /tmp/sglang-target-ids \
  --moe-trace-expert-routes \
  --moe-trace-ids-only
```

Use a fresh output directory for each IDs-only run. A pre-existing batch
manifest is rejected rather than overwritten. No draft model, prefill,
warmup/capture, or graph-padding rows are dumped. Every real target verify
candidate is recorded, including rejected branches. Positions can repeat in
a tree; candidate indices preserve distinct rows instead of deduplicating them.

`--moe-trace-max-decode-tokens` limits **candidate input rows per request** in
this mode, rather than emitted/accepted output tokens. Zero is unlimited.
Already-executed overlap lookahead forwards are retained after logical request
completion. Request completion in the manifest is therefore independent of
whether later executed frames contain that request ID.

IDs-only output uses format version 2 with forward-aligned batch chunks:

```text
/tmp/sglang-target-ids/
├── manifest.json
└── rank-000/
    ├── chunk-000000.meta.json
    └── chunk-000000.safetensors
```

The asynchronous writer aggregates complete forwards up to 8 MiB or 32 forwards,
and flushes a partial chunk after 0.5 seconds of inactivity or on shutdown.
The bounded queue honors the existing `block`/`drop` policy and records drop
counts in the run manifest. Only CPU payloads whose D2H copy has completed are
submitted. The manifest commits each data/metadata pair with SHA-256 digests.

Each sidecar lists the ordered forward IDs, phases (`decode` or `target_verify`),
chunk row ranges, request IDs, request row splits, and source capture indices.
For example, requests A and B with three candidates each have row splits
`[0, 3, 6]`. A graph padded to eight rows still exports only those six rows.
The tensors are `input_token_ids`, `positions`, `candidate_indices`,
`site_valid`, and `site.<id>.expert_ids` (int32, `[rows, routed_top_k]`).
Unvisited sites have `site_valid=false` and expert IDs `-1`.

Read complete forwards or reconstruct an individual request:

```python
from sglang.srt.moe_trace.loader import MoeTraceBatchLoader

loader = MoeTraceBatchLoader("/tmp/sglang-target-ids")
for frame, tensors in loader.iter_forwards():
    print(frame["forward_id"], frame["request_ids"], frame["row_splits"])
    print(tensors["site.0.expert_ids"])

trace = loader.load_trace("REQUEST_ID")
# Per-row forward_ids and candidate_indices retain the execution timeline.
print(trace["forward_ids"], trace["candidate_indices"])
```

The original per-request format and `MoeTraceLoader` continue to apply when
`--moe-trace-ids-only` is omitted.

## Run a small smoke request

Use several output tokens because the token produced by the prefill pass is not
part of the trace. Keeping `max_new_tokens` small bounds the activation output.

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "rid": "moe-trace-smoke",
    "text": "Give one short fact about Saturn.",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 4,
      "ignore_eos": true
    }
  }'
```

Without IDs-only mode, only standard decode steps are persisted. Prefill, including `prefill_last`, is
excluded. The routing computation that produces the first generated token is
therefore not traced; that token can still appear as the input to the first
decode row.

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--moe-trace-output-dir PATH` | unset | Enables the trace configuration and selects its root directory. One or both feature flags are also required. |
| `--moe-trace-expert-routes` | false | Records logical expert IDs and final routing weights. Supported on CUDA and NPU. |
| `--moe-trace-ids-only` | false | Records target IDs without weights/activations in v2 batch chunks. Requires `--moe-trace-expert-routes`; supports EAGLE-family verification and concurrent TP requests. |
| `--moe-trace-router-inputs` | false | Records the hidden state immediately before each gate projection as groupwise int4. CUDA only. |
| `--moe-trace-max-decode-tokens N` | `0` | Maximum traced decode rows per request. `0` means unlimited. Use a small value for activation tests. |
| `--moe-trace-activation-group-size N` | `128` | Even, positive quantization group size for router inputs. It is validated for every enabled trace configuration, including route-only NPU tracing. |
| `--moe-trace-queue-depth N` | `2` | Maximum number of host batches waiting for the asynchronous writer. |
| `--moe-trace-overflow-policy {block,drop}` | `block` | Blocks inference result processing or drops a trace batch when the writer queue is full. `block` preserves every accepted trace row. |

## Output layout

The writer creates one request directory using a sanitized request ID plus a
stable hash:

```text
/tmp/sglang-moe-trace/
└── moe-trace-smoke-<hash>/
    ├── manifest.json
    └── rank-000/
        ├── chunk-000000.safetensors
        └── chunk-000001.safetensors
```

`manifest.json` contains the original request ID, format version, completion
status, site metadata, quantization metadata, chunk shapes, sizes, and SHA-256
digests. A request status is one of `open`, `complete`, `failed`, or
`truncated`.

Every chunk contains row-aligned request metadata:

- `input_token_ids`: input token for each decode row.
- `positions`: decode position for each row.
- `site_valid`: bool mask with shape `[rows, num_sites]` indicating whether
  each registered MoE site wrote that row.

Optional per-site tensors use the site ID from the manifest:

- `site.<id>.expert_ids`: int32 logical routed expert IDs with shape
  `[rows, routed_top_k]`.
- `site.<id>.expert_weights`: FP32 final routing weights with shape
  `[rows, routed_top_k]`.
- `site.<id>.activation_q`: uint8-packed int4 router inputs with shape
  `[rows, ceil(hidden_size / 2)]`.
- `site.<id>.scales`: FP16 group scales with shape
  `[rows, ceil(hidden_size / group_size)]`.

For tensor parallel execution, routing inputs and decisions are replicated. A
single TP leader writes the trace, preventing multiple ranks from racing on the
same request manifest.

## Load and validate a trace

`MoeTraceLoader` validates the format version, the tensor shapes recorded in
the manifest, file sizes, and SHA-256 digests before returning CPU tensors.

```python
import json
from pathlib import Path

from sglang.srt.moe_trace.loader import MoeTraceLoader

root = Path("/tmp/sglang-moe-trace")
manifest_path = next(root.glob("*/manifest.json"))
request_id = json.loads(manifest_path.read_text())["request_id"]

loader = MoeTraceLoader(root)
manifest = loader.manifest(request_id)
trace = loader.load_trace(request_id)

print(manifest["status"], manifest["total_rows"])
for site in manifest["sites"]:
    site_id = site["site_id"]
    ids_key = f"site.{site_id}.expert_ids"
    weights_key = f"site.{site_id}.expert_weights"
    if ids_key in trace:
        print(site["layer_id"], trace[ids_key], trace[weights_key])
```

Pass `verify=False` to `iter_chunks` or `load_trace` only when SHA-256
verification is intentionally unnecessary.

## Dequantize CUDA router inputs

Use the hidden size from the site metadata and the group size from the manifest.

```python
from sglang.srt.moe_trace.codec import dequantize_packed_int4

site = manifest["sites"][0]
site_id = site["site_id"]
router_inputs = dequantize_packed_int4(
    trace[f"site.{site_id}.activation_q"],
    trace[f"site.{site_id}.scales"],
    hidden_size=site["hidden_size"],
    group_size=manifest["quantization"]["group_size"],
)
```

The codec uses symmetric per-row, per-group quantization with range `[-7, 7]`.
Two signed int4 values are stored per byte in two's-complement form, with the
first value in the low nibble. Dequantization is approximate and returns FP16
by default.

## Current limitations

- Traces cover decode and, in IDs-only mode, target verification; prefill and
  `prefill_last` are excluded.
- Speculative decoding requires IDs-only mode and a supported EAGLE-family
  algorithm; DFLASH, DSPARK, UNO, NGRAM, and FROZEN_KV_MTP are not yet supported.
- Diffusion-LLM decoding and pipeline parallelism are rejected at startup.
- NPU supports route traces only and requires TP-only execution.
- Router-input quantization is CUDA-only.
- Backends that do not expose the logical IDs and weights consumed by the MoE
  experts are rejected at startup.
- Trace I/O can affect latency. Bound the number of rows and prefer route-only
  collection when activations are not required.

## Run focused tests

Run tests in the project environment:

```bash
python -m unittest discover \
  -s test/registered/unit/moe_trace \
  -p 'test_*.py' -v
```

On a CUDA host, run the kernel and graph-replay coverage:

```bash
python -m unittest discover \
  -s test/registered/kernels/moe_trace \
  -p 'test_*.py' -v
```

NPU runtime, graph replay, and D2H-to-writer behavior still require validation
on real Ascend hardware.
`test_recorder_ids_graph.py` exercises multi-request target graph replay on
CUDA and Ascend NPU; each test skips when its accelerator is unavailable.
