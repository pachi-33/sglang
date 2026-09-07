#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_single_node_moe.sh <2|4|8>

Examples:
  ./run_single_node_moe.sh 8
  MODEL_PATH=/models/GLM-5.2 SERVER_PORT=8000 ./run_single_node_moe.sh 4
  ENABLE_SPECULATIVE=0 QUANTIZATION= ./run_single_node_moe.sh 2

Frequently used overrides:
  MODEL_PATH                 Model directory (default: /home/chenxu/glm5_2_weight/)
  SERVED_MODEL_NAME          API model name (default: glm52)
  SERVER_HOST / SERVER_PORT  Listen address and port (default: 127.0.0.1:6677)
  MAX_RUNNING_REQUESTS       Global request concurrency (default: 5 per NPU)
  MAX_PREFILL_TOKENS         Global prefill budget (default: 64000 per NPU)
  ENABLE_SPECULATIVE         Enable NEXTN/MTP decoding: 1 or 0 (default: 1)
  QUANTIZATION               Quantization method; empty disables the CLI option
  API_KEY                    Add --api-key when non-empty
EOF
}

NPU_COUNT="${1:-8}"
case "${NPU_COUNT}" in
  2|4|8) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: NPU count must be 2, 4, or 8; got '${NPU_COUNT}'." >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/home/chenxu/glm5_2_weight/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm52}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-6677}"
QUANTIZATION="${QUANTIZATION-modelslim}"
API_KEY="${API_KEY:-}"

PER_NPU_MAX_RUNNING_REQUESTS="${PER_NPU_MAX_RUNNING_REQUESTS:-5}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-$((NPU_COUNT * PER_NPU_MAX_RUNNING_REQUESTS))}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$((NPU_COUNT * 64000))}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-65536}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.86}"

ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-1}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-5}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-6}"
SPECULATIVE_DRAFT_KV_CACHE_DTYPE="${SPECULATIVE_DRAFT_KV_CACHE_DTYPE:-bf16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

case "${ENABLE_SPECULATIVE}" in
  0|1) ;;
  *)
    echo "ERROR: ENABLE_SPECULATIVE must be 0 or 1." >&2
    exit 2
    ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

# Use this checkout by default instead of a hard-coded shared repository.
export PYTHONPATH="${SGLANG_REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-300}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-68}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-60}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export ASCEND_USE_FIA="${ASCEND_USE_FIA:-1}"
export SGLANG_NPU_USE_MLAPO="${SGLANG_NPU_USE_MLAPO:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# DeepEP low-latency decode needs capacity for every token proposed by one
# attention-DP rank. Round up in case MAX_RUNNING_REQUESTS is not divisible by DP.
PER_RANK_REQUESTS=$(( (MAX_RUNNING_REQUESTS + NPU_COUNT - 1) / NPU_COUNT ))
TOKENS_PER_REQUEST=1
if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
  TOKENS_PER_REQUEST="${SPECULATIVE_NUM_DRAFT_TOKENS}"
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}"
else
  unset SGLANG_ENABLE_OVERLAP_PLAN_STREAM
fi

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$((PER_RANK_REQUESTS * TOKENS_PER_REQUEST))}"
export DEEPEP_HCCL_BUFFSIZE="${DEEPEP_HCCL_BUFFSIZE:-1000}"

launch_args=(
  -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${SERVER_HOST}"
  --port "${SERVER_PORT}"
  --nnodes 1
  --node-rank 0
  --tp-size "${NPU_COUNT}"
  --dp-size "${NPU_COUNT}"
  --enable-dp-attention
  --enable-dp-lm-head
  --attention-backend ascend
  --device npu
  --trust-remote-code
  --watchdog-timeout 9000
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --load-balance-method round_robin
  --moe-a2a-backend deepep
  --deepep-mode auto
  --enable-metrics
)

if [[ -n "${QUANTIZATION}" ]]; then
  launch_args+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${API_KEY}" ]]; then
  launch_args+=(--api-key "${API_KEY}")
fi

if [[ "${ENABLE_SPECULATIVE}" == "1" ]]; then
  launch_args+=(
    --speculative-algorithm NEXTN
    --speculative-draft-kv-cache-dtype "${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
  )
fi

echo "============================================================"
echo "Launching single-node MoE inference"
echo "model path               : ${MODEL_PATH}"
echo "served model name        : ${SERVED_MODEL_NAME}"
echo "listen address           : ${SERVER_HOST}:${SERVER_PORT}"
echo "NPU / TP / attention DP  : ${NPU_COUNT} / ${NPU_COUNT} / ${NPU_COUNT}"
echo "max running requests     : ${MAX_RUNNING_REQUESTS} (${PER_RANK_REQUESTS} per rank)"
echo "max prefill tokens       : ${MAX_PREFILL_TOKENS}"
echo "chunked prefill size     : ${CHUNKED_PREFILL_SIZE}"
echo "speculative decoding     : ${ENABLE_SPECULATIVE}"
echo "DeepEP tokens per rank   : ${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
echo "HCCL / GLOO interface    : ${HCCL_SOCKET_IFNAME} / ${GLOO_SOCKET_IFNAME}"
echo "============================================================"

exec "${PYTHON_BIN}" "${launch_args[@]}"
