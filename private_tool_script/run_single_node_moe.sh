#!/usr/bin/env bash

set -Eeuo pipefail

# ==================== 修改这里 ====================
# 2 卡: NUM_NPUS=2，ASCEND_RT_VISIBLE_DEVICES=0,1
# 4 卡: NUM_NPUS=4，ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# 8 卡: NUM_NPUS=8，ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_NPUS=2
export ASCEND_RT_VISIBLE_DEVICES=14,15

MODEL_PATH=/home/weights/Qwen3-30B-A3B-W8A8 
SERVED_MODEL_NAME=qwen3
SERVER_HOST=127.0.0.1
SERVER_PORT=8818
MEM_FRACTION_STATIC=0.7

# Decode CUDA Graph；在 NPU 上底层实际使用 NPUGraph。
CUDA_GRAPH_BACKEND_DECODE=full #full disabled
CUDA_GRAPH_MAX_BS_DECODE=5

# 1: 开启 NEXTN 投机解码；0: 关闭投机解码。
ENABLE_SPECULATIVE=0

# 1: 主模型 KV Cache 使用 FP8；0: 使用 BF16。
ENABLE_FP8_KV_CACHE=0
# ==================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../python${PYTHONPATH:+:${PYTHONPATH}}"

# 每卡最多 5 个并发请求，每卡分配约 64000 个 prefill token 预算。
MAX_RUNNING_REQUESTS=$((NUM_NPUS * 5))
MAX_PREFILL_TOKENS=$((NUM_NPUS * 64000))

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1
export TRANSFORMERS_VERBOSITY=error

export DEEPEP_HCCL_BUFFSIZE=1000

case "${ENABLE_FP8_KV_CACHE}" in
  1) KV_CACHE_DTYPE=fp8_e4m3 ;;
  0) KV_CACHE_DTYPE=bf16 ;;
  *)
    echo "ERROR: ENABLE_FP8_KV_CACHE must be 0 or 1." >&2
    exit 2
    ;;
esac

case "${ENABLE_SPECULATIVE}" in
  1)
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    # 5 个请求/rank × 每轮 6 个 NEXTN draft token = 30。
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=30
    set -- \
      --speculative-algorithm NEXTN \
      --speculative-draft-kv-cache-dtype bf16 \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 6
    ;;
  0)
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=5
    set --
    ;;
  *)
    echo "ERROR: ENABLE_SPECULATIVE must be 0 or 1." >&2
    exit 2
    ;;
esac

cat <<EOF
============================================================
Single-node MoE inference configuration
------------------------------------------------------------
[Model]
MODEL_PATH                                      = ${MODEL_PATH}
SERVED_MODEL_NAME                               = ${SERVED_MODEL_NAME}
QUANTIZATION                                    = modelslim
TRUST_REMOTE_CODE                               = enabled

[Server]
SERVER_HOST                                     = ${SERVER_HOST}
SERVER_PORT                                     = ${SERVER_PORT}
ENABLE_METRICS                                  = enabled
WATCHDOG_TIMEOUT                                = 9000

[Parallel]
ASCEND_RT_VISIBLE_DEVICES                       = ${ASCEND_RT_VISIBLE_DEVICES}
NUM_NPUS                                        = ${NUM_NPUS}
NNODES                                          = 1
NODE_RANK                                       = 0
TP_SIZE                                         = ${NUM_NPUS}
DP_SIZE                                         = ${NUM_NPUS}
ENABLE_DP_ATTENTION                             = enabled
ENABLE_DP_LM_HEAD                               = enabled
LOAD_BALANCE_METHOD                             = round_robin

[Memory and scheduling]
MAX_RUNNING_REQUESTS                            = ${MAX_RUNNING_REQUESTS}
MEM_FRACTION_STATIC                             = ${MEM_FRACTION_STATIC}
MAX_PREFILL_TOKENS                              = ${MAX_PREFILL_TOKENS}
CHUNKED_PREFILL_SIZE                            = 65536
ENABLE_FP8_KV_CACHE                             = ${ENABLE_FP8_KV_CACHE}
KV_CACHE_DTYPE                                  = ${KV_CACHE_DTYPE}
PYTORCH_NPU_ALLOC_CONF                          = ${PYTORCH_NPU_ALLOC_CONF}

[CUDA Graph / NPU Graph]
CUDA_GRAPH_BACKEND_DECODE                       = ${CUDA_GRAPH_BACKEND_DECODE}
CUDA_GRAPH_MAX_BS_DECODE                        = ${CUDA_GRAPH_MAX_BS_DECODE}
CUDA_GRAPH_BS_DECODE                            = auto, up to ${CUDA_GRAPH_MAX_BS_DECODE}
PREFILL_CUDA_GRAPH                              = disabled

[Ascend and communication]
DEVICE                                          = npu
ATTENTION_BACKEND                               = ascend
ASCEND_USE_FIA                                  = ${ASCEND_USE_FIA}
SGLANG_NPU_USE_MLAPO                            = ${SGLANG_NPU_USE_MLAPO}
STREAMS_PER_DEVICE                              = ${STREAMS_PER_DEVICE}
HCCL_CONNECT_TIMEOUT                            = ${HCCL_CONNECT_TIMEOUT}
HCCL_EXEC_TIMEOUT                               = ${HCCL_EXEC_TIMEOUT}
HCCL_OP_EXPANSION_MODE                          = ${HCCL_OP_EXPANSION_MODE}
ACL_DEVICE_SYNC_TIMEOUT                         = ${ACL_DEVICE_SYNC_TIMEOUT}
HCCL_SOCKET_IFNAME                              = ${HCCL_SOCKET_IFNAME}
GLOO_SOCKET_IFNAME                              = ${GLOO_SOCKET_IFNAME}
DEEPEP_HCCL_BUFFSIZE                            = ${DEEPEP_HCCL_BUFFSIZE}
MOE_A2A_BACKEND                                 = deepep
DEEPEP_MODE                                     = auto
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = ${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}

[Speculative decoding]
ENABLE_SPECULATIVE                              = ${ENABLE_SPECULATIVE}
SPECULATIVE_ALGORITHM                           = NEXTN (used when enabled)
SPECULATIVE_DRAFT_KV_CACHE_DTYPE                = bf16 (used when enabled)
SPECULATIVE_NUM_STEPS                           = 5 (used when enabled)
SPECULATIVE_EAGLE_TOPK                          = 1 (used when enabled)
SPECULATIVE_NUM_DRAFT_TOKENS                    = 6 (used when enabled)
SGLANG_ENABLE_OVERLAP_PLAN_STREAM               = ${SGLANG_ENABLE_OVERLAP_PLAN_STREAM}

[Runtime]
PYTHONPATH                                      = ${PYTHONPATH}
TRANSFORMERS_VERBOSITY                          = ${TRANSFORMERS_VERBOSITY}
HTTP_PROXY / HTTPS_PROXY                        = unset
HCCL_IF_IP / HCCL_SOCKET_FAMILY / RANK_TABLE    = unset
============================================================
EOF

echo "[Launch command]"
set -x
exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size "${NUM_NPUS}" \
  --dp-size "${NUM_NPUS}" \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --attention-backend ascend \
  --device npu \
  --trust-remote-code \
  --watchdog-timeout 9000 \
  --cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}" \
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}" \
  --disable-prefill-cuda-graph \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --quantization modelslim \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  --chunked-prefill-size 65536 \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --load-balance-method round_robin \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  "$@" \
  --enable-metrics
