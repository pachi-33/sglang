#!/bin/bash

# ===== Cleanup =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

pkill -9 python  2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 VLLM   2>/dev/null || true

# ===== Environment =====
export PYTHONPATH=/mnt/share/chenxu_will_delete/GLM5_2/sglang/python:$PYTHONPATH

export DEEPEP_HCCL_BUFFSIZE=1536
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# [FIA]
export ASCEND_USE_FIA=1

# [MLAPO]
export SGLANG_NPU_USE_MLAPO=1

# [DEEPEP]
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=30

# [Prefill Delay]
#export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
#export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

# [MTP]
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export TRANSFORMERS_VERBOSITY=error

# [多机]
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME=data0.3001

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

# ===== Model Config =====
MODEL_PATH=/home/chenxu/glm5_2_weight/
SERVED_MODEL_NAME=glm52
SERVER_HOST=141.61.33.21
SERVER_PORT=6677

# ===== Cluster Config ===========================================
# 每台机器: IP + HCCL 网卡名 (一一对应)
NODE_IPS=(
  "141.61.33.21"
  "141.61.33.23"
  "141.61.33.24"
  "141.61.33.25"
)
HCCL_IFS=(
  "eth4"
  "enp34s0f1"
  "enp34s0f1"
  "enp34s0f1"
)

NUM_NPUS_PER_NODE=8          # 每机 NPU 数
# ================================================================

MASTER_ADDR="${NODE_IPS[0]}"
MASTER_PORT="5567"
DIST_INIT_ADDR="${MASTER_ADDR}:${MASTER_PORT}"

NNODES=${#NODE_IPS[@]}
TP_SIZE=$(( NNODES * NUM_NPUS_PER_NODE ))
DP_SIZE=$(( NNODES * NUM_NPUS_PER_NODE ))                    # DP 并行度


# ===== Auto-detect node rank by matching local IPs =============
LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')

NODE_RANK=""
for i in "${!NODE_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${NODE_IPS[$i]}" || "$LOCAL_HOST2" == "${NODE_IPS[$i]}" ]]; then
    NODE_RANK="$i"
    SERVER_HOST="${NODE_IPS[$i]}"
    export HCCL_SOCKET_IFNAME="${HCCL_IFS[$i]}"
    break
  fi
done

if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: local IPs [${LOCAL_HOST1} ${LOCAL_HOST2}] not found in NODE_IPS=[${NODE_IPS[*]}]"
  exit 1
fi

echo "========================================"
echo "Launching GLM5.2 ${NNODES} Nodes"
echo "node-rank       : ${NODE_RANK}"
echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
echo "dist-init-addr  : ${DIST_INIT_ADDR}"
echo "nnodes          : ${NNODES}"
echo "tp-size         : ${TP_SIZE}"
echo "dp-size         : ${DP_SIZE}"
echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
echo "========================================"

# ===== Launch =====
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --tp-size "${TP_SIZE}" \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --watchdog-timeout 9000 \
  --max-running-requests 160 \
  --mem-fraction-static 0.86 \
  --quantization modelslim \
  --max-prefill-tokens 2048000 \
  --chunked-prefill-size 65536 \
  --kv-cache-dtype "fp8_e4m3" \
  --dp ${DP_SIZE} \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --load-balance-method round_robin \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --speculative-algorithm NEXTN \
  --speculative-draft-kv-cache-dtype bf16 \
  --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --enable-metrics
