#!/usr/bin/env bash
# GLM-5.2 A3 resident-weight baseline: one 2-node P instance + one 2-node D instance.
# Reference: https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2#prefill-decode-disaggregation-deployment
# 四台机器使用同一份配置，无参数启动；按本机 IP 自动识别 P0 / P1 / D0 / D1。
# Usage: serve_glm52_pd_baseline.sh [auto|router] [--dry-run]
set -eo pipefail

# ==================== 必填：修改这五项，四台机器保持一致 ====================
# 在引号内填实际值；已有同名环境变量时，环境变量优先。
MODEL_PATH=${MODEL_PATH:-"/home/weights/GLM-5.2-w8a8"} # 本机 GLM-5.2 ModelSlim 权重目录
P0_IP=${P0_IP:-"61.47.19.68"}           # P rank 0：P rendezvous、HTTP、bootstrap 和 KV store 所在机器
P1_IP=${P1_IP:-"61.47.19.69"}           # P rank 1：加入同一个 P 实例，不是第二个独立 P 服务
D0_IP=${D0_IP:-"61.47.19.70"}           # D rank 0：D rendezvous 和 HTTP 所在机器
D1_IP=${D1_IP:-"61.47.19.71"}           # D rank 1：加入同一个 D 实例

# ==================== 可选：默认按官方 PD 参数，按容量和模型能力调整 ====================
COMM_IFACE=${COMM_IFACE:-""} # 通常留空，自动选持有本机匹配 IP 的通信网卡
ENABLE_MTP=${ENABLE_MTP:-1} # 默认开启，使用手册 MTP 参数；checkpoint 无 NextN 权重时设 0
MTP_DRAFT_QUANTIZATION=${MTP_DRAFT_QUANTIZATION:-unquant}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-180000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-524288} # P：全组值，运行时按 Attention DP8 除为 65536
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-180000}   # P：与 chunk 分开配置，沿用手册
P_MAX_RUNNING_REQUESTS=${P_MAX_RUNNING_REQUESTS:-64}
D_MAX_RUNNING_REQUESTS=${D_MAX_RUNNING_REQUESTS:-128}
D_CUDA_GRAPH_MAX_BS=${D_CUDA_GRAPH_MAX_BS:-4}
# MAX_RUNNING_REQUESTS 环境变量可覆盖当前机器对应角色的并发值，一般不必设置。
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ==================== 端口：四台及 Router 保持一致 ====================
P_PORT=${P_PORT:-8000}
D_PORT=${D_PORT:-8003}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
P_DIST_PORT=${P_DIST_PORT:-5000} # P0 上的 rendezvous
D_DIST_PORT=${D_DIST_PORT:-5000} # D0 上的 rendezvous，与 P 不在同一台机器
ROUTER_PORT=${ROUTER_PORT:-6688}
MF_STORE_PORT=${MF_STORE_PORT:-24707} # 四台 KV store 均默认指向 tcp://P0_IP:此端口

# ==================== 非标准安装位置才需要修改 ====================
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
# LOG_DIR 可用环境变量覆盖，默认在 /tmp 下，须为绝对路径。
# ASCEND_MF_STORE_URL / ASCEND_MF_TRANSFER_PROTOCOL 按 fabric 环境覆盖，四台需一致。

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  cat <<EOF
Usage: $0 [auto|router] [--dry-run]
  无参数 / auto : 从本机 IPv4 自动识别 P0、P1、D0、D1；四台使用相同命令。
  --dry-run     : 只检查配置并打印命令，不加载模型或创建日志。
  router        : 单独启动一次 Router，只登记 P0、D0，不要求本机匹配节点 IP。
必填：MODEL_PATH、P0_IP、P1_IP、D0_IP、D1_IP；均在脚本顶部，支持环境变量覆盖。
P 和 D 各为 nnodes=2 / TP32 / EP32 的实例，全部专家权重常驻 NPU。
EOF
}
ROLE=auto
DRY_RUN=0
role_given=0
for arg in "$@"; do
  case "$arg" in
    auto|router)
      [[ $role_given == 0 ]] || fail "Specify only one role"
      ROLE=$arg; role_given=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "Unknown argument: $arg" ;;
  esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
[[ -n $P0_IP && -n $D0_IP ]] || fail "Set P0_IP and D0_IP to reachable node IPv4 addresses"
if [[ $ROLE == auto ]]; then
  # 通过 Python 标准库调用系统接口，枚举全部 IPv4；不需要 ip / ifconfig / hostname。
  if ! node_info=$(python3 "$SCRIPT_DIR/pd_node.py" \
    --node P0_IP "$P0_IP" prefill 0 \
    --node P1_IP "$P1_IP" prefill 1 \
    --node D0_IP "$D0_IP" decode 0 \
    --node D1_IP "$D1_IP" decode 1 \
    --node-rank "${NODE_RANK:-}" --comm-iface "$COMM_IFACE"); then
    fail "Automatic node identification failed; no server was started"
  fi
  {
    IFS= read -r ROLE
    IFS= read -r NODE_RANK
    IFS= read -r NODE_IP
    IFS= read -r COMM_IFACE
  } <<< "$node_info"
  printf 'Auto node: IP=%s role=%s node_rank=%s interface=%s\n' "$NODE_IP" "$ROLE" "$NODE_RANK" "$COMM_IFACE"
fi

LOG_DIR=${LOG_DIR:-/tmp/sglang-glm52-pd-baseline-$ROLE-${NODE_RANK:-router}-$(date +%Y%m%d_%H%M%S)-$$}
[[ $LOG_DIR == /* ]] || fail "LOG_DIR must be an absolute path"
if [[ $ROLE == prefill ]]; then
  MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$P_MAX_RUNNING_REQUESTS}
else
  MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$D_MAX_RUNNING_REQUESTS}
fi

# Only stdlib/model metadata is needed for dry-run. No weights or extensions are built.
python3 - "$ROLE" "$MODEL_PATH" "$ASCEND_RT_VISIBLE_DEVICES" "$CONTEXT_LENGTH" \
  "$CHUNKED_PREFILL_SIZE" "$MAX_PREFILL_TOKENS" "$MAX_RUNNING_REQUESTS" \
  "$MEM_FRACTION_STATIC" "$ENABLE_MTP" "$D_CUDA_GRAPH_MAX_BS" \
  "$P_PORT" "$D_PORT" "$BOOTSTRAP_PORT" "$P_DIST_PORT" "$D_DIST_PORT" \
  "$ROUTER_PORT" "$MF_STORE_PORT" "$P0_IP" "$D0_IP" <<'PY_CONFIG'
import ipaddress
import json
import math
import sys
from pathlib import Path

(
    role, model, devices, context, chunk, prefill_tokens, requests, fraction, mtp,
    graph_bs, *ports_and_ips,
) = sys.argv[1:]
for port in ports_and_ips[:-2]:
    if not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError(f"Invalid TCP port: {port}")
for value in ports_and_ips[-2:]:
    address = ipaddress.IPv4Address(value)
    if address.is_loopback or address.is_unspecified or address.is_multicast or value == "255.255.255.255":
        raise ValueError("P0_IP and D0_IP must be non-loopback unicast IPv4 addresses")
if ports_and_ips[-2] == ports_and_ips[-1]:
    raise ValueError("P0_IP and D0_IP must be distinct node addresses")
if role != "router":
    visible = devices.split(",")
    if len(visible) != 16 or len(set(visible)) != 16 or not all(d.isdecimal() for d in visible):
        raise ValueError("Each A3 node must expose 16 distinct NPU devices")
    for name, value in (("CONTEXT_LENGTH", context), ("MAX_RUNNING_REQUESTS", requests)):
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if role == "prefill":
        if not chunk.isdecimal() or int(chunk) <= 0 or int(chunk) % (8 * 128):
            raise ValueError("CHUNKED_PREFILL_SIZE must be positive and divisible by DP8 * page size 128")
        if not prefill_tokens.isdecimal() or int(prefill_tokens) <= 0:
            raise ValueError("MAX_PREFILL_TOKENS must be a positive integer")
    elif not graph_bs.isdecimal() or int(graph_bs) <= 0:
        raise ValueError("D_CUDA_GRAPH_MAX_BS must be a positive integer")
    if not math.isfinite(float(fraction)) or not 0 < float(fraction) < 1:
        raise ValueError("MEM_FRACTION_STATIC must be in (0, 1)")
    if mtp not in ("0", "1"):
        raise ValueError("ENABLE_MTP must be 0 or 1")
    config = json.loads((Path(model) / "config.json").read_text())
    if config.get("model_type") != "glm_moe_dsa" or "GlmMoeDsaForCausalLM" not in config.get("architectures", []):
        raise ValueError("MODEL_PATH must contain GLM-5.2 / GlmMoeDsaForCausalLM")
    experts = config.get("n_routed_experts")
    if type(experts) is not int or experts <= 0 or experts % 32:
        raise ValueError("n_routed_experts must be positive and divisible by EP=32")
PY_CONFIG

if [[ $ROLE == router ]]; then
  command=(python3 -u -m sglang_router.launch_router
    --pd-disaggregation --policy round_robin
    --prefill "http://$P0_IP:$P_PORT" "$BOOTSTRAP_PORT"
    --decode "http://$D0_IP:$D_PORT"
    --host 0.0.0.0 --port "$ROUTER_PORT")
else
  command=(python3 -u -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name glm-5
    --device npu --attention-backend ascend --quantization modelslim
    --load-format safetensors --dtype bfloat16 --trust-remote-code
    --disaggregation-mode "$ROLE" --disaggregation-transfer-backend ascend
    --host 0.0.0.0 --nnodes 2 --node-rank "$NODE_RANK" --tp-size 32 --ep-size 32
    --page-size 128 --kv-cache-dtype bfloat16 --context-length "$CONTEXT_LENGTH"
    --mem-fraction-static "$MEM_FRACTION_STATIC" --max-running-requests "$MAX_RUNNING_REQUESTS"
    --moe-a2a-backend deepep --disable-shared-experts-fusion
    --load-balance-method round_robin)
  if [[ $ROLE == prefill ]]; then
    command+=(--port "$P_PORT" --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
      --dist-init-addr "$P0_IP:$P_DIST_PORT" --attn-dp-size 8
      --enable-dp-lm-head --moe-dense-tp-size 1 --deepep-mode normal
      --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS")
    mtp_steps=1; mtp_tokens=2
  else
    command+=(--port "$D_PORT" --dist-init-addr "$D0_IP:$D_DIST_PORT"
      --attn-dp-size 32 --deepep-mode low_latency
      --cuda-graph-backend-decode full --cuda-graph-max-bs-decode "$D_CUDA_GRAPH_MAX_BS"
      --tokenizer-worker-num 4 --watchdog-timeout 9000)
    mtp_steps=3; mtp_tokens=4
  fi
  if [[ $ENABLE_MTP == 1 ]]; then
    python3 "$SCRIPT_DIR/glm52_mtp_config.py" "$MODEL_PATH" \
      --draft-quantization "$MTP_DRAFT_QUANTIZATION"
    command+=(--speculative-algorithm NEXTN --speculative-num-steps "$mtp_steps"
      --speculative-eagle-topk 1 --speculative-num-draft-tokens "$mtp_tokens"
      --speculative-draft-model-path "$MODEL_PATH"
      --speculative-draft-model-quantization "$MTP_DRAFT_QUANTIZATION"
      --speculative-draft-attention-backend ascend)
  fi
fi

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ $ROLE != router ]]; then
  export ASCEND_MF_STORE_URL=${ASCEND_MF_STORE_URL:-tcp://$P0_IP:$MF_STORE_PORT}
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
  export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SGLANG_SET_CPU_AFFINITY=1
  export STREAMS_PER_DEVICE=32 TRANSFORMERS_VERBOSITY=error
  if [[ $ROLE == prefill ]]; then
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1 DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export TASK_QUEUE_ENABLE=2 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 SGLANG_NPU_USE_MULTI_STREAM=0
  else
    export TASK_QUEUE_ENABLE=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 SGLANG_NPU_USE_MULTI_STREAM=1
    export HCCL_BUFFSIZE=650 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
  fi
  printf 'KV store: %s; communication interface: %s\n' "$ASCEND_MF_STORE_URL" "$COMM_IFACE"
fi
printf 'Baseline: resident weights; P=2 nodes/TP32/DP8/EP32; D=2 nodes/TP32/DP32/EP32\n'
printf 'Logs: %s\nCommand:' "$LOG_DIR"
printf ' %q' "${command[@]}"
printf '\n'
if [[ $DRY_RUN == 1 ]]; then exit 0; fi

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/server.log") 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ASCEND_LAUNCH_BLOCKING
if [[ $ROLE == router ]]; then exec "${command[@]}"; fi
for env_file in "$CANN_ENV" "$ATB_ENV"; do
  [[ -f $env_file ]] || fail "Missing Ascend environment script: $env_file"
  source "$env_file"
done
exec "${command[@]}"
