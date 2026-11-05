#!/usr/bin/env bash
# One A3 prefill node (offloaded experts), one two-node A3 decode instance.
# 三台机器使用同一份配置，均直接运行本脚本；按本机 IP 自动识别 P / D0 / D1。
# Usage: serve_glm52_pd.sh [auto|prefill|decode|router] [--dry-run]
set -eo pipefail

# ==================== 必填：修改这四项，三台机器保持一致 ====================
# 在引号内填实际值；已有同名环境变量时，环境变量优先。
MODEL_PATH=${MODEL_PATH:-"/home/weights/GLM-5.2-w8a8"} # 本机 GLM-5.2 ModelSlim 权重目录，例如 /models/GLM-5.2
P_IP=${P_IP:-"61.47.19.69"}             # P 机器实际绑定的、其他节点可达的通信 IPv4
D0_IP=${D0_IP:-"61.47.19.70"}           # D 第 0 台机器的通信 IPv4（也是 D rendezvous / HTTP 地址）
D1_IP=${D1_IP:-"61.47.19.71"}           # D 第 1 台机器的通信 IPv4

# ==================== 通常不用改：按需确认容量和模型能力 ====================
COMM_IFACE=${COMM_IFACE:-""} # auto 时留空：从匹配 IP 自动找网卡；显式角色入口需填写
ENABLE_MTP=${ENABLE_MTP:-0}  # 仅当 checkpoint 含 NextN 权重时设为 1；三台一致
MTP_DRAFT_QUANTIZATION=${MTP_DRAFT_QUANTIZATION:-unquant}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-180000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
HOST_POOL_GIB=${HOST_POOL_GIB:-48} # 仅 P，每 rank 的 GiB；16 rank 共 768 GiB，需额外装载/系统内存
EXECUTION_SLOTS=${EXECUTION_SLOTS:-""} # 仅 P，留空自动计算两个整层 bank 的槽数
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-65536} # 仅 P
P_MAX_RUNNING_REQUESTS=${P_MAX_RUNNING_REQUESTS:-64}
D_MAX_RUNNING_REQUESTS=${D_MAX_RUNNING_REQUESTS:-128}
# MAX_RUNNING_REQUESTS 环境变量可覆盖本机对应角色的并发值；一般不必设置。
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ==================== 端口：默认即可，改动时三台保持一致 ====================
P_PORT=${P_PORT:-8000}
D_PORT=${D_PORT:-8003}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
DIST_PORT=${DIST_PORT:-5000}
ROUTER_PORT=${ROUTER_PORT:-6688}
MF_STORE_PORT=${MF_STORE_PORT:-24707} # KV store；默认 tcp://P_IP:此端口

# ==================== 仅非标准安装位置需要修改 ====================
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
CUSTOMIZE_ENV=${CUSTOMIZE_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash}
CUSTOM_TRANSFORMER_ENV=${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash}
# RUNTIME_DIR / LOG_DIR 可用环境变量覆盖，默认在 /tmp 下，见角色识别后的路径设置。
# ASCEND_MF_STORE_URL / ASCEND_MF_TRANSFER_PROTOCOL 按 fabric 环境覆盖；三台需一致。
# MAX_JOBS 可覆盖 P 端扩展编译并发，默认 4。

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  cat <<EOF
Usage: $0 [auto|prefill|decode|router] [--dry-run]
  无参数 / auto : 从本机 IPv4 自动选择 P、D0、D1；三台使用相同命令。
  --dry-run     : 只检查配置并打印命令，不加载模型或创建运行文件。
  router        : 单独启动一次 Router，不要求本机 IP 匹配 P/D。
  prefill/decode: 兼容手工角色；需 COMM_IFACE，decode 另需 NODE_RANK=0/1。
先填写脚本顶部 MODEL_PATH、P_IP、D0_IP、D1_IP，或设置同名环境变量。
EOF
}
ROLE=auto
DRY_RUN=0
role_given=0
for arg in "$@"; do
  case "$arg" in
    auto|prefill|decode|router)
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
[[ -n ${P_IP:-} && -n ${D0_IP:-} ]] || fail "Set P_IP and D0_IP to reachable node IPs"
if [[ $ROLE == auto ]]; then
  # 通过 Python 标准库调用系统接口，枚举全部 IPv4；不需要 ip / ifconfig / hostname。
  if ! node_info=$(python3 "$SCRIPT_DIR/pd_node.py" \
    --node P_IP "$P_IP" prefill 0 \
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
RUNTIME_DIR=${RUNTIME_DIR:-/tmp/sglang-glm52-pd-$ROLE-runtime}
LOG_DIR=${LOG_DIR:-/tmp/sglang-glm52-pd-$ROLE-$(date +%Y%m%d_%H%M%S)-$$}
[[ $RUNTIME_DIR == /* && $LOG_DIR == /* ]] || fail "RUNTIME_DIR and LOG_DIR must be absolute paths"

# Build commands before preparing any accelerator environment. Dry-run needs
# only Python's standard library and never loads torch_npu, CANN or the model.
if [[ $ROLE == router ]]; then
  command=(python3 -u -m sglang_router.launch_router
    --pd-disaggregation --policy round_robin
    --prefill "http://$P_IP:$P_PORT" "$BOOTSTRAP_PORT"
    --decode "http://$D0_IP:$D_PORT"
    --host 0.0.0.0 --port "$ROUTER_PORT")
else
  [[ -n ${MODEL_PATH:-} && -f $MODEL_PATH/config.json ]] || fail "Set MODEL_PATH to a local GLM-5.2 ModelSlim checkpoint"
  [[ -n ${COMM_IFACE:-} && $COMM_IFACE != lo ]] || fail "Set COMM_IFACE to this node's inter-node communication interface"
  [[ $ENABLE_MTP == 0 || $ENABLE_MTP == 1 ]] || fail "ENABLE_MTP must be 0 or 1"
  command=(python3 -u -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name glm-5
    --device npu --attention-backend ascend --quantization modelslim
    --load-format safetensors --dtype bfloat16 --trust-remote-code
    --disaggregation-mode "$ROLE" --disaggregation-transfer-backend ascend
    --host 0.0.0.0 --page-size 128 --kv-cache-dtype bfloat16
    --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
    --disable-shared-experts-fusion --load-balance-method round_robin)
  if [[ $ROLE == prefill ]]; then
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$P_MAX_RUNNING_REQUESTS}
    command+=(--port "$P_PORT" --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
      --nnodes 1 --node-rank 0 --tp-size 16 --attn-dp-size 4 --ep-size 16
      --enable-dp-lm-head --moe-dense-tp-size 1
      --moe-a2a-backend ascend_ops --deepep-mode auto
      --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}'
      --disable-overlap-schedule
      --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$CHUNKED_PREFILL_SIZE"
      --max-running-requests "$MAX_RUNNING_REQUESTS"
      --moe-offload-config "$LOG_DIR/offload.json" --watchdog-timeout 600)
    mtp_steps=1
    mtp_tokens=2
  else
    [[ ${NODE_RANK:-} == 0 || ${NODE_RANK:-} == 1 ]] || fail "Decode requires NODE_RANK=0 or NODE_RANK=1"
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$D_MAX_RUNNING_REQUESTS}
    command+=(--port "$D_PORT" --nnodes 2 --node-rank "$NODE_RANK"
      --dist-init-addr "$D0_IP:$DIST_PORT"
      --tp-size 32 --attn-dp-size 32 --ep-size 32
      --moe-a2a-backend deepep --deepep-mode low_latency
      --max-running-requests "$MAX_RUNNING_REQUESTS"
      --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 4
      --tokenizer-worker-num 4 --watchdog-timeout 9000)
    mtp_steps=3
    mtp_tokens=4
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

# Validate scalar settings and write only the P configuration. No hotspot
# profile or decode-cache knobs belong in a prefill-only offload configuration.
python3 - "$ROLE" "${MODEL_PATH:-}" "${HOST_POOL_GIB:-48}" \
  "${EXECUTION_SLOTS:-}" "$RUNTIME_DIR" "$LOG_DIR" "$DRY_RUN" \
  "$P_PORT" "$D_PORT" "$BOOTSTRAP_PORT" "$DIST_PORT" "$ROUTER_PORT" "$MF_STORE_PORT" \
  "${ASCEND_RT_VISIBLE_DEVICES:-}" "${CONTEXT_LENGTH:-16384}" \
  "${CHUNKED_PREFILL_SIZE:-8192}" "${MAX_RUNNING_REQUESTS:-128}" \
  "${MEM_FRACTION_STATIC:-0.8}" <<'PY_CONFIG'
import json
import math
import sys
from pathlib import Path

role, model, host, slots, runtime, logs, dry_run = sys.argv[1:8]
for port in sys.argv[8:14]:
    if not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError(f"Invalid TCP port: {port}")
if role != "router":
    devices = sys.argv[14].split(",")
    if len(devices) != 16 or len(set(devices)) != 16 or not all(d.isdecimal() for d in devices):
        raise ValueError("Each A3 node must expose 16 distinct NPU devices")
    for name, value in zip(("CONTEXT_LENGTH", "CHUNKED_PREFILL_SIZE", "MAX_RUNNING_REQUESTS"), sys.argv[15:18]):
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    fraction = float(sys.argv[18])
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("MEM_FRACTION_STATIC must be in (0, 1)")
    model_config = json.loads((Path(model) / "config.json").read_text())
    if model_config.get("model_type") != "glm_moe_dsa" or "GlmMoeDsaForCausalLM" not in model_config.get("architectures", []):
        raise ValueError("MODEL_PATH must contain GLM-5.2 / GlmMoeDsaForCausalLM")
    experts = model_config.get("n_routed_experts")
    ep_size = 16 if role == "prefill" else 32
    if type(experts) is not int or experts <= 0 or experts % ep_size:
        raise ValueError(f"n_routed_experts must be positive and divisible by EP={ep_size}")
if role == "prefill":
    if not host.isdecimal() or int(host) <= 0:
        raise ValueError("HOST_POOL_GIB must be a positive integer per rank")
    config = {
        "mode": "prefill_only",
        "cache_ratio": 0,
        "host_pool_gib": int(host),
        "native_extension_directory": str(Path(runtime) / "native_acl"),
    }
    bank_slots = 2 * experts // ep_size
    if slots:
        if not slots.isdecimal() or int(slots) < bank_slots:
            raise ValueError(f"EXECUTION_SLOTS must be at least {bank_slots} for two layer banks")
        config["execution_slots"] = int(slots)
    output = json.dumps(config, indent=2) + "\n"
    print(output, end="")
    print(f"Execution slots: {slots or bank_slots} (two banks need {bank_slots})")
    print(f"Host pool reservation: {host} GiB/rank, {16 * int(host)} GiB/node; allow extra RAM for loading and the OS")
    if dry_run == "0":
        Path(logs).mkdir(parents=True, exist_ok=True)
        (Path(logs) / "offload.json").write_text(output)
PY_CONFIG

export HCCL_BUFFSIZE=1024
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ $ROLE != router ]]; then
  export ASCEND_MF_STORE_URL=${ASCEND_MF_STORE_URL:-tcp://$P_IP:$MF_STORE_PORT}
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
  export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SGLANG_SET_CPU_AFFINITY=1
  export STREAMS_PER_DEVICE=32
  export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 ACL_DEVICE_SYNC_TIMEOUT=300
  if [[ $ROLE == prefill ]]; then
    export TASK_QUEUE_ENABLE=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
    export HCCL_OP_EXPANSION_MODE=AIV INF_NAN_MODE_FORCE_DISABLE=1
  else
    export TASK_QUEUE_ENABLE=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_NPU_USE_MULTI_STREAM=1 HCCL_BUFFSIZE=1024
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
  fi
  printf 'KV store: %s; communication interface: %s\n' "$ASCEND_MF_STORE_URL" "$COMM_IFACE"
fi
printf 'Runtime: %s\nLogs: %s\nCommand:' "$RUNTIME_DIR" "$LOG_DIR"
printf ' %q' "${command[@]}"
printf '\n'
if [[ $DRY_RUN == 1 ]]; then exit 0; fi

mkdir -p "$RUNTIME_DIR" "$LOG_DIR"
exec > >(tee "$LOG_DIR/server.log") 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ASCEND_LAUNCH_BLOCKING
if [[ $ROLE == router ]]; then exec "${command[@]}"; fi

# Dependency paths are configured at the top of this script.
for env_file in \
  "$CANN_ENV" "$ATB_ENV" "$CUSTOMIZE_ENV" "$CUSTOM_TRANSFORMER_ENV"; do
  [[ -f $env_file ]] || fail "Missing Ascend environment script: $env_file"
  source "$env_file"
done
export TORCH_EXTENSIONS_DIR="$RUNTIME_DIR/torch_extensions"
if [[ $ROLE == prefill ]]; then
  # Reuse the existing offload extension build, isolated from installed packages
  # and the source tree. Choose a fresh RUNTIME_DIR after dependency upgrades.
  mkdir -p "$RUNTIME_DIR/python" "$RUNTIME_DIR/native_acl"
  python3 - "$RUNTIME_DIR/python" <<'PY_BUILD'
import importlib.util
import shutil
import sys
from pathlib import Path

source = Path(importlib.util.find_spec("memfabric_hybrid").origin).parent
target = Path(sys.argv[1]) / "memfabric_hybrid"
if not target.exists():
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
PY_BUILD
  export PYTHONPATH="$SGLANG_ROOT/python:$RUNTIME_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$RUNTIME_DIR/python/memfabric_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export MAX_JOBS=${MAX_JOBS:-4}
  python3 - "$RUNTIME_DIR/python/memfabric_hybrid/lib" <<'PY_EXT'
import sys
from memfabric_hybrid._acc_offload_build import ensure_acc_offload_ext

assert ensure_acc_offload_ext("A3", install_dir=sys.argv[1])
PY_EXT
  python3 -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"
fi
exec "${command[@]}"
