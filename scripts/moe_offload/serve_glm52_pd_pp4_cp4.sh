#!/usr/bin/env bash
# GLM-5.2 A3：单机 P 卸载 + 双机 D 常驻；三台用同一份脚本按 IP 自动识别。
# P: PP4 × TP4 / attention DP1 / CP4 zigzag / stage 内 EP4。
# D: 两机 TP32 / attention DP32 / EP32，沿用 serve_glm52_pd_baseline.sh。
# Usage: bash serve_glm52_pd_pp4_cp4.sh [auto|router] [--dry-run]
set -eo pipefail

# ==================== 必填：三台保持相同配置，示例地址必须换成实际值 ====================
MODEL_PATH=${MODEL_PATH:-"/home/weights/GLM-5.2-w8a8"} # 各机同版本的 ModelSlim W8A8 权重目录
P_IP=${P_IP:-"61.47.19.68"}   # 单机 P 的通信 IPv4；P HTTP / bootstrap / KV store
D0_IP=${D0_IP:-"61.47.19.70"} # D rank 0 的通信 IPv4；D HTTP / rendezvous
D1_IP=${D1_IP:-"61.47.19.71"} # D rank 1 的通信 IPv4

# ==================== 按机器容量调整 ====================
HOST_POOL_GIB=${HOST_POOL_GIB:-56} # 仅 P，每 rank；16 rank 共 896 GiB，另留系统及装载内存
EXECUTION_SLOTS=${EXECUTION_SLOTS:-""} # 仅 P；留空自动使用双 bank，256 专家时为 128
PP_PARTITION=${SGLANG_PP_LAYER_PARTITION:-18,20,24,16} # 四个 stage 的层数，总和须等于模型层数
P_MEM_FRACTION_STATIC=${P_MEM_FRACTION_STATIC:-0.6}
D_MEM_FRACTION_STATIC=${D_MEM_FRACTION_STATIC:-0.75}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-180000} # P/D 相同
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192} # 仅 P，必须为 128 的倍数
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-$CHUNKED_PREFILL_SIZE}
P_MAX_RUNNING_REQUESTS=${P_MAX_RUNNING_REQUESTS:-128}
D_MAX_RUNNING_REQUESTS=${D_MAX_RUNNING_REQUESTS:-128}
D_CUDA_GRAPH_MAX_BS=${D_CUDA_GRAPH_MAX_BS:-4}
# 如需单机覆盖，可设置 MAX_RUNNING_REQUESTS / MEM_FRACTION_STATIC。
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ==================== 通信及端口：三台与 Router 保持一致 ====================
COMM_IFACE=${COMM_IFACE:-""} # 留空自动选择持有本机匹配 IP 的网卡，无需 ip / ifconfig
P_PORT=${P_PORT:-8000}
D_PORT=${D_PORT:-8003}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
P_DIST_PORT=${P_DIST_PORT:-5000}
D_DIST_PORT=${D_DIST_PORT:-5000}
MF_STORE_PORT=${MF_STORE_PORT:-24707} # 默认三台都连接 tcp://P_IP:MF_STORE_PORT
ROUTER_PORT=${ROUTER_PORT:-6688}

# ==================== 非标准安装位置才需要修改 ====================
PYTHON_BIN=${PYTHON_BIN:-python3} # 默认使用当前激活环境，也可指定 Python 可执行文件绝对路径
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
CUSTOMIZE_ENV=${CUSTOMIZE_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash}
CUSTOM_TRANSFORMER_ENV=${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash}
# 可覆盖 RUNTIME_DIR / LOG_DIR（绝对路径），以及三台一致的 ASCEND_MF_STORE_URL / ASCEND_MF_TRANSFER_PROTOCOL。
# P 的扩展编译并发可用 MAX_JOBS 覆盖，默认 4。升级 CANN 等依赖后请使用新的 RUNTIME_DIR。

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  cat <<EOF
Usage: $0 [auto|router] [--dry-run]
  无参数 / auto : 按本机 IPv4 识别 P、D0、D1，三台运行相同命令。
  --dry-run     : 检查配置、打印命令；不加载 NPU、不创建运行文件。
  router        : P、D ready 后在可达两者的机器上单独运行一次 Router。
必填：MODEL_PATH、P_IP、D0_IP、D1_IP（在脚本顶部填写或用同名环境变量覆盖）。
P 使用 PP4 × TP4 / DP1 / CP4 / stage 内 EP4 卸载，D 使用双机 TP32 / DP32 / EP32 常驻。
固定关闭 MTP；真实 PD 请求发送到 Router 的 ROUTER_PORT，不要携带 fake bootstrap 字段。
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
[[ ${ENABLE_MTP:-0} == 0 ]] || fail "PP4 prefill-only offload requires ENABLE_MTP=0 on P and D"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
if [[ $ROLE == auto ]]; then
  # libc getifaddrs + Python 标准库，可识别别名 IP，不依赖网络命令。
  if ! node_info=$("$PYTHON_BIN" "$SCRIPT_DIR/pd_node.py" \
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

RUNTIME_DIR=${RUNTIME_DIR:-/tmp/sglang-glm52-pd-pp4-cp4-runtime}
LOG_DIR=${LOG_DIR:-/tmp/sglang-glm52-pd-pp4-cp4-$ROLE-${NODE_RANK:-router}-$(date +%Y%m%d_%H%M%S)-$$}
[[ $RUNTIME_DIR == /* && $LOG_DIR == /* ]] || fail "RUNTIME_DIR and LOG_DIR must be absolute paths"
if [[ $ROLE == prefill ]]; then
  MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$P_MAX_RUNNING_REQUESTS}
  MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-$P_MEM_FRACTION_STATIC}
else
  MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$D_MAX_RUNNING_REQUESTS}
  MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-$D_MEM_FRACTION_STATIC}
fi

# Dry-run 只读 config.json。正常启动时仅 P 生成 offload.json。
"$PYTHON_BIN" - "$ROLE" "$MODEL_PATH" "$HOST_POOL_GIB" "$EXECUTION_SLOTS" \
  "$RUNTIME_DIR" "$LOG_DIR" "$DRY_RUN" "$PP_PARTITION" "$ASCEND_RT_VISIBLE_DEVICES" \
  "$CONTEXT_LENGTH" "$CHUNKED_PREFILL_SIZE" "$MAX_PREFILL_TOKENS" "$MAX_RUNNING_REQUESTS" \
  "$MEM_FRACTION_STATIC" "$D_CUDA_GRAPH_MAX_BS" "$P_IP" "$D0_IP" "$D1_IP" \
  "$P_PORT" "$D_PORT" "$BOOTSTRAP_PORT" "$P_DIST_PORT" "$D_DIST_PORT" "$MF_STORE_PORT" "$ROUTER_PORT" <<'PY_CONFIG'
import ipaddress
import json
import math
import sys
from itertools import accumulate
from pathlib import Path

(role, model, host, slots, runtime, logs, dry, partition, devices, context,
 chunk, prefill, requests, fraction, graph_bs, p_ip, d0_ip, d1_ip, *ports) = sys.argv[1:]
for port in ports:
    if not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError(f"Invalid TCP port: {port}")
node_ips = (p_ip, d0_ip) if role == "router" else (p_ip, d0_ip, d1_ip)
for value in node_ips:
    address = ipaddress.IPv4Address(value)
    if address.is_loopback or address.is_unspecified or address.is_multicast or value == "255.255.255.255":
        raise ValueError("P/D IPs must be non-loopback unicast IPv4 addresses")
if len(set(node_ips)) != len(node_ips):
    raise ValueError("P/D node IPs must be distinct")
if role != "router":
    visible = devices.split(",")
    if len(visible) != 16 or len(set(visible)) != 16 or not all(d.isdecimal() for d in visible):
        raise ValueError("Each A3 node must expose 16 distinct NPU devices")
    for name, value in (("CONTEXT_LENGTH", context), ("MAX_RUNNING_REQUESTS", requests)):
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(float(fraction)) or not 0 < float(fraction) < 1:
        raise ValueError("MEM_FRACTION_STATIC must be in (0, 1)")
    cfg = json.loads((Path(model) / "config.json").read_text())
    if cfg.get("model_type") != "glm_moe_dsa" or "GlmMoeDsaForCausalLM" not in cfg.get("architectures", []):
        raise ValueError("MODEL_PATH must contain GLM-5.2 / GlmMoeDsaForCausalLM")
    experts = cfg.get("n_routed_experts")
    ep_size = 4 if role == "prefill" else 32
    if type(experts) is not int or experts <= 0 or experts % ep_size:
        raise ValueError(f"n_routed_experts must be positive and divisible by EP={ep_size}")
if role == "decode":
    if not graph_bs.isdecimal() or int(graph_bs) <= 0:
        raise ValueError("D_CUDA_GRAPH_MAX_BS must be a positive integer")
if role == "prefill":
    for name, value in (("HOST_POOL_GIB", host), ("CHUNKED_PREFILL_SIZE", chunk),
                        ("MAX_PREFILL_TOKENS", prefill)):
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if int(chunk) % 128:
        raise ValueError("CHUNKED_PREFILL_SIZE must be divisible by page size 128")
    layers, first, frequency = (cfg.get(k) for k in ("num_hidden_layers", "first_k_dense_replace", "moe_layer_freq"))
    if type(layers) is not int or layers <= 0 or type(first) is not int or first < 0 or type(frequency) is not int or frequency <= 0:
        raise ValueError("Invalid num_hidden_layers / first_k_dense_replace / moe_layer_freq")
    try:
        parts = [int(x) for x in partition.split(",")]
    except ValueError as exc:
        raise ValueError("SGLANG_PP_LAYER_PARTITION must contain four positive integers") from exc
    if len(parts) != 4 or any(x <= 0 for x in parts) or sum(parts) != layers:
        raise ValueError(f"SGLANG_PP_LAYER_PARTITION must contain four positive lengths summing to {layers}")
    bounds = [0, *accumulate(parts)]
    counts = [sum(i >= first and i % frequency == 0 for i in range(start, end))
              for start, end in zip(bounds, bounds[1:])]
    if not all(counts):
        raise ValueError("Each PP stage must contain at least one routed MoE layer")
    required_slots = 2 * experts // ep_size
    if slots and (not slots.isdecimal() or int(slots) < required_slots):
        raise ValueError(f"EXECUTION_SLOTS must be at least {required_slots} for two full local-layer banks")
    config = {
        "mode": "prefill_only",
        "cache_ratio": 0,
        "execution_slots": int(slots) if slots else required_slots,
        "host_pool_gib": int(host),
        "native_extension_directory": str(Path(runtime) / "native_acl"),
    }
    hidden, intermediate = cfg.get("hidden_size"), cfg.get("moe_intermediate_size")
    if type(hidden) is int and hidden > 0 and type(intermediate) is int and intermediate > 0:
        rank_layer_bytes = 3 * hidden * intermediate * (experts // ep_size)
        estimate = [rank_layer_bytes * count / 1024**3 for count in counts]
        print(f"W8 weight-only Host GiB/rank by stage: {estimate}; metadata/alignment/loading need extra RAM")
        if max(estimate) >= int(host):
            raise ValueError("HOST_POOL_GIB leaves no room for the largest stage's W8 weights and metadata")
    print(json.dumps(config, indent=2))
    print(f"Stage routed layers: {counts}; experts/rank/layer: {experts // ep_size}")
    print(f"Host pool reservation: {host} GiB/rank, {16 * int(host)} GiB/node; allow extra RAM for loading and the OS")
    if dry == "0":
        Path(logs).mkdir(parents=True, exist_ok=True)
        (Path(logs) / "offload.json").write_text(json.dumps(config, indent=2) + "\n")
PY_CONFIG

if [[ $ROLE == router ]]; then
  command=("$PYTHON_BIN" -u -m sglang_router.launch_router
    --pd-disaggregation --policy round_robin
    --prefill "http://$P_IP:$P_PORT" "$BOOTSTRAP_PORT"
    --decode "http://$D0_IP:$D_PORT"
    --host 0.0.0.0 --port "$ROUTER_PORT")
else
  command=("$PYTHON_BIN" -u -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name glm-5
    --device npu --attention-backend ascend --quantization modelslim
    --load-format safetensors --dtype bfloat16 --trust-remote-code
    --disaggregation-mode "$ROLE" --disaggregation-transfer-backend ascend
    --host 0.0.0.0 --page-size 128 --kv-cache-dtype bfloat16
    --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --disable-shared-experts-fusion --load-balance-method round_robin)
  if [[ $ROLE == prefill ]]; then
    command+=(--port "$P_PORT" --dist-init-addr "$P_IP:$P_DIST_PORT"
      --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
      --nnodes 1 --node-rank 0 --pp-size 4 --tp-size 4 --dp-size 1 --attn-dp-size 1
      --enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag
      --ep-size 4 --moe-dp-size 1 --moe-dense-tp-size 1
      --moe-a2a-backend ascend_ops --deepep-mode auto
      --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}'
      --disable-overlap-schedule
      --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
      --moe-offload-config "$LOG_DIR/offload.json" --watchdog-timeout 600)
  else
    command+=(--port "$D_PORT" --dist-init-addr "$D0_IP:$D_DIST_PORT"
      --nnodes 2 --node-rank "$NODE_RANK" --tp-size 32 --attn-dp-size 32 --ep-size 32
      --moe-a2a-backend deepep --deepep-mode low_latency
      --cuda-graph-backend-decode full --cuda-graph-max-bs-decode "$D_CUDA_GRAPH_MAX_BS"
      --tokenizer-worker-num 4 --watchdog-timeout 9000)
  fi
fi

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export HCCL_BUFFSIZE=1024
# 这是实际 PD 部署，不启用独立 P 假发送测试开关。
unset SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE
if [[ $ROLE != router ]]; then
  export SGLANG_HOST_IP="$NODE_IP"
  export ASCEND_MF_STORE_URL=${ASCEND_MF_STORE_URL:-tcp://$P_IP:$MF_STORE_PORT}
  export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SGLANG_SET_CPU_AFFINITY=1
  export STREAMS_PER_DEVICE=32 TRANSFORMERS_VERBOSITY=error
  if [[ $ROLE == prefill ]]; then
    export SGLANG_PP_LAYER_PARTITION="$PP_PARTITION"
    export HCCL_OP_EXPANSION_MODE=AIV HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 ACL_DEVICE_SYNC_TIMEOUT=300
    export INF_NAN_MODE_FORCE_DISABLE=1 TASK_QUEUE_ENABLE=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
    printf 'PP partition: %s\n' "$SGLANG_PP_LAYER_PARTITION"
  else
    unset SGLANG_PP_LAYER_PARTITION
    export TASK_QUEUE_ENABLE=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 SGLANG_NPU_USE_MULTI_STREAM=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
  fi
  printf 'KV store: %s; communication interface: %s\n' "$ASCEND_MF_STORE_URL" "$COMM_IFACE"
fi
printf 'P=1 node PP4×TP4/CP4/EP4 offload; D=2 nodes TP32/DP32/EP32 resident; MTP=off\n'
printf 'Runtime: %s\nLogs: %s\nCommand:' "$RUNTIME_DIR" "$LOG_DIR"
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
if [[ $ROLE == prefill ]]; then
  for env_file in "$CUSTOMIZE_ENV" "$CUSTOM_TRANSFORMER_ENV"; do
    [[ -f $env_file ]] || fail "Missing Ascend environment script: $env_file"
    source "$env_file"
  done
  mkdir -p "$RUNTIME_DIR/python" "$RUNTIME_DIR/native_acl"
  "$PYTHON_BIN" - "$RUNTIME_DIR/python" <<'PY_BUILD'
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
  export TORCH_EXTENSIONS_DIR="$RUNTIME_DIR/torch_extensions"
  export MAX_JOBS=${MAX_JOBS:-4}
  "$PYTHON_BIN" - "$RUNTIME_DIR/python/memfabric_hybrid/lib" <<'PY_EXT'
import sys
from memfabric_hybrid._acc_offload_build import ensure_acc_offload_ext

assert ensure_acc_offload_ext("A3", install_dir=sys.argv[1])
PY_EXT
  "$PYTHON_BIN" -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"
fi
exec "${command[@]}"
