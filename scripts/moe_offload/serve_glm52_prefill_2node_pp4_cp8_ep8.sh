#!/usr/bin/env bash
# A3 双机各 16 die，PP4 / TP8 / CP8 / EP8，仅 Prefill 权重卸载。
# 两机分别执行本脚本；不依赖其他启动脚本。先激活 torch_npu / SGLang 环境。
set -eo pipefail

MODEL_PATH=${MODEL_PATH:-/home/weights/GLM-5.2-w8a8}
# 两边填写相同的 P0_IP / P1_IP，仅 NODE_RANK 分别设为 0 / 1。
P0_IP=${P0_IP:-}
P1_IP=${P1_IP:-}
NODE_RANK=${NODE_RANK:-}
COMM_IFACE=${COMM_IFACE:-enp196s0f0}
SERVER_HOST=${SERVER_HOST:-0.0.0.0}
SERVER_PORT=${SERVER_PORT:-8000}
DIST_PORT=${DIST_PORT:-5000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
MF_STORE_PORT=${MF_STORE_PORT:-24707}

# 两个比例只设置一个。都不设置时卸载 75%，每层常驻 25%。
# OFFLOAD=1 表示全卸载；OFFLOAD=0 请运行原全 HBM 基线。
PREFILL_OFFLOAD_RATIO=${PREFILL_OFFLOAD_RATIO:-}
PREFILL_RESIDENT_RATIO=${PREFILL_RESIDENT_RATIO:-}
HOST_POOL_GIB=${HOST_POOL_GIB:-32} # 每进程预留，两机各 16 个进程
EXECUTION_SLOTS=${EXECUTION_SLOTS:-}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-180000}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-4096}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-180000}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.72}
CP_STRATEGY=${CP_STRATEGY:-interleave}
P_MOE_A2A_BACKEND=${P_MOE_A2A_BACKEND:-deepep}
FAKE_PREFILL_TEST=${FAKE_PREFILL_TEST:-1}
export SGLANG_PP_LAYER_PARTITION=${SGLANG_PP_LAYER_PARTITION:-18,20,24,16}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

PYTHON_BIN=${PYTHON_BIN:-python3}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
CUSTOMIZE_ENV=${CUSTOMIZE_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash}
CUSTOM_TRANSFORMER_ENV=${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash}

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  cat <<EOF
Usage: P0_IP=61.47.19.69 P1_IP=61.47.19.70 NODE_RANK=0 bash $0 [--dry-run]
另一台执行相同命令，NODE_RANK=1；两边都必须启动。
  --dry-run : 只读 config.json、检查参数和打印命令，不创建文件或初始化 NPU。
固定双机 PP4 / TP8 / attention DP1 / CP8 / EP8，每机两个完整 PP stage。
默认 interleave、DeepEP normal INT8、chunk=4096；CP_STRATEGY=zigzag 可做对照。
P_MOE_A2A_BACKEND=ascend_ops 可选择 alltoallv。
PREFILL_OFFLOAD_RATIO=0.75 表示卸载 75%；也可单独设置 PREFILL_RESIDENT_RATIO=0.25。
常驻专家取整后打印实际比例；只调整 routed experts，不卸载 attention/shared expert。
HOST_POOL_GIB 默认 32 GiB/进程，即每机预留 512 GiB，加载和系统还需要额外内存。
两边的配置指纹应相同；本脚本不远程检查另一台机器。
FAKE_PREFILL_TEST=1 默认保留 fake sender 测试的 radix cache。
两边就绪后，请求直发节点 0 的 /generate，max_new_tokens=1、
bootstrap_host=2.2.2.2、bootstrap_room=0；不需要 D / Router。
实际 PD 联调可设置 FAKE_PREFILL_TEST=0 并使用正常 bootstrap 参数。
EOF
}
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "Unknown argument: $arg" ;;
  esac
done
[[ $NODE_RANK == 0 || $NODE_RANK == 1 ]] || fail "Set NODE_RANK to 0 or 1"
[[ -n $P0_IP && -n $P1_IP ]] || fail "Set both P0_IP and P1_IP to reachable node IPv4 addresses"
[[ -n $COMM_IFACE && $COMM_IFACE != lo ]] || fail "COMM_IFACE must name a non-loopback communication interface"
[[ $CP_STRATEGY == interleave || $CP_STRATEGY == zigzag ]] || fail "CP_STRATEGY must be interleave or zigzag"
[[ $FAKE_PREFILL_TEST == 0 || $FAKE_PREFILL_TEST == 1 ]] || fail "FAKE_PREFILL_TEST must be 0 or 1"
[[ ${ENABLE_MTP:-0} == 0 ]] || fail "This prefill-only test requires ENABLE_MTP=0"
[[ -f $MODEL_PATH/config.json ]] || fail "Set MODEL_PATH to a local GLM-5.2 ModelSlim W8A8 checkpoint"
if [[ $NODE_RANK == 0 ]]; then LOCAL_HOST_IP=$P0_IP; else LOCAL_HOST_IP=$P1_IP; fi
RUNTIME_DIR=${RUNTIME_DIR:-/tmp/sglang-glm52-prefill-2node-pp4-cp8-ep8-node$NODE_RANK-runtime}
LOG_DIR=${LOG_DIR:-/tmp/sglang-glm52-prefill-2node-pp4-cp8-ep8-node$NODE_RANK-$(date +%Y%m%d_%H%M%S)-$$}
[[ $RUNTIME_DIR == /* && $LOG_DIR == /* ]] || fail "RUNTIME_DIR and LOG_DIR must be absolute paths"
case "$P_MOE_A2A_BACKEND" in
  deepep)
    moe_args=(--moe-a2a-backend deepep --deepep-mode normal
      --deepep-dispatcher-output-dtype int8)
    ;;
  ascend_ops)
    moe_args=(--moe-a2a-backend ascend_ops --deepep-mode auto
      --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}')
    ;;
  *) fail "P_MOE_A2A_BACKEND must be deepep or ascend_ops" ;;
esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1

# Preflight imports only the standard library and the host-only ratio/slot helper.
"$PYTHON_BIN" - "$MODEL_PATH" "$SGLANG_ROOT" "$NODE_RANK" "$P0_IP" "$P1_IP" \
  "$HOST_POOL_GIB" "$EXECUTION_SLOTS" "$PREFILL_OFFLOAD_RATIO" "$PREFILL_RESIDENT_RATIO" \
  "$SGLANG_PP_LAYER_PARTITION" "$ASCEND_RT_VISIBLE_DEVICES" "$CONTEXT_LENGTH" \
  "$CHUNKED_PREFILL_SIZE" "$MAX_PREFILL_TOKENS" "$MAX_RUNNING_REQUESTS" "$MEM_FRACTION_STATIC" \
  "$SERVER_PORT" "$DIST_PORT" "$BOOTSTRAP_PORT" "$MF_STORE_PORT" "$CP_STRATEGY" \
  "$P_MOE_A2A_BACKEND" "$FAKE_PREFILL_TEST" "$RUNTIME_DIR" "$LOG_DIR" "$DRY_RUN" <<'PY_CONFIG'
import hashlib
import ipaddress
import json
import math
import sys
from decimal import Decimal
from itertools import accumulate
from pathlib import Path
from runpy import run_path

(model, root, node_rank, p0, p1, host, slots, offload, resident_ratio, partition,
 devices, context, chunk, prefill, requests, fraction, server_port, dist_port,
 bootstrap_port, store_port, cp, backend, fake, runtime, logs, dry) = sys.argv[1:]
rank, ep = int(node_rank), 8
for address in (p0, p1):
    ip = ipaddress.IPv4Address(address)
    if ip.is_loopback or ip.is_unspecified or ip.is_multicast or str(ip) == "255.255.255.255":
        raise ValueError("P0_IP/P1_IP must be non-loopback unicast IPv4 addresses")
if p0 == p1:
    raise ValueError("P0_IP and P1_IP must name different nodes")
ports = (server_port, dist_port, bootstrap_port, store_port)
if any(not p.isdecimal() or not 1 <= int(p) <= 65535 for p in ports):
    raise ValueError("TCP ports must be integers in [1, 65535]")
if len(set(map(int, ports))) != len(ports):
    raise ValueError("Server, distributed, bootstrap and MemFabric ports must be distinct")
visible = devices.split(",")
if (len(visible) != 16 or not all(d.isdecimal() for d in visible)
        or len(set(map(int, visible))) != 16):
    raise ValueError("Expose 16 distinct A3 NPU devices on each node")
for name, value in (("HOST_POOL_GIB", host), ("CONTEXT_LENGTH", context),
                    ("CHUNKED_PREFILL_SIZE", chunk), ("MAX_PREFILL_TOKENS", prefill),
                    ("MAX_RUNNING_REQUESTS", requests)):
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
if int(chunk) % 128:
    raise ValueError("CHUNKED_PREFILL_SIZE must be divisible by 128")
if not math.isfinite(float(fraction)) or not 0 < float(fraction) < 1:
    raise ValueError("MEM_FRACTION_STATIC must be in (0, 1)")
cfg = json.loads((Path(model) / "config.json").read_text())
if cfg.get("model_type") != "glm_moe_dsa" or "GlmMoeDsaForCausalLM" not in cfg.get("architectures", []):
    raise ValueError("MODEL_PATH must contain GLM-5.2 / GlmMoeDsaForCausalLM")
experts = cfg.get("n_routed_experts")
if type(experts) is not int or experts <= 0 or experts % ep:
    raise ValueError("n_routed_experts must be positive and divisible by EP8")
layers, first, frequency = (cfg.get(k) for k in ("num_hidden_layers", "first_k_dense_replace", "moe_layer_freq"))
if type(layers) is not int or layers <= 0 or type(first) is not int or first < 0 or type(frequency) is not int or frequency <= 0:
    raise ValueError("Invalid num_hidden_layers / first_k_dense_replace / moe_layer_freq")
parts = [int(x) for x in partition.split(",")]
if len(parts) != 4 or any(x <= 0 for x in parts) or sum(parts) != layers:
    raise ValueError(f"SGLANG_PP_LAYER_PARTITION must have 4 positive lengths summing to {layers}")
bounds = [0, *accumulate(parts)]
counts = [sum(i >= first and i % frequency == 0 for i in range(start, end))
          for start, end in zip(bounds, bounds[1:])]
if not all(counts):
    raise ValueError("Each PP stage must contain at least one routed MoE layer")
options = run_path(str(Path(root) / "python/sglang/srt/offloading/moe/npu/options.py"))
if offload and resident_ratio:
    raise ValueError("Set only one of PREFILL_OFFLOAD_RATIO and PREFILL_RESIDENT_RATIO")
if not offload and not resident_ratio:
    offload = "0.75"
if offload:
    value = Decimal(offload)
    if not value.is_finite() or not 0 < value <= 1:
        raise ValueError("PREFILL_OFFLOAD_RATIO must be in (0, 1]; use the resident baseline for 0")
    resident_ratio = float(Decimal(1) - value)
ratio = options["parse_prefill_resident_ratio"](float(resident_ratio))
if slots and (not slots.isdecimal() or int(slots) <= 0):
    raise ValueError("EXECUTION_SLOTS must be a positive integer")
slot_budget = int(slots) if slots else None
local_experts = experts // ep
layouts = [options["prefill_weight_slots"](local_experts, n, ratio, slot_budget) for n in counts]
resident = layouts[0][0]
output = {
    "mode": "prefill_only", "cache_ratio": 0, "host_pool_gib": int(host),
    "prefill_resident_ratio": ratio,
    "native_extension_directory": str(Path(runtime) / "native_acl"),
}
if slot_budget is not None:
    output["execution_slots"] = slot_budget
elif not resident:
    output["execution_slots"] = layouts[0][1]
print(json.dumps(output, indent=2))
print(f"Node {rank}: PP stages {[2 * rank, 2 * rank + 1]}; stage routed layers: {counts}")
print(f"Experts/rank/layer: resident={resident}, cold={local_experts - resident}, total={local_experts}")
print(f"Effective offload ratio: {(local_experts - resident) / local_experts:.6f}; requested resident ratio: {ratio}")
print(f"HBM slots/rank by stage: {[n for _, n in layouts]} (resident prefix + two cold banks)")
hidden, intermediate = cfg.get("hidden_size"), cfg.get("moe_intermediate_size")
if type(hidden) is int and hidden > 0 and type(intermediate) is int and intermediate > 0:
    # Current symmetric W8A8: BF16 scales, FP32 offsets; Host keeps FP32 scale extents.
    n = 2 * intermediate + hidden
    expert_bytes = 3 * hidden * intermediate + 6 * n
    host_bytes = 3 * hidden * intermediate + 8 * n
    host_gib = [host_bytes * local_experts * count / 1024**3 for count in counts]
    print(f"Estimated cold H2D/layer/rank: {expert_bytes * (local_experts - resident) / 1024**3:.6f} GiB")
    print(f"Estimated expert HBM GiB/rank by stage: {[round(expert_bytes * n / 1024**3, 4) for _, n in layouts]}")
    print(f"Estimated Host data GiB/rank by stage: {[round(x, 4) for x in host_gib]}; resident copies remain on Host")
    if max(host_gib) >= int(host):
        raise ValueError("HOST_POOL_GIB leaves no room for the largest stage's weights and metadata")
print(f"Host pool reservation: {host} GiB/rank, {16 * int(host)} GiB/node; allow extra RAM for loading and OS")
common = dict(model_config=cfg, pp_partition=parts, nnodes=2, pp=4, tp=8, ep=8,
              cp=cp, backend=backend, resident_ratio=ratio, slots=slot_budget,
              host_gib=int(host), context=int(context), chunk=int(chunk),
              prefill=int(prefill), requests=int(requests), fraction=float(fraction),
              p0=p0, p1=p1, ports=list(map(int, ports)), fake=int(fake))
signature = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
print(f"Common configuration SHA256 (compare both nodes): {signature}")
if dry == "0":
    Path(logs).mkdir(parents=True, exist_ok=True)
    (Path(logs) / "offload.json").write_text(json.dumps(output, indent=2) + "\n")
    (Path(logs) / "launch_config.json").write_text(
        json.dumps(dict(common=common, signature=signature, node_rank=rank), indent=2) + "\n")
PY_CONFIG

command=("$PYTHON_BIN" -u -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name glm-5
  --device npu --base-gpu-id 0 --attention-backend ascend --quantization modelslim
  --load-format safetensors --dtype bfloat16 --trust-remote-code
  --host "$SERVER_HOST" --port "$SERVER_PORT" --dist-init-addr "$P0_IP:$DIST_PORT"
  --nnodes 2 --node-rank "$NODE_RANK" --pp-size 4 --tp-size 8 --dp-size 1 --attn-dp-size 1
  --enable-prefill-cp --attn-cp-size 8 --cp-strategy "$CP_STRATEGY"
  --ep-size 8 --moe-dp-size 1 --moe-dense-tp-size 1 --enable-dp-lm-head
  --load-balance-method round_robin "${moe_args[@]}"
  --disaggregation-mode prefill --disaggregation-transfer-backend ascend
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
  --disable-overlap-schedule --disable-shared-experts-fusion
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled
  --page-size 128 --kv-cache-dtype bfloat16
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --moe-offload-config "$LOG_DIR/offload.json" --watchdog-timeout 1200)

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_HOST_IP="$LOCAL_HOST_IP"
# Both store variable names must resolve to the same common endpoint.
export ASCEND_MF_STORE_URL="tcp://$P0_IP:$MF_STORE_PORT"
export MF_CONFIG_STORE_URL="$ASCEND_MF_STORE_URL"
export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
export SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE="$FAKE_PREFILL_TEST"
printf 'Node: %s\nLocal IP: %s\nPP partition: %s\nKV store: %s\nRuntime: %s\nLogs: %s\nCommand:' \
  "$NODE_RANK" "$LOCAL_HOST_IP" "$SGLANG_PP_LAYER_PARTITION" "$ASCEND_MF_STORE_URL" "$RUNTIME_DIR" "$LOG_DIR"
printf ' %q' "${command[@]}"
printf '\n'
if [[ $DRY_RUN == 1 ]]; then exit 0; fi

[[ -d /sys/class/net/$COMM_IFACE ]] || fail "Communication interface does not exist: $COMM_IFACE"
mkdir -p "$RUNTIME_DIR/python" "$RUNTIME_DIR/native_acl" "$LOG_DIR"
exec > >(tee "$LOG_DIR/server.log") 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ASCEND_LAUNCH_BLOCKING
for env_file in "$CANN_ENV" "$ATB_ENV" "$CUSTOMIZE_ENV" "$CUSTOM_TRANSFORMER_ENV"; do
  [[ -f $env_file ]] || fail "Missing Ascend environment script: $env_file"
  source "$env_file"
done

# Build a node-local runtime, as in the existing single-node offload launchers.
# After upgrading CANN / torch_npu / MemFabric, choose a fresh RUNTIME_DIR.
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

if not ensure_acc_offload_ext("A3", install_dir=sys.argv[1]):
    raise RuntimeError("Failed to prepare the A3 MemFabric offload extension")
PY_EXT
"$PYTHON_BIN" -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SGLANG_SET_CPU_AFFINITY=1
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1200}
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
if [[ $P_MOE_A2A_BACKEND == deepep ]]; then
  export DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-$HCCL_BUFFSIZE}
  export DEEPEP_NORMAL_LONG_SEQ_ROUND=${DEEPEP_NORMAL_LONG_SEQ_ROUND:-72}
  export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}
  export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=${DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ:-1}
  export MOE_EXPERT_TOKEN_NUMS_TYPE=1 DEEP_NORMAL_MODE_USE_INT8_QUANT=1
fi
export STREAMS_PER_DEVICE=32 TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ENABLE_PROFILING=${ENABLE_PROFILING:-0}
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
exec "${command[@]}"
