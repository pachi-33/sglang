#!/usr/bin/env bash
# 在已激活 torch_npu 的 A3 环境中运行单机 EP16 测试。
# bash scripts/moe_offload/run_glm52_prefill_dfc_h2d.sh --mode both --tokens-per-rank 128 512
set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../.." && pwd)
benchmark="$root/python/sglang/srt/offloading/moe/test/test_glm52_prefill_dfc_h2d.py"
python_bin=${PYTHON_BIN:-python3}
backend=${P_MOE_A2A_BACKEND:-ascend_ops}
native_dir=${SGLANG_MOE_NATIVE_EXTENSION_DIR:-}
dry_run=0
test_args=()
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    -h|--help)
      echo "启动脚本支持 --dry-run 预览命令。测试参数如下。"
      exec "$python_bin" "$benchmark" --help ;;
    --backend) backend=${2:?--backend requires a value}; shift ;;
    --backend=*) backend=${1#*=} ;;
    --native-extension-dir) native_dir=${2:?--native-extension-dir requires a path}; shift ;;
    --native-extension-dir=*) native_dir=${1#*=} ;;
    *) test_args+=("$1") ;;
  esac
  shift
done

# 指定的目录用于加载已部署的扩展。默认沿用服务脚本的构建缓存，在启动 16 个 rank 前构建一次。
build=()
if [[ -z $native_dir ]]; then
  native_dir="${RUNTIME_DIR:-/tmp/sglang-glm52-prefill-pp4-cp4-runtime}/native_acl"
  build=("$python_bin" -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$native_dir")
fi
launch=("$python_bin" -m torch.distributed.run --standalone --nproc-per-node=16
  "$benchmark" --native-extension-dir "$native_dir" --backend "$backend" "${test_args[@]}")
if (( dry_run )); then
  if (( ${#build[@]} )); then
    printf ' %q' "${build[@]}"
    printf '\n'
  fi
  printf ' %q' "${launch[@]}"
  printf '\n'
  exit 0
fi

# 加载当前安装提供的 CANN、ATB 和自定义算子环境。
for env_file in \
  "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" \
  "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}" \
  "${CUSTOMIZE_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash}" \
  "${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash}"; do
  if [[ -f $env_file ]]; then
    source "$env_file"
  fi
done
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-${COMM_IFACE:-lo}}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${COMM_IFACE:-lo}}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024} HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-300}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32} INF_NAN_MODE_FORCE_DISABLE=${INF_NAN_MODE_FORCE_DISABLE:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
unset ASCEND_LAUNCH_BLOCKING

if [[ $backend == deepep || $backend == both ]]; then
  export DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-$HCCL_BUFFSIZE}
  export DEEPEP_NORMAL_LONG_SEQ_ROUND=${DEEPEP_NORMAL_LONG_SEQ_ROUND:-72}
  export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}
  export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=${DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ:-1}
  export MOE_EXPERT_TOKEN_NUMS_TYPE=${MOE_EXPERT_TOKEN_NUMS_TYPE:-1}
  export DEEP_NORMAL_MODE_USE_INT8_QUANT=${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}
fi

if (( ${#build[@]} )); then
  export MAX_JOBS=${MAX_JOBS:-4}
  "${build[@]}"
fi
exec "${launch[@]}"
