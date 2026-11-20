#!/usr/bin/env bash
# 在已激活 torch_npu 的 A3 环境中运行单机 EP16 测试。
# bash scripts/moe_offload/run_glm52_prefill_dfc_h2d.sh --profile --mode both --tokens-per-rank 128 512
set -eo pipefail

# 测试默认值。可直接修改或用环境变量覆盖，命令行参数优先。
backend=${P_MOE_A2A_BACKEND:-ascend_ops}       # --backend，ascend_ops / deepep / both
mode=${DFC_MODE:-baseline}                      # --mode，baseline / overlap / both
tokens_per_rank="${TOKENS_PER_RANK:-128}"       # --tokens-per-rank，每个 rank 的输入 token 数，多组用空格分隔
ep_load="${EP_LOAD:-balanced}"                 # --ep-load，balanced / 3to1 / 7to1，多组用空格分隔
warmup=${DFC_WARMUP:-1}                        # --warmup，每组在采集前执行的预热轮数
iterations=${DFC_ITERATIONS:-1}                # --iterations，每组预热后的执行轮数
profile=${DFC_PROFILE:-0}                      # --profile，1 时每个 rank 连续采集所有 workload 的正式轮次
output_dir=${DFC_OUTPUT_DIR:-/tmp/glm52-dfc}   # --output，结果根目录
native_dir=${SGLANG_MOE_NATIVE_EXTENSION_DIR:-} # --native-extension-dir，留空时在 runtime_dir/native_acl 构建

# 启动与构建
python_bin=${PYTHON_BIN:-python3}
runtime_dir=${RUNTIME_DIR:-/tmp/sglang-glm52-prefill-pp4-cp4-runtime}
rdzv_endpoint=${RDZV_ENDPOINT:-127.0.0.1:29500}
dry_run=0                                     # --dry-run，默认执行，传入后只打印命令
max_build_jobs=4                              # MAX_JOBS 未设置时使用的编译并发数
ascend_env_files=(
  "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
  "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
  "${CUSTOMIZE_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash}"
  "${CUSTOM_TRANSFORMER_ENV:-/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash}"
)

# NPU 和通信默认值在 Ascend 环境脚本加载后生效。
configure_npu_env() {
  local comm_iface=${COMM_IFACE:-lo}
  export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
  export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$comm_iface}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$comm_iface}
  export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
  export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
  export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300}
  export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
  export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-300}
  export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
  export INF_NAN_MODE_FORCE_DISABLE=${INF_NAN_MODE_FORCE_DISABLE:-1}
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
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/../.." && pwd)
benchmark="$root/python/sglang/srt/offloading/moe/test/test_glm52_prefill_dfc_h2d.py"
test_args=()
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    -h|--help)
      echo "用 --dry-run 预览最终命令，用 --no-profile 关闭采样。测试参数如下。"
      exec "$python_bin" "$benchmark" --help ;;
    --backend) backend=${2:?--backend requires a value}; shift ;;
    --backend=*) backend=${1#*=} ;;
    --native-extension-dir) native_dir=${2:?--native-extension-dir requires a path}; shift ;;
    --native-extension-dir=*) native_dir=${1#*=} ;;
    --profile) profile=1 ;;
    --no-profile) profile=0 ;;
    *) test_args+=("$1") ;;
  esac
  shift
done

# 指定的目录用于加载已部署的扩展。默认沿用服务脚本的构建缓存，在启动 16 个 rank 前构建一次。
build=()
if [[ -z $native_dir ]]; then
  native_dir="$runtime_dir/native_acl"
  build=("$python_bin" -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$native_dir")
fi
launch=("$python_bin" -m torch.distributed.run
  --nnodes=1
  --nproc-per-node=16
  --rdzv-backend=static
  --rdzv-endpoint="$rdzv_endpoint"
  --rdzv-id="glm52-dfc-$$"
  "$benchmark" --native-extension-dir "$native_dir" --backend "$backend")

# 在调用者省略选项时补上默认值，保留 argparse 对多值选项和参数校验的处理。
add_default() {
  local arg
  for arg in "${test_args[@]}"; do
    if [[ $arg == "$1" || $arg == "$1="* ]]; then
      return 0
    fi
  done
  launch+=("$@")
}
read -r -a token_counts <<< "$tokens_per_rank"
read -r -a loads <<< "$ep_load"
add_default --mode "$mode"
add_default --tokens-per-rank "${token_counts[@]}"
add_default --ep-load "${loads[@]}"
add_default --warmup "$warmup"
add_default --iterations "$iterations"
add_default --output "$output_dir"
case "$profile" in
  0) ;;
  1) launch+=(--profile) ;;
  *) echo "DFC_PROFILE 必须为 0 或 1" >&2; exit 2 ;;
esac
launch+=("${test_args[@]}")

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
for env_file in "${ascend_env_files[@]}"; do
  if [[ -f $env_file ]]; then
    source "$env_file"
  fi
done
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
configure_npu_env

if (( ${#build[@]} )); then
  export MAX_JOBS=${MAX_JOBS:-$max_build_jobs}
  "${build[@]}"
fi
exec "${launch[@]}"
