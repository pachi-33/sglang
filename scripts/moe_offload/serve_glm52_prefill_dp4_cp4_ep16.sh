#!/usr/bin/env bash
# A3 单机 16 die，PP1 / attention DP4 × CP4 / EP16，仅 Prefill 权重卸载。
set -e

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  cat <<EOF
Usage: MODEL_PATH=/models/GLM-5.2-w8a8 $0 [--dry-run]
默认每层常驻 25% 的本地 routed experts，每个 attention DP 组实际 chunk 上限 16384。
PREFILL_RESIDENT_RATIO 可调整常驻比例；CHUNKED_PREFILL_SIZE 是除以 DP 数之前的值。
P_MOE_A2A_BACKEND=deepep 可选择 DeepEP normal；默认 Ascend Ops alltoallv。
--dry-run 只检查配置并打印启动命令。
EOF
  exit 0
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LOG_DIR=${LOG_DIR:-/tmp/sglang-glm52-prefill-dp4-cp4-ep16-$(date +%Y%m%d_%H%M%S)-$$}
exec bash "$script_dir/serve_glm52_prefill_pp4_cp4.sh" --dp4-cp4-ep16 "$@"
