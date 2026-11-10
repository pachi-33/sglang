#!/usr/bin/env bash
set -euo pipefail

# 直连 P0。前缀比例大于 0 时，P 启动前需设置 SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE=1。
# N=4 KV_CACHE_HIT_RATE=0.90 INPUT_LEN=128000 CONCURRENCY=1 bash bench_4group_shared_prefix.sh
PYTHON_BIN=${PYTHON_BIN:-python}
HOST=${HOST:-61.47.19.69}
PORT=${PORT:-8000}
MODEL=${MODEL:-/home/weights/GLM-5.2-w8a8}
N=${N:-4}
INPUT_LEN=${INPUT_LEN:-128000}
KV_CACHE_HIT_RATE=${KV_CACHE_HIT_RATE:-0.90}
CONCURRENCY=${CONCURRENCY:-1}
READY_CHECK_TIMEOUT_SEC=${READY_CHECK_TIMEOUT_SEC:-60}
NUM_GROUPS=4

# 用同一个 seed 和固定长度生成两次，保证四条预热前缀与正式请求逐 token 一致。
PARAMS=$("$PYTHON_BIN" - "$NUM_GROUPS" "$N" "$INPUT_LEN" "$KV_CACHE_HIT_RATE" \
    "$CONCURRENCY" "$PORT" "$READY_CHECK_TIMEOUT_SEC" "${SEED:-}" <<'PY'
import secrets
import sys
from decimal import Decimal, InvalidOperation

groups, n, input_len, ratio, concurrency, port, ready_timeout, seed = sys.argv[1:]
try:
    groups, n, input_len, concurrency, port, ready_timeout = map(
        int, (groups, n, input_len, concurrency, port, ready_timeout)
    )
    ratio = Decimal(ratio)
    seed = int(seed) if seed else secrets.randbits(32)
    if min(n, input_len, concurrency) < 1:
        raise ValueError("N、INPUT_LEN、CONCURRENCY 必须是正整数")
    if not ratio.is_finite() or not 0 <= ratio <= 1:
        raise ValueError("KV_CACHE_HIT_RATE 必须在 [0, 1] 内，例如 0.90")
    if not 1 <= port <= 65535 or ready_timeout < 0:
        raise ValueError("PORT 必须在 [1, 65535] 内，READY_CHECK_TIMEOUT_SEC 必须非负")
    if not 0 <= seed < 2**32:
        raise ValueError("SEED 必须在 [0, 2**32) 内")
except (ValueError, InvalidOperation) as exc:
    sys.exit(f"配置错误：{exc}")

prefix_len = int(input_len * ratio)
print(prefix_len, input_len - prefix_len, groups * n, seed)
PY
)
read -r PREFIX_LEN QUESTION_LEN TOTAL_REQUESTS SEED <<< "$PARAMS"

RESULT_ROOT=${RESULT_ROOT:-.}
mkdir -p -- "$RESULT_ROOT"
RUN_DIR=$(mktemp -d "${RESULT_ROOT%/}/prefill-4groups.XXXXXX")

COMMON_ARGS=(
    --backend sglang
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL"
    --tokenizer "$MODEL"
    --ready-check-timeout-sec "$READY_CHECK_TIMEOUT_SEC"
    --dataset-name generated-shared-prefix
    --gsp-num-groups "$NUM_GROUPS"
    --gsp-group-distribution uniform
    --gsp-system-prompt-len "$PREFIX_LEN"
    --gsp-range-ratio 1.0
    --gsp-output-len 1
    --tokenize-prompt
    --warmup-requests 0
    --request-rate inf
    --seed "$SEED"
    --extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'
    --cache-report
)

run_bench() {
    local per_group=$1 question_len=$2 concurrency=$3 output_file=$4 expected=$5
    "$PYTHON_BIN" -m sglang.bench_serving "${COMMON_ARGS[@]}" \
        --gsp-prompts-per-group "$per_group" \
        --gsp-question-len "$question_len" \
        --max-concurrency "$concurrency" \
        --output-file "$output_file" \
        --output-details

    "$PYTHON_BIN" - "$output_file" "$expected" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
try:
    with open(path, encoding="utf-8") as report:
        completed = json.loads(report.readline())["completed"]
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f"无法读取压测结果 {path}: {exc}")
if completed != expected:
    sys.exit(f"压测成功 {completed}/{expected} 条，请检查 {path}")
PY
}

printf 'P0=%s:%s | %s 组 × %s 条 = %s 条正式请求 | 并发=%s\n' \
    "$HOST" "$PORT" "$NUM_GROUPS" "$N" "$TOTAL_REQUESTS" "$CONCURRENCY"
printf '目标前缀占比=%s | 每条输入=%s 前缀=%s 后缀=%s | seed=%s\n' \
    "$KV_CACHE_HIT_RATE" "$INPUT_LEN" "$PREFIX_LEN" "$QUESTION_LEN" "$SEED"
printf '结果目录 %s\n' "$RUN_DIR"

if (( PREFIX_LEN > 0 )); then
    echo '[1/2] 串行预热四组，每组只发共享前缀'
    run_bench 1 0 1 "$RUN_DIR/warmup.jsonl" "$NUM_GROUPS"
else
    echo '[1/2] 前缀长度为 0，跳过预热'
fi

echo '[2/2] 正式压测，实际命中率看 Cache Hit Details'
run_bench "$N" "$QUESTION_LEN" "$CONCURRENCY" "$RUN_DIR/benchmark.jsonl" "$TOTAL_REQUESTS"
