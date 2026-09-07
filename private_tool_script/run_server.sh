# high performance cpu

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

sysctl -w vm.swappiness=0

sysctl -w kernel.numa_balancing=0

sysctl -w kernel.sched_migration_cost_ns=50000

# bind cpu

export SGLANG_SET_CPU_AFFINITY=1  

unset https_proxy

unset http_proxy

unset HTTPS_PROXY

unset HTTP_PROXY

unset ASCEND_LAUNCH_BLOCKING

# cann

source /usr/local/Ascend/ascend-toolkit/set_env.sh

source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

export SGLANG_ENABLE_SPEC_V2=1

export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export HCCL_BUFFSIZE=1000

export HCCL_OP_EXPANSION_MODE=AIV

export HCCL_SOCKET_IFNAME=lo

export GLOO_SOCKET_IFNAME=lo

export TRANSFORMERS_VERBOSITY=error

# MODEL_PATH=/home/weights/Qwen3.6-27B-W8A8
MODEL_PATH=/home/weights/Qwen3-Next-80B-A3B-Instruct

export SGLANG_NPU_PROFILING=0

export SGLANG_NPU_PROFILING_BS=16

export PYTHONPATH=/home/y00951466/sglang/python:$PYTHONPATH

export ASCEND_RT_VISIBLE_DEVICES=15

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 1 \
        --nnodes 1 \
        --dp-size 1 \
        --cuda-graph-bs 8 \
        --max-running-requests 128 \
        --quantization modelslim \
        --device npu --host 61.47.19.76 --port 8818

  

# exit

  

# python -m sglang.bench_serving \

# --dataset-name random --backend sglang \

# --model /home/weights/GLM-5.2-0610-Provider-w4a8 \

# --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \

# --host 61.47.19.67 --port 8810 --max-concurrency 104 \

# --random-input-len 1024 --random-output-len 1024 \

# --num-prompts 456 --random-range-ratio 1