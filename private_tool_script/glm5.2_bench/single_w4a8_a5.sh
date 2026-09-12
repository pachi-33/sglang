
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh


export ASCEND_LAUNCH_BLOCKING=1


export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
# export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TRANSFORMERS_VERBOSITY=error

MODEL_PATH=/home/weights/GLM-5.2-W4A8C8-A5-0731
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_BS=16
# export PYTHONPATH=/home/y00951466/sglang/python:$PYTHONPATH
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

# export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 8 \
        --nnodes 1 \
        --dp-size 2 \
	--enable-dp-attention \
        --chunked-prefill-size 2048 \
        --max-prefill-tokens 32768 \
        --trust-remote-code \
        --mem-fraction-static 0.85 \
        --served-model-name GLM-5.2-w4a8 \
        --enable-prefill-delayer \
        --prefill-delayer-max-delay-passes 100 \
        --cuda-graph-bs-decode 8 \
        --cuda-graph-bs-prefill 8 \
        --max-running-requests 128 \
        --quantization modelslim \
        --moe-a2a-backend deepep --deepep-mode auto \
        --load-balance-method round_robin \
        --device npu --host 127.0.0.1 --port 8818 \
	--speculative-algorithm NEXTN --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5
	# --speculative-draft-model-quantization unquant


# python -m sglang.bench_serving \
# --dataset-name random --backend sglang \
# --model /home/weights/GLM-5.2-0610-Provider-w4a8 \
# --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
# --host 61.47.19.67 --port 8810 --max-concurrency 104 \
# --random-input-len 1024 --random-output-len 1024 \
# --num-prompts 456 --random-range-ratio 1
