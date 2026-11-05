# ============================================================
# Before running, update the following variables:
#   IPS: IP addresses of each node in the cluster
#   IP_MASTER: rank 0 node IP address with port
#   MODEL_PATH: path to the model weights directory
#   HCCL_SOCKET_IFNAME: network interface name for HCCL
#   GLOO_SOCKET_IFNAME: network interface name for Gloo
# ============================================================

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
# MTP OVERLAP
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV

# Run command ifconfig on two nodes, find out which inet addr has same IP with your node IP. That is your public interface, which should be added here
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0

# DEEPEP
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1


IPS=('61.47.19.69' '61.47.19.71')
IP_MASTER="${IPS[0]}:5000"

MODEL_PATH=/home/weights/GLM-5.2-w8a8
export HCCL_BUFFSIZE=1024
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
for i in "${!IPS[@]}";
do
    if [[ "$LOCAL_HOST1" == "${IPS[$i]}" || "$LOCAL_HOST2" == "${IPS[$i]}" ]];
    then
        echo "${IPS[$i]}"
        python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 32 --nnodes 2 --node-rank $i --dist-init-addr $IP_MASTER \
        --chunked-prefill-size 32768 --max-prefill-tokens 65536 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.70 \
        --port 30000 \
        --served-model-name glm-5 \
        --cuda-graph-max-bs-decode 2 \
        --attn-dp-size 16 \
        --enable-dp-lm-head --moe-dense-tp-size 1 \
        --moe-a2a-backend ascend_ops --deepep-mode auto \
        --moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}' \
        --disable-overlap-schedule \
        --speculative-draft-model-quantization unquant \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4  \
        --disable-radix-cache
        NODE_RANK=$i
        break
    fi
done