unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# export PYTHONPATH=/mnt/share/l00951279/sglang/python:$PYTHONPATH
# export PYTHONPATH=/home/y00951466/sglang-my/python:$PYTHONPATH/
export PYTHONPATH=/home/y00951466/sglang-a5-optim/python:$PYTHONPATH/

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_BUFFSIZE=600
export HCCL_OP_EXPANSION_MODE=""AIV""
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export SGLANG_HICACHE_HYBM_RESERVE_GB=30
export SGLANG_HICACHE_IO_ASCENDC=1
export SGLANG_HICACHE_HOST_MEM=hybm

# source /mnt/share/chenxu/SFA/vendors/custom_transformer/bin/set_env.bash

export ACL_DEVICE_SYNC_TIMEOUT=300

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32

# MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W4A4C8-mxfp4-A5-0731
MODEL_PATH=/home/weights/GLM-5.2-W4A8C8-A5-0731
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=0

export TRANSFORMERS_VERBOSITY=error

export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

SERVED_MODEL_NAME=glm52
SERVER_HOST=127.0.0.1
SERVER_PORT=8818

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--served-model-name "${SERVED_MODEL_NAME}" \
--host "${SERVER_HOST}" \
--port "${SERVER_PORT}" \
--tp-size 8 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--max-running-requests 24 \
--mem-fraction-static 0.85 \
--quantization modelslim \
--max-prefill-tokens 2048000 \
--chunked-prefill-size 16384 \
--enable-metrics \
--speculative-algorithm EAGLE \
--speculative-num-steps 4 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 5 \
--kv-cache-dtype "fp8_e4m3"
# --max-total-tokens 327680 \
# --cuda-graph-max-bs-decode 8 \
# --cuda-graph-max-bs-prefill 8 \
