export PYTHONPATH=/home/y00951466/sglang/python:$PYTHONPATH

python -m sglang.bench_serving \
    --dataset-name random --backend sglang \
    --model /home/weights/Qwen3.6-27B-W8A8 \
    --host 61.47.19.76 --port 8818 --max-concurrency 104 \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 128 --random-range-ratio 1