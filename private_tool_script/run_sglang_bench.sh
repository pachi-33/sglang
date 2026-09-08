export PYTHONPATH=/home/y00951466/sglang/python:$PYTHONPATH

python -m sglang.bench_serving \
    --dataset-name random --backend sglang \
    --model /home/weights/Qwen3.5-35B-A3B-w8a8-mtp \
    --dataset-path /home/y00951466/ShareGPT_V3_unfiltered_cleaned_split.json \
    --host 127.0.0.1 --port 8818 --max-concurrency 10 \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 128 --random-range-ratio 1 \
    --request-rate inf