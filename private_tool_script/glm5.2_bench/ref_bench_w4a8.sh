python -m sglang.bench_serving \
--dataset-name random --backend sglang \
--model /home/weights/GLM-5.2-W4A8C8-A5-0731 \
--dataset-path /home/y00951466/ShareGPT_V3_unfiltered_cleaned_split.json \
--host 127.0.0.1 --port 8818 --max-concurrency 8 \
--random-input-len 1024 --random-output-len 1024 \
--num-prompts 64 --random-range-ratio 1