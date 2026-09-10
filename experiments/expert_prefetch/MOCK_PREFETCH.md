# V100 分层 LRU Mock Expert Prefetch

该基础设施把专家预测与数据搬运解耦。服务启动时把 ExpertPack 的 9,728 个
payload 全量校验并放入 pinned CPU 内存，随后关闭 pack 文件。运行时的 demand
load 和 prefetch 都是 pinned CPU DRAM 到 V100 VRAM 的 H2D，不再读取 SSD。

## Cache 布局

`--expert-cache-ratio R` 控制 layers 1–38 各自独立的严格 LRU 热区：

```text
C = floor(256 * R)
S = 256 - C
total_slots = 38 * C + S
```

`S` 是跨层复用的 prefill 临时区，不参与 decode 淘汰。默认 `R=0.40`，对应
每层 102 个热专家、154 个临时 slot、总计 4,030 个专家和约 6,800.7 MiB。
`--expert-cache-mib` 是安全上限，不改变 ratio 推导出的容量。

Demand hit 会更新本层 MRU。预取 miss 插入 MRU，允许错误预测污染缓存；预取
已驻留专家不会更新热度。任何一层都不能淘汰另一层的专家。

## Record/Replay

Mock 请求成对执行：

1. `record` 使用 greedy 推理记录内存中的 `[N,40,8]` 路由 oracle。
2. 服务同步 CUDA，清空所有专家映射和 LRU，但保留物理 cache tensor。
3. `replay` 使用相同请求和 mock 超参执行，并逐 token 对照 Record。

Mock 仅在 decode row 1 以后生效。layer L 的 router-ready event 触发
layer `L + lead_layers` 的低优先级 prefetch stream；compute stream 随即继续
当前层尾部和后续层计算。目标层未命中的 demand copy 使用独立高优先级 stream。

启动服务：

```bash
CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 \
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.single_gpu_api \
  --model-dir /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16 \
  --expert-pack-manifest /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json \
  --expert-source pinned-memory --expert-cache-policy layer-lru \
  --expert-cache-ratio 0.40 --expert-cache-mib 7168 \
  --enable-mock-expert-prefetch --capacity 2048 \
  --host 127.0.0.1 --port 8818
```

运行 paired benchmark：

```bash
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.bench_serving \
  --dataset-name random --backend sglang \
  --model /home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16 \
  --dataset-path /home/yaozhenyang/dev/sglang-v100/ShareGPT_V3_unfiltered_cleaned_split.json \
  --host 127.0.0.1 --port 8818 --max-concurrency 1 \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 128 --random-range-ratio 1 --request-rate inf \
  --random-input-token-ids --seed 0 \
  --mock-expert-prefetch --mock-prefetch-recall 0.5 \
  --mock-prefetch-top-k 8 --mock-prefetch-lead-layers 2 \
  --mock-prefetch-seed 0
```

Bench 的 `duration`、TTFT、ITL 和吞吐只使用 Replay。`wall_duration` 和
`record_duration` 单独记录。`top_k=0, recall=0` 是同样采用成对执行和冷 cache
起点的 demand-only control。

每次 replay 返回聚合的候选、useful/on-time/late/wasted、layer-local hit/miss、
eviction、prefetch/demand H2D、可用窗口、暴露等待和 overlap ratio。服务参数
`--mock-prefetch-log` 可保存逐 row、逐 target layer 的 CUDA Event 原始记录。
