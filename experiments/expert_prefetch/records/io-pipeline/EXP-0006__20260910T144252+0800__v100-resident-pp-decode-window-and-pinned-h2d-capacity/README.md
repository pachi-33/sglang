# EXP-0006: V100 resident PP decode window and pinned H2D capacity

## 实验身份

- 类型：`io-pipeline`
- 开始时间：`2026-09-10T14:42:52+08:00`
- 结束时间：`2026-09-10T14:54:42+08:00`
- 状态：`completed`
- 测量设备：Tesla V100-SXM2-16GB `GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96`
- PP 辅助设备：NVIDIA GeForce RTX 4070 SUPER `GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4`
- Git：`lab/expert-prefetch@0ff9268cf4da181547c6de1e75e88d93d05d33f2`，dirty=false

## 问题与假设

测量 V100 上 batch=1 单 token decode 时，从当前层 router 完成到下一层 routed expert 开始之间的真实计算窗口；再用相同 ExpertPack payload 的 pinned CPU→V100 H2D 延迟估计该窗口能够容纳的专家数量。

假设：常驻专家模型中的相邻层计算能够提供稳定、可量化的预取窗口，至少一档候选数量能在 95% 的 offload-relevant 窗口内完成 H2D。

容量结论只表示 **H2D-only 上界**。它不包含预测器、router ID D2H、SSD 读取、checksum、CPU 调度和实际 copy/compute 争用。

## 对照与变量

- 对照组：同一常驻 PP session、同一 prompt、关闭 timing 的 257-token greedy 生成。
- 实验组：reset 后开启 back worker timing 的 257-token greedy 生成，其中 256 个实际 decode step 被记录。
- 独立变量：timing 开关；H2D candidate 数量 `1,2,4,8,16,24,32`。
- 固定变量：batch=1、greedy、忽略 EOS、输入 1024 tokens、capacity 2048、PP split 17、同一模型和 token 序列。
- PP 布局：4070 front layers 0–16；V100 back layers 17–39；`expert_offload=None`，两侧 routed experts 全部常驻。
- H2D：每专家 `1,769,488 bytes`，pinned staging，独立 transfer stream，8 个 SoA component copy；每档 warmup 20 次、正式 200 次。
- 干扰因素：操作系统调度、GPU 温度/频率、PCIe 竞争，以及事件采集本身的小量开销。

## 输入与身份

- 模型：`/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16`
- config SHA-256：`e1be0a1fb619901c1f97afeb75beb2a6581be706e4eb6166d10690104e0b7634`
- checkpoint index SHA-256：`5bd3e4596cf3d20483079f01df50c78946a2fe2aa27251acaa13dff3a06e27ed`
- ExpertPack manifest：`/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4-expertpack-v1/manifest.json`
- ExpertPack manifest SHA-256：`bd194286aed4b16814370d80c73878bf3049937de9c88890da6733b7ca6b54c3`
- ShareGPT：`/home/yaozhenyang/dev/sglang-v100/ShareGPT_V3_unfiltered_cleaned_split.json`
- ShareGPT SHA-256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`
- 数据集复现：seed 0、request index 0、严格 1024 tokens。
- prompt token SHA-256：`4e7bf8d35d419417f9274499ef041fe6cc5de9210c9e1ffc6f05bc8f9af823e8`
- EXP-0001 oracle：`records/trace-analysis/EXP-0001__20260908T170025+0800__v100-random-1024x256-expert-activation-dataset/artifacts/EXP-0001.expert-activation-dataset.npz`

## 指标与通过条件

- timed/control 的 257 个 token 完全一致，前 256 个 token 与 EXP-0001 request 0 一致。
- `decode_timing.jsonl` 恰好包含 `256 × 23 = 5,888` 行，每步覆盖 layers 17–39。
- 相邻层窗口共 `256 × 22 = 5,632` 个，其中 targets 18–38 的 offload-relevant 窗口共 `5,376` 个。
- CUDA Event 时间有限、非负，并满足五个边界单调。
- H2D 每档 200 个有效样本，传输量为 `k × 1,769,488 bytes`。
- 对每个 transition 和 candidate `k`，`fit_rate=P(deadline_window >= H2D_p95(k))`；`max_k_95` 是 fit rate 至少 95% 的最大候选数。
- 分别记录两张 GPU 的设备信息、显存状态和实验后的进程清理。

## 操作步骤

1. 记录实验前设备快照，确认没有遗留计算进程。
2. 在一个常驻 PP session 中运行 16-step decode warmup、关闭 timing 的 control 和开启 V100 timing 的 measurement。
3. PP controller 自动关闭两个 worker 后，确认两张卡没有遗留进程。
4. 仅暴露 V100，运行 pinned H2D 基准。
5. 生成逐层、窗口、H2D 和容量汇总并执行验收检查。
6. 记录实验后设备快照、结果、结论和清理状态。

完整命令见 [commands.sh](commands.sh)。

## 结果

- 正确性：timed/control 的 257 个 token 完全一致；前 256 个 token 与 EXP-0001 request 0 完全一致。
- 原始计时：5,888 行，完整覆盖 256 个 decode step × V100 layers 17–39；所有五事件边界有限、非负且单调。
- V100 NVFP4 layers 17–38 的层总时间：mean `1.316 ms`、p50 `1.306 ms`、p95 `1.509 ms`、p99 `1.927 ms`。
- 主要 deadline 窗口 `router_ready(L) → routed_expert_start(L+1)`：5,376 个 offload-relevant 样本，mean `1.394 ms`、p50 `1.396 ms`、p95 `1.549 ms`，最小值 `1.252 ms`。
- H2D p95：1 个专家 `0.566 ms`，2 个 `1.112 ms`，4 个 `2.201 ms`，8 个 `4.398 ms`；p50 有效带宽约 `3.00 GiB/s`。
- 容量：1 和 2 专家的总体 fit rate 均为 `100%`；4 专家为 `0.0186%`（1/5,376）；8 个及以上为 `0%`。21 个 offload-relevant transitions 的 `max_k_95` 全部为 `2`。
- `38→39` 单独报告，`max_k_95=2`，但 layer 39 routed expert 为 FP16 常驻，未计入主要容量结论。
- timing 开关墙钟对照：control `15.250 s`，timed `15.704 s`，单次观测开销 `2.98%`。
- 常驻确认：front/back READY 均为 `expert_offload_enabled=false`、`routed_experts_resident=true`，没有构造 ExpertPackStore，也没有 SSD expert read 或运行时 H2D expert install。
- 峰值 reserved memory：4070 front `11.13 GiB`，V100 back `12.03 GiB`。

轻量结果：

- [验收报告](results/acceptance_report.json)
- [实验汇总](results/experiment_summary.json)
- [逐层计时](results/layer_timing_summary.csv)
- [相邻层窗口](results/router_window_summary.csv)
- [H2D 曲线](results/h2d_summary.csv)
- [逐 transition 容量](results/prefetch_capacity.csv)
- [PP 运行摘要](results/pp_run_summary.json)
- [H2D 运行摘要](results/h2d_run_summary.json)

原始数据：

- `artifacts/decode_timing.jsonl`，SHA-256 `8331ce4a2ad6b070b9924b56620f5ee4d53db93ea1cf74e2a8f2d51ffe2db4bf`
- `artifacts/h2d_samples.npz`，SHA-256 `f9cb81fe2f499607c8d316e3535ba2fd10f88bb1380be36e6c604a99124486ae`
- 原始运行日志与实验前后 GPU 快照位于 `logs/`。

## 结论与后续

接受“存在稳定可量化窗口”的假设。按本机约 3.00 GiB/s 的 pinned H2D 曲线，当前层 router 完成后，到下一层 routed expert 开始前，95% 保证只能容纳 2 个完整 ExpertPack payload。Top-8 全量 miss 无法在一个相邻层窗口内补齐；后续真实预取策略应优先测试高置信度 Top-2，并把更大的候选集提前超过一层发出。

这仍是乐观上界。加入预测器、route D2H、SSD、checksum、Python 调度和 copy/compute 争用后，可用容量只会下降；下一实验需要在 demand-loading control 上测端到端 useful/late/wasted 和真实 stall。

## 资源清理

- [x] PP controller、front worker 和 back worker 已关闭
- [x] H2D benchmark 进程已关闭
- [x] V100 无遗留计算进程；结束显存 `1 MiB`
- [x] RTX 4070 SUPER 无遗留计算进程；结束显存 `2 MiB`
- [x] 原始日志和产物位置已记录
