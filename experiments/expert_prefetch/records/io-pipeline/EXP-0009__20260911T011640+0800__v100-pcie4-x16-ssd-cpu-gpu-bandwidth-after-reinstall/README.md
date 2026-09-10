# EXP-0009: V100 PCIe4 x16 SSD-CPU-GPU bandwidth after reinstall

## 实验身份

- 类型：`io-pipeline`
- 开始时间：`2026-09-11T01:16:40+08:00`
- 结束时间：`2026-09-11T01:30:06+08:00`
- 状态：`completed`
- 设备：Tesla V100-SXM2-16GB GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96
- Git：`lab/expert-prefetch@a5f26cd5ec8d91f3add35c2130e3d1248861de7a`，dirty=false

## 问题与假设

问题：V100 从原 PCIe 3.0 x4 插槽迁移到主板 PCIe 4.0 x16 插槽并重装系统后，SSD→pinned CPU、pinned CPU→V100 和二者重叠的端到端带宽分别是多少？

假设：V100 因设备能力上限在负载下协商为 PCIe 3.0 x16；pinned H2D 带宽超过 10 GB/s，不再是约 7 GB/s NVMe 读取的瓶颈；三缓冲流水线的端到端带宽至少达到隔离 SSD→CPU 带宽的 90%。

## 对照与变量

- 对照组：隔离测量 `O_DIRECT SSD→pinned CPU` 和 `pinned CPU→V100`；与 EXP-0006 同 UUID V100 在旧 PCIe 3.0 x4 安装下约 3.00 GiB/s 的 ExpertPack H2D 结果作历史参考，但不合并样本。
- 实验组：`O_DIRECT SSD→3 个 16 MiB pinned buffer→V100`，读线程和单 CUDA transfer stream 流水重叠。
- 独立变量：传输路径（SSD→CPU、CPU→GPU、SSD→CPU→GPU）。
- 固定变量：V100 UUID、单 GPU 可见、8 GiB 实体测试文件、每轮 8 GiB useful bytes、3 个正式样本/路径、16 MiB SSD chunk、256 MiB H2D buffer、无 GPU expert cache、无模型/API/prefetch。
- pinned staging：流水线 48 MiB；隔离 H2D source 256 MiB。GPU destination 最大 256 MiB。
- 同口径补充：复用 EXP-0006 的真实 Qwen3.5 ExpertPack、每专家 `1,769,488 bytes`、8 个 SoA component copy，candidate `1/2/4/8/16/24/32`、每档 warmup 20/正式 200；用于比较换槽前后 H2D，文件加载不计时。
- 干扰因素：NVMe controller cache、SSD 温度/后台活动、CPU 调度、PCIe 电源状态；`O_DIRECT` 绕过 Linux page cache，但无法关闭盘内缓存。

## 指标与通过条件

- 记录每轮 elapsed、useful bytes、GB/s/GiB/s，以及 P50/P95。
- 在持续 H2D 负载中记录 PCIe generation、width、P-state。
- SSD 和 H2D 每路径每轮必须精确搬运 8 GiB；端到端最后一个 GPU buffer 必须仍为零，验证传输内容。
- 通过条件：工作态 PCIe 3.0 x16；H2D >10 GB/s；流水线带宽 ≥ SSD 隔离带宽的 90%。
- 本实验不运行模型、路由或预取，因此 token correctness、issued/useful/late/wasted、hit/miss、TTFT、ITL 和模型 peak GPU memory 不适用；会记录微基准显存与进程清理。

## 操作步骤

1. 记录实验前 GPU/PCIe、CPU、SSD、文件系统和进程快照。
2. 用 direct write 生成 8 GiB 非稀疏零填充测试文件。
3. 运行 3 轮隔离 SSD→CPU、隔离 H2D 和三缓冲 SSD→GPU 流水线。
4. 保存原始样本与汇总，检查字节数、内容和假设阈值。
5. 删除临时测试文件，确认无临时服务或 GPU compute 进程，填写结论并关闭实验。

## 结果

- 工作态链路：`P0, PCIe generation 3, width 16`；V100 设备与插槽均报告最大 Gen3 x16。
- 两次主测试合并共 6 轮，每路径每轮精确传输 8 GiB：
  - SSD→pinned CPU：p50 `1.345493 s`，`6.384 GB/s`（`5.946 GiB/s`）；p95 `1.350445 s`。
  - pinned CPU→V100：p50 `0.652703 s`，`13.161 GB/s`（`12.257 GiB/s`）；p95 `0.652726 s`。
  - 三缓冲 SSD→CPU→V100：p50 `1.357158 s`，`6.329 GB/s`（`5.895 GiB/s`）；p95 `1.376240 s`。
- 流水线/隔离 SSD 带宽比 `99.14%`，通过 ≥90% 条件；端到端已经是 SSD 受限。
- 3 条内容校验全部通过；正式轮次与 warmup 外，每条路径 useful bytes 均与配置一致。
- 第二次主测试在正确位置采样的峰值 GPU allocated memory 为 `256 MiB`。首轮带宽有效，但其 `2.5 GiB` 峰值被计时后的全量零值归约 workspace 污染；首轮 summary/raw 保留且不作为显存结论。
- 同口径真实 ExpertPack H2D：Top-8 `14,155,904` payload bytes 的 p50 从 EXP-0006 的 `4.395648 ms` 降至 `1.187616 ms`，有效带宽从 `2.9993` 升至 `11.1010 GiB/s`，提升 `3.701x`。Top-1 到 Top-32 各档提升 `3.63x–3.71x`。

轻量结果：

- [两轮合并主结果](results/bandwidth_combined_summary.json)
- [三条路径汇总](results/path_summary.csv)
- [第二轮完整 summary](results/bandwidth_summary_rerun.json)
- [真实 ExpertPack H2D summary](results/expertpack_h2d_summary.json)
- [换槽前后 ExpertPack 对比](results/expertpack_h2d_comparison.csv)

原始数据位于 `artifacts/bandwidth_samples.npz`、`artifacts/bandwidth_samples_rerun.npz` 和 `artifacts/expertpack_h2d_samples.npz`。设备、存储、文件系统和实验前后进程快照位于 `logs/`。

## 结论与后续

接受假设。V100 安装在新插槽后以其设备上限 PCIe 3.0 x16 工作，连续 pinned H2D 达 `12.257 GiB/s`，真实碎片化 ExpertPack H2D 达约 `11.1 GiB/s`；对比旧 x4 安装，同口径提升约 `3.70x`。

三缓冲把端到端带宽保持在 SSD 单独直读的 `99.14%`，因此当前 `SSD→CPU→V100` 顺序大块路径的瓶颈已从 H2D 转移到 NVMe。这个结论是合成最大带宽，不等同于随机小对象、checksum、路由、cache miss 和计算争用下的服务吞吐；真实专家预取实验仍应使用 demand-loading control 测等待时间。

## 资源清理

- [x] 本实验没有启动 API/CLI 服务；所有微基准进程均已正常退出
- [x] V100 无遗留 compute process；结束显存 `1 MiB`
- [x] 8 GiB 临时测试文件已删除；原始样本、summary 和日志位置已记录
