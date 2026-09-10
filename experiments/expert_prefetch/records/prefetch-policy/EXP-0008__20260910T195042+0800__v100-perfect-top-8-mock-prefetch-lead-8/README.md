# EXP-0008: V100 perfect Top-8 mock prefetch lead 8

## 实验身份

- 类型：`prefetch-policy`
- 开始时间：`2026-09-10T19:50:42+08:00`
- 结束时间：`2026-09-10T19:59:00+08:00`
- 状态：`completed`
- 设备：Tesla V100-SXM2-16GB GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96
- Git：`lab/expert-prefetch@eb965adef81b3655f51d29155a5f47e82f566f9e`，dirty=false

## 问题与假设

在相同的分层 LRU cache 和请求上，验证完美 Top-8 oracle、提前8层的 mock
prefetch 能否在不产生错误候选的情况下减少 demand miss，并利用更长窗口隐藏
pinned CPU DRAM 到 V100 VRAM 的 H2D。Record/Replay token 必须完全一致。

## 对照与变量

- 对照组：recall=0、top-k=0、lead=8 的 paired demand-only Replay。
- 实验组：recall=1.0、top-k=8、lead=8、seed=0。
- 独立变量：是否按完美 Top-8 oracle 发出提前8层的 prefetch。
- 固定变量：同一 V100、同一模型和 ExpertPack、严格1024 token输入、16 token
  输出、greedy、ignore EOS、concurrency=1、cache ratio=0.40、dataset seed=0。
- 干扰因素：mock 不包含真实预测器计算；服务启动和bench探测会预热kernel；
  Record不计入性能，但会生成本请求的route oracle。

## 指标与通过条件

- 两组Record/Replay completion token IDs完全一致。
- Treatment实现recall=1、wasted=0、late尽量为0，runtime SSD reads=0。
- 对照不发出prefetch，记录两组Replay-only TTFT、ITL、E2E、吞吐和峰值显存。
- Treatment保存465个eligible `(decode row,target layer)` 的CUDA timing，所有时间
  有限且非负，并报告H2D、可用窗口、暴露等待和overlap ratio。

## 操作步骤

1. 记录V100与4070初始状态，仅向服务暴露目标V100。
2. 启动ratio=0.40、capacity=2048的single-GPU API并等待READY。
3. 运行1-request 1024×16 demand-only control。
4. 运行相同输入的recall=1.0、top-k=8、lead=8 treatment。
5. 保存health、bench JSONL、日志和逐command CUDA timing。
6. 关闭服务并确认两张GPU均无遗留计算进程。

## 结果

- 机器可读汇总：[results/experiment_summary.json](results/experiment_summary.json)
- Control/Treatment：[results/control_bench.jsonl](results/control_bench.jsonl)、
  [results/treatment_bench.jsonl](results/treatment_bench.jsonl)
- READY/运行后快照：[results/health_ready.json](results/health_ready.json)、
  [results/health_after.json](results/health_after.json)
- 原始CUDA timing：`artifacts/prefetch_timing.jsonl`（本机保留、Git忽略），
  treatment包含465个完整且无重复的decode row/target layer command。

两组使用相同prompt SHA-256
`4e7bf8d35d419417f9274499ef041fe6cc5de9210c9e1ffc6f05bc8f9af823e8`，
各完成一个严格1024×16 paired请求，Record/Replay token均完全一致。

| 指标 | Demand-only control | Recall 1.0 / Top-k 8 / Lead 8 |
|---|---:|---:|
| Replay TTFT | 4557.70 ms | 4263.26 ms |
| Replay median ITL | 69.09 ms | 69.68 ms |
| Replay E2E | 5635.15 ms | 5332.88 ms |
| 输出吞吐 | 2.839 tok/s | 3.000 tok/s |
| demand hot hit | 4481 | 836 |
| demand prefetch hit | 0 | 3720 |
| demand miss | 79 | 4 |
| demand H2D | 139,789,552 B | 7,077,952 B |
| prefetch candidates/useful/wasted | 0/0/0 | 3720/3720/0 |
| prefetch cache hit/H2D experts | 0/0 | 3645/75 |
| prefetch H2D bytes | 0 B | 132,711,600 B |
| prefetch on-time/late | 0/0 | 3720/0 |
| runtime ExpertPack reads | 0 | 0 |

Treatment精确实现recall=1.0，所有候选集合均与目标层真实Top-8相同。Prefill
已经使3645个候选驻留，剩余75个由prefetch H2D完成，另有4个无法由lead=8
覆盖的demand miss。Treatment的prefetch与demand合计仍传输79个专家、
139,789,552 B，与control总H2D完全相同，只改变了传输时机。

465个command覆盖decode row 1..15、target layer 8..38。仅39个command需要
实际H2D；可用窗口中位数13.374 ms，copy command的H2D中位数0.565 ms。
汇总prefetch H2D为42.066 ms，其中41.381 ms被隐藏，overlap ratio为98.37%，
报告的暴露等待为6.356 ms。所有时间均有限且非负。

启动读取9728条pack记录，READY后fd关闭，两组runtime SSD read均为0。GPU峰值
reserved为14,417,920,000 B，距显存总量仍有2,510,422,016 B。

## 结论与后续

接受正确性和预取覆盖假设：lead=8的完美Top-8预取把79次demand miss中的75次
移到异步prefetch路径，全部on-time且无错误候选，并隐藏98.37%的prefetch H2D。

本次没有观察到decode延迟收益：median ITL反而增加0.58 ms（0.84%），可视为
单请求波动范围。E2E减少302.27 ms主要来自TTFT减少294.44 ms，而prefill阶段不
执行prefetch，因此不能把该差值归因于预取。该实验说明当前102-expert热cache下，
decode本身只有79次、约140 MB总H2D；完美预取主要改变等待位置。后续需要增加
请求数，并与lead=1/2/4及更小cache ratio组合，才能判断延迟收益。

## 资源清理

- [x] 临时API以SIGINT关闭，exit code 0
- [x] V100和RTX 4070 SUPER均无遗留计算进程
- [x] 原始timing与日志保留在本实验的`artifacts/`和`logs/`
