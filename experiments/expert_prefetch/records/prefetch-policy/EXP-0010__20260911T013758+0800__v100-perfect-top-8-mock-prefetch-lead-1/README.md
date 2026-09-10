# EXP-0010: V100 perfect Top-8 mock prefetch lead 1

## 实验身份

- 类型：`prefetch-policy`
- 开始时间：`2026-09-11T01:37:58+08:00`
- 结束时间：`2026-09-11T01:47:01+08:00`
- 状态：`completed`
- 设备：Tesla V100-SXM2-16GB GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96
- Git：`lab/expert-prefetch@a5f26cd5ec8d91f3add35c2130e3d1248861de7a`，dirty=true

## 问题与假设

在相同的分层 LRU cache 和请求上，验证完美 Top-8 oracle 只提前1层时，是否
仍能在目标专家使用前完成 pinned CPU DRAM 到 V100 VRAM 的 H2D，并将结果与
同配置的 demand-only control 以及 EXP-0008 的 lead=8 结果比较。Record/Replay
token 必须完全一致。

## 对照与变量

- 对照组：recall=0、top-k=0、lead=1 的 paired demand-only Replay。
- 实验组：recall=1.0、top-k=8、lead=1、seed=0。
- 独立变量：是否按完美 Top-8 oracle 发出提前1层的 prefetch。
- 固定变量：同一 V100、同一模型和 ExpertPack、严格1024 token输入、16 token
  输出、greedy、ignore EOS、concurrency=1、cache ratio=0.40、dataset seed=0。
- 干扰因素：mock 不包含真实预测器计算；服务启动和bench探测会预热kernel；
  Record不计入性能；代码目录的dirty状态来自独立且已完成的EXP-0009台账和
  带宽脚本，tracked模型推理代码仍对应`a5f26cd5ec`。

## 指标与通过条件

- 两组Record/Replay completion token IDs完全一致。
- Treatment实现recall=1、wasted=0，记录on-time/late，runtime SSD reads=0。
- 对照不发出prefetch，记录两组Replay-only TTFT、ITL、E2E、吞吐和峰值显存。
- Treatment保存570个eligible `(decode row,target layer)` 的CUDA timing，所有时间
  有限且非负，并报告H2D、可用窗口、暴露等待和overlap ratio。

## 操作步骤

1. 记录V100与4070初始状态，仅向服务暴露目标V100。
2. 启动ratio=0.40、capacity=2048的single-GPU API并等待READY。
3. 运行1-request 1024×16 demand-only control。
4. 运行相同输入的recall=1.0、top-k=8、lead=1 treatment。
5. 保存health、bench JSONL、日志和逐command CUDA timing。
6. 关闭服务并确认两张GPU均无遗留计算进程。

## 结果

- 机器可读汇总：[results/experiment_summary.json](results/experiment_summary.json)
- Control/Treatment：[results/control_bench.jsonl](results/control_bench.jsonl)、
  [results/treatment_bench.jsonl](results/treatment_bench.jsonl)
- READY/运行后快照：[results/health_ready.json](results/health_ready.json)、
  [results/health_after.json](results/health_after.json)
- 原始CUDA timing：`artifacts/prefetch_timing.jsonl`（本机保留、Git忽略），
  treatment包含570个完整且无重复的decode row/target layer command。

两组使用相同prompt SHA-256
`4e7bf8d35d419417f9274499ef041fe6cc5de9210c9e1ffc6f05bc8f9af823e8`，
各完成一个严格1024×16 paired请求，Record/Replay token均完全一致。

| 指标 | Demand-only control | Recall 1.0 / Top-k 8 / Lead 1 |
|---|---:|---:|
| Replay TTFT | 1992.93 ms | 1993.88 ms |
| Replay median ITL | 64.88 ms | 66.99 ms |
| Replay E2E | 2984.47 ms | 3097.89 ms |
| 输出吞吐 | 5.361 tok/s | 5.165 tok/s |
| demand hot hit | 4481 | 0 |
| demand prefetch hit | 0 | 4560 |
| demand miss | 79 | 0 |
| demand H2D | 139,789,552 B | 0 B |
| prefetch candidates/useful/wasted | 0/0/0 | 4560/4560/0 |
| prefetch cache hit/H2D experts | 0/0 | 4481/79 |
| prefetch H2D bytes | 0 B | 139,789,552 B |
| prefetch on-time/late | 0/0 | 4560/0 |
| runtime ExpertPack reads | 0 | 0 |

Treatment精确实现recall=1.0，570个command覆盖decode row 1..15、target layer
1..38；候选集合逐组等于真实Top-8。全部4560次候选访问均on-time，79个原本
由control demand加载的专家全部转移到prefetch路径，decode demand miss降为0。
两组总H2D仍相同，均为79个专家、139,789,552 B，只改变传输时机。

43个command需要实际H2D，其余527个为全cache-hit command。可用窗口中位数
1.718 ms，copy command的H2D中位数0.156 ms；汇总prefetch H2D为12.276 ms，
其中11.575 ms被隐藏，overlap ratio为94.29%，报告的暴露等待为9.274 ms。
所有570个command的时间有限且非负。

启动读取9728条pack记录，READY后fd关闭，两组runtime SSD read均为0。GPU峰值
reserved为14,417,920,000 B，距显存总量仍有2,510,422,016 B。

## 结论与后续

接受正确性与deadline覆盖假设：lead=1已经足以让完美Top-8把全部79次decode
demand miss提前执行，所有预测均on-time，且94.29%的prefetch H2D被当前指标
判定为隐藏。

本次仍未得到decode性能收益。Treatment相对同服务control的TTFT只增加0.95 ms，
但median ITL增加2.11 ms（3.25%），E2E增加113.42 ms（3.80%），输出吞吐下降
约3.66%。因为TTFT基本一致，差异主要位于decode和Replay尾部。570个command中
只有43个实际复制，527个全命中command仍执行Python scheduler、LRU、CUDA Event
和统计路径；结果进一步支持当前工程调度/测量开销抵消demand等待收益的判断。

本实验只有每组一个16-token请求，只用于功能与趋势验证。主机在EXP-0009前后
经历硬件重装，当前H2D速度与EXP-0008不同，因此lead=1和lead=8的绝对时间不能
直接横向归因；可靠比较需要在当前拓扑上交替重跑多个1024×256请求。

## 资源清理

- [x] 临时API以SIGINT关闭，exit code 0
- [x] 当前系统枚举到的V100无遗留计算进程
- [x] 原始timing与日志保留在本实验的`artifacts/`和`logs/`
