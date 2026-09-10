# EXP-0007: V100 layered LRU mock prefetch overlap validation

## 实验身份

- 类型：`prefetch-policy`
- 开始时间：`2026-09-10T19:17:07+08:00`
- 结束时间：`2026-09-10T19:35:58+08:00`
- 状态：`completed`
- 设备：V100 GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96
- Git：`lab/expert-prefetch@9856014ec48a03904fd62523c62807f07189bf2d`，dirty=false

## 问题与假设

验证单卡 V100 能否同时容纳完整 pinned host expert repository、ratio=0.40 的
分层 GPU LRU cache 和完整模型，并验证 router-ready 后提交的 mock prefetch H2D
能够与 compute stream 的后续层计算重叠。若实现正确，Record/Replay token 必须
完全相同、runtime SSD read 必须为零，且代表性配置至少有一个 CUDA H2D 区间被
部分或全部掩盖。

## 对照与变量

- 对照组：paired demand-only Replay，recall=0、top-k=0、lead=2。
- 实验组：paired mock prefetch，recall=0.5、top-k=8、lead=2、seed=0。
- 独立变量：是否发出 mock prefetch 以及对应候选召回率。
- 固定变量：同一 V100、同一模型/ExpertPack、严格 1024 token 输入、16 token
  输出、greedy、ignore EOS、concurrency=1、cache ratio=0.40、seed=0。
- 干扰因素：mock 不包含真实预测器计算；startup 会读取 SSD，但运行时只从
  pinned host DRAM 传输；V100 copy engine 的 stream priority 不保证抢占当前 copy。

## 指标与通过条件

- 服务成功 READY，startup_pack_reads=9728，runtime_pack_reads=0。
- Record/Replay completion token IDs 完全一致。
- 每层 cache capacity=102，scratch=154，总 cache=4030 experts。
- 对照组不发出 prefetch H2D；实验组报告 issued/useful/on-time/late/wasted。
- 原始 command timing 中至少存在有效、有限且非负的 H2D/window/stall 数据。
- 记录 Replay-only TTFT、ITL、吞吐、H2D bytes、LRU eviction 和峰值显存。

## 操作步骤

1. 记录两张 GPU 的初始状态，只向进程暴露目标 V100。
2. 启动 single-GPU API，等待 pinned repository 和模型 READY。
3. 运行 1-request 1024x16 demand-only paired control。
4. 运行相同输入的 recall=0.5/top-k=8/lead=2 paired treatment。
5. 保存 health、server/bench 日志、bench JSONL 和逐 command timing。
6. 关闭服务并确认 V100 与 4070 均无遗留计算进程。

Preflight 首次使用默认 served model basename，bench 按绝对模型路径请求时由模型
身份校验返回 404；该请求未进入 Record、未改变 cache。正式运行在服务命令中固定
`--served-model-name` 为同一绝对模型路径后开始。

## 结果

- 完整机器可读汇总：[results/experiment_summary.json](results/experiment_summary.json)
- Control bench：[results/control_bench.jsonl](results/control_bench.jsonl)
- Treatment bench：[results/treatment_bench.jsonl](results/treatment_bench.jsonl)
- READY/运行后快照：[results/health_ready.json](results/health_ready.json)、
  [results/health_after.json](results/health_after.json)
- 原始 CUDA command timing：`artifacts/prefetch_timing.jsonl`（本机保留、Git忽略），
  共2行，control无command，treatment含555个decode `(row,target layer)` command。

两组各完成一个严格1024×16 paired请求，prompt SHA-256均为
`4e7bf8d35d419417f9274499ef041fe6cc5de9210c9e1ffc6f05bc8f9af823e8`；
control和treatment内部的Record/Replay token均完全一致。

| 指标 | Demand-only control | Recall 0.5 / Top-k 8 / Lead 2 |
|---|---:|---:|
| Replay TTFT | 4257.41 ms | 4261.77 ms |
| Replay median ITL | 68.37 ms | 92.29 ms |
| Replay E2E | 5327.67 ms | 5815.45 ms |
| 输出吞吐 | 3.003 tok/s | 2.751 tok/s |
| demand hot hit | 4481 | 2271 |
| demand prefetch hit | 0 | 2220 |
| demand miss | 79 | 69 |
| demand H2D | 139,789,552 B | 122,094,672 B |
| prefetch candidates/useful/wasted | 0/0/0 | 4440/2220/2220 |
| prefetch H2D experts/bytes | 0/0 B | 1452/2,569,296,576 B |
| prefetch on-time/late | 0/0 | 2220/0 |
| runtime ExpertPack reads | 0 | 0 |

Mock候选精确实现0.5 recall。548/555个command实际发出H2D；其中523个
command记录到正的隐藏H2D。可用窗口中位数为4.999 ms，带copy command的H2D
中位数为1.670 ms；汇总H2D 814.622 ms，隐藏723.372 ms，暴露113.067 ms，
overlap ratio为88.80%。所有555个command的时间有限且非负，row 1..15、
target layer 2..38完整且无重复。

启动阶段读取并校验9728条pack记录；READY后pack fd关闭，runtime读取严格为0。
cache总容量4030专家、每层102、scratch 154，占7,131,036,640 B；pinned host
repository为17,213,579,264 B。峰值reserved为14,417,920,000 B，距显存总量
仍有2,510,422,016 B。

## 结论与后续

接受功能与并行性假设：实现能在单张V100上完成pinned repository、分层LRU、
Record/Replay、demand fallback和异步prefetch闭环，并以CUDA Event观测到真实的
H2D隐藏；4070未参与。

本次代表配置没有带来性能收益：相对control，E2E增加487.78 ms（9.16%），
median ITL增加23.92 ms（34.99%），尽管demand miss减少10次。直接原因是为了
获得50% recall同时注入了2220个错误候选，产生约2.57 GB prefetch流量和1449次
prefetch eviction；当前短请求中cache pollution和传输/调度代价超过了减少10次
demand miss的收益。该结果只有每组1个短请求，用于功能验收，不作为稳定性能
估计；后续应单独扫描top-k、recall、lead和ratio，并增加样本量。

## 资源清理

- [x] 临时 API/CLI 进程以SIGINT关闭，exit code 0
- [x] V100和RTX 4070 SUPER均无遗留计算进程
- [x] 大体积日志和产物位置已记录；raw timing与日志保留在本实验目录
