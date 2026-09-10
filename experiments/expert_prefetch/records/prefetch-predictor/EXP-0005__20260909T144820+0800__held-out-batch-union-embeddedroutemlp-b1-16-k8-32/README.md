# EXP-0005: Held-out batch-union EmbeddedRouteMLP B1-16 K8-32

## 实验身份

- 类型：`prefetch-predictor`
- 开始时间：`2026-09-09T14:48:20+08:00`
- 结束时间：`2026-09-09T14:59:16+08:00`
- 状态：`completed`
- 设备：Offline analysis of V100 GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 artifacts
- Git：`lab/expert-prefetch@1c6f6f75ffe674c7b62f870d574e1ce4a7ea7b84`，dirty=true（创建时另一个独立实验 EXP-0004 尚未归档）

## 问题与假设

冻结 EXP-0003 的 Top-32 排名，将 B 个 held-out test 请求在相同输出位置、
相同层的真实 Top-8 和预测 Top-K 分别跨请求去重。固定定义：

- `M = |真实union ∩ 预测union|`，即预测正确的去重专家数；
- `N = |真实union|`，即batch真实去重专家数；
- `M/N` 是batch union recall，主指标为逐batch、token、layer的宏平均。

假设：K增大时M和`M/N`单调不减；跨请求候选共享会使batch union recall
高于单请求召回，但预测union规模和未命中候选也会增加。

## 对照与变量

- 对照组：B=1，必须逐值复现 EXP-0003 test Recall@K。
- 独立变量：请求 batch 大小 `B∈{1,2,4,8,16}`；单请求候选预算
  `K∈{8,16,24,32}`。
- 固定变量：EXP-0002 epoch 6 checkpoint、EXP-0003 Top-32 排名、EXP-0001
  seed-0 test split 的16个请求、目标 row 1..255、40层、history tokens=8。
- 抽样：对16个 test 请求的每个 B 精确枚举全部无放回组合，共14,827组；
  K档位复用相同请求组合。
- 干扰因素：test split 仅16个请求；结果不代表 B>16，不测在线 predictor
  latency、cache 状态或 SSD/H2D 行为。

## 指标与通过条件

- 主指标：offload layers 1..38、row 1..255 上逐 cell 宏平均 `M/N`，其中
  M为预测正确专家数，N为真实专家数。
- 辅助指标：预测union规模、`N-M` missed、预测union中未命中的wasted专家、
  完整覆盖率，以及`sum(M)/sum(N)`。
- 同时报告 all/cold-start/steady-state 和逐层结果。
- 验收：请求身份严格对齐；B=1复现 EXP-0003；N与K无关；随K增加，M和
  M/N单调不减；`0≤M≤N`；集合基数满足理论边界；输出原子发布。

## 操作步骤

1. 校验 EXP-0003 prediction artifact 与 test split 的SHA、shape和请求身份。
2. 将真实Top-8和预测Top-K编码为4个uint64的256-bit集合。
3. 对每个B精确枚举全部请求组合，按同一位置和层执行跨请求OR。
4. 对K=8/16/24/32计算预测正确数M、真实数N、M/N和辅助指标。
5. 保存权威`batch_recall_summary.json`、CSV、详细结果和实际评估的请求子集。
6. 复核无partial文件；本实验不启动API、不使用GPU。

## 结果

精确枚举了 B=1/2/4/8/16 的全部 14,827 个 test 请求组合。主结果为
row 1..255、offload layers 1..38 上的逐 cell 宏平均 `M/N`：

| B | K=8 M/N | K=16 M/N | K=24 M/N | K=32 M/N |
|---:|---:|---:|---:|---:|
| 1 | 34.5882% | 50.5061% | 60.2586% | 67.1727% |
| 2 | 37.0136% | 53.9324% | 64.2315% | 71.4317% |
| 4 | 40.8099% | 58.9791% | 69.7817% | 77.0838% |
| 8 | 46.3331% | 65.6193% | 76.5040% | 83.4085% |
| 16 | 53.2803% | 73.0096% | 83.1992% | 89.0690% |

B=16时，真实专家数N平均为90.6702：

| K | 预测正确M | 真实N | M/N | 预测union规模 | missed | wasted |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 48.2543 | 90.6702 | 53.2803% | 76.7029 | 42.4159 | 28.4486 |
| 16 | 66.1710 | 90.6702 | 73.0096% | 122.7877 | 24.4992 | 56.6167 |
| 24 | 75.4315 | 90.6702 | 83.1992% | 154.8755 | 15.2387 | 79.4441 |
| 32 | 80.7680 | 90.6702 | 89.0690% | 177.9791 | 9.9022 | 97.2110 |

- B=1全层Recall逐值复现EXP-0003，所有shape、ID、集合基数、K前缀单调性
  和请求身份检查通过。
- 另用Python set独立抽查200个随机 `(B,K,position,layer)` cell，M/N/H与
  bitset实现完全一致；0个partial文件。
- EXP-0003 Top-32 artifact SHA-256：
  `ba344540414bf89a9a4b7515055b0657e2eaaadadd4d527881eaa9a865178718`。
- Test split SHA-256：
  `e9fe59e7401b36548623ab820cc9aa671d3f1b50473a4d9eb08581e2a46abe10`。
- 权威口径结果见`results/batch_recall_summary.json`和
  `results/batch_recall.csv`。轻量原始汇总见`results/batch_union_summary.json`，完整scope结果见
  `results/batch_union_metrics.{json,csv}`和`results/batch_union_by_layer.csv`；
  实际枚举的请求组合保存在`artifacts/evaluated_batch_subsets.npz`。

## 结论与后续

假设成立。随着B增长，其他请求的候选会覆盖一部分当前请求的真实专家，所有K
档位的batch recall均提升；随着K增长，覆盖继续上升，但预测union和wasted
候选也快速增加。当前最高档B=16/K=32仍有约10.92%的真实union未覆盖，因此
该预测器不能单独替代demand-loading fallback。

后续做cache replay时应继续以`M/N`约束覆盖，并将`N-M`映射为仍需同步换入
的专家；预测union中未命中的候选是潜在无用预取。是否产生真实SSD/H2D流量
还取决于cache命中。

## 资源清理

- [x] 本实验未启动API，8818端口关闭，无评估进程残留
- [x] 本实验只离线读取NPZ且未调用CUDA；结束复核环境无法访问NVIDIA驱动，
  因此未把实时GPU进程查询作为实验结论
- [x] 日志位于`logs/`，枚举子集位于`artifacts/`，轻量结果位于`results/`
