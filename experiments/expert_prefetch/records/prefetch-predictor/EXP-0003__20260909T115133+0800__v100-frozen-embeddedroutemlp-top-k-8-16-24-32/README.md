# EXP-0003: V100 frozen EmbeddedRouteMLP top-k 8 16 24 32

## 实验身份

- 类型：`prefetch-predictor`
- 开始时间：`2026-09-09T11:51:33+08:00`
- 结束时间：`2026-09-09T11:57:15+08:00`
- 状态：`completed`
- 设备：Tesla V100-SXM2-16GB `GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96`
- Git：`lab/expert-prefetch@4059b1c33db499e7c9f349b0b3a9a1ea27c40cb0`，dirty=false
- 算法：`embedded-route-mlp/v0001`
- 冻结模型：EXP-0002 epoch 6，SHA-256 `d84c4f7c0aceb368d2254dd1aaee1a713f1bd76974cd151b40af83ddaac8c524`

## 问题与假设

固定同一个模型和同一组logits，预测候选数从8扩大到16、24、32时，实际Top-8
专家的召回率如何变化？

召回率固定定义为：

```text
Recall@K = |真实Top-8 ∩ 预测Top-K| / 8
```

假设：Recall随K单调增加，但每增加8个候选带来的边际召回下降，同时错误候选数
上升。实验不根据结果重新训练、选择模型或改变排序。

## 对照与变量

- 对照：同一冻结模型的Top-8；另记录逐层频率和“上一token Top-8加频率补齐”基线。
- 独立变量：`candidate_count ∈ {8,16,24,32}`。
- 固定变量：EXP-0001 request-level seed-0 split、validation/test请求、row 1..255、
  40层、`t=8`、`k=4`、`lead_layers=0`、同一checkpoint和一次Top-32排序。
- 模型状态：只读eval、FP16 AMP；不训练、不更新参数。

## 指标与通过条件

- 对全部40层和offload layers 1..38分别报告Recall@8/16/24/32。
- 报告每个token-layer平均命中、错误候选、漏掉的真实专家、precision和完整覆盖率。
- 分开报告cold-start和steady-state，并保存逐层结果和Top-32原始预测。
- Recall必须随K单调不降，候选必须唯一且位于0..255。
- K=8/16/32的整体结果必须与EXP-0002冻结checkpoint结果一致。

## 操作步骤

1. 校验checkpoint SHA和GPU状态。
2. 对validation和test各执行一次Top-32推理。
3. 对同一排序截取前8/16/24/32并计算指标。
4. 原子保存指标JSON和候选NPZ，校验shape、唯一性及SHA。
5. 确认两张GPU无遗留计算进程并完成账本。

## 结果

同一个Top-32排序的前缀结果如下。平均命中数是每个token-layer的8个真实专家
中被覆盖的数量。

| K | Validation Recall | Validation平均命中 | Test Recall | Test平均命中 | Test offload Recall |
|---:|---:|---:|---:|---:|---:|
| 8 | 31.3662% | 2.5093 / 8 | 34.3417% | 2.7473 / 8 | 34.5882% |
| 16 | 46.7639% | 3.7411 / 8 | 50.2083% | 4.0167 / 8 | 50.5061% |
| 24 | 56.4817% | 4.5185 / 8 | 59.9359% | 4.7949 / 8 | 60.2586% |
| 32 | 63.5857% | 5.0869 / 8 | 66.8540% | 5.3483 / 8 | 67.1727% |

Test上的边际Recall提升为：K 8→16 `+15.8667 pp`，16→24
`+9.7276 pp`，24→32 `+6.9181 pp`。扩大候选数的边际收益递减。

| K | Test precision | 平均错误候选 | 平均漏掉真实专家 | 8个真实专家全部覆盖率 |
|---:|---:|---:|---:|---:|
| 8 | 34.3417% | 5.2527 | 5.2527 | 0.2917% |
| 16 | 25.1042% | 11.9833 | 3.9833 | 5.1379% |
| 24 | 19.9786% | 19.2051 | 3.2051 | 9.9522% |
| 32 | 16.7135% | 26.6517 | 2.6517 | 14.9203% |

- K=8/16/32的整体Recall与EXP-0002逐值一致。
- Validation/Test预测shape均为`[16,255,40,32]`，全部候选唯一且位于0..255。
- Top-32预测artifact为10,408,182 bytes，SHA-256
  `ba344540414bf89a9a4b7515055b0657e2eaaadadd4d527881eaa9a865178718`。
- 轻量结果见`results/topk_summary.json`；逐层、cold-start、steady-state和基线
  明细见本地`results/topk_metrics.json`。

## 结论与后续

假设成立。扩大候选集合能稳定提高真实Top-8覆盖率，但precision下降且错误候选
快速增加。K=16在test上已经覆盖约4.02/8个专家；继续扩大到32只再覆盖约1.33
个专家，却额外增加16个候选。

这里的“错误候选”只是集合层面的未命中，尚不能直接等同于SSD无用读取：候选
可能已经在GPU cache，也可能在真正发出I/O前被过滤。下一步应把K=8/16/24/32
带入相同cache状态的离线replay，测量增量hit、issued/useful/wasted及读取字节，
再决定在线默认候选预算。

## 资源清理

- [x] 本实验未启动API服务
- [x] V100和RTX 4070 SUPER无遗留计算进程
- [x] 结束后显存为V100 1 MiB、RTX 4070 SUPER 2 MiB
- [x] 原始Top-32预测位于`artifacts/`，轻量指标位于`results/`

归档复核通过：预测和详细指标文件的大小及SHA-256、四档Recall的分母与单调性、
metadata/INDEX/summary的一致性。资源清理结论依据实验结束时保存的
`results/gpu_after.csv`和空的`results/compute_processes_after.csv`；本次归档环境
无法访问NVIDIA驱动，未能再次查询实时GPU状态。归档时尝试重跑pytest，当前
Python环境缺少pytest，因此未完成此次测试复跑；未改动评测代码。
