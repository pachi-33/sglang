# EXP-0002: V100 EmbeddedRouteMLP v0001 default t8 k4 seed0

## 实验身份

- 类型：`prefetch-predictor`
- 开始时间：`2026-09-09T10:32:44+08:00`
- 结束时间：`2026-09-09T10:43:14+08:00`
- 状态：`completed`
- 设备：Tesla V100-SXM2-16GB `GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96`
- Git：`lab/expert-prefetch@a639034c4aa276bba9858bab2effdbdb825c9c7d`，dirty=false
- 算法：`embedded-route-mlp/v0001`

## 问题与假设

问题：只使用前8个路由位置的全部40层结果，以及本位置目标层之前最多4层的
路由结果，能否预测当前层Top-8专家集合？

可证伪假设：默认EmbeddedRouteMLP在冻结test上的Recall@8比最强非学习基线
至少高5个百分点；对实际offload的layer 1..38单独计算时也满足这一条件。

## 对照与变量

- 对照组：训练集逐层频率Top-8、同层上一token Top-8、本token上一层Top-8。
- 独立变量：使用`embedded-route-mlp/v0001`学习预测路由集合。
- 固定变量：EXP-0001 request-level seed-0 96/16/16 split；目标row 1..255、
  layer 0..39；`t=8`、`k=4`、`lead_layers=0`、embedding dim 32、hidden
  dim 512、dropout 0.1、seed 0、FP16 AMP、AdamW、batch 512。
- 模型选择：最多20 epoch，以validation Recall@8选择checkpoint，patience 3；
  checkpoint冻结后只执行一次test评测。
- 干扰因素：这是单seed初始实验；训练GPU吞吐不代表在线逐层推理延迟。

## 指标与通过条件

- 数据加载、因果边界、训练、checkpoint重载和test评测全部成功，无NaN/Inf。
- 主要指标：整体及layer 1..38的Recall@8。
- 辅助指标：Recall@16/32、exact-set accuracy、Jaccard、set cross-entropy、
  逐层Recall、训练/评测吞吐、epoch和峰值GPU显存。
- test Recall@8和offload Recall@8均比最强对照至少高0.05则接受假设。
- 本实验不接入ExpertStore，不产生issued/useful/late/wasted或端到端TTFT/ITL；
  这些指标在后续online prefetch实验与demand-loading控制共同测量。

## 操作步骤

1. 记录GPU初始状态并确认只暴露目标V100。
2. 计算validation/test非学习基线。
3. 在train上训练，以validation Recall@8保存最佳checkpoint并early stop。
4. 重载冻结checkpoint，最后一次评测validation和test。
5. 归档指标、epoch日志、checkpoint和GPU前后状态。
6. 确认无遗留GPU进程，更新实验结论和算法版本实验索引。

## 结果

训练共9个epoch，在epoch 6达到最佳validation Recall@8，之后连续3个epoch
没有改善并early stop。训练与冻结checkpoint评测总计84.44秒；模型含
5,791,744个参数。

| split | predictor Recall@8 | 最强基线 | 提升 | predictor offload Recall@8 | 最强offload基线 | 提升 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 31.3662% | 23.5377% | +7.8285 pp | 31.5662% | 24.0902% | +7.4760 pp |
| test | 34.3417% | 25.0748% | +9.2669 pp | 34.5882% | 25.7750% | +8.8132 pp |

最强基线均为“同层上一token”。冻结test的Recall@16/32分别为50.2083%和
66.8540%，exact-set accuracy为0.2917%，mean Jaccard为22.9728%。逐层结果
见`results/predictor_metrics.json`。

- 训练吞吐约111k samples/s；冻结评测约370k samples/s。
- 峰值GPU allocated/reserved：639,026,176 / 855,638,016 bytes。
- checkpoint：`artifacts/best_model.pt`，23,172,936 bytes，SHA-256
  `d84c4f7c0aceb368d2254dd1aaee1a713f1bd76974cd151b40af83ddaac8c524`。
- 训练、checksum、checkpoint重载和唯一一次test评测均成功，无NaN/Inf。

## 结论与后续

假设成立。整体和offload层的test提升分别为9.27和8.81个百分点，均超过预设
5个百分点门槛。纯路由历史包含可学习信号，EmbeddedRouteMLP值得继续实验。

这是单seed、`lead_layers=0`的离线准确率结果，尚不能说明SSD预取可以及时完成。
下一步依次扫描`t`、`k`和`lead_layers`，随后以三seed复验冻结配置，再进入
ExpertStore replay和V100在线demand-loading对照。

## 资源清理

- [x] 本实验未启动API服务
- [x] V100和RTX 4070 SUPER无遗留计算进程
- [x] 结束后显存为V100 1 MiB、RTX 4070 SUPER 2 MiB
- [x] checkpoint位于`artifacts/`，轻量指标位于`results/`
