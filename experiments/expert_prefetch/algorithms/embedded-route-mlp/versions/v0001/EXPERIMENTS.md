# EmbeddedRouteMLP v0001 experiments

每次训练或评测前，先通过 `experiments/expert_prefetch/new_experiment.py` 分配
全局 `EXP-NNNN`，再在下表登记。实验记录目录是原始结果和结论的权威来源。

| EXP ID | 状态 | 目的/变量 | 配置 | 关键结果 | 实验记录 |
|---|---|---|---|---|---|
| EXP-0002 | completed | 非学习基线和v0001默认`t=8,k=4,seed=0` | `default_config.yaml` | test Recall@8 34.34%，较最强基线+9.27 pp | [记录](../../../../records/prefetch-predictor/EXP-0002__20260909T103244+0800__v100-embeddedroutemlp-v0001-default-t8-k4-seed0/README.md) |

建议顺序：非学习基线 → `t`扫描 → `k`扫描 → embedding/hidden容量 →
`lead_layers`扫描 → 三seed冻结模型测试 → V100在线预取对照。
