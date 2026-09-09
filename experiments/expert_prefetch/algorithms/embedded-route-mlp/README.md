# EmbeddedRouteMLP

EmbeddedRouteMLP 根据已经完成的路由选择预测当前目标层的 Top-8 专家。它将
每个 `(layer_id, expert_id)` 映射到低维向量，避免为每个路由槽构造稠密的
256维 multi-hot 输入。

## 版本

| 版本 | 状态 | 说明 |
|---|---|---|
| [v0001](versions/v0001/DESIGN.md) | experimenting | layer-specific route embedding、因果上下文、两层MLP和集合交叉熵；EXP-0002通过 |

已被实验引用的版本保持不可变。改变输入或目标语义、embedding身份、网络结构、
损失函数或在线推理契约时，复制当前目录并分配下一个 `vNNNN`。预先定义的
`t`、`k`、`lead_layers`、embedding/hidden size和优化器超参扫描不产生新版本。
