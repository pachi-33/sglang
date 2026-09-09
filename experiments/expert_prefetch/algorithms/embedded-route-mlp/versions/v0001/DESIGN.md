# EmbeddedRouteMLP v0001

## 身份和状态

- 算法：`embedded-route-mlp`
- 版本：`v0001`
- 状态：`implemented`
- 任务：根据因果可见的历史路由预测当前 token、当前目标层的 Top-8全局专家ID
- 数据：EXP-0001 request-level seed-0 split，96/16/16 requests

## 数据契约

输入数组 `expert_ids` 的轴为：

```text
[request, output_attribution_position, layer_id, router_rank]
```

每条请求有256行。row 0是prompt最后位置产生首个输出token时的路由；row
1..255是decode步骤。v0001使用row 0作为row 1的历史，不把row 0作为训练目标。
初始训练和评测目标为row 1..255、layer 0..39；预取收益单独统计offload的
layer 1..38。

数据严格按请求切分，不能把同一请求的不同token分到不同集合。训练代码不得
使用当前行的 `sampled_token_ids`、目标层路由、未来层路由或未来token路由。

## 因果特征

对请求 `r`、位置 `s`、目标层 `l`，目标为：

```text
Y(r,s,l) = expert_ids[r,s,l,:]
```

可见历史由三个超参控制：

- `t`：前 `t` 个位置的全部40层路由。
- `k`：本位置最多使用的前序层数量。
- `lead_layers`：预测相对目标route提前的层数。记为 `delta`。

目标层 `l` 的当前token特征只能来自：

```text
[max(0, l-delta-k), l-delta)
```

`delta=0` 表示紧邻目标route前预测，此时最多看到 `l-k .. l-1`。缺失的历史
token或层槽使用零向量，并附加validity mask。历史token槽按“最近到最远”、
每个token内按layer 0..39的固定顺序展开。

## 路由编码

物理专家身份为 `(layer_id, global_expert_id)`。模型维护：

```text
route_embedding[40, 256, d]
```

一层的Top-8路由编码为8个embedding的均值：

```text
z(route) = mean(route_embedding[layer_id, top8_expert_ids], axis=rank)
```

这等价于对该层256维multi-hot做共享线性投影后除以8，但不等价于完全无约束的
稠密FlatMLP。v0001将Top-8作为无序集合；router rank仍保留在原数据中，可在
后续版本引入rank-specific embedding。

目标层使用单独的 `target_layer_embedding[40,d]`。最终输入为所有路由编码、
目标层embedding和validity mask的拼接。

## 网络和损失

默认 `t=8`、`k=4`、`d=32` 时，路由槽数量为 `8*40+4=324`，MLP输入维度为：

```text
d * (40*t + k + 1) + t + k = 10,412
```

网络：

```text
Linear(input_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 256)
Softmax（仅推理/指标计算；训练向loss传未归一化logits）
```

目标集合包含8个专家。默认损失为均匀集合交叉熵：

```text
loss = -mean(log_softmax(logits)[target_top8])
```

预测结果为logits最大的 `candidate_count` 个专家；路由准确率默认使用前8个。

## 推理接口

模型前向接口必须显式接收已经裁剪好的因果特征和mask，不允许在模型内部访问
完整trace：

```python
logits = model(
    history_expert_ids,
    history_valid_mask,
    current_layer_expert_ids,
    current_layer_valid_mask,
    target_layer_ids,
)
```

输出shape为 `[batch,256]`。离线实现先验证准确率和延迟；在线接入时只对layer
1..38调用ExpertStore预取，并保留demand-loading作为正确性回退。

## 固定评测

主要指标：`Recall@8`。同时记录Recall@16/32、exact-set accuracy、Jaccard、
set cross-entropy、逐层结果、cold-start (`s<t`) 和steady-state (`s>=t`)。
预取实验还需记录issued/useful/late/wasted、cache hit/miss、SSD和H2D数据及等待
时间、预测器延迟、TTFT、ITL和峰值显存。

所有超参数只能根据validation选择。配置冻结后test只运行一次；正式结论至少
报告三个训练seed的均值和标准差。在线V100结果必须与相同输入的demand-loading
控制进行逐token正确性比较。

## 新版本边界

以下改变必须创建 `v0002` 或更高版本：

- 使用token ID、hidden state或router weight作为输入；
- 将Top-8 rank编码进embedding；
- 改成per-layer独立模型、Transformer、RNN或其他网络；
- 改变目标、损失或在线预测时允许看到的信息；
- 修复会改变既有checkpoint输出的实现错误。
