# Expert Prefetch 实验账本

本目录记录 `lab/expert-prefetch` 分支上的专家预取实验。每次实验必须先创建记录，再执行命令；失败或没有得到预期结果的实验也保留，以免重复踩坑。

## 标识与目录

实验使用全局递增编号，目录结构为：

```text
records/<experiment_type>/EXP-<NNNN>__<YYYYMMDDTHHMMSS+0800>__<slug>/
├── metadata.json       # 可机器读取的身份、时间、环境与状态
├── README.md           # 假设、变量、步骤、结果和结论
├── commands.sh         # 实际执行命令
├── results/            # 小体积汇总数据，可提交 Git
├── logs/               # 原始日志，默认不提交 Git
└── artifacts/          # trace、profile 等大文件，默认不提交 Git
```

三个主要管理维度如下：

- **实验编号**：`EXP-0001` 起全局递增，永不复用，即使实验失败或取消。
- **实验类型**：使用小写 kebab-case；同类实验进入同一个类型目录。
- **时间**：目录名使用 Asia/Singapore 时区的开始时间，元数据使用带时区的 RFC 3339 时间。

推荐的首批实验类型：

| 类型 | 用途 |
|---|---|
| `baseline` | 固定当前 demand-loading 的正确性和性能基线 |
| `trace-analysis` | 分析 token/layer 的专家激活和重用规律 |
| `prefetch-policy` | 比较预取候选、窗口、提前量和回退策略 |
| `prefetch-predictor` | 评估预测器准确率、召回率和覆盖率 |
| `cache-policy` | 研究预取与 GPU expert cache 淘汰的相互作用 |
| `io-pipeline` | 研究 SSD 读取、pinned staging 和 H2D 流水线 |
| `end-to-end` | 记录完整 CLI/API 请求的 TTFT、ITL 和吞吐 |
| `correctness` | 路由一致性、错误注入和回归验证 |

新类型可以直接创建，但必须在首个该类型实验的 README 中解释边界，之后保持同一语义。

## 创建实验

从仓库根目录执行：

```bash
python experiments/expert_prefetch/new_experiment.py \
  --type baseline \
  --title "V100 demand-loading baseline" \
  --device "V100 GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"
```

脚本会在持有文件锁时分配下一个全局编号、创建记录目录，并向 [INDEX.csv](INDEX.csv) 追加一行。`metadata.json` 自动记录分支、commit、工作树是否有未提交改动、主机和开始时间。

## 执行规则

每次实验至少记录以下内容：

1. 先在 `README.md` 写清假设、对照组、独立变量、固定变量和验收指标。
2. 将可复现命令写入 `commands.sh`；涉及模型、ExpertPack、cache 大小、随机种子和请求形状时必须给出完整参数。
3. 原始输出进入 `logs/` 或 `artifacts/`；提炼后的 CSV、JSON 和图表进入 `results/`。
4. 结束后填写结束时间、状态、关键结果和结论，并同步更新 `INDEX.csv`。
5. GPU/API 测试结束后关闭临时服务，记录服务退出方式，并确认测试 GPU 没有遗留计算进程。

状态统一使用：`planned`、`running`、`completed`、`failed`、`inconclusive` 或 `aborted`。实验失败描述实验结果；基础设施故障导致没有形成有效观测时使用 `inconclusive`。

## 比较实验的最低要求

预取实验需要同时保留 demand-loading 对照，并固定模型、prompt/token 序列、GPU、cache 容量和生成参数。至少报告：

- 生成 token 是否与对照一致；
- expert request、hit、miss、prefetch issued/useful/late/wasted 数量；
- SSD read 和 H2D 字节与等待时间；
- cold/warm TTFT、ITL 及峰值显存；
- 被改变的单一变量和无法控制的干扰因素。

跨 V100 与 RTX 4070 SUPER 的结果分别归档，不能直接合并为同一组样本。

## 算法版本

会迭代的预测与预取算法统一放在 `algorithms/<algorithm>/versions/vNNNN/`。
每个版本目录至少包含设计、默认配置、实验索引和小体积汇总结果。版本目录
用于说明“运行了什么算法”，`records/` 下的全局 `EXP-NNNN` 记录仍是具体实验
数据和结论的权威来源。

- `vNNNN` 从 `v0001` 开始递增；已经被实验引用的版本不得原地改变行为。
- 输入/目标语义、特征编码、网络结构、损失或推理契约改变时创建新版本。
- `t`、`k`、hidden size、学习率等预先声明的超参扫描保留在同一算法版本内。
- 每次训练或评测仍须先用 `new_experiment.py` 分配全局实验编号，再将该
  `EXP-NNNN` 和结果链接登记到版本目录的 `EXPERIMENTS.md`。
- checkpoint、逐步预测和原始日志保存在对应实验的 `artifacts/`、`logs/`；
  版本目录的 `results/` 只保存跨实验对比表、图和最终小结。
