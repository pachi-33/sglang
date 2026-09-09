# Expert-prefetch algorithm registry

本目录管理专家预测和预取算法的版本身份。实验编号回答“哪一次运行”，算法
版本回答“运行的具体算法语义是什么”。

```text
algorithms/<algorithm>/
├── README.md
└── versions/
    └── vNNNN/
        ├── DESIGN.md
        ├── default_config.yaml
        ├── EXPERIMENTS.md
        └── results/
            └── README.md
```

具体训练、评测、日志和 checkpoint 必须保存在全局实验账本的 `records/`
目录。版本目录只保存稳定设计、默认配置、实验链接以及适合 Git 管理的汇总
结果。

当前算法族：

| 算法 | 当前版本 | 状态 |
|---|---|---|
| [embedded-route-mlp](embedded-route-mlp/README.md) | `v0001` | EXP-0002初始实验通过 |
