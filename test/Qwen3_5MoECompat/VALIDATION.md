# Qwen3.5 MoE V100 独立层与四层无状态精度验收（历史）

本页保留 2026-09-07 的 M0–M5 无状态基线证据及其源码 hash，不能当作当前工作树的
重新扫描结果。当前分支已增加 SM70/SM89 算子合同、单请求 cache 和可配置双 worker
流水线；对应实现范围、验收记录与复现命令见
[双卡与 Cache 验收](VALIDATION_SM70_SM89_PIPELINE.md)。下面的 93 项是历史完整 discovery
计数，与后续扩展的 `unit/` 套件计数不同。

最终 v4 扫描完成原始 **0～39 全部 40 层**，每层独立加载真实权重和独立输入。
所有预先冻结的部件预算、形状、有限值和逐 expert 检查通过，覆盖 **130 个 W8A8 投影、
29,184 个 NVFP4 expert 矩阵、1,536 个 FP16 expert 矩阵**。T32 强制路由
`expert_id=8*token+slot` 实际执行每层全部 256 experts，共 10,240 个 expert 实例。

环境：指定 `sglang-v100`、Torch 2.3.1+cu121、Triton 2.3.1、指定 UUID 的 SM70 V100。
合同：`qwen35-m3-fp8-block-semantic-v4`。全部 40 份报告拥有相同的源码 SHA256：

```text
ff462157f05883b765b4f5110b64fa8f43fb798a153c2703eed3bb3d026694cb
```

hash 在 CLI 启动时冻结，覆盖 `python/sglang/srt/layers/qwen3_5/**/*.py`、reference 和扫描器。
原始 JSON 包含 shape、NRMSE、max、P99、有限值、预算及逐 expert 误差；汇总见
[layer_scan_summary.json](reports/layer_scan_summary.json)，扫描日志见
[layer_scan_validation.txt](reports/layer_scan_validation.txt)。

## 冻结预算与最差结果

NRMSE 为 `||actual-reference||₂/||reference||₂`。局部算子使用相同输入，由独立 reference
计算预期；上下游独立传播的误差另行记录。

| 检查 | 最差 NRMSE | 预算 | 位置 |
|---|---:|---:|---|
| W8A8 数学 oracle | 2.2602488e-05 | 2e-3 | 第 39 层，self_attn.o_proj_math |
| W8A8 K128 语义 oracle | 1.5385138e-05 | 2e-3 | 第 14 层，linear_attn.out_proj_semantic |
| 独立 expert 输出 | 0.00040272632 | 5e-3 | 第 39 层，expert 106 |
| Attention/GDN 分支 | 0.0044264463 | 5e-3 | 第 14 层 |
| 自然路由 MoE | 0.00028615829 | 5e-3 | 第 9 层 |
| 均衡 256-expert combine | 0.00012724988 | 5e-3 | 第 0 层 |
| GDN FP32 recurrent output | 3.3614495e-07 | 1e-4 | 第 14 层 |
| GDN FP32 recurrent state | 4.9934272e-07 | 1e-4 | 第 0 层 |

W8A8 同时使用完整反量化 FP32 数学 oracle 和 K128 partial → activation scale →
weight scale → 顺序 FP32 累加的语义 oracle，各受 2e-3 预算约束。NVFP4 reference 使用
局部 FP16 操作数、FP32 GEMM，之后才施加 global reciprocal，保留 G1/SwiGLU/G2/combine
的 FP16 边界。早期 reference 先缩放操作数会改变临界 A4 编码，现已修正，预算没有放宽。
不声称逐指令模拟 HMMA 或全部 FlashInfer fast-math 变体。

相同 router logits 下 Top-8 IDs 全部精确匹配；独立 router 投影产生的 8/9 名近邻交换与
score margin 在各层 JSON 单独报告。A8 producer 的 captured FP16 边界、bytes、scales
分别核对；A8 零组保留 signed zero，NVFP4 零组 canonical zero。

## 40 层逐层结果

expert 列为该层 256 个 expert 的最大 NRMSE。“整层传播”是诊断，**未设预算，未标通过**。

| 层 | 分支 / expert dtype | W8A8 语义最大 | Attention/GDN | 自然 MoE | 单 expert 最大 | 整层传播（诊断） |
|---:|---|---:|---:|---:|---:|---:|
| [0](reports/layer_scan/layer_00.json) | GDN / FP16 | 1.021e-05 | 1.920e-04 | 9.881e-05 | 3.104e-04 | 2.272e-04 |
| [1](reports/layer_scan/layer_01.json) | GDN / NVFP4 | 7.458e-06 | 8.407e-04 | 8.803e-05 | 2.168e-05 | 2.596e-03 |
| [2](reports/layer_scan/layer_02.json) | GDN / NVFP4 | 8.770e-06 | 1.645e-04 | 9.030e-05 | 2.482e-05 | 1.357e-03 |
| [3](reports/layer_scan/layer_03.json) | Full / NVFP4 | 7.279e-06 | 1.988e-03 | 7.452e-05 | 3.005e-05 | 5.330e-03 |
| [4](reports/layer_scan/layer_04.json) | GDN / NVFP4 | 7.849e-06 | 3.536e-04 | 8.086e-05 | 7.432e-05 | 3.187e-03 |
| [5](reports/layer_scan/layer_05.json) | GDN / NVFP4 | 6.035e-06 | 1.939e-04 | 7.426e-05 | 3.390e-05 | 2.600e-03 |
| [6](reports/layer_scan/layer_06.json) | GDN / NVFP4 | 5.294e-06 | 1.642e-04 | 8.716e-05 | 3.449e-05 | 1.952e-03 |
| [7](reports/layer_scan/layer_07.json) | Full / NVFP4 | 1.180e-05 | 1.240e-03 | 8.766e-05 | 3.828e-05 | 5.899e-03 |
| [8](reports/layer_scan/layer_08.json) | GDN / NVFP4 | 7.297e-06 | 2.650e-04 | 1.246e-04 | 3.146e-05 | 2.383e-03 |
| [9](reports/layer_scan/layer_09.json) | GDN / NVFP4 | 5.874e-06 | 5.151e-04 | 2.862e-04 | 3.129e-05 | 3.344e-03 |
| [10](reports/layer_scan/layer_10.json) | GDN / NVFP4 | 6.495e-06 | 2.375e-04 | 7.479e-05 | 1.472e-05 | 3.166e-03 |
| [11](reports/layer_scan/layer_11.json) | Full / NVFP4 | 6.694e-06 | 1.128e-03 | 8.318e-05 | 1.606e-05 | 5.386e-03 |
| [12](reports/layer_scan/layer_12.json) | GDN / NVFP4 | 6.817e-06 | 2.356e-04 | 8.369e-05 | 2.623e-05 | 2.434e-03 |
| [13](reports/layer_scan/layer_13.json) | GDN / NVFP4 | 6.570e-06 | 3.557e-04 | 1.126e-04 | 3.507e-05 | 2.761e-03 |
| [14](reports/layer_scan/layer_14.json) | GDN / NVFP4 | 1.539e-05 | 4.426e-03 | 8.080e-05 | 2.432e-05 | 9.187e-03 |
| [15](reports/layer_scan/layer_15.json) | Full / NVFP4 | 7.805e-06 | 1.658e-03 | 6.684e-05 | 4.037e-05 | 6.165e-03 |
| [16](reports/layer_scan/layer_16.json) | GDN / NVFP4 | 6.086e-06 | 2.687e-04 | 8.326e-05 | 2.652e-05 | 3.378e-03 |
| [17](reports/layer_scan/layer_17.json) | GDN / NVFP4 | 7.238e-06 | 3.062e-04 | 7.896e-05 | 4.436e-05 | 2.906e-03 |
| [18](reports/layer_scan/layer_18.json) | GDN / NVFP4 | 8.140e-06 | 2.841e-04 | 9.622e-05 | 1.509e-05 | 3.347e-03 |
| [19](reports/layer_scan/layer_19.json) | Full / NVFP4 | 7.620e-06 | 1.491e-03 | 7.345e-05 | 2.951e-05 | 9.359e-03 |
| [20](reports/layer_scan/layer_20.json) | GDN / NVFP4 | 6.546e-06 | 3.885e-04 | 8.052e-05 | 1.897e-05 | 3.230e-03 |
| [21](reports/layer_scan/layer_21.json) | GDN / NVFP4 | 8.924e-06 | 3.409e-04 | 1.323e-04 | 4.089e-05 | 3.108e-03 |
| [22](reports/layer_scan/layer_22.json) | GDN / NVFP4 | 1.289e-05 | 1.635e-04 | 1.038e-04 | 2.261e-05 | 1.449e-03 |
| [23](reports/layer_scan/layer_23.json) | Full / NVFP4 | 1.211e-05 | 1.299e-03 | 8.869e-05 | 2.837e-05 | 6.951e-03 |
| [24](reports/layer_scan/layer_24.json) | GDN / NVFP4 | 6.878e-06 | 1.281e-04 | 6.887e-05 | 3.084e-05 | 2.691e-03 |
| [25](reports/layer_scan/layer_25.json) | GDN / NVFP4 | 6.001e-06 | 7.434e-04 | 5.654e-05 | 4.328e-05 | 1.315e-02 |
| [26](reports/layer_scan/layer_26.json) | GDN / NVFP4 | 6.746e-06 | 1.073e-04 | 6.355e-05 | 2.839e-05 | 3.394e-03 |
| [27](reports/layer_scan/layer_27.json) | Full / NVFP4 | 1.201e-05 | 2.064e-03 | 7.455e-05 | 2.522e-05 | 1.058e-02 |
| [28](reports/layer_scan/layer_28.json) | GDN / NVFP4 | 6.891e-06 | 2.066e-04 | 8.491e-05 | 3.291e-05 | 4.510e-03 |
| [29](reports/layer_scan/layer_29.json) | GDN / NVFP4 | 8.261e-06 | 8.337e-04 | 7.611e-05 | 1.814e-05 | 8.912e-03 |
| [30](reports/layer_scan/layer_30.json) | GDN / NVFP4 | 7.830e-06 | 3.678e-04 | 8.776e-05 | 2.175e-05 | 6.877e-03 |
| [31](reports/layer_scan/layer_31.json) | Full / NVFP4 | 8.221e-06 | 1.654e-03 | 8.042e-05 | 2.160e-05 | 9.411e-03 |
| [32](reports/layer_scan/layer_32.json) | GDN / NVFP4 | 8.165e-06 | 2.164e-04 | 6.831e-05 | 2.990e-05 | 1.237e-02 |
| [33](reports/layer_scan/layer_33.json) | GDN / NVFP4 | 9.124e-06 | 1.101e-04 | 5.783e-05 | 3.881e-05 | 3.304e-03 |
| [34](reports/layer_scan/layer_34.json) | GDN / NVFP4 | 7.469e-06 | 6.510e-04 | 9.566e-05 | 1.589e-05 | 4.703e-03 |
| [35](reports/layer_scan/layer_35.json) | Full / NVFP4 | 1.188e-05 | 1.656e-03 | 6.731e-05 | 1.838e-05 | 1.515e-02 |
| [36](reports/layer_scan/layer_36.json) | GDN / NVFP4 | 7.869e-06 | 2.993e-04 | 7.774e-05 | 3.901e-05 | 5.302e-03 |
| [37](reports/layer_scan/layer_37.json) | GDN / NVFP4 | 9.624e-06 | 1.694e-04 | 4.803e-05 | 7.844e-06 | 2.958e-03 |
| [38](reports/layer_scan/layer_38.json) | GDN / NVFP4 | 5.414e-06 | 1.474e-04 | 4.369e-05 | 1.945e-05 | 2.979e-03 |
| [39](reports/layer_scan/layer_39.json) | Full / FP16 | 8.726e-06 | 1.986e-03 | 1.060e-04 | 4.027e-04 | 1.905e-03 |

整层独立传播最大 NRMSE 为 **0.015154175（第 35 层）**。
这包含上游 FP16/量化边界及路由变化的传播；局部预算不能代替整层或完整模型质量验收。

原始 v4 的 `expert_execution_backend` 是通用 NVFP4 标签，第 0/39 层该字符串不准确：
实际使用 checkpoint 指定的 Triton FP16 grouped gate/up/down，与独立 FP16 reference
比较并执行全 256 experts。上表和汇总纠正标签，原始测量文本保留以便追溯。

## 单元、集成与后端验收

最终 unittest discovery 共运行 **93 项，92 通过、1 跳过**。跳过的是显式启用的单层扫描
smoke，已由单独 40 层 CLI 覆盖；没有因设备或 checkpoint 缺失而跳过 GPU 验收。
日志为 [final_regression.txt](reports/final_regression.txt)。unittest 报告的 356.771 秒
包含等待 GPU 锁的时间，不能用于 kernel 性能比较。

四层集成使用真实原始 0～3 层、全部 experts、embedding/final norm/head，另验第 39 层。
覆盖空输入、ragged、重复调用、63/64/65 边界、膨胀 max_seqlen、显式与默认 logits 行，
以及 embedding/norm/head 独立 oracle。packed 与分序列调用 hidden/logits NRMSE 均为 0。

| 四层 T2048 案例 | Peak allocated | 上限 |
|---|---:|---:|
| 单序列 | 5,460,760,064 B（5.09 GiB） | 12 GiB |
| `[1]×1983+[65]` | 9,595,500,032 B（8.94 GiB） | 12 GiB |

格式后完整四层 profiler 已通过全部 6 种 token 数：计算事件精确匹配本进程 Triton PTX，
无框架或未知 compute kernel。完整计时、kernel 数和当前源码 hash 见 [PERFORMANCE.md](PERFORMANCE.md)。

复现命令见 [README.md](README.md)，完整设计图见 [DESIGN.md](DESIGN.md)，
逐算子清单见 [CHECKLIST.md](CHECKLIST.md)。本页历史测量范围为文本 TP=1，最多 2048
tokens，无 KV/Conv/SSM 跨调用 cache。它涵盖独立 40 层与原始 0～3 四层集成，不包含
后续完整 40 层 cache 流水线的测量。模型生成质量评测、旧 ModelRunner 服务、视觉和 MTP
仍不在当前双卡 CLI 的验收范围。

扫描证据对应提交 `41d7d83452` 的源码。随后仅将扫描器输出的第 0/39 层 backend 标签
改为 FP16；没有更改计算、reference、预算或原始测量值。重跑时源码 hash 会因这项 metadata
表达式改变而不同，不能将新 hash 冒充原始扫描 hash。
