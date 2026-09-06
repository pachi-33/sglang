# V100 性能与计算后端记录

以下计时使用 `sglang-v100`（Torch 2.3.1+cu121 / Triton 2.3.1）和指定 SM70 V100。
计时排除加载、首次编译，计入 activation quant、routing、GEMM、SwiGLU、combine；
MoE 完整路径还计入 shared expert 和 residual。原始 JSON 保留显存、kernel histogram、
代码 hash、重复次数与设备信息，不能将单部件耗时解读为完整 40 层模型性能。

## NVFP4 MoE 融合

真实原始第 1 层、全部 256 experts，比较融合与同量化、同 FP16 舍入边界的显式未融合 Triton 路径。
18 个测试组合全部有限、NRMSE 为 0；完整路径 kernel 数为 18 对 24。默认始终使用成对
GEMM1/SwiGLU/A4 融合，未融合路径仅用于诊断。

| 路由 | T | 融合 ms | 未融合 ms | 延迟降低 | 融合 / 未融合峰值 MiB |
|---|---:|---:|---:|---:|---:|
| 自然 | 1 | 1.174 | 2.243 | 47.7% | 537.93 / 562.06 |
| 自然 | 4 | 2.290 | 3.362 | 31.9% | 538.01 / 562.13 |
| 自然 | 32 | 6.896 | 7.970 | 13.5% | 540.52 / 565.31 |
| 自然 | 128 | 9.628 | 11.211 | 14.1% | 549.15 / 576.19 |
| 自然 | 512 | 11.459 | 13.047 | 12.2% | 583.66 / 619.71 |
| 自然 | 2048 | 28.062 | 30.474 | 7.9% | 721.71 / 793.81 |
| 均衡强制 | 1 | 1.160 | 2.140 | 45.8% | 537.94 / 562.06 |
| 均衡强制 | 4 | 2.186 | 3.208 | 31.8% | 538.01 / 562.14 |
| 均衡强制 | 32 | 10.462 | 12.352 | 15.3% | 540.53 / 565.31 |
| 均衡强制 | 128 | 10.499 | 12.455 | 15.7% | 549.16 / 576.19 |
| 均衡强制 | 512 | 11.130 | 12.857 | 13.4% | 583.69 / 619.74 |
| 均衡强制 | 2048 | 23.235 | 25.291 | 8.1% | 721.84 / 793.93 |
| 8 个热点 expert | 1 | 1.152 | 2.152 | 46.5% | 537.94 / 562.06 |
| 8 个热点 expert | 4 | 1.160 | 2.148 | 46.0% | 538.01 / 562.14 |
| 8 个热点 expert | 32 | 1.190 | 2.185 | 45.5% | 540.53 / 565.31 |
| 8 个热点 expert | 128 | 2.375 | 3.295 | 27.9% | 549.16 / 576.19 |
| 8 个热点 expert | 512 | 6.308 | 7.377 | 14.5% | 583.69 / 619.74 |
| 8 个热点 expert | 2048 | 22.778 | 24.682 | 7.7% | 721.84 / 793.93 |

原始结果：[moe_benchmark.json](reports/moe_benchmark.json)。

80/160/320 个 persistent CTA 的候选均通过精度、确定性和 scratch 重用验证。320 在 18 个
组合中有 13 个最快，相比 160 的最大退步为 1.43%，因此默认选择 320；scratch 上限
为 `320×32×32×2 = 640 KiB`。历史 cap 调优使用分开的 shared/residual add；上表是随后
重新测量的融合 epilogue 完整路径，两套结果保留各自测量语义。

候选记录：[cap80](reports/moe_benchmark_cap80.json)、[cap160](reports/moe_benchmark_cap160.json)、
[cap320](reports/moe_benchmark_cap320.json)、[选择依据](reports/moe_cta_cap_tuning_comparison.json)。

## 合并输入投影

真实第 0 层合并 QKV/Z 与 B/A，第 3 层合并 QGate/K/V。CPU 整理 packed bytes/scales，
设备 component weights 共用 merged storage；后继算子直接读取带行 stride 的投影视图。
全部 12 个组合与分开投影的 FP16 输出逐位相同。

| 分支 | T | 分开 ms | 合并 ms | kernel 数（分开 → 合并） |
|---|---:|---:|---:|---:|
| gdn | 1 | 0.390 | 0.282 | 5 → 3 |
| gdn | 4 | 0.391 | 0.289 | 5 → 3 |
| gdn | 32 | 0.396 | 0.289 | 5 → 3 |
| gdn | 128 | 0.908 | 0.795 | 5 → 3 |
| gdn | 512 | 2.803 | 2.643 | 5 → 3 |
| gdn | 2048 | 8.932 | 8.833 | 5 → 3 |
| full | 1 | 0.382 | 0.225 | 4 → 2 |
| full | 4 | 0.383 | 0.227 | 4 → 2 |
| full | 32 | 0.388 | 0.224 | 4 → 2 |
| full | 128 | 0.656 | 0.523 | 4 → 2 |
| full | 512 | 1.793 | 1.712 | 4 → 2 |
| full | 2048 | 6.475 | 6.391 | 4 → 2 |

原始结果：[projection_packing_benchmark.json](reports/projection_packing_benchmark.json)。

上述 MoE 与投影对照记录于最终统一格式整理前；报告 hash 对应测量当时的原始文件。
isort/Black 后已确认除 import 排列与内部 helper docstring 外 AST 不变，最终单元和集成回归
另行记录。不得将原报告 hash 声称为格式整理后的字节级源码 hash。

## 四层完整调用与 Triton 审计

最终格式源码重新运行了完整 `forward_no_cache(input_ids=...)`，包含原始 0～3 层、
真实 embedding、final norm、全部 experts 和一行 LM-head logits。6 个案例的 shape 与有限值
均通过，所有 CUDA compute events 精确匹配本进程私有缓存中生成的 Triton PTX entry；
未出现 PyTorch/ATen、cuBLAS、CUTLASS 或未知 compute kernel。内存事件单独分类，此次
六个案例均为空。生成的 PTX 验证为 SM70，并包含 FP16 HMMA、FP32 accumulation。

| T | 四层完整调用中位 ms | 计算 kernel 数 | Peak allocated MiB |
|---:|---:|---:|---:|
| 1 | 7.856 | 136 | 4989.63 |
| 4 | 10.100 | 136 | 4990.19 |
| 32 | 18.868 | 136 | 4992.81 |
| 128 | 33.710 | 472 | 5001.88 |
| 512 | 74.425 | 1480 | 5039.08 |
| 2048 | 242.240 | 5524 | 5202.14 |

报告：[backend_audit.json](reports/backend_audit.json)；源码 SHA1：
`2849c5cb8cc23eb3779183677b66c94c25610c4b`。该报告取代旧的格式整理前审计，
匹配最终生产源码。可运行 `bench.backend_audit` 重新核查精确 PTX entry allowlist。

BT16 WY 每 chunk 有 14 个有序 kernel，T2048 为 128 chunks，三个 GDN 层的大量阶段
启动导致四层调用总计 5,524 个 compute launches；这仍是主要性能限制。当前结果是内存
有界、正确性通过的 SM70 实现，尚不能用作完整 40 层吞吐或缓存解码性能指标。
