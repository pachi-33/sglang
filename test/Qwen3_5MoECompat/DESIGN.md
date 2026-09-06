# Qwen3.5 MoE：SM70 量化融合与无 cache 设计

本文件取代早期“离线展开 FP16、先接缓存和多卡服务”的建议。当前交付按
`feat/qwen3/compat` 分支实现，精度和性能的已通过项目以
[MILESTONES.md](MILESTONES.md) 及实际测试报告为准；设计图不表示所有验收已完成。

结构和计算语义对照本地 SGLang checkout
`4349538c02e1566a1424510d5ac3ae853f49feef`，模型为
`Qwen-AgentWorld-35B-A3B-NVFP4_fp16`。该 checkout 未被当作官方最新 main。

## 固定范围

- 文本、TP=1、SM70 V100；全部 Python、测试及 benchmark 使用 `sglang-v100` 环境。
- 40 层分别加载真实权重并独立验证；集成使用原始第 0～3 层、全部 256 experts、
  真实 embedding、final norm 和独立 LM head。
- 总 token 数与每序列长度至多 2048。四层集成 CUDA peak allocated 上限为 12 GiB。
- 每次调用包含完整序列。Conv 左边界、GDN FP32 state 均从零开始；不接收或保留跨调用状态。
- 完整 40 层串联推理、旧 ModelRunner 服务、视觉、MTP、分布式和运行时 offload 属于后续范围。

## 模型与权重

| 结构 | 数量或形状 | 权重与激活计算 |
|---|---|---|
| Embedding | `[248320,2048]` | FP16，Triton gather |
| Decoder | 40 层；`[GDN,GDN,GDN,Full] × 10` | 输入、输出 `[T,2048]` |
| Decoder / final norm | 80 + 1 个，2048 维 | Gemma RMSNorm，乘 `(1+weight)`，eps `1e-6` |
| GDN QKV / Z / out | 每 GDN 层 3 个；`8192×2048`、`4096×2048`、`2048×4096` | FP8 W8A8，共 90 个矩阵 |
| GDN B / A | 每层各 `[32,2048]` | FP16 |
| GDN Conv | `[8192,1,4]`，无 bias | depthwise causal Conv + SiLU；每序列左侧补零 |
| GDN Q / K / V | Q/K `[T,16,128]`；V `[T,32,128]` | Q/K L2Norm；每两个 V heads 共用一个 Q/K head |
| GDN state | 每序列 `[32,128,128]` | FP32，物理顺序 `[V,K]` |
| GDN output norm / gate | 每 head 128 维，Z `[T,32,128]` | 普通 RMSNorm，乘 `weight` 和 `SiLU(Z)` |
| Full Q+gate / K / V / O | 每 Full 层 4 个；`8192×2048`、两份 `512×2048`、`2048×4096` | FP8 W8A8，共 40 个矩阵 |
| Full Q/K norm 与 RoPE | Q16、KV2、D256；旋转前 64 维 | Gemma RMSNorm 后先舍入 FP16，再做 partial NeoX RoPE，theta `1e7` |
| Full attention | causal GQA；Q16 / KV2 | Triton QK、softmax、PV；无 KV cache |
| Router | 每层 `[256,2048]` | FP16 GEMM，FP32 softmax 与归一化 Top-8 |
| Routed gate/up/down | 每层 256 experts；gate/up `[512,2048]`，down `[2048,512]` | 第 1～38 层 NVFP4 W4A4，共 29,184 个矩阵；第 0/39 层 FP16，共 1,536 个矩阵 |
| Shared expert | 每层 `2048→512→2048`，另有 `2048→1` scalar gate | FP16 |
| LM head | 独立 `[248320,2048]` | Triton FP16 GEMM，默认只投影各序列末 token |

FP8 矩阵合计 `30×3 + 10×4 = 130`。模型目录名中的 `_fp16` 不表示所有矩阵均为 FP16。
压缩权重在 CPU 上整理，再一次性转入 GPU；生产路径只在 tile 内解码，不保留完整展开副本。

## 量化和舍入合同

W8A8 权重保留原始 E4M3FN 编码及 FP16 `[N/128,K/128]` scale。
activation 每行每 128 元素编码，scale 为 FP32。每个 K128 分块内先做 FP16 HMMA，
FP32 累加其 partial，再乘该块 activation / weight scale；跨块结果仍在 FP32 中累加。
任意 W8A8 scale 不能提前折入 FP16 操作数。

NVFP4 使用 checkpoint 静态 global multiplier `G`：

```text
u = FP32(x) * G
sf = E4M3FN_RNE_sat(max(abs(u_group16)) * FP32(1/6))
q = E2M1_RNE_sat(u / decode(sf))
reconstruction = decode(q) * decode(sf) / G
```

scale 形成使用 FP32 倒数乘法，A8 同理使用 `FP32(1/448)`；payload 的归一化商使用
round-to-nearest FP32 除法。CPU / CUDA reference 必须明确实现这些舍入点。
FP4 偶数 K 在低 nibble；local scale 为零时整组编码归零。编码采用有限饱和、RNE，
独立 codec 测试覆盖 signed zero、subnormal、midpoint、饱和和全部有效码。
A8 保留负零编码，包括 scale 为零的整组；不能把 NVFP4 的零组规范套用到 A8。

NVFP4 的 `decode(FP4) × decode(FP8 local scale)` 最大绝对值为 2688，能精确表示为 FP16，
可直接形成片上 HMMA 操作数。global reciprocal 仍在 FP32 中施加。

同层 routed gate/up 的输入 global scale 相同，每 token 只生成一次 A4 payload；
GEMM1 按路由 token map 读取。成对 gate/up tile 在同一 CTA 中计算，保留
`GEMM1→FP16→SwiGLU→FP16→A4` 边界，直接输出 down 所需的 packed activation。
down 的输入 global scale 按 expert 选择。GEMM2 先舍入 FP16，再乘路由权重；
combine 按固定 Top-8 顺序在 FP32 中累加。

Full Attention output gate 的边界为 `FP16(attention16 × sigmoid(gate32))`。
Shared expert scalar gate 则先把 sigmoid 舍入 FP16，再乘 shared output；两者不能混用。
每次 residual add 也先产生 FP16 sum，再用于后续 FP32 norm。

## GDN 的两条路径

每条序列按实际长度选择算法：长度至多 64 使用 FP32 recurrent kernel，
长度大于 64 使用真实 BT16 WY 分解，包含 Gram、FP32 三角求解、U/W、R、state 更新和输出。
调用方提供的 `max_seqlen` 只控制工作范围，不能改变同一条序列的数值路径。
混合 batch 先产生 WY 输出，再用无 state 输出的 recurrent kernel 覆盖短序列；
该 kernel 跳过长序列和空序列，循环次数等于短序列实际长度。
单独的 recurrent / chunk 接口仍返回各自的 output 和 state，用于精度对照。
它不以 recurrent kernel 冒充 chunk，也不调用 PyTorch 矩阵计算。

记 `G=cumsum(g)`、`D=exp(G)`、`S0[V,K]` 为当前 chunk 前的 state：

```text
Lij = lower_strict(beta_i * dot(K_i,K_j) * exp(G_i-G_j))
A16 = FP16((I+L)^-1)             # 三角求解内部全部 FP32
U16 = FP16(A16 @ FP16(beta*V))
W16 = FP16(A16 @ FP16(beta*D*K))
H16 = FP16(S0)
R32 = FP32(U16) - W16 @ H16.T
R16 = FP16(R32)
RD16 = FP16(R32 * exp(G_last-G_i))
S1 = exp(G_last)*S0 + RD16.T @ K16
C16 = FP16(causal(Q16 @ K16.T) * exp(G_i-G_j))
O16 = FP16((D*(Q16 @ H16.T) + C16 @ R16) / sqrt(128))
```

未来位置和尾块必须在 `exp` 前屏蔽。`RD16` 从 R32 计算，不能改为从 R16 再乘 decay。
V100 的 Triton 2.3 布局限制要求部分矩阵阶段通过独立 kernel 和临时工作区衔接。
有界实现逐 BT16 完成 WY 和输出后复用当前 chunk 的工作区，保留本次调用的 FP32 state。
工作区设计不应按 `batch × max_chunks` 保存所有序列的 state history。
WY 临时工作区按至多 32 条序列复用，完整输出保留绝对 token offset；
每条序列的 FP32 state 只属于当前调用。全短序列的模型路径直接使用无 state 输出的递推。

## 代码接口与主干对应

| 本次模块 | 职责 | 主干结构对应 |
|---|---|---|
| `models/qwen3_5_moe.py` | `EntryClass`、模型装配、`forward_no_cache` | `Qwen3_5MoeForConditionalGeneration` |
| `layers/qwen3_5/config.py`、`checkpoint.py` | 旧 Transformers 的嵌套 config 注册、按层加载、压缩权重和 scale 校验 | 配置、主干 `load_weights` 映射 |
| `layers/qwen3_5/runner.py` | 选定原始层的 stateless 编排 | GDN / Full decoder 层及 residual、MoE 连接 |
| `weights.py`、`quantization.py` | 类型化压缩张量与 W8A8 / A4 接口 | mixed compressed-tensors 量化语义 |
| `dense.py`、`ops.py`、`model_ops.py` | FP16 Linear、norm、投影布局和融合 producer | Linear、Gemma / gated RMSNorm、QKVZBA 与 Q/gate 布局 |
| `attention.py`、`gdn.py`、`moe.py` | 部件接口和计算调度 | causal GQA、Gated DeltaNet、Qwen2 sparse MoE |
| `kernels/` | SM70 Triton 计算 | 本期性能相关算子的实现位置 |
| `test/.../reference/` | 独立数学和量化 reference | 不导入生产 kernel helper 来定义预期结果 |

```python
forward_no_cache(
    *, input_ids=None, hidden_states=None,
    positions, cu_seqlens, max_seqlen,
    logits_indices=None,
)
```

`input_ids` 和 `[T,2048]` FP16 `hidden_states` 二选一；返回 final norm 后的 hidden 和所选 logits。
`positions[T]`、`cu_seqlens[B+1]` 为同设备 contiguous int32/int64；`max_seqlen` 是调用方提供的
Python int。offset 内容是受信任的调用 metadata：起点 0、终点 T、非递减，各长度不超过 max。
公共接口检查 shape/dtype/device 及能由 shape 判断的矛盾，不在热路径把 offsets 读回 CPU。

无 token 时返回零行 logits；非空 packed batch 中的空序列默认对应零 logits 行。
默认和显式 logits 行选择均在 GPU 上完成。Python/PyTorch 只负责配置、分配、加载、视图及测试参考；
真正不复制的 view 可以保留，不能为了避免 view 而新增无意义的数据复制 kernel。

## 验收方法

单矩阵 NRMSE 上限 `2e-3`，attention / MoE 部件 `5e-3`，GDN FP32 recurrent 内部 output / state
relative L2 上限 `1e-4`，GDN chunk 长度 2048 的 NRMSE 上限 `5e-3`。同时记录最大误差、P99、
NaN/Inf、形状和序列长度；不得通过放宽预算绕过错误。

40 层扫描每次使用独立 T32 输入和 `expert_id[token,slot]=8*token+slot`，覆盖每层全部 256 experts。
自然 router、热点、空专家、尾块和相同分数的确定性选择另行检查。参考按单矩阵或单 expert 临时展开，
attention 按 query、head 按 vocab 分块，避免参考实现成为显存瓶颈。

四层集成检查精度连接、单序列 / ragged / empty、Conv / chunk 边界、packed 与分序列一致性、
重复调用、原始第 39 层及实际 peak allocated。四层通过不能推断完整模型的生成质量。

性能测量覆盖 T=1/4/32/128/512/2048，排除编译和加载，计入 quant、routing、GEMM、SwiGLU、combine。
融合与未融合的比较必须保留相同量化和 FP16 边界。成对 GEMM1 / SwiGLU / A4 融合始终是默认要求；
未融合路径仅用于明确指定的诊断比较。在满足融合要求的候选中，只有正确且实测有收益的配置进入默认调度；
profiler 和当前 V100 进程编译的 PTX 用于核实 Triton / SM70 HMMA 覆盖。

完整模型计算图见 [model_design.mmd](model_design.mmd)。
