# Qwen3.5 MoE SM70/SM89 单请求 Cache 与双卡流水线验收

本页记录 `feat/qwen3/pp` 分支最终代码的实现合同与验收结果。完整双卡验收 JSON 已保存于
[pipeline_acceptance.json](reports/pipeline_acceptance.json)，顶层 `ok=true`。
历史独立 40 层扫描与四层 stateless profiler 保留在 [VALIDATION.md](VALIDATION.md)
和 [PERFORMANCE.md](PERFORMANCE.md)，其源码 hash 不代表当前 cache/pipeline 工作树。

## 环境与执行合同

| 项目 | 固定值 |
|---|---|
| 分支 | `feat/qwen3/pp`；实现与本页报告一同提交 |
| 最终验证日期 | 2026-09-08（Asia/Singapore） |
| 模型 | `/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16` |
| Python | `/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python` |
| 运行库 | Torch 2.3.1+cu121、Triton 2.3.1；旧 Transformers 4.43.2 tokenizer 兼容 |
| Front（默认） | RTX 4070 SUPER / SM89，`GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4` |
| Back（默认） | V100-SXM2-16GB / SM70，`GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96` |
| 分层（默认） | Front：embedding、layers 0–16、final norm/head；Back：layers 17–39 |
| 请求 | batch=1、一个活跃请求、`prompt + R ≤2048`（R 为生成上限）、greedy |

Controller 不执行 CUDA 计算。两个独立 Python worker 在启动前各自固定 GPU UUID，
避免 Triton 2.3.1 的进程级 target cache 混用架构。JSON 元数据与连续 FP16 binary
payload 使用版本化帧；层 16→17 经 CPU 传递 `[P,2048]` prefill 或 4 KiB decode。
后半层只返回末行，交由 4070 final norm/head。没有 NCCL、P2P 或 pipeline micro-batch。
`front_uuid`、`back_uuid` 与 `split_layer` 均为显式配置，role 不绑定 GPU 架构；默认采用
实测通过的 4070-front 17/23，旧 V100-front 20/20 仍可显式选择。

生命周期是 `BEGIN → PREFILL → DECODE* → RESET/END`。命令校验 epoch、step_id、
expected_prefix_len、token_count，双方 ACK 的 consumed_len 一致后才推进。
整数与 shape 字段必须是精确 JSON integer，不能用 bool/float 相等比较冒充。
执行异常令 cache poisoned，重试同一步被拒绝，必须 reset；任一侧半步失败会 reset 两侧。

两个 worker 在默认 17/23 切分、容量 2048 时分别保留：

| 状态 | 每层布局 | 4070 front（17 层） | V100 back（23 层） |
|---|---|---:|---:|
| GDN recurrent | FP32 `[32,128,128]` | 13 层，26 MiB | 17 层，34 MiB |
| GDN Conv tail | FP16 `[3,8192]` | 13 层，0.609375 MiB | 17 层，0.796875 MiB |
| Full Attention K/V | 各 FP16 `[2048,2,256]` | 4 层，16 MiB | 6 层，24 MiB |
| 合计 | 显式外部 single-request cache | 42.609375 MiB | 58.796875 MiB |

整段 fresh prefill 保存 Conv 前 QKV tail（短序列左补零）、GDN FP32 final state、
Q/K norm 和 RoPE 后 K，以及投影 V。GDN ≤64 使用 recurrent，>64 使用 chunk/WY。
后续仅单 token decode：原位推进 Conv/GDN，在 KV 位置 L 写入后读取 `[0:L+1]`，
固定 GQA `query_head // 8`、scale `1/16`。Reset 清 GDN/Conv，KV 存储通过有效长度归零失效。

首 token 来自 prompt logits；生成 R>0 个 token 只执行 R−1 次 decode，最后采样 token
不写 cache，但仍计入公开的总上下文长度。因此容量限制是 `prompt + R ≤2048`；
最后一个 token 不占 cache slot 只影响 decode 次数，不把总上下文放宽到 2049。
请求结束立即 reset。CLI 应用 checkpoint chat template，也支持 raw/stdin。
二维 tokenizer merges 只在内存转成旧格式，已用固定 raw/chat fixture 与 Transformers
4.57.3 逐 ID 对照。Head `[248077,248320)` 被屏蔽，EOS IDs 为 `{248046,248044}`。

## 最终验收结果及来源

以下 unit 与真实层计数来自协调者对最终代码的双端回归；pipeline 数值逐项对应已落盘的
[验收 JSON](reports/pipeline_acceptance.json)。复核脚本为
`integration/pipeline_acceptance.py`，下节给出同一命令。

| 检查 | 最终结果 |
|---|---|
| SM89 顺序门 | 现有算子 unit gate 在 cache/流水线实现前通过；设备合同显式支持 SM89，没有用 monkey patch 绕过 V100 gate |
| 双端完整 unit 回归 | SM89 126/126，SM70 126/126；无 skip/error |
| 真实层 cache | `integration.test_stateful_runner` 两卡各 5/5；包含 GDN/Full Attention prefill/decode 对照与生命周期检查 |
| 修复后的 quantization 模块 | SM70 7/7；SM89 7/7 且 FP8 GEMM filtered memcheck 0 errors |
| 双卡完整模型短序列 | 8 步 cached/stateless greedy token 均一致 |
| A→reset→B→reset→A | A1/A2 IDs 均 `[11,271]`；B 为 `[271,248068]` |
| Chat 连续两次 | IDs 均 `[90700,8340,25,271,16,13,220,2972]`，输出有限 |
| HTTP API 与物理角色交换 | 默认 4070 front、V100 back；worker READY 的 capability 和 0–16/17–39 范围正确；native/completions/chat 均为 HTTP 200，auth 401/200 合同通过 |
| 完整双卡 acceptance | `ok=true`；容量/epoch/step 错误、半步失败恢复和 reset 显存检查通过 |

8 步 cached/stateless 对照 token IDs：

```text
[11, 271, 40, 1044, 4313, 310, 958, 279]
```

所有步骤 greedy token 一致。下面按 step_id 记录完整模型误差传播诊断；router slots
统计 40 层各 8 个 Top-8 位置的 ID 差异，完整逐层 IDs 与概率差异保存在 JSON 中。

| Step | Hidden NRMSE | Hidden max abs | Logits NRMSE | Logits max abs | Router mismatch slots | Router max prob abs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0.0880324941 | 0.328125 | 0.0644059171 | 0.97265625 | 68 | 0.0213265419 |
| 2 | 0.0606885125 | 0.224853515625 | 0.0797930527 | 0.75390625 | 55 | 0.0192497969 |
| 3 | 0.0504031262 | 0.265625 | 0.0501716776 | 0.51495361328125 | 79 | 0.0195047855 |
| 4 | 0.0452422605 | 0.123046875 | 0.0385278462 | 0.48828125 | 58 | 0.0231304467 |
| 5 | 0.0434926619 | 0.11328125 | 0.0402171735 | 0.478271484375 | 58 | 0.0187364966 |
| 6 | 0.0405510709 | 0.103515625 | 0.0417209150 | 0.48828125 | 124 | 0.0309149623 |
| 7 | 0.0335850120 | 0.21484375 | 0.0273354563 | 0.39453125 | 60 | 0.0208339095 |

这些诊断不能描述为满足局部 kernel 的 5e-3 预算，也不意味着 hidden/logits 或 router
逐项相同。`--validate-stateless` 启用完整前缀及 router 对照；生产生成不捕获 router。

| 显存测量 | 4070 SUPER front allocated | V100 back allocated |
|---|---:|---:|
| 权重与 capacity-2048 cache 加载后 | 11,611,956,736 B | 12,546,822,656 B |
| 2048 fresh prefill peak | 11,910,991,872 B | 12,845,352,960 B |
| 首 decode peak（1-token prefill，prefix_len=1） | 11,682,299,392 B | 12,616,668,160 B |
| 预热后三轮 reset 后稳定值 | 11,612,463,104 B | 12,546,831,872 B |

以上 17/23 案例无 OOM；三轮 reset 的 allocated 与 reserved 都逐字节稳定，reserved 分别为
4070 SUPER 12,119,441,408 B、V100 12,949,913,600 B。首 decode 列是 1-token prefill
后首次 decode 完成时的累计 peak，不是单独区间峰值；它不能改称 2048-token prefill 后
decode。容量 2048 已满时追加 decode 按合同拒绝。脚本分别测量 full-capacity prefill
与短 prefill/decode，报告中的 prefix_len 应与峰值一起解释。

选择 17 层不是只按权重估算：4070-front 18/22 曾实际加载并完成短 prefill/decode，
但在 2048 prefill 的 GDN 临时张量分配处 OOM（请求再分配 20 MiB 时失败）。因此生产
默认回退为 17/23；18/22 不属于通过配置。

错误恢复字段均包含实际 reset 后的新请求 token。真实半步故障中，front 的 step 1
先返回 `consumed_len=2`，随后 back 因注入的 prefix mismatch 拒绝并 poison；公共
`generate_ids` 清理路径自动将两侧归零，恢复请求返回 `[11]`。容量、重复 step 与错误
epoch 路径也分别 reset 两侧并以 `[11]` 恢复。

## SM89 FP8 scale 越界定位与回归

原双卡 `Hello --max-new-tokens 2 --validate-stateless` 在一次 cached decode 后，
后半 worker 的完整两 token 前缀重算发生 illegal memory access。独立随机输入重现同样的
`prefill T1 → stateless T1 → decode → stateless T2` 序列后，filtered compute-sanitizer
将首次故障定位到 FP8 GEMM activation scale 的 4-byte global read，而非 GDN state 写入。
原实现报告 984 次错误，访问越过一个 2 MiB allocation 的末尾。

仅把 activation-scale backing 补到 32 行、仍保持逻辑 `[2,32]`，故障就消失。正式修复
不依赖 padding：scale 地址先用 `min(row,m−1)` 限制在合法行，再用 `where(row<m,scale,0)`
清零无效行，规避 SM89/Triton 2.3.1 一维 masked-load lowering。M=0 由 wrapper 提前返回。
新增 storage-tail 测试覆盖 M=1/2/3/17/31/33；没有修改 GDN/Conv 数学或 cache 布局。

修复后的 20 层后半独立复现与 quantization 模块均通过 filtered memcheck。精确命令
（repository root）：

```bash
CUDA_VISIBLE_DEVICES=GPU-75341d61-b0b3-969b-8ef8-4b750d11ade4 \
PYTHONPATH=python:. \
/usr/local/cuda-12.8/bin/compute-sanitizer \
  --tool memcheck --kernel-name kns=fp8_block128_gemm_kernel \
  --report-api-errors no --print-limit 4 --show-backtrace no \
  /home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m unittest test.Qwen3_5MoECompat.unit.test_quantization -v
```

最终重跑输出摘要：`Ran 7 tests in 2.660s`、`OK`、`ERROR SUMMARY: 0 errors`。
此过滤器只审计 FP8 GEMM，不表示完整模型所有 kernel 都接受过 memcheck。

## 复现与结果留存

两卡 `unit/` 独立进程命令及 real-layer cache 测试命令见 [README.md](README.md)。
GPU correctness tests 各自持有 UUID 锁，历史 V100-only 性能 benchmark/scan 与 V100 共用锁。
不要给自带锁的命令套第二层 flock，也不要让双卡验收与占满显存的其他任务同时运行。

完整 CLI 短序列对照：

```bash
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m sglang.srt.layers.qwen3_5.pipeline \
  --raw-prompt --prompt Hello --max-new-tokens 8 \
  --print-token-ids --validate-stateless
```

包含 A/B/A、chat 双跑、2048 prefill、首 decode、reset 显存稳定性、容量/epoch/step
与半步失败恢复的独立验收脚本：

```bash
PYTHONPATH=python:. \
/home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python \
  -m test.Qwen3_5MoECompat.integration.pipeline_acceptance \
  --output test/Qwen3_5MoECompat/reports/pipeline_acceptance.json
```

输出是单份 JSON，记录配置、worker 硬件信息、逐步误差/路由诊断、显存计数及异常恢复。
本页数值对应已保存报告。默认 A 为 `Hello`，B 为 `Explain cache reuse in one sentence.`，
chat 为 `请用一句话解释 KV 缓存。`；不要把此前不同 B prompt 的交互式结果混入这份报告。

HTTP 服务入口为 `-m sglang.srt.layers.qwen3_5.pipeline_api`，提供 `/generate`、
`/v1/models`、`/v1/completions` 与 `/v1/chat/completions`。真实 smoke 在父进程
默认 4070-front 17/23 配置下完成：completion `Hello` 返回 IDs `[11,271]`，chat
返回 `[90700,8340]`，worker READY 确认 front 位于 SM89、back 位于 SM70。两卡使用
相同算子源码但独立 JIT target。API CLI 同样暴露 `--front-uuid`、`--back-uuid` 和
`--split-layer`；GPU index 顺序不会影响 UUID 选卡。

当前支持范围为单请求连续状态、文本 greedy CLI 与非流式 HTTP API；多轮 append、
prefix sharing、radix attention、通用 sstate、分页 KV、服务端调度、视觉/MTP、
API streaming、批并行均不在此合同内。
短序列 token 一致和有限输出是工程验收，不替代广泛 prompt 的生成质量评测。
