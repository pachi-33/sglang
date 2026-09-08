# Qwen3.5 ExpertPack 单 GPU 与历史兼容路径核对清单

本清单用于代码审阅、复现和验收。当前交付路径是文本模型、TP=1、单进程、单张
V100/SM70 或 4070/SM89：一个完整 40 层 runner 维护一个请求，整段 fresh prefill 后逐
token decode，上限 2048；layers 1–38 的 NVFP4 routed experts 按需换入。双 worker 分层
流水线和 packed-sequence 无状态接口仅保留为历史 oracle，不参与 ExpertPack 运行。
不包含视觉、MTP、radix/sstate 管理、分页 KV、NCCL/P2P、批调度或 ModelRunner 服务。
所有 Python 命令使用 `sglang-v100` 环境；单卡测试暴露一个指定 UUID，流水线由
controller 在启动前给两个 worker 分别设置 UUID。ExpertPack CLI/API 必须只暴露一张
受支持 GPU；V100 使用 7168 MiB cache，12 GB 4070 必须显式使用 3584 MiB profile。

## ExpertPack 当前实现与验收状态

| 核对项 | 生产源码 | 验证源码/证据 | 当前判定 |
|---|---|---|---|
| v1 文件与身份 | `expert_pack/format.py`、`build.py`、`validate.py` | `unit/test_expert_pack.py`；完整 validator | 已验证：layers 1–38 共 9,728 records；整包/payload/padding/source bytes 全通过 |
| typed cache 与读取 | `expert_pack/store.py` | `unit/test_expert_pack_store.py`；单卡 acceptance | 正常路径已验证：V100 7168 MiB/4247 slots、SM89 3584 MiB/2123 slots、16 staging、2 workers、LFU/LRU、SHA、event/lease |
| 选择性 checkpoint | `checkpoint.py`、`runner.py` | `unit/test_single_gpu.py`；单卡 acceptance | 已验证：中间层不保留 routed tensors；layers 0/39 与其余权重常驻 |
| logical→slot kernel | `moe.py`、`kernels/moe.py` | `unit/test_moe.py`；`reports/expert_offload_layer1_v100.json` | 已验证：原 global IDs 保持不变；SM70/SM89 nonidentity mapping 和 slot 2048；V100 T=1/32/2048 输出与 router IDs/weights 精确一致，热命中零读取/H2D |
| 40 层 CLI/backend | `single_gpu.py` | V100 acceptance；V100/SM89 HTTP smoke 与 serving reports | 已验证：两种架构黄金 token；V100 A/B/A、2048、capacity 恢复并保留大于 1 GiB 余量；SM89 最后一条 988→128 请求的 `total_memory-peak_reserved` 为 1.18 GiB，尚未做 2048 验收 |
| 单卡 HTTP | `single_gpu_api.py`、`pipeline_api.py` | CPU 单测；`reports/single_gpu_api_{smoke,concurrency,checksum_failure}_v100.json` | 已验证：三入口 200、活跃请求期间 429、FAILED health/后续请求 503 |
| fatal 故障传播 | store/runner/backend/API FAILED latch | CPU contracts；`reports/single_gpu_api_{checksum,short_read,h2d}_failure_v100.json` | 已验证：checksum、短读及 H2D/CUDA error 均 poison cache、锁存 FAILED，首/health/后续均 503 |
| 长稳压 | 全路径 | 计划中的 stress | 待验证 |


路径约定：`layers/...` 表示 `python/sglang/srt/layers/...`；`kernels/...`
表示 `python/sglang/srt/layers/qwen3_5/kernels/...`；无目录前缀的生产文件名表示
`python/sglang/srt/layers/qwen3_5/` 下的文件；`models/...` 表示 `python/sglang/srt/models/...`；
`unit/...`、`integration/...`、`reference/...` 与 `bench/...` 均表示
`test/Qwen3_5MoECompat/` 下的对应子目录。下表中的相对路径均按这个约定展开。

## 入口、权重和运行边界

| 核对项 | 生产源码 | 验证源码 | 验收点 |
|---|---|---|---|
| 配置和模型入口 | `python/sglang/srt/models/qwen3_5_moe.py`、`layers/qwen3_5/config.py` | `unit/test_model_config.py` | 嵌套 `text_config` 可在旧 Transformers 注册前加载；固定 40 层配置和 RoPE 合同被拒绝/接受。 |
| 精确 checkpoint 清单 | `layers/qwen3_5/manifest.py`、`layers/qwen3_5/checkpoint.py` | `unit/test_checkpoint.py` | 仅接受固定文本张量名、shape 和 dtype；payload 读取前完成 header 校验。 |
| 压缩 `Weight` 合同 | `layers/qwen3_5/weights.py` | `unit/test_checkpoint.py`、`unit/test_quantization.py` | FP16、FP8、NVFP4 的物理存储、scale 和 global scale 不允许混用。 |
| 无状态编排和 metadata | `layers/qwen3_5/runner.py` | `integration/test_stateless_model.py`、`unit/test_gdn.py` | 隐藏态为连续 FP16 `[T,2048]`，`T<=2048`；`positions=[T]`、`cu_seqlens` 为同卡连续 int32/int64。调用者信任 `cu` 的内容：首项为 0、末项为 T、单调、每段长度不超过 `max_seqlen`，内核不为热路径回读同步验证。无 KV 或 recurrent state 跨调用保存。 |
| 原始 0--3 层端到端入口 | `models/qwen3_5_moe.py` | `integration/test_stateless_model.py` | embedding、4 个原始层、final norm 与选择行 LM head 在同一次无 cache 调用中连接。 |
| SM70/SM89 设备与锁合同 | `runner.py`、`unit/test_environment.py` | `unit/` | runner 接受 capability `(7,0)`/`(8,9)`；GPU 测试严格校验单一 UUID/capability 配对，各卡使用独立 UUID 锁。V100TestCase 是兼容别名。 |
| 单请求外部 cache | `runner.py` | `integration/test_stateful_runner.py` | runner 所有权、capacity、前缀长度、空/重复 prefill、单 token decode、容量耗尽；执行失败 poison，reset 清 Conv/GDN 与 KV 有效长度。 |
| 双 worker 分层流水线 | `pipeline.py` | `unit/test_pipeline.py`、`integration/pipeline_acceptance.py` | 默认 4070 embed/0–16/norm/head、V100 17–39；front/back UUID 与 split 显式可配；独立子进程、带版本 JSON/FP16 帧、CPU staging，epoch/step/prefix/consumed_len 一致后才推进。半步失败 reset 两侧。 |
| 文本 greedy CLI | `pipeline.py` | `unit/test_pipeline.py`、`integration/pipeline_acceptance.py` | chat template/raw/stdin、二维 merges 内存兼容、逐 tokenizer ID 对照；屏蔽 `[248077,248320)`，EOS `{248046,248044}`；R 个输出只执行 R−1 次 decode。 |
| 单请求 HTTP API | `pipeline_api.py` | `unit/test_pipeline_api.py`、真实 HTTP smoke | 持久双 worker；`/generate`、models、OpenAI completions/chat；仅 greedy、非流式、n=1。API key 可选；并发请求返回 429；输出 text、usage、finish reason 和精确 token IDs；关闭时由 controller 向独立 session worker 发送 SHUTDOWN。 |

## 量化、线性层和投影打包

| 算子/边界 | 生产源码 | 独立参考或测试 | 核对点 |
|---|---|---|---|
| E4M3FN 编解码、A8 K128 quantize | `layers/qwen3_5/quantization.py`、`kernels/quantization.py` | `reference/codec.py`、`unit/test_reference_codec.py`、`unit/test_quantization.py` | 有限饱和/RNE、NaN 编码、signed zero、FP32 reciprocal scale 形成。 |
| W8A8 K128 GEMM | `layers/qwen3_5/quantization.py`、`kernels/quantization.py` | `reference/model.py`、`unit/test_reference_model.py`、`unit/test_quantization.py` | 每 K128 partial 的 FP16 E4M3 decode、FP32 partial/scale/cross-K 累加、输出 FP16。部分 M tile 的 scale 地址先限制到合法行再清零无效行；storage-tail 回归覆盖 M=1/2/3/17/31/33，避免 SM89/Triton 2.3.1 masked-load 越界。 |
| FP16 linear、embedding、LM head | `layers/qwen3_5/dense.py`、`kernels/dense.py` | `unit/test_dense_ops.py` | 隐藏维 `H=2048`；embedding/LM head 为 `[248320,2048]`，默认只选择每序列末 token logits；显式 `logits_indices` 可选择任意行或全部行，此时输出和显存按所选行数增长。 |
| Gemma RMSNorm、ordinary RMSNorm、residual add | `layers/qwen3_5/ops.py`、`kernels/ops.py` | `unit/test_dense_ops.py`、`unit/test_model_ops.py` | Gemma 使用 `FP16(x * rsqrt(mean(x²)+1e-6) * (1+w))`；GDN ordinary norm 使用 `w` 而非 `1+w`；residual add 先产生 FP16 sum，`residual_add_gemma_rms_norm` 以该 sum 再做 norm。 |
| GDN QKV+Z / B+A 打包 | `layers/qwen3_5/checkpoint.py`、`runner.py` | `unit/test_projection_packing.py` | CPU 排列为 `[QKV,Z]`、`[B,A]`；GPU 只传 merged storage，公开 component `Weight` 是别名视图。 |
| Full QGate+K+V 打包 | `layers/qwen3_5/checkpoint.py`、`runner.py` | `unit/test_projection_packing.py` | CPU 排列为 `[QGate,K,V]`；component bytes 和 scales 共享 merged storage。 |
| 投影性能/精度对照 | `bench/benchmark_projection_packing.py` | 已提交的 `reports/*.json` | layer 0/3，T=1/4/32/128/512/2048；separate/merged FP16 输出完全相等，并记录 CUDA kernel 数与中位延迟。 |

## GDN 路径

| 算子/边界 | 生产源码 | 测试 | 核对点 |
|---|---|---|---|
| Conv4 + SiLU | `kernels/gdn.py` | `unit/test_gdn.py` | packed sequence 左边界清零；投影 column view 可有大于逻辑宽度的 token stride；输出连续。 |
| Conv 单 token decode | `kernels/gdn.py`、`runner.py` | `unit/test_gdn.py` | 读取 `[3,8192]` 原始 QKV tail 和当前行，计算 Conv/SiLU 后原位推进；短 prompt 左补零；对照完整卷积输出与 tail。 |
| Q/K/V 布局和 L2Norm | `kernels/model_ops.py`、`kernels/gdn.py` | `unit/test_gdn.py` | `Q/K=[T,16,128]`、`V=[T,32,128]`，Q/K L2Norm 在 FP32 中计算。 |
| A/B gate | `kernels/gdn.py` | `unit/test_gdn.py` | B/A merged 输出的 stride=64 被接受；softplus 稳定；beta 有明确 FP16 边界。 |
| 短序列递推 | `kernels/gdn.py` | `unit/test_gdn.py` | 实际长度 1--64 使用 FP32 state recurrence，65 及以上不被短路径覆盖。 |
| GDN cache continuation | `kernels/gdn.py`、`runner.py` | `unit/test_gdn.py`、`integration/test_stateful_runner.py` | prefill 保存 FP32 `[32,128,128]` final state；decode 接受非零 initial state 并原位更新。纯 recurrent relative L2 ≤1e-4，chunk-prefill 接 decode NRMSE ≤5e-3；out/state 不允许 storage 别名。 |
| BT16 WY chunk | `kernels/gdn_chunk.py`、`kernels/gdn.py` | `unit/test_gdn.py` | Gram、FP32 三角求解、U/W、R/state/output 分阶段；ragged tail 和 O(B) workspace 被覆盖。每个 sequence tile 的每个 BT16 chunk 依次发射 14 个阶段 launch；`max_seqlen=2048` 时为 128 chunks，即最多 `14×128×ceil(B/32)` 个有序 launch（短 sequence 在掩码中空转）。这是内存有界实现，不能被描述为低 launch-count 路径。`stream_gdn16` 是内部 `max_seqlen>0` helper，空输入由公开 `chunk_gdn` 处理。 |
| Z norm/SiLU/A8 producer | `kernels/model_ops.py` | `unit/test_model_ops.py` | `[T,4096]` Z column view 的 token stride 被显式传入；FP16 boundary 与 A8 bytes/scales 分开核对。 |

## Full Attention 路径

| 算子/边界 | 生产源码 | 测试 | 核对点 |
|---|---|---|---|
| QGate/K norm + partial NeoX RoPE | `kernels/model_ops.py` | `unit/test_model_ops.py` | Q projection 每 head 是 `Q[256], Gate[256]`；Q/K source 可为 merged row view；输出 Q/K/Gate 连续。 |
| Causal GQA slab | `kernels/attention.py`、`kernels/attention_slab.py` | `unit/test_attention.py` | Q16/KV2/D256、packed ragged causal mask、int32/int64 cu；V 可为 token stride 9216 的 merged view。 |
| 连续 KV decode attention | `kernels/attention.py`、`runner.py` | `unit/test_attention.py`、`integration/test_stateful_runner.py` | 先写 K/V 位置 L，再读取 `[0:L+1]`；Q head //8 映射 KV head，scale=1/16，无 torch.cat。长度覆盖 1/3/16/17/63/64/65/511/512/513/2048，未用区 NaN 验证逻辑边界。 |
| Attention gate + O A8 producer | `kernels/model_ops.py` | `unit/test_model_ops.py` | `FP16(attention × sigmoid(gate))` 是 producer boundary，A8 codec 独立检查。 |

## MoE 路径

固定形状：router 为 `[256,2048]`；每层 routed gate/up 为 `[256,1024,2048]`
（逻辑顺序 gate 后 up，每支 intermediate `I=512`），down 为 `[256,2048,512]`；
每 token 固定 Top-8。第 0/39 层 routed 权重 FP16，中间第 1--38 层为 raw NVFP4。

| 算子/边界 | 生产源码 | 独立参考或测试 | 核对点 |
|---|---|---|---|
| Router、stable Top-8 | `layers/qwen3_5/moe.py` 的 `route_topk`、`kernels/moe.py` 的 `normalized_top8_kernel` | `reference/moe.py`、`unit/test_moe.py` | router FP16 GEMM 后以 FP32 softmax；选中 8 路由权重在 FP32 归一化；相等 logit 必须 lower expert ID 优先。 |
| 稳定 expert-major dispatch | `layers/qwen3_5/moe.py` 的 `_build_dispatch`、`kernels/moe.py` 的 `dispatch_*` | `unit/test_moe.py` | route 数为 `R=T×8`；count/prefix/scatter/inverse 按 expert-major 稳定重排。`dispatch_stable_scatter_kernel` 每个 expert CTA 都按 256-route chunk 扫描全部 R，因此 scatter 本身是 O(E×R)，`E=256`；容量按 32 对齐，不可误称为 O(R)。 |
| 一次 A4 输入 codec | `layers/qwen3_5/quantization.py`、`kernels/quantization.py`、`layers/qwen3_5/moe.py` | `reference/model.py`、`reference/codec.py`、`unit/test_quantization.py`、`unit/test_moe.py` | routed hidden `x=[T,2048]` 仅为 gate/up 生成一次 static-global A4；gate/up input global 必须物理 `[256]` 且相同。强制 fused G1 不先 gather/materialize expert-major A4，而是从原始 per-token A4 用 `source_ids` token map 直接加载；仅显式 unfused baseline materialize expert-major G1/SwiGLU A4。 |
| paired G1 + SwiGLU + direct A4 | `layers/qwen3_5/moe.py` 的 `_paired_gemm1_swiglu_a4`、`kernels/moe.py` 的 `nvfp4_paired_gemm1_swiglu_a4_kernel` | `reference/moe.py`、`unit/test_moe.py` | 同一 CTA 计算 gate/up；每支 G1 后先 FP16，SwiGLU 后再 FP16，直接量化为 GEMM2 所需 A4，不 materialize 完整 gate/up/SwiGLU FP16 tensor。A4 local scale 是 E4M3，packed E2M1 偶数 K 在低 nibble。 |
| persistent G1 cap/scratch | 同上 | `unit/test_moe.py`、`bench/benchmark_moe.py` | 仅允许 CTA cap 80/160/320，默认 320；每 program 一个 `32×32` FP16 scratch（2 KiB），最大 `320×2 KiB=640 KiB`。这是 scratch 上限，不等于所有临时/输出 allocation。 |
| G2 与 expert-specific down globals | `layers/qwen3_5/moe.py` 的 `_grouped_gemm`、`kernels/moe.py` 的 `nvfp4_grouped_gemm_kernel` | `reference/moe.py`、`unit/test_moe.py` | down A4 input global 和 down weight global 都是物理 `[256]`，按 expert row 选择；local E2M1×E4M3 decode 后 FP32 GEMM，global reciprocal 在累加后施加；G2 输出先舍入 FP16，再乘 route weight。 |
| 固定顺序 FP32 combine、shared/residual | `layers/qwen3_5/moe.py` 的 `execute_experts`/`fused_*_moe`、`kernels/moe.py` 的 `route_combine*_kernel` | `reference/moe.py`、`unit/test_moe.py`、`integration/layer_scan.py` | inverse route 恢复后按固定 Top-8 顺序 FP32 累加。shared expert 为 FP16 `2048→512→2048`；shared scalar gate 的 sigmoid 先 FP16 再乘 shared output；routed+shared 和 residual add 保留各自 FP16 边界。 |
| checkpoint 规定的 FP16 routed 路径（layer 0/39） | `layers/qwen3_5/moe.py` 的 `execute_fp16_experts`/`fused_fp16_moe`、`kernels/moe.py` 的 `fp16_*` | `reference/moe.py`、`unit/test_moe.py` | 仍使用全部 256 physical experts、同一 stable dispatch/Top-8/combine 合同；这是 checkpoint 规定的 FP16 层，不是 NVFP4 的降级 fallback，也不使用 NVFP4 global-scale 路径。 |
| ExpertPack slot-aware NVFP4（layer 1–38） | `layers/qwen3_5/expert_pack/store.py`、`layers/qwen3_5/moe.py`、`kernels/moe.py` | `unit/test_expert_pack_store.py`、`unit/test_moe.py`、单卡 acceptance | router IDs/weights 保持 global 256-expert 语义；lease 提供 `expert_to_slot[256]`；cache 可大于 2048 slots，所有物理 slot 地址先转 int64；完整生成黄金 token 已通过。 |
| 单请求 Expert Activation Trace | `layers/qwen3_5/expert_trace.py`、`runner.py`、`single_gpu.py` | `unit/test_expert_trace.py`、`unit/test_single_gpu.py`、SM70/SM89 API smoke | 请求显式开启后按 sampled-output attribution 保存 `[N,40,8] uint8` global expert IDs；row 0 对应 prompt 最后位置，后续 row 对应前一生成 token 的 decode；JSON commit marker 与 NPZ SHA-256 已验证，trace 发布失败会熔断单卡 backend。 |
| MoE benchmark | `bench/benchmark_moe.py` | 已提交的 `reports/*.json` | real layer 1、natural/forced routes、T=1/4/32/128/512/2048、fused 与 explicit unfused 对照。 |

## 实层扫描和后端审计

| 核对项 | 源码 | 验证 | 当前判读 |
|---|---|---|---|
| 40 层独立扫描 | `integration/layer_scan.py` | `integration/test_layer_scan.py`、CLI `--layers 0-39` | 投影同时记录 `_math` 与 `_semantic` 两个 2e-3 gate；组成层使用 block-semantic FP8 reference；MoE 强制 256 expert route 逐 expert 报告。 |
| M3 参考 source hash | `integration/layer_scan.py` | 每层 JSON `source_sha256` | hash 覆盖生产 Qwen3.5 Python、独立 reference 与 scanner，报告必须和 source hash/contract 一起解释。 |
| 完整后端 PTX 审计 | `bench/backend_audit.py` | `unit/test_backend_audit.py`、backend audit JSON | profiler CUDA compute events 必须精确匹配本进程 PTX entry；memory event 单列；framework/cublas/cutlass/未知 compute 失败。专门化 entry 名需原样写进 allowlist。 |
| 完整 40 层 cached/stateless 对照 | `pipeline.py`、`integration/pipeline_acceptance.py` | 手动双卡 acceptance CLI | 固定短序列逐步 greedy 一致；hidden/logits NRMSE 和最大误差、router IDs/概率差异为独立诊断，不能用局部 5e-3 预算冒充完整模型误差验收。 |
| 双卡生命周期和显存 | `integration/pipeline_acceptance.py` | 手动双卡 acceptance CLI | A→reset→B→reset→A、chat 双跑、2048 fresh prefill、首 decode、重复 reset、epoch/step/容量错误与半步失败恢复；结果写入单份 JSON。 |
| 格式/导入 | `python/sglang/srt/layers/qwen3_5/`、`models/qwen3_5_moe.py`、`test/Qwen3_5MoECompat/` | `isort==5.13.2 --check`、`black==24.10.0 --check`、`py_compile` | 格式检查不替代 GPU 精度或性能验收。 |

## 执行前后核对

1. ExpertPack 运行前确认 manifest `complete=true`、pack size/identity 和源 checkpoint 的 config/index/shard inventory；完整离线校验与运行时逐记录 SHA 是不同证据。
2. 单卡入口只暴露一个受支持 UUID，并确认 `device_count=1` 与 capability `(7,0)`/`(8,9)` 配对；4070 命令必须覆盖默认 cache 为 3584 MiB。
3. 验收证据必须分别检查黄金 token、cache 实际字节、Store error 和 `total_memory-peak_reserved>=1 GiB`；A/B/A、2048 full-capacity 与故障恢复仍以 V100 完整 acceptance 为准。
4. 实时 HTTP 200/429 与 checksum/短读/H2D→FAILED/503 以对应 V100 JSON 为证；SM89 当前覆盖正常 offload、流式 API 与 benchmark。连接取消、fatal 后新进程重启与长稳压仍保持待验证，不能用 mock backend 单测提升状态。
5. 历史路径运行前确认 required interpreter、`PYTHONPATH=python:.` 和单卡测试的 UUID/capability 配对；双 worker 分别启动，不能在同一进程切换 Triton target。
6. GPU unittest 自己持有对应 `/tmp/qwen35-gpu-<UUID>-sm<capability>.lock`；不要再套 shell `flock`。
7. standalone layer scan 和 benchmark 自己持锁；同样不要外层加锁。
8. 检查 M3 JSON 的 `math_contract`、`source_sha256`、每个 `_math`/`_semantic` projection gate、强制 256-expert route 和 nonfinite 字段。
9. 检查 projection/MoE/backend benchmark 的 token 覆盖、actual GPU、kernel histogram 与 peak allocation；不要把不同 source hash 或不同 UUID 的报告混用。
10. M3/M5 历史证据见 [VALIDATION.md](VALIDATION.md)，双卡 cache 分支记录见 [VALIDATION_SM70_SM89_PIPELINE.md](VALIDATION_SM70_SM89_PIPELINE.md)。既有四层或双卡数字不能充当 ExpertPack 单 GPU 验收。
