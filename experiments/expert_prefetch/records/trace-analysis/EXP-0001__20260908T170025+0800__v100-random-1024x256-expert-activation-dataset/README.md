# EXP-0001: V100 random 1024x256 expert activation dataset

## 实验身份

- 类型：`trace-analysis`
- 开始时间：`2026-09-08T17:00:25+08:00`
- 结束时间：`2026-09-08T19:26:02+08:00`
- 状态：`completed`
- 设备：Tesla V100-SXM2-16GB `GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96`
- Git：`lab/expert-prefetch@fdf52e5e1962eb6d70f06f038cef09df7818bf48`，dirty=false

## 问题与假设

为后续 decode 阶段专家预测算法建立固定输入和可复现的路由真值数据集。假设当前 ExpertPack demand-loading 路径能够连续完成 128 个严格 1024-token 输入、256-token 输出的请求，并为每个输出位置提交完整 40 层 Top-8 全局专家 ID。

**结论：假设成立。** 128/128 请求成功，全量 artifact、路由 shape、token 归因和身份校验通过。

## 对照与变量

- 对照组：当前无预取的 demand-loading ExpertStore。
- 独立变量：本实验不改变换入策略，只采集路由真值。
- 固定变量：Qwen-AgentWorld checkpoint 和 ExpertPack、V100、7168 MiB cache、16 个 staging slots、2 个 I/O worker、capacity 2048、greedy、ignore EOS、seed 0、并发 1。
- 输入：ShareGPT 首轮文本经 tokenizer 编码后重复或截断为 1024 个 ID，直接以 token ID 数组发送，服务端 usage 确认全部为 1024。
- Trace 语义：每请求记录 256 个输出归因位置；prefill 只保留最后位置，后续 255 行为 decode。
- 干扰因素：bench 先执行一次不采 trace 的启动探测，正式样本开始前 cache 已预热一次。

## 身份摘要

- ShareGPT SHA-256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`
- config SHA-256：`e1be0a1fb619901c1f97afeb75beb2a6581be706e4eb6166d10690104e0b7634`
- checkpoint index SHA-256：`5bd3e4596cf3d20483079f01df50c78946a2fe2aa27251acaa13dff3a06e27ed`
- ExpertPack manifest SHA-256：`bd194286aed4b16814370d80c73878bf3049937de9c88890da6733b7ca6b54c3`
- ExpertPack payload SHA-256：`5d53114a227ed9d7a86e656a557b5c6ffbb9d10f46ddb5ca27671397ccdbe1f5`
- Dataset manifest SHA-256：`99bfb6879608c8fa8eb231b0f16979a91aed4e9c7725af4a17a238c1aeecc07a`

## 结果

### 数据集

- 128 个请求，逻辑 shape 为 `[128,256,40,8]`。
- 32,768 个输出位置、1,310,720 个 token-layer 位置、10,485,760 个 Top-8 专家选择。
- 128 个 JSON 与 128 个 NPZ，总计 10,245,882 bytes；0 个 partial 或未索引 artifact。
- 所有 trace 均为 `status=ok`，prompt/completion 为 1024/256，expert ID 位于 0..255。
- phase、`model_input_positions=[1023..1278]` 和 `model_input_token_ids[1:]==sampled_token_ids[:-1]` 全部通过。
- 原始 trace 位于 `artifacts/traces/`；请求顺序及 artifact SHA 见 `results/dataset_manifest.jsonl`。

### Bench 表征

- 128/128 成功，正式 benchmark 用时 7693.48 秒。
- 请求吞吐 0.016637 req/s，输入吞吐 17.037 token/s，输出吞吐 4.259 token/s。
- mean/median/p99 TTFT：6073.77/6483.62/8029.10 ms。
- mean/median/p99 ITL：211.80/197.71/396.17 ms。
- sampled completion 严格为 32,768 token；流式文本重分词为 32,683，预测数据使用 trace 内 sampled token ID，不使用文本重分词计数。

### Demand-loading 与显存

- cache hit/miss：7,938,286/3,012,510，hit rate 72.4905%，eviction 3,008,263。
- SSD 读取和 H2D 均为 5,330,600,294,880 bytes（4.848 TiB）。
- pack read/checksum/staging 累计时间：595.033/2573.699/90.179 秒。
- I/O、checksum、CUDA、epoch、fatal 错误均为 0；停止前 store 为 READY。
- 最后一条正式请求 peak reserved 14,803,795,968 bytes，距显存总量余量 2,124,546,048 bytes。

### Cold/Warm 辅助表征

使用与正式 request 0 相同 prompt SHA 的无 trace A/A 请求，结果不属于 128 份数据集：

- cold：TTFT 12764.88 ms，mean ITL 92.32 ms，E2E 36307.17 ms。
- warm：TTFT 4612.65 ms，mean ITL 68.89 ms，E2E 22180.33 ms。

## 结论与后续

EXP-0001 可作为 V100 decode 专家预测实验的固定真值集。后续算法实验应引用本实验编号和 dataset manifest SHA，保持 request 顺序、GPU、cache 容量及输入不变；预测输出按 `(request_index, output_token, layer_id, topk_rank)` 与本数据集比较。

## 资源清理

- [x] 正式采集和辅助测量 API 进程均已关闭，8818 端口关闭
- [x] V100 和 RTX 4070 SUPER 均无遗留计算进程
- [x] 测试后显存：V100 1 MiB，RTX 4070 SUPER 2 MiB
- [x] 大体积日志和产物保存在 `logs/` 和 `artifacts/traces/`
