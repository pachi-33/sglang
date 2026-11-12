# PP chunk 流水时间线补丁

在每个 PP stage 的各 NPU rank 记录 prefill chunk 的 forward 起止，本地保存 JSONL，离线汇总为 Chrome Trace / Perfetto 可打开的 `trace.json`。不加载另一个模型，不开启完整 profiler，不修改 SGLang 源文件。

当前适配 `experiment/ascend-ops-moe-offload-review` 的 `58e353d08c` 执行边界。启动前按函数内容检查兼容性；只改了其他模块的提交不影响检查。支持 NPU eager、PP>1、非 speculative 的 `EXTEND` prefill，支持单机和跨机，MoE 后端可用 Ascend Ops 或 DeepEP。不能把它当作任意上游版本都可直接使用的补丁。

## 时间口径

`CHUNK_FORWARD` 在 PP 已建立输入依赖之后、`Scheduler.run_batch` 调用前向计算流记录 start；调用返回后向同一流记录 end。读出的是两个事件真正执行的设备时间，不是 Python 返回时间，也不是轮询发现完成的时间。

该跨度包括前向内 Attention/FFN、CP/EP 通信依赖、权重等待、Host 下发间断及此调用中的其他准备/结果工作。它不是纯 AI Core 计算时长；最后 stage 的实际输出处理可能与其他 stage 不同。异步分支只覆盖计算流已经等待的依赖，不额外排空所有 H2D/通信流。`run_batch` 内可能发起缓存前缀的 P→D KV 传输，本补丁不为它另计持续时间；它对该调用的实际扰动仍可能体现在记录里。

- **交接间隔**：同一个 chunk，下游 forward 开始 − 上游 forward 结束。含 PP 传输、发送前调度和下游排队，不能作为纯 HCCL 时长或带宽分母。
- **本 stage 空档**：该 stage 的 forward 区间并集之外的时间。可能包含通信、CPU 调度、KV 传输或其他工作，不等于设备空闲。
- **stage 汇总**：同一 chunk 在该 stage 各 CP/TP rank 的最早开始到最晚结束。另保留每张 NPU 的明细轨道。stage 包络可能重叠，统计采用区间并集。
- **整体窗口**：本次采集最早的已记录 forward 开始到最晚结束，不包括 HTTP 排队、tokenization 和客户端收首 token，不能直接叫 TTFT。

跨 stage 的交接和各 stage 空档可能重复覆盖，不能相加当总损失。不同 stage 的包络受 rank 间偏差影响，交接差值可能为负；保留原值并标注，绝不强行归零。

## 请求完成标记

如果服务端提供 `_pp_prefill_completion_event` 钩子，新补丁会在原有 NPU 时间线旁增加每 rank 的 **Host completion** 轨道；旧服务端没有这个钩子时仍只记录原有 forward。只有 control start 到 finish 的窗口内才记录，不需要完整 profiler。

| 标记 | 含义 |
|---|---|
| `result_processed` | 最后一个 prefill chunk 的 CPU 后处理已经返回，请求已进入待完成 KV 传输队列；中间 chunk 不记录 |
| `release_ready` | 调度器已收到该请求的 PP 完成共识，本地请求仍在 inflight 队列；不表示资源已经释放 |
| `response_enqueued` | 完成队列处理已返回，响应已经交给输出路径；**不表示 HTTP 已发完或客户端已收到** |

这些是 Host 单调时钟的观察点，用同一组 Host→参考时钟锚点映射到全局时间轴，保留时钟误差。它们不是额外 NPU event，也不将 CPU 后处理时间冒充设备计算结束。比较同一 rid 的最后一个 NPU forward 结束、`result_processed`、`release_ready` 和 `response_enqueued`，可区分结果后处理与完成确认等待；HTTP 返回到下一条 POST 的间隔仍需客户端时间戳。

完成标记可能晚于最后一个 forward。`summary.csv` 的占比和窗口仍只按 forward 计算，不把这些标记算作计算或扩展占比分母。中途开始采集也可能看到没有对应 forward 的请求标记，不能据此补造缺失的计算区间。

## 安装

把整个目录同步到所有 P 节点，建议放在 SGLang 仓库 `tools/pp_chunk_trace`。在运行服务的同一个 Python 环境中：

```bash
cd /path/to/sglang/tools/pp_chunk_trace
python3 install.py install
python3 install.py status
python3 -m pp_chunk_trace.compat /path/to/sglang
```

安装只在当前 Python 的 site-packages 注册一个有开关的 `.pth` 文件。它引用这个目录的绝对路径，所以安装后不要搬动目录；重装前先卸载旧注册。`--target SITE_PACKAGES` 可显式选择目录。不要仅为这份补丁更换服务的 Python 环境。

安装不自动开启记录。只有 `PP_CHUNK_TRACE_ENABLE=1` 时，Python 子进程启动才注册导入钩子，在 SGLang Scheduler 加载后包装方法。父进程和 spawn 子进程均适用；源码版本不匹配会明确拒绝启用。不要只在部分 PP/CP/TP rank 启用：trace 身份随现有 PP proxy 元数据传递，所有 P rank 必须使用同一补丁和运行编号。

## 单机采集

在启动 P 服务的终端设置，随后执行原启动命令：

```bash
export PP_CHUNK_TRACE_ENABLE=1
export PP_CHUNK_TRACE_RUN_ID=offload-128k-001
export PP_CHUNK_TRACE_DIR=/tmp/pp-chunk-traces
unset PP_CHUNK_TRACE_CLOCK_ADDR

# 执行原有 P 节点启动命令，参数保持不变。
```

服务 ready 后，在同一机器、同一个 Python 环境的另一个终端执行：

```bash
python3 -m pp_chunk_trace.control start \
  --directory /tmp/pp-chunk-traces --run-id offload-128k-001
```

看到 `pending: []`、`errors: []` 后，再发起要测的请求。启动预热不记录；start 之前的请求不能用于此次完整流水对比。可以在一次采集中运行多个串行或并发请求。

待所有目标请求完成，再执行：

```bash
python3 -m pp_chunk_trace.control finish \
  --directory /tmp/pp-chunk-traces --run-id offload-128k-001
```

finish 等待在途事件写出，补充结束时钟校准并返回状态。若返回非零、`pending` 或 `errors` 非空，保留日志并按不完整采集处理。不要先强杀进程再 finish。每次进程运行只支持一个 start/finish 窗口；下一轮重启服务并换 RUN_ID，避免拼入旧数据。

## 双机/多机采集

跨机器使用一台参考时钟服务。先在各 P 节点可访问的机器上运行，`REF_IP` 换成该机器的实际 IP：

```bash
python3 -m pp_chunk_trace.clocks serve --host REF_IP --port 29681
```

所有参与同一次流水的 P 节点统一设置：

```bash
export PP_CHUNK_TRACE_ENABLE=1
export PP_CHUNK_TRACE_RUN_ID=baseline-128k-001
export PP_CHUNK_TRACE_DIR=/tmp/pp-chunk-traces
export PP_CHUNK_TRACE_CLOCK_ADDR=REF_IP:29681

# 各节点执行各自原有的 P 服务启动命令。
```

服务就绪后，**每台 P 机器分别执行一次 control start**，都返回成功后发请求；请求完成后，每台分别执行 control finish。参考时钟服务在所有节点 finish 完成前保持运行。它重启会改变 clock_id，合并器会拒绝混合。

参考服务只交换有长度限制的时间采样 JSON，不接收模型输入；部署在可信的实验网络。协议没有认证，不能放到公网。所有校准使用单调时钟，不依赖机器显示的日期或 NTP 墙钟一致。

每张 NPU 用独立空 stream 的事件建立 device→Host 锚点；Host 通过多次往返取最小 RTT 样本映射到参考机器。保留校准误差和起止锚点，离线做线性映射。每个 chunk 不进行时钟网络交互、barrier 或 synchronize。若误差大于交接间隔，CSV 标 `uncertain`；该结果不能用来判断微秒级网络性能。长时间漂移及网络非对称无法由此完全消除。

## 汇集与转换

将各机器的对应 RUN_ID 目录拷贝到一个本地目录，保留各文件名即可，不要求共享盘。例如：

```text
collected/baseline/node0/*.jsonl
collected/baseline/node1/*.jsonl
```

在任意有 Python 3.9+ 的机器上运行（转换不需要 torch、NPU 或 SGLang）：

```bash
cd /path/to/sglang/tools/pp_chunk_trace
python3 -m pp_chunk_trace.merge /path/to/collected/baseline \
  --output /path/to/baseline-timeline
```

基线与卸载实验分别合并，不把两次独立实验的起止直接相减。若输入目录包含多个 RUN_ID，需要用 `--run-id baseline-128k-001` 选择。

输出：

| 文件 | 内容 |
|---|---|
| `trace.json` | PP stage 汇总轨道、NPU rank 明细、chunk 标签、空档和交接箭头 |
| `summary.csv` | 公共窗口、forward 并集、forward 外占比、填充/排空和中间空档 |
| `chunks.csv` | 每个设备每个 forward 的起止、耗时和实际 token 数 |
| `handoffs.csv` | 每次跨 stage 交接差值、时钟误差及完整性标记 |
| `completions.csv` | 每 rank、每请求的 Host 完成标记、相对时间和时钟误差；旧服务端为空表 |
| `warnings.json` | 缺 rank、缺 stage、截断、错误、时钟不确定性等 |

用 Chrome tracing 或 Perfetto 打开 `trace.json`。纵轴按 `DP lane / PP rank` 分组，默认汇总轨道显示 `R2 C5 (16384 tokens)`；展开看各 CP/TP rank。横轴是统一相对时间。事件详情有原始 rid、token 范围、attempt 和 batch_id，不保存 prompt 内容。chunk 序号来自该请求真实调度过的 token 范围，不用固定 chunk size 除法猜测；重试同一范围保留序号并增加 attempt。

token 数表示 CP 切分前、该 attention-DP 请求组的真实逻辑 chunk 大小。CP4 的四张 NPU 都会标同一 chunk 的 16384 tokens，这不表示每张卡都独立处理了 16384 个有效 token，不能跨 CP/TP 行累加 token 数。

一次 forward 若打包多个请求，只画一个执行条，并列所有请求片段；统计不重复加算。可见的 stage 包络与每设备条是两套视图，不应再求和。

## 开销与故障边界

默认每 rank 预分配 128 对 timing events。可以在启动前用 `PP_CHUNK_TRACE_MAX_PENDING=256` 调整；容量用完时丢记录并在 finish 标出，不阻塞模型等待空槽。记录线程只在 end event 完成后读取设备时间并写本地 JSONL；事件未完成不复用。Host 完成标记使用独立有界队列，容量为 `PP_CHUNK_TRACE_MAX_PENDING × 8`，不等待未完成的 forward 事件；排满时丢弃并在 finish 标记采集不完整，不阻塞生产线程。Host 标记只入队，由记录线程写盘，不增加设备同步或逐条日志。

主执行路径新增两个 event record、CPU 元数据快照和小型 PP 消息头，没有新增 collective、逐 chunk wait/synchronize、trace 专用全流排空或逐条打印。记录仍有实际开销，是否可接受需要在 NPU 上对照同负载不开/开补丁；CPU 检查不能证明开销百分比。

异步设备错误可能使部分记录没有结束；不会用 Host 观察时间补造设备完成时间。收到 finish 后不再接纳新的记录，但推理服务可继续运行。强杀进程可能丢失尚未写出的尾部，合并器明确提示没有 session_end；不完整 stage 不输出气泡占比，避免把丢记录画成真实空档。clock 校准失败不降级到另一时钟域假装对齐。不同 rank 的 RUN_ID 或 chunk 身份不匹配会明确拒绝继续该记录路径，因此务必在全部 P rank 一致启用。

本版不细分层内 H2D/算子气泡，不测纯 PP 链路带宽，不统计请求控制消息或 P→D KV 传输区间，也不重排生产 stream。要解释某个空档的根因，结合一份短 profiler；该补丁首先回答“哪一个 stage、哪一个 chunk、哪里出现交接和空档”。

## 关闭与卸载

```bash
# 下次启动服务不启用；已运行进程需重启才还原包装。
unset PP_CHUNK_TRACE_ENABLE

cd /path/to/sglang/tools/pp_chunk_trace
python3 install.py uninstall
python3 install.py status
```

卸载只移除本工具拥有的 `.pth` 注册，不改服务源码、其他补丁、已保存的 trace 或测试目录。

## CPU 检查

```bash
cd /path/to/sglang/tools/pp_chunk_trace
python3 -B -m unittest discover -s tests -v
```

覆盖独立时钟/协议、安装和子进程导入、PP 身份传播、异步事件采集和离线合并。实际 NPU 的事件计时、跨机校准误差、forward 时长与 profiler 一致性、以及 TTFT 扰动仍需服务器实测。
