# PP prefill 请求提前收尾

本功能默认关闭。开启后，在原有 PP 释放共识交换结束时，立即处理已经确认完成的请求，不再等另一批结果的 D2H 同步和后处理。请求、激活、输出和释放确认的通信顺序保持不变。它不是后台持续轮询，也不保证去除全部流水线气泡。

## 使用

所有 P 节点、所有 PP/CP/TP rank 必须使用同一版代码和相同开关。重新启动 P 服务，例如在仓库根目录：

```bash
export SGLANG_PP_PREFILL_EARLY_RELEASE=1
bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh
```

如果自定义启动脚本内部也给此变量赋值，以脚本内最终值为准，外部环境变量不能覆盖脚本中的无条件 `export`。

启动日志应包含 `PP prefill early release: True`。基线设为 `0` 并重启；该变量在 PP prefill 循环入口读取，不支持运行中切换。当前要求关闭 speculative decoding；开启该功能同时启用 speculative 会明确拒绝。

bench 保持 `--request-rate inf --max-concurrency 4`，输出长度为 1，保持缓存预热、chunk size 和输入不变。客户端在单条 HTTP 响应结束时已会补入下一条请求，无需修改补入机制。

## 完成条件和改动边界

- 保留原来的两阶段 PP 完成共识，包含任一 stage 失败则整体失败的语义，并保留本地传输终态复核。
- `None` 表示没有共识，不能传给非 PP 的无筛选退役模式；空字典表示完成本轮共识但没有可释放请求。
- CPU 控制消息和设备收发的相对顺序保持不变，包括 `release → requests → transferred` 的 CPU FIFO、proxy 接收/发送、output 环的奇偶顺序以及 PP2 NPU 专用交换。
- 唯一调整是：拿到释放确认后，先调用原请求退役函数，再等待无关结果的 D2H 和后处理。KV 释放、缓存句柄结束和响应输出仍由原函数完成。
- 请求只有最终 chunk 结果处理完成后才进入 inflight。本轮 release 来自此前的传输状态快照；本轮刚后处理、刚进入 inflight 的请求不能借旧共识提前释放。
- 真实 PD 成功响应仍须通过 KV 传输完成检查；失败沿用原失败处理路径。fake bootstrap 也不跳过 PP 共识。
- 仍可能等待原有 proxy/output 通信和释放共识，不能把该功能描述成“响应早于所有设备等待”。

## 修复的死锁

旧版开启此开关时，把整段 release 交换和控制发送移到 proxy/output 接收之前。空转后首个非空 batch 可出现：上游卡在本轮 HCCL `isend` 初始化，等待下游接收激活；下游却先等上游下一轮 release；上游要等当前发送返回才能进入下一轮。仅保持 CPU FIFO 顺序不足以避免跨 CPU/HCCL 的等待环。

修复后，release 的发送、接收和 commit 都回到原位置，位于本轮 proxy/output 交换之后。不通过增加超时、改变端口或强制预热通信域掩盖这个依赖问题。

## 时间标记与对照

更新并按 `tools/pp_chunk_trace/README.md` 安装/启动 chunk trace 后，增加以下 Host 时间标记：

- `result_processed`：最终 chunk 结果已在本 stage 后处理并进入 inflight 队列。
- `release_ready`：该请求出现在收到的 PP 释放确认中；仍需原本地状态检查。
- `response_enqueued`：原退役函数返回该请求，输出路径已执行。它不表示 HTTP 已送达，也不保证每个 CP/TP rank 都是对外响应者。

比较同一 RID 的 NPU 最终 forward 结束、上述三个标记以及下一条请求进入 PP0 的时间。HTTP 收到响应和下一条 POST 的精确边界仍需客户端时间戳，不能从 `response_enqueued` 推定。

重点检查并发 4 下周期性的 PP0 空档、正式阶段吞吐、平均/P95 TTFT。排除单独预热及启动时间。不要把 NPU forward envelope 当作硬件利用率。

## 验证

```bash
python3 -B test/registered/unit/managers/test_pp_prefill_early_release.py
python3 -B test/registered/unit/managers/test_pp_prefill_tensor_protocol.py
(cd tools/pp_chunk_trace && python3 -B -m unittest discover -s tests)
```

CPU 回归提取生产调度代码，模拟 CPU FIFO 和设备 proxy/output 配对，包括首次 `isend` 等待接收、空转后首个请求、连续/稀疏请求、输出回传、PP2 专用交换和不同 PP/异步深度；另检查已确认请求先于无关 D2H 返回、未确认请求不释放和旧开关路径不变。

这些测试不能替代 NPU/HCCL 验证。服务器上先完整重启，检查首次请求能完成；等待空闲后再次请求，再测连续并发 4，确认所有请求返回且没有超时。实际 PD 还需核对 KV 传输完成后才响应。用同环境 A/B 和原 chunk trace 验证收益，不把协议模拟当成硬件性能结果。
