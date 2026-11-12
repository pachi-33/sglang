# PP prefill 请求提前收尾

本功能默认关闭。开启后，在原有 PP 释放共识交换结束时，立即处理已经确认完成的请求，不再等另一批结果的 D2H 同步和后处理。请求、激活、输出和释放确认的通信顺序保持不变。它不是后台持续轮询，也不保证去除全部流水线气泡。

## PR #38309 的另外两个开关

本分支合入了 [PR #38309](https://github.com/sgl-project/sglang/pull/38309)，来源提交为 `8cc6cf74ea75e456e743c96eea9d3a8b0abf5d79`。下面两个开关在代码中默认关闭，当前 `serve_glm52_prefill_pp4_cp4.sh` 已显式开启它们：

| 开关 | 行为 |
| --- | --- |
| `SGLANG_DISAGG_LAYERWISE=1` | 在支持的模型 forward 中，attention 写好一层 KV 后提交该层传输；最终的状态、元数据仍由结果后处理发送。 |
| `SGLANG_PP_EARLY_RELEASE_KV=1` | 每次 PP 循环开头，用当前 stage 已汇总的传输状态收尾请求，不等待原来的全 PP 确认回传。对 PP0 来说，此时只有本 stage 的完成状态。 |

这两个开关与本文原来的 `SGLANG_PP_PREFILL_EARLY_RELEASE` 不同。原开关保留全 PP 完成确认，仅调整确认之后的收尾顺序；PR 的 `SGLANG_PP_EARLY_RELEASE_KV` 会改变等待条件。下文关于“保留全 PP 共识”的说明，适用于 PR 的这个开关关闭时。

PR 没有把尾 stage 的结果回传改成广播，也没有取消 `next_mb_id` 决定的结果处理顺序。真实 PD 中，P 的 HTTP 响应交给 router，客户端的生成结果来自 D；直接请求 P 的 fake benchmark 则把 P 的 HTTP 结束作为释放并发名额的时刻，两者不能等同。

需要测试时，在所有 P rank 的启动环境中设置对应开关，例如：

```bash
export SGLANG_DISAGG_LAYERWISE=1
export SGLANG_PP_EARLY_RELEASE_KV=1
bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh
```

分层传输需要真实 decode 节点才能验证；fake sender 不会实际发送分层 KV。合入时保留了本分支的 PP 通信顺序修复、KV checksum、CP 索引处理和普通传输的异步索引拷贝。CPU 检查不能替代 NPU/HCCL 下的传输正确性与性能验证。

### Ascend 缓冲区与后台传输修复

逐层发送现在会包含同一层的全部已注册缓冲区（K、V、index K 等），沿用 Ascend 的 PP 层映射。同步传输接口在独立后台线程执行：线程先等待这一层 attention 的设备事件，再读取 KV；forward 只提交任务，不等待网络传输，也不再在 MoE matmul 后等待传输。

每个请求分别保存任务的完成状态。结果后处理、最后一次发送和释放 KV 前，会等待该请求的后台任务；这仍可能等待本请求的慢传输，但不会等待另一个请求后来覆盖的全局完成事件。后台任务也参与取消确认、内存整理和后端退出的检查，避免仍在读写的页被复用。

逐层传输的索引范围现已使用 `req.start_send_idx:min(req.extend_end, len(req.origin_input_ids))`，避免分块 prefill 读取尚未分配的后续 KV 索引；decode 目标页偏移则相对于 `req.disagg_decode_prefix_len` 计算。保留现有最终 chunk 推进发送进度的逻辑，因此后续 chunk 仍可能重复发送前面已计算的 KV。

此前 review 指出的 PP 提前释放确认丢失问题尚未处理，不能据此认为这两个开关的所有场景都已验证。

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
