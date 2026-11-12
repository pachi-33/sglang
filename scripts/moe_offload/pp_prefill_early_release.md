# PP prefill 请求提前收尾

本功能默认关闭，只改变 PP prefill 的 CPU 控制阶段顺序。开启后，在下一批的 proxy 接收、输出回传或 D2H 等待之前，处理已经获全 PP 完成确认的请求，并推进请求/传输状态发送。它不是后台持续轮询，也不保证去除全部流水线气泡。

## 使用

所有 P 节点、所有 PP/CP/TP rank 必须使用同一版代码和相同开关。重新启动 P 服务，例如在仓库根目录：

```bash
export SGLANG_PP_PREFILL_EARLY_RELEASE=1
bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh
```

启动日志应包含 `PP prefill early release: True`。基线设为 `0` 并重启；该变量在 PP prefill 循环入口读取，不支持运行中切换。第一版要求关闭 speculative decoding；开启该功能同时启用 speculative 会明确拒绝。

bench 保持 `--request-rate inf --max-concurrency 4`，输出长度为 1，保持缓存预热、chunk size 和输入不变。客户端在单条 HTTP 响应结束时已会补入下一条请求，无需修改补入机制。

## 完成条件和改动边界

- 保留原来的两阶段 PP 完成共识，包含任一 stage 失败则整体失败的语义。
- `None` 表示没有共识，不能传给非 PP 的无筛选退役模式；空字典表示完成本轮共识但没有可释放请求。
- 保持每条 CPU 链的 `release → requests → transferred` 消息顺序，以及所有 CP/TP 成员相同的 collective 顺序。
- HCCL 输出张量收发、奇偶 rank 收发顺序、forward fence 和 proxy send 不变。
- 仍由原函数释放 KV、结束缓存句柄并输出响应。真实 PD 必须等当前请求的 KV 传输完成；fake bootstrap 也不跳过 PP 共识。
- 本轮刚后处理、刚进入 inflight 的请求要等后续共识，不提前释放。

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
python3 -B -m unittest discover -s tools/pp_chunk_trace/tests
```

CPU 测试覆盖控制消息顺序、阻塞前退役、默认关闭路径、未获共识不释放，以及多 PP/异步深度下的协议进展模拟。这不能替代服务器上的 NPU/HCCL 验证；需用同环境 A/B 确认实际收益与长时间运行正确性。
