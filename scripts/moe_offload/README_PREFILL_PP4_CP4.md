# 单机 A3，仅 Prefill 卸载测试

`serve_glm52_prefill_pp4_cp4.sh` 使用 16 die，固定为 PP4 × TP4、Attention DP1 / CP4 zigzag、stage 内 EP4。只启动 P 服务，关闭 MTP，测试每请求输出 1 个 token。脚本沿用 Ascend KV manager，请求使用已有 fake sender 跳过真实 D 对端；不需要 D / Router。

先激活安装了 SGLang 依赖、torch_npu 和 MemFabric 的 Python 环境，设置本机 GLM-5.2 ModelSlim **W8A8** 权重目录：

```bash
export MODEL_PATH=/models/GLM-5.2-w8a8
bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh --dry-run
bash scripts/moe_offload/serve_glm52_prefill_pp4_cp4.sh
```

`--dry-run` 只读取模型 `config.json` 并检查、打印配置；不需要 NPU 依赖，不创建运行文件。正常启动会在 `LOG_DIR` 生成 `offload.json`，使用 `mode=prefill_only`、`cache_ratio=0`、两个完整本地专家层的 bank；256 专家 / EP4 时，默认 `execution_slots=128`。实际格式和容量继续由运行时代码检查。

按需设置：

| 环境变量 | 默认值与用途 |
| --- | --- |
| `HOST_POOL_GIB` | 每 rank 64 GiB；16 rank 的物理 Host pool 合计 **1024 GiB**，系统和 checkpoint 加载还需额外内存 |
| `SGLANG_PP_LAYER_PARTITION` | `18,20,24,16`；须为四个正数，和等于模型层数，各 stage 至少有一个 routed MoE 层 |
| `CHUNKED_PREFILL_SIZE` / `MAX_PREFILL_TOKENS` | `8192` / 与 chunk 相同；按目标输入长度及 HBM 容量调整 |
| `CONTEXT_LENGTH` / `MAX_RUNNING_REQUESTS` | `16384` / `8` |
| `MEM_FRACTION_STATIC` | `0.8` |
| `SERVER_PORT` | P HTTP 默认 `8000`，默认监听 `0.0.0.0` |
| `P_IP` / `COMM_IFACE` | 单机内部通信默认 `127.0.0.1` / `lo`，不用 `ip`、`ifconfig` 或 `hostname` 工具 |
| `PYTHON_BIN` | 当前环境的 `python3`；仅需要固定解释器时填写绝对路径，已有 `PYTHONPATH` 会保留 |
| `RUNTIME_DIR` / `LOG_DIR` | `/tmp` 下的扩展缓存 / 日志目录；升级 CANN、torch_npu、MemFabric 后换新的 `RUNTIME_DIR` |

CANN 等安装位置非标准时，设置脚本顶部的四个 `*_ENV` 路径。更小的 `HOST_POOL_GIB` 必须能容纳本 stage 的所有本地专家及量化元数据；脚本会先做 W8 权重本体估算，不能把这项估算当成完整内存预算。

服务 ready 后，请求直发 P 的 `/generate`：

```bash
curl -sS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "请解释流水线并行、上下文并行和专家并行如何协同完成长文本的预填充计算。",
    "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": true},
    "bootstrap_host": "2.2.2.2",
    "bootstrap_room": 0,
    "stream": false
  }'
```

其他机器发请求时，将 URL 中 `127.0.0.1` 换成 P 机器可达的实际 IP。`bootstrap_host=2.2.2.2` 是源码中的 **fake 测试标记**，保持这个值，不换成真实 IP。仅设置输出长度为 1 不能自动绕过 PD 握手。

独立 P 启动脚本会显式设置 `SGLANG_TEST_FAKE_PREFILL_RADIX_CACHE=1`，以便测试跨请求 prefix KV cache；脚本内的设置会覆盖启动前同名环境变量。运行时的开关默认值仍是关闭，其他启动方式需要在启动服务前自行设置为 `1`。

开关仅影响 Prefill 模式下使用 `bootstrap_host=2.2.2.2` 的请求。打开后，此类请求计算出的 P 侧 KV 可以写入本地 radix cache；fake sender 仍然不会发送 KV，也不需要 D。请求仍须携带 `bootstrap_room`。服务启动时的 fake 预热请求也受此开关影响，因此它适用于隔离的基准测试进程。

测试命中时，等待预热请求**成功完成**后，再发相同 token 前缀的请求；通过响应 `meta_info.cached_tokens` 或服务日志 `#cached-token` 观察实际命中。`--page-size 128` 会按页对齐缓存，实际命中数不一定恰好等于目标前缀长度。普通 Prefill / fake Decode 默认行为，以及 embedding override 请求的缓存保护保持不变。

`scripts/moe_offload/simple_bench.sh` 的 `COMMON_ARGS` 已包含以下两项，可用于直连独立 P；使用前将地址、端口和输入长度改成当前 P 服务配置：

```bash
--extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'
--cache-report
```

脚本中的 `KV_CACHE_HIT_RATE` 是生成共享前缀的目标比例。真正的命中率由服务返回的 cached token 数决定；请求失败、KV 容量不足或生成的前缀不同都会影响结果。使用真实 PD Router 测试时须从脚本移除 fake bootstrap 字段。`simple_bench.sh` 内的 `start_profile` 使用独立的固定地址，执行该步骤前也需改成实际 P 服务地址或跳过它。

可先发送两个共用 512 个 token 前缀的请求，检查第二个请求的缓存计数（将 URL 换成实际 P 服务地址）：

```bash
python3 - <<'PY_SMOKE'
import json
import random
import urllib.request

url = "http://127.0.0.1:8000/generate"
rng = random.SystemRandom()
prefix = [rng.randrange(100, 3000) for _ in range(512)]
for index in range(2):
    body = {
        "input_ids": prefix + [10 + index] * 128,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
        "bootstrap_host": "2.2.2.2", "bootstrap_room": 0, "stream": False,
    }
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    meta = result.get("meta_info", {})
    if result.get("error") or (meta.get("finish_reason") or {}).get("type") == "abort":
        raise RuntimeError(result)
    print(f"request={index + 1} prompt_tokens={meta.get('prompt_tokens')} cached_tokens={meta.get('cached_tokens')}")
PY_SMOKE
```

在缓存开启、没有驱逐且 KV 充足的情况下，第二个请求应能看到非零命中；这个测试使用 page 对齐的共享前缀。若依旧为零，请核对实际服务是否使用当前代码、是否通过该脚本启动，以及请求是否成功完成。

第一个 token 在 prefill 结束后由 logits 采样产生，不需要 decode forward。长输入仍可包含多个 prefill chunk。极短输入可能回退到不切 CP 的路径；验收 CP4 时用目标长度输入，并检查 forward / CP 元数据。与同拓扑常驻结果比较 token ID / logits，确认只输出 1 token、没有本请求的 decode forward，且连续请求结束后 KV / metadata / bank 正常回收。

本配置需要目标 NPU 上的数值、PP/CP 通信和性能验收。Fake sender 不传输真实 KV，独立 P 测试通过不代表真实 P→D 传输、D 多 token 生成或 Router 联调已经通过。
