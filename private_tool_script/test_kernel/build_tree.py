import torch
import torch_npu
import sgl_kernel_npu  # noqa: F401  # 加载 libsgl_kernel_npu.so 并注册 torch.ops.npu.*

if not hasattr(torch.ops.npu, "build_tree_kernel_efficient"):
    raise RuntimeError(
        "sgl_kernel_npu 已导入，但 torch.ops.npu.build_tree_kernel_efficient "
        "仍未注册；请确认当前 Python 环境安装的是包含 build_tree 的 "
        "sgl_kernel_npu，并检查其 lib/libsgl_kernel_npu.so。"
    )

torch.npu.set_device(0)
device = "npu:0"

parent_list = torch.tensor(
    [[-1, 0, 1, 2]], dtype=torch.int64, device=device
)
top_scores_index = torch.tensor(
    [[0, 1, 2, 3]], dtype=torch.int64, device=device
)
seq_lens = torch.tensor(
    [6], dtype=torch.int64, device=device
)
tree_mask = torch.ones(
    (55,), dtype=torch.bool, device=device
)

# 以下假定它们是纯输出缓冲区。
# 如果源码要求调用前初始化，应改成与原调用端完全一致。
positions = torch.empty((5,), dtype=torch.int64, device=device)
retrieve_index = torch.empty((1, 5), dtype=torch.int64, device=device)
retrieve_next_token = torch.empty((1, 5), dtype=torch.int64, device=device)
retrieve_next_sibling = torch.empty((1, 5), dtype=torch.int64, device=device)

torch.npu.synchronize()
print("before build_tree", flush=True)

torch.ops.npu.build_tree_kernel_efficient(
    parent_list,
    top_scores_index,
    seq_lens,
    tree_mask,
    positions,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    1, # topk
    4,  # spec_steps
    5,  # num_verify_tokens
    0,  # tree_mask_mode
)

torch.npu.synchronize()
print("after build_tree", flush=True)

for name, tensor in [
    ("tree_mask", tree_mask),
    ("positions", positions),
    ("retrieve_index", retrieve_index),
    ("retrieve_next_token", retrieve_next_token),
    ("retrieve_next_sibling", retrieve_next_sibling),
]:
    print(name, tensor.cpu().tolist(), flush=True)
