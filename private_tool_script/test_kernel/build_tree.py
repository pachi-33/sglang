import torch
import torch_npu

# 在这里添加项目实际使用的自定义算子注册代码。
# 仅 import torch_npu 不一定会注册这个自定义算子。
# 例如：import <实际注册模块>
# 或：torch.ops.load_library("<实际动态库路径>")

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
