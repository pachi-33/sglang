import torch
import torch_npu
import sgl_kernel_npu

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

positions = torch.empty((5,), dtype=torch.int64, device=device)
# 与推理代码一致：三个 retrieve 输出来自同一个连续 buffer。
retrieve_buf = torch.full((3, 1, 5), -1, dtype=torch.int64, device=device)
retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf

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
    1,  # topk
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


# ASCEND_LAUNCH_BLOCKING=1 python3 build_tree.py