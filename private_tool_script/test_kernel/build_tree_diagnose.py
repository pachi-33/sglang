"""Replay the observed bs=1, seq_len=6 build_tree call in one fresh process.

Run in the same A5 container/Python environment as the server. Compare separate
vs shared output storage and default vs new streams in separate invocations.
PASS requires device synchronization AND exact output validation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def emit(stage, **fields):
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layout", choices=("separate", "shared"), default="shared")
    parser.add_argument("--stream", choices=("default", "new"), default="new")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    import torch
    import torch_npu
    import sgl_kernel_npu

    package_file = Path(sgl_kernel_npu.__file__).resolve()
    library_file = package_file.parent / "lib" / "libsgl_kernel_npu.so"
    sha256 = hashlib.sha256()
    with library_file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    emit(
        "ENV",
        python=sys.executable,
        torch_version=torch.__version__,
        torch_npu_version=getattr(torch_npu, "__version__", "unknown"),
        package=str(package_file),
        library=str(library_file),
        sha256=sha256.hexdigest(),
        device=device,
        device_name=torch.npu.get_device_name(args.device),
        options=vars(args),
        environment={name: os.environ.get(name) for name in (
            "ASCEND_HOME_PATH", "LD_LIBRARY_PATH", "PYTHONPATH",
            "ASCEND_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE", "PER_STREAM_QUEUE",
            "ASCEND_RT_VISIBLE_DEVICES", "STREAMS_PER_DEVICE",
        )},
    )
    if not hasattr(torch.ops.npu, "build_tree_kernel_efficient"):
        raise RuntimeError("build_tree_kernel_efficient is not registered")

    stream = (
        torch.npu.default_stream(args.device)
        if args.stream == "default"
        else torch.npu.Stream(device=args.device)
    )
    with torch.inference_mode(), torch.npu.stream(stream):
        parent = torch.tensor([[-1, 0, 1, 2]], dtype=torch.int64, device=device)
        selected = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([6], dtype=torch.int64, device=device)
        mask = torch.ones(55, dtype=torch.bool, device=device)
        positions = torch.empty(5, dtype=torch.int64, device=device)
        if args.layout == "shared":
            retrieve_buf = torch.full((3, 1, 5), -1, dtype=torch.int64, device=device)
            index, next_token, sibling = retrieve_buf.unbind(0)
        else:
            index, next_token, sibling = [
                torch.full((1, 5), -1, dtype=torch.int64, device=device)
                for _ in range(3)
            ]

        tensors = dict(parent_list=parent, top_scores_index=selected,
                       seq_lens=seq_lens, tree_mask=mask, positions=positions,
                       retrieve_index=index, retrieve_next_token=next_token,
                       retrieve_next_sibling=sibling)
        for name, tensor in tensors.items():
            emit("TENSOR", name=name, shape=list(tensor.shape),
                 dtype=str(tensor.dtype), stride=list(tensor.stride()),
                 storage_offset=tensor.storage_offset(),
                 data_ptr=hex(tensor.data_ptr()), ptr_mod_32=tensor.data_ptr() % 32)
        emit("INPUT_SYNC_BEGIN")
        torch.npu.synchronize(args.device)
        emit("INPUT_READY", topk=1, depth=4, draft_token_num=5, mask_mode=0)

        expected = {
            "positions": [6, 7, 8, 9, 10],
            "retrieve_index": [[0, 1, 2, 3, 4]],
            "retrieve_next_token": [[1, 2, 3, 4, -1]],
            "retrieve_next_sibling": [[-1, -1, -1, -1, -1]],
            "tree_mask": [True] * 55,
        }
        for row in range(5):
            for col in range(5):
                expected["tree_mask"][row * 11 + 6 + col] = col <= row

        for iteration in range(args.iterations):
            mask.fill_(True)
            for output in (positions, index, next_token, sibling):
                output.fill_(-1)
            emit("CALL_BEGIN", iteration=iteration)
            torch.ops.npu.build_tree_kernel_efficient(
                parent, selected, seq_lens, mask, positions,
                index, next_token, sibling, 1, 4, 5, 0,
            )
            emit("CALL_RETURNED", iteration=iteration)
            torch.npu.synchronize(args.device)
            emit("DEVICE_COMPLETED", iteration=iteration)
            for name, want in expected.items():
                actual = tensors[name].cpu().tolist()
                if actual != want:
                    raise AssertionError(f"{name}: expected {want}, got {actual}")
            emit("PASS", iteration=iteration)


if __name__ == "__main__":
    main()


# ASCEND_LAUNCH_BLOCKING=1 python3 build_tree_diagnose.py --layout separate --stream default
# ASCEND_LAUNCH_BLOCKING=1 python3 build_tree_diagnose.py --layout shared --stream default
# ASCEND_LAUNCH_BLOCKING=1 python3 build_tree_diagnose.py --layout shared --stream new