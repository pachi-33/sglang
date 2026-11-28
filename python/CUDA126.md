# CUDA 12.6 on Linux x86-64

This branch targets Ada GPUs with Python 3.11 and PyTorch 2.14.1+cu126. The `pp-lab` guest has an environment named `sglang-cu126`. Activate it before working in this checkout.

## Install from the checkout

Start in a conda environment with Python 3.11, a C++20 host compiler and the CUDA 12.6 toolkit available through `CUDA_HOME`. Run these commands from the repository root.

```bash
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install 'cmake>=3.31,<4' ninja scikit-build-core setuptools setuptools-rust setuptools-scm wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.14.1+cu126' 'torchvision==0.29.1+cu126' \
  'torchaudio==2.11.0+cu126' 'torchcodec==0.17.0+cu126'

export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1 -DSGL_KERNEL_ENABLE_SM90A=OFF" \
  python -m pip install --no-build-isolation ./python/sglang/kernels/aot
SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-build-isolation -e ./python
python -m pip check
```

The locally built kernel has the version `0.4.11+cu126`. The Python package uses FlashInfer 0.6.18.post1 with cuDNN frontend 1.28.0. Newer frontend releases pull in the CUDA 13 CUTLASS DSL on Linux. The base dependencies omit the upstream DeepEP, DeepGEMM, DeepSeek recipe, FA3 and FlashMLA wheels. Automatic backend selection checks the same FlashInfer minimum version as engine startup. The Qwen AWQ pipeline uses SGLang's Torch native attention backend.

## Run PP=4 in the guest

The 535-series vGPU driver needs a newer NCCL for this workload. The prepared conda environment preloads NCCL 2.32.6 from its own `opt/nccl-vgpu` directory and selects shared-memory transport. PyTorch's pinned NCCL 2.29.3 stays installed. To reproduce the guest settings in another such environment, install the auxiliary library and export these variables in the shell used to launch SGLang.

```bash
python -m pip install --target "$CONDA_PREFIX/opt/nccl-vgpu" 'nvidia-nccl-cu12==2.32.6'
export LD_PRELOAD="$CONDA_PREFIX/opt/nccl-vgpu/nvidia/nccl/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
```

After `conda activate sglang-cu126` in `pp-lab`, set `MODEL_DIR` to the directory containing the Qwen2.5-0.5B-Instruct-AWQ checkpoint and run

```bash
sglang serve --model-path "$MODEL_DIR" --host 127.0.0.1 \
  --tp-size 1 --pp-size 4 --dtype float16 --quantization awq \
  --attention-backend torch_native --sampling-backend pytorch --disable-custom-all-reduce \
  --context-length 256 --max-running-requests 1 --max-total-tokens 512 \
  --mem-fraction-static 0.80 --chunked-prefill-size 128
```
