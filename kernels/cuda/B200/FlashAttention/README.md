# FlashAttention — Blackwell B200 (SM100a)

Custom CUDA kernels implementing FlashAttention on NVIDIA Blackwell B200 using native `tcgen05` UMMA instructions, TMEM, and TMA.

## Kernel 1 — Baseline

Online softmax, tiled QK → P → PV using TMEM + UMMA (`tcgen05`).

- **Tile sizes:** BLOCK_M = BLOCK_N = HEAD_DIM = 128, MMA_K = 16
- **Layout:** K-major no-swizzle SMEM for Q/K, MN-major for V
- **Data type:** bf16 inputs/outputs, fp32 accumulators

### Results

| Config | Time | TFLOPS |
|--------|------|--------|
| batch=4, Q=[128,128], KV=[128,128] | 0.011 ms | 1.6 |

Measured on NVIDIA B200 (Modal), average over 20 iterations.

## Usage

```bash
modal run scripts/run.py::main --task kernels/cuda/B200/FlashAttention/1_baseline.cu --gpu B200
```

## Build

Requires CUDA 12.8+ with `sm_100a` support. Compiles with:

```
nvcc -arch=sm_100a -gencode arch=compute_100a,code=sm_100a -O3 -lcublas -lcuda
```
