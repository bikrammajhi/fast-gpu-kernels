# A100 GEMM Kernel Lab

High-performance BF16 GEMM kernels targeting the NVIDIA A100 (Ampere) tensor cores.

## Hardware

- **GPU:** NVIDIA A100-SXM4-80GB (or A100 80GB PCIe)
- **Tensor Core Peak:** 312 TFLOPS (BF16, SM80)
- **Architecture:** Ampere (`sm_80`)
- **Shared Memory:** 164 KB / SM (configurable)

## Files

```
kernels/cuda/A100/
├── common.h        — PTX helpers, MMA/LDMATRIX intrinsics, launch utilities
├── matmul_v1.cu    — v1: basic smem + ldmatrix, no pipelining
├── matmul_v2.cu    — placeholder for next version
└── benchmark.cu    — harness: timing, cuBLAS baseline, % SOL reporting
```

## Building & Running

```bash
modal run scripts/run.py --task kernels/cuda/A100/benchmark.cu
```

Or compile locally:

```bash
nvcc -O3 -arch=sm_80 -lcublas -o benchmark benchmark.cu
./benchmark
```

## Adding a New Kernel Version

1. Create `matmul_vN.cu` with your kernel
2. Define `matmul_vN_launch(A, B, C, M, N, K)` — the standard interface
3. In `benchmark.cu`:
   - Add `#include "matmul_vN.cu"`
   - Add `{ "vN", matmul_vN_launch }` to the `kernels[]` array

That's it. The harness handles timing, cuBLAS comparison, and % SOL automatically.

## Kernel Interface

Every kernel version must export:

```cpp
inline void matmul_vN_launch(
    const __nv_bfloat16* A,   // M x K, row-major
    const __nv_bfloat16* B,   // K x N, row-major
    __nv_bfloat16* C,         // M x N, row-major
    int M, int N, int K
);
```

## Benchmark Results

Tested on A100-SXM4-80GB (312 TFLOPS BF16 Tensor Core peak).

| N | v1 TFLOPS | v1 %SOL | cuBLAS TFLOPS | cuBLAS %SOL |
|---|---|---|---|---|
| 128 | 0.3 | 0.1% | 0.8 | 0.2% |
| 256 | 1.5 | 0.5% | 5.0 | 1.6% |
| 512 | 7.1 | 2.3% | 34.0 | 10.9% |
| 1024 | 30.1 | 9.6% | 111.6 | 35.8% |
| 2048 | 47.9 | 15.4% | 158.6 | 50.8% |
| 4096 | 59.9 | 19.2% | 268.4 | 86.0% |
| 8192 | 64.0 | 20.5% | 294.3 | 94.3% |

**Peak v1 performance: 64.0 TFLOPS at N=8192 (20.5% SOL)**

## Kernel Design: matmul_v1

### Tile Sizes

| Parameter | Value | Rationale |
|---|---|---|
| BLOCK_M | 128 | Matches tensor core tile, good occupancy |
| BLOCK_N | 128 | Same as BLOCK_M |
| BLOCK_K | 64 | Balances compute vs memory per k-tile |
| NUM_WARP_M | 2 | 2x2 warp layout = 4 warps = 128 threads |
| NUM_WARP_N | 2 | |
| SMEM_STRIDE | 64 | = BLOCK_K, no padding |

### Memory Hierarchy

```
Global Memory (HBM)
    │  GMEM2SMEM_COPY: vectorized 16-byte loads (uint4)
    ▼
Shared Memory (16 KB A + 16 KB B = 32 KB)
    │  LDMATRIX_X4/X2: warp-cooperative smem→register loads
    ▼
Registers (accumulator in FP32)
    │  MMA_M16N8K16: bf16 tensor core mma instruction
    ▼
Registers (result)
    │  __float2bfloat16_rn + vectorized store
    ▼
Global Memory (HBM)
```

### Execution Flow

```
For each BLOCK_K tile in K:
    1. GMEM → SMEM (vectorized, all threads cooperate)
    2. __syncthreads()
    3. For each MMA_K step in BLOCK_K:
       a. Load B from smem → registers (LDMATRIX_X2)
       b. For each MMA_M step:
          - Load A from smem → registers (LDMATRIX_X4)
          - MMA_M16N8K16: accumulate into FP32 registers
    4. __syncthreads()
    5. Advance A/B pointers by BLOCK_K

Epilogue:
    - Convert FP32 accumulators → BF16
    - Vectorized store (bf16x2) to global memory
```

## Known Bottlenecks (v1)

| Issue | Impact | Fix |
|---|---|---|
| Single-buffered smem | Stalls on every k-tile load | `cp.async` + multi-stage pipeline |
| No smem swizzle | Bank conflicts on ldmatrix | Add swizzle to smem layout |
| Synchronous gmem→smem | No overlap of load/compute | `cp.async` with commit/wait groups |
| No threadblock rasterization | Poor L2 cache reuse | Swizzle block IDs |
| Scalar epilogue stores | Suboptimal global memory writes | Vectorized epilogue |

## Theoretical Limits

**A100 BF16 Tensor Core:**
- 312 TFLOPS peak
- 16x8x16 MMA instruction = 256 multiply-accumulates = 512 FLOPs
- 128 SMs, each can issue MMA every ~4 cycles
- Memory bandwidth: 2 TB/s HBM2e

**Roofline at N=8192:**
- FLOPs: 2 * 8192^3 = 1.10e12
- Bytes: 3 * 8192^2 * 2 = 386.5 MB (read A, B; write C)
- Arithmetic intensity: 1.10e12 / 386.5e6 = 2846 FLOPs/byte
- This is **compute-bound** (>> 312 FLOPs/byte at 2 TB/s)
- cuBLAS achieves 94.3% SOL, confirming compute-bound regime
