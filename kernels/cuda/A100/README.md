# A100 BF16 GEMM Kernel Lab

Hand-optimized BF16 matrix multiplication kernels on NVIDIA A100-SXM4-40GB.
From 20.6% to 83.1% of cuBLAS peak — seven kernels, each adding one optimization.

**Hardware**: A100-SXM4-40GB | 312 TFLOPS BF16 peak | 108 SMs | 1.5 TB/s HBM2e

---

## Results (N=16384, A100-SXM4-40GB)

| Kernel | Technique | TFLOPS | % Peak | Δ |
|--------|-----------|--------|--------|---|
| v1 | Baseline: sync loads, 2×2 warps, 1-stage | 64.2 | 20.6% | — |
| v2 | + `cp.async` + 2-stage pipeline | 73.2 | 23.5% | +14% |
| v3 | + shared memory padding (+8) | 152.1 | 48.7% | **+108%** |
| v4 | + XOR swizzle replaces padding | 153.9 | 49.3% | +1% |
| v7s3 | + `swizzle_better` + `ldmatrix.x4` + 3-stage | 219.5 | 70.3% | +43% |
| v10 | + lambda-local register declarations | 252.6 | 81.0% | +15% |
| v11a | + 4×2 warps (256 threads) | 258.7 | 82.9% | +2% |
| cuBLAS | Reference | 300.4 | 96.3% | — |

## Full Sweep

```
N       v1      v2      v3      v4     v7s3   v10    v11a  cuBLAS
───── ────── ────── ────── ────── ────── ────── ────── ──────
 128    0.2    0.3    0.4    0.4    0.4    0.4    0.4    0.5
 256    1.2    1.3    2.2    2.3    2.5    2.5    2.5    3.6
 512    5.6    6.1   12.1   12.9   14.2   14.2   14.4   26.2
1024   23.6   26.5   57.7   62.2   70.7   70.8   71.5   86.5
2048   37.3   41.9  102.8  114.1  114.8  130.2  131.8  124.2
4096   47.0   53.0  180.5  196.7  227.4  229.7  230.3  269.8
8192   64.2   72.7  162.2  213.4  215.1  247.2  254.1  295.2
16384  64.2   73.2  152.1  153.9  219.5  252.6  258.7  300.4
```

## Optimization Map

```
v1: Baseline (64 TFLOPS, 20.6%)
 │  sync loads, 2x2 warps, single-stage
 ▼
v2: + cp.async + 2-stage pipeline (73 TFLOPS, 23.5%)  [+14%]
 │  overlap global loads with compute
 ▼
v3: + smem padding +8 (152 TFLOPS, 48.7%)  [+108%]
 │  ELIMINATES BANK CONFLICTS
 ▼
v4: + XOR swizzle (154 TFLOPS, 49.3%)  [+1%]
 │  same perf, saves 16KB smem
 ▼
v7s3: + swizzle_better + ldmatrix.x4 + 3-stage (220 TFLOPS, 70.3%)  [+43%]
 │  clean rewrite, multi-stage pipeline
 ▼
v10: + lambda-local register declarations (253 TFLOPS, 81.0%)  [+15%]
 │  compiler sees regs better, better scheduling
 ▼
v11a: + 4x2 warps, 256 threads (259 TFLOPS, 82.9%)  [+2%]
 │  more warps = more ILP + latency hiding
 ▼
▓▓▓ cuBLAS: 300 TFLOPS (96.3%) ▓▓▓
```

## Key Takeaways

| Optimization | Impact | Difficulty |
|-------------|--------|------------|
| Bank conflict fix (padding) | **+108%** | Trivial (1 constant) |
| Multi-stage + ldmatrix.x4 | +43% | Medium |
| Lambda-local registers | +15% | Trivial (move 2 lines) |
| `cp.async` pipeline | +14% | Low |
| More warps (4×2) | +2% | Low |
| XOR swizzle | +1% | Medium |

**The lesson**: Profile before optimizing. Bank conflicts gave us a 2× speedup with a one-line change. Meanwhile, the multi-stage pipeline and compiler hints unlocked the biggest gains at scale.

---

## Project Structure

```
kernels/cuda/A100/
├── README.md               This file
├── BLOG.md                 Detailed writeup of the optimization journey
├── VIZ.md                  Visual reference for the kernel internals
├── SWIZZLE_DEEP_DIVE.md    Deep dive on XOR swizzle for bank conflict elimination
├── common.h                PTX helpers: cp_async, LDMATRIX_X4, MMA_M16N8K16, swizzle
├── benchmark.cu            Benchmark harness: timing, correctness, cuBLAS comparison
├── matmul_v1.cu            Baseline: sync loads, 2x2 warps, single-stage
├── matmul_v2.cu            + cp.async + 2-stage pipeline
├── matmul_v3.cu            + shared memory padding (bank conflicts)
├── matmul_v4.cu            + XOR swizzle replaces padding
├── matmul_v7.cu            + swizzle_better + ldmatrix.x4 + multi-stage
├── matmul_v10.cu           + lambda-local register declarations
├── matmul_v11.cu           + 4x2 warps (256 threads)
└── experiments/            Side experiments and earlier iterations
    ├── matmul_v5.cu        v3 variant (threadblock swizzle exploration)
    ├── matmul_v6.cu        v4 variant (swizzle on store + load)
    ├── matmul_v8.cu        + tunable GROUP_M parameter
    ├── matmul_v9.cu        + tile/warp exploration (256x128, 4x4)
    ├── matmul_v12.cu       + double-buffered register loads
    ├── matmul_v13.cu       v11a + v12 combined (best: v13d)
    ├── matmul_v14.cu       smem epilogue experiment (regression)
    ├── matmul_v21.cu       Alternative scheduling
    ├── matmul_v22.cu       Alternative scheduling
    ├── matmul_v23.cu       Alternative scheduling
    ├── matmul_v24.cu       L2 cache tiling experiments
    ├── matmul_v25.cu       L2 cache tiling experiments
    ├── matmul_v26.cu       2D tiled CTA scheduling (correctness issues)
    ├── matmul_v27.cu       Hilbert-curve CTA scheduling
    └── matmul_streamk.cu   Stream-K decomposition (atomicAdd bottleneck)
```

## Building and Running

```bash
modal run scripts/run.py --task kernels/cuda/A100/benchmark.cu
```
