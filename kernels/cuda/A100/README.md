# A100 BF16 GEMM Kernel Lab

Hand-optimized BF16 GEMM on NVIDIA A100-SXM4-40GB.

**Hardware:** A100-SXM4-40GB | 312 TFLOPS BF16 peak | 108 SMs | 1.5 TB/s HBM2e
**Target:** N = 16384, bf16
**Best kernel:** 259 TFLOPS (82.9% of cuBLAS at 300 TFLOPS)

---

## Results (N=16384)

| Kernel | Technique | TFLOPS | % Peak | Δ |
|--------|-----------|--------|--------|---|
| v1 | Baseline (sync loads, 2x2 warps, single-stage) | 64.2 | 20.6% | — |
| v2 | + `cp.async` + 2-stage pipeline | 73.2 | 23.5% | +14% |
| v3 | + SMEM padding (+8) | 152.1 | 48.7% | **+108%** |
| v4 | + XOR swizzle replaces padding | 153.9 | 49.3% | +1% |
| v7s3 | + `ldmatrix.x4` + 3-stage pipeline | 219.5 | 70.3% | +43% |
| v10 | + lambda-local register declarations | 252.6 | 81.0% | +15% |
| v11a | + 4x2 warps (256 threads) | 258.7 | 82.9% | +2% |
| cuBLAS | Reference | 300.4 | 96.3% | — |

## Full Sweep

| N | v1 | v2 | v3 | v4 | v7s3 | v10 | v11a | cuBLAS |
|---|------|------|------|------|--------|--------|--------|--------|
| 128 | 0.2 | 0.3 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.5 |
| 256 | 1.2 | 1.3 | 2.2 | 2.3 | 2.5 | 2.5 | 2.5 | 3.6 |
| 512 | 5.6 | 6.1 | 12.1 | 12.9 | 14.2 | 14.2 | 14.4 | 26.2 |
| 1024 | 23.6 | 26.5 | 57.7 | 62.2 | 70.7 | 70.8 | 71.5 | 86.5 |
| 2048 | 37.3 | 41.9 | 102.8 | 114.1 | 114.8 | 130.2 | 131.8 | 124.2 |
| 4096 | 47.0 | 53.0 | 180.5 | 196.7 | 227.4 | 229.7 | 230.3 | 269.8 |
| 8192 | 64.2 | 72.7 | 162.2 | 213.4 | 215.1 | 247.2 | 254.1 | 295.2 |
| 16384 | 64.2 | 73.2 | 152.1 | 153.9 | 219.5 | 252.6 | 258.7 | 300.4 |

## Directory

```
kernels/cuda/A100/
├── matmul_v1.cu ... matmul_v11.cu
├── benchmark.cu
├── common.h
├── docs/
└── experiments/
```

## Run

```bash
modal run scripts/run.py --task kernels/cuda/A100/benchmark.cu
```
