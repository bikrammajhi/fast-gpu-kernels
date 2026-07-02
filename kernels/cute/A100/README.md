# A100 BF16 GEMM Kernel Lab

Hand-written BF16 GEMM kernels for NVIDIA A100-SXM4-40GB, implemented with NVIDIA CuTe.

**Target:** M = N = K = 8192, bf16  
**Hardware:** A100-SXM4-40GB (108 SMs, SM80)  
**Peak TFLOPS:** 312 (bf16)  
**Best kernel:** 211 TFLOPS (80.1% of cuBLAS at 263 TFLOPS)

---

## Results (8192×8192×8192, bf16)

| # | Kernel | Key Optimization | TFLOP/s | % of cuBLAS | Δ |
|---|--------|-----------------|---------|-------------|---|
| 1 | v1 | Baseline: sync gmem→smem, row-major smem | 45.9 | 16.9% | — |
| 2 | v2 | + `UniversalCopy<uint128_t>` vector loads | 58.4 | 22.2% | +26% |
| 3 | v3 | + SMEM padding `kPad=8` | 134.5 | 50.5% | **+131%** |
| 4 | v4 | + `Swizzle<3,3,3>` | 115.3 | 42.9% | −14% |
| 5 | v5 | + `cp.async` CACHEALWAYS, single-stage | 170.8 | 64.1% | **+48%** |
| 6 | v6 | swizzle, single-stage | 180.2 | 68.0% | +5% |
| 7 | v7 | + 2-stage smem (`bP=2`), pipelined K-loop | 172.9 | 65.0% | −4% |
| 8 | v8 | + 3-stage smem (`bP=3`) | 200.4 | 75.8% | **+16%** |
| 9 | ptx_gemm | Hand-written PTX (`ldmatrix.x4`, `mma.sync`) | 211.0 | 80.1% | +7% |

> **Performance note:** Bank conflicts (v3) and `cp.async` (v5) are the two largest single-K steps. v7 regresses at 8192 — 3 stages (v8) is the correct buffer depth. CuTe abstraction caps at ~200 TFLOPS; inline PTX (+11 TFLOPS) breaks through.

## Directory

```
kernels/cute/A100/
├── matmul_v1.cu ... matmul_v8.cu
├── ptx_gemm.cu
├── benchmark.cu
├── docs/
└── experiments/
```

## Run

```bash
bash kernels/cute/A100/scripts/bench_all.sh
bash kernels/cute/A100/scripts/bench_all_8192.sh
```
