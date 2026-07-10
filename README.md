# fast-gpu-kernels

Hand-optimized BF16 GEMM kernels for NVIDIA A100 / H100 / B200, benchmarked on Modal.

**Hardware:** A100-SXM4-40GB | H100 80GB HBM3 | B200 (sm_100a, Blackwell)
**Target:** M = N = K = 16384, bf16 | Peak: 312 TFLOPS (A100), ~988 TFLOPS (H100), ~1478 TFLOPS (B200)

---

## Benchmarks

Each table shows peak TFLOPS at the largest GEMM and the delta versus the previous iteration.

### Hand-written CUDA kernels (A100)

| Kernel | Technique | TFLOPS | % Peak | Δ |
|--------|-----------|--------|--------|---|
| v1 | Baseline | 64.2 | 20.6% | — |
| v2 | + `cp.async` 2-stage | 73.2 | 23.5% | +14% |
| v3 | + SMEM padding (+8) | 152.1 | 48.7% | **+108%** |
| v4 | + XOR swizzle | 153.9 | 49.3% | +1% |
| v7s3 | + `ldmatrix.x4` + 3-stage | 219.5 | 70.3% | +43% |
| v10 | + lambda-local regs | 252.6 | 81.0% | +15% |
| v11a | + 4x2 warps (256T) | **258.7** | 82.9% | +2% |
| cuBLAS | Reference | 300.4 | 96.3% | — |

**Full sweep (A100)**

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

---

### CuTe kernels (A100)

| # | Kernel | Key Optimisation | TFLOP/s | % of cuBLAS | Δ |
|---|--------|------------------|---------|-------------|---|
| 1 | v1 | Baseline | 45.9 | 16.9% | — |
| 2 | v2 | + vector loads | 58.4 | 22.2% | +26% |
| 3 | v3 | + SMEM padding | 134.5 | 50.5% | **+131%** |
| 4 | v4 | + `Swizzle<3,3,3>` | 115.3 | 42.9% | -14% |
| 5 | v5 | + `cp.async` CACHEALWAYS | 170.8 | 64.1% | **+48%** |
| 6 | v6 | swizzle, single-stage | 180.2 | 68.0% | +5% |
| 7 | v7 | + 2-stage smem, pipelined K-loop | 172.9 | 65.0% | -4% |
| 8 | v8 | + 3-stage smem | 200.4 | 75.8% | **+16%** |
| 9 | ptx_gemm | + inline PTX | **211.0** | 79.4% | +7% |
| cuBLAS | Reference | 263.4 | — | — | — |

---

### CuTe DSL kernels (H100 / B200)

#### B200 (sm_100a, Blackwell) (M=N=K=8192, Float16)

| Version | Kernel time (us) | Throughput (TFLOPs) | Speedup vs v1 |
|---------|------------------|---------------------|---------------|
| v1      | 2400.18          | 458.10              | 1.00x         |
| v2      | 1229.82          | 894.04              | 1.95x         |
| v3      | 762.88           | 1441.26             | 3.15x         |
| v4      | 652.78           | 1684.34             | 3.68x         |
| v5      | 597.49           | 1840.22             | 4.02x         |
| v6      | 617.95           | 1779.28             | 3.89x         |

All versions pass numerical verification against a PyTorch `einsum` reference.
**CuTe DSL v5 reaches ~125% of cuBLAS peak (1840 vs 1478 TFLOPs).**

```bash
modal run scripts/cute_dsl/run.py::main --task B200/matmul_v6.py --gpu B200
```

---

### CuTe kernels (H100)

H100-specific CuTe kernels using WGMMA/TMA are located in `kernels/cute/H100/`:

| Kernel | Description | Duration | TFLOPS |
|--------|-------------|----------|--------|
| matmul_v1 | Baseline WGMMA | 24.6800 ms | 356.4 |
| matmul_v2 | WGMMA with prefetch | 24.0442 ms | 365.8 |
| matmul_v3 | WGMMA with cluster sync | 24.0581 ms | 365.6 |
| matmul_v4 | WGMMA with TMA barriers | 24.0498 ms | 365.7 |

```bash
modal run scripts/cute/run.py::main --task kernels/cute/H100/matmul_v4.cu --gpu H100
```

---

## Requirements

- Python 3.12+
- [Modal](https://modal.com) account + `modal setup`
- Git for cloning CUTLASS

## Setup

```bash
git clone https://github.com/bikrammajhi/fast-gpu-kernels.git
cd fast-gpu-kernels
pip install -e ".[dev]"
modal setup
```

## Run

```bash
modal run scripts/run.py::main --task kernels/cuda/A100/benchmark.cu --gpu A100
modal run scripts/cute/run.py::main --task kernels/cute/A100/benchmark.cu --gpu A100
modal run scripts/cute_dsl/run.py::main --task B200/matmul_v6.py --gpu B200
```

### GPU selection

Edit the default GPU in `scripts/run.py:108` or pass `--gpu`:

| GPU | Modal Name |
|-----|-----------|
| B200 | B200 |
| H200 | H200 |
| H100 | H100 (default) |
| RTX PRO 6000 | RTXPRO6000 |
| A100 80GB | A100-80GB |
| A100 40GB | A100-40GB |
| L40S | L40S |
| A10 | A10 |
| L4 | L4 |
| T4 | T4 |
