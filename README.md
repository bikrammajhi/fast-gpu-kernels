# fast-gpu-kernels

Hand-optimized BF16 GEMM kernels for NVIDIA A100 / H100 / B200, benchmarked on Modal.

**Hardware:** A100-SXM4-40GB | H100 80GB HBM3 | B200 (sm_100a, Blackwell)
**Target:** M = N = K = 16384, bf16 | Peak: 312 TFLOPS (A100), ~988 TFLOPS (H100), ~1478 TFLOPS (B200)

---

## Benchmarks

Results expressed as % of cuBLAS at the largest GEMM shape, with delta versus the previous iteration.

### A100 — Hand-written CUDA

| Kernel | Technique | TFLOPS | % of cuBLAS | Δ |
|--------|-----------|--------|-------------|---|
| cuBLAS | Reference | 300.4 | — | — |
| v1 | Baseline | 64.2 | 21.4% | — |
| v2 | + `cp.async` 2-stage | 73.2 | 24.4% | +14% |
| v3 | + SMEM padding (+8) | 152.1 | 50.6% | **+108%** |
| v4 | + XOR swizzle | 153.9 | 51.2% | +1% |
| v7s3 | + `ldmatrix.x4` + 3-stage | 219.5 | 73.0% | +43% |
| v10 | + lambda-local regs | 252.6 | 84.1% | +15% |
| v11a | + 4x2 warps (256T) | **258.7** | 86.1% | +2% |

### A100 — CuTe

| # | Kernel | Key Optimisation | % of cuBLAS | Δ |
|---|--------|------------------|-------------|---|
| cuBLAS | Reference | — | — | — |
| 1 | v1 | Baseline | 16.9% | — |
| 2 | v2 | + vector loads | 22.2% | +26% |
| 3 | v3 | + SMEM padding | 50.5% | **+131%** |
| 4 | v4 | + `Swizzle<3,3,3>` | 42.9% | −14% |
| 5 | v5 | + `cp.async` CACHEALWAYS | 64.1% | **+48%** |
| 6 | v6 | swizzle, single-stage | 68.0% | +5% |
| 7 | v7 | + 2-stage smem, pipelined K-loop | 65.0% | −4% |
| 8 | v8 | + 3-stage smem | 75.8% | **+16%** |
| 9 | ptx_gemm | + inline PTX | **79.4%** | +7% |

### H100 — CuTe (WGMMA / TMA)

| Kernel | Description | % of cuBLAS |
|--------|-------------|-------------|
| cuBLAS | Reference (16384³, bf16) | ~75% of peak |
| matmul_v1 | Baseline WGMMA | ~49% |
| matmul_v2 | WGMMA with prefetch | ~50% |
| matmul_v3 | WGMMA with cluster sync | ~50% |
| matmul_v4 | WGMMA with TMA barriers | ~50% |

> **Baseline:** H100 cuBLAS at 16384×16384×16384, bf16 ≈ 741 TFLOPS (~75% of 988 TFLOPS peak). Kernel times: 24.0–24.7 ms.

### H100 — CuTe DSL

| Kernel | % of cuBLAS |
|--------|-------------|
| cuBLAS | Reference (4096³, bf16) |
| DSL v2 | ~88% |

> **Baseline:** H100 cuBLAS at 4096×4096×4096, bf16 = 776.0 TFLOPS. DSL v2: 200.91 µs, 684.07 TFLOPS.

### B200 — CuTe DSL

| Version | Kernel time (us) | % of cuBLAS | Speedup vs v1 |
|---------|------------------|-------------|---------------|
| cuBLAS | Reference (8192³, Float16) | — | — |
| v1 | 2400.18 | 31% | 1.00x |
| v2 | 1229.82 | 60% | 1.95x |
| v3 | 762.88 | 97% | 3.15x |
| v4 | 652.78 | 114% | 3.68x |
| v5 | 597.49 | 125% | 4.02x |
| v6 | 617.95 | 120% | 3.89x |

All B200 versions pass numerical verification against a PyTorch `einsum` reference.  
**CuTe DSL v5 reaches ~125% of cuBLAS peak.**

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
modal run scripts/benchmark_modal.py::main --gpu H100
modal run scripts/cute_dsl/run.py::main --task H100/matmul_v2.py --gpu H100
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
