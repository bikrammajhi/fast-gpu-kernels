# fast-gpu-kernels

Hand-optimized BF16 GEMM kernels for NVIDIA A100 / H100, benchmarked on Modal.

**Hardware:** A100-SXM4-40GB | H100 (switch via `--gpu`)
**Target:** M = N = K = 16384, bf16 | Peak: 312 TFLOPS (A100), 395 TFLOPS (Blackwell)

---

## Benchmarks

Each table shows peak TFLOPS at the largest GEMM (16384) and the delta versus the previous iteration.

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

### CuTe kernels (A100 / H100)

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

## Key Takeaways

| Optimisation | Impact |
|-----------|--------|
| Bank conflict fix (padding) | **+108%** CUDA, **+131%** CuTe |
| Multi-stage + `ldmatrix.x4` | +43% |
| `cp.async` CACHEALWAYS | +48% CuTe |
| Hand-written PTX | +7% vs CuTe abstraction ceiling |

**Rule:** profile first. One constant (`kPad = 8`) can double throughput.

---

## Requirements

- Python 3.12+
- [Modal](https://modal.com) account + `modal setup`
- Git for cloning cutlass

## Setup

```bash
git clone https://github.com/bikrammajhi/fast-gpu-kernels.git
cd fast-gpu-kernels
pip install -e ".[dev]"
modal setup
```

## Run

```bash
modal run scripts/run.py --task kernels/cuda/A100/benchmark.cu --gpu A100
modal run scripts/run.py --task kernels/cute/H100/benchmark.cu --gpu H100
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
