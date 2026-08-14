# fast-gpu-kernels

Hand-optimized BF16 GEMM kernels for NVIDIA A100 / H100 / B200, benchmarked on Modal.

**Hardware:** A100-SXM4-40GB | H100 80GB HBM3 | B200 (sm_100a, Blackwell)
**Target:** M = N = K = 16384, bf16 | Peak: 312 TFLOPS (A100), ~988 TFLOPS (H100), ~2,250 TFLOPS (B200)

---

## Benchmarks

Peak TFLOPS at the largest benchmarked GEMM shape, with delta versus the previous iteration.

### A100 — Hand-written CUDA

Problem: `M = N = K = 16384`, bf16

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

Problem: `M = N = K = 16384`, bf16

| # | Kernel | Key Optimisation | TFLOPS | % of cuBLAS | Δ |
|---|--------|------------------|--------|-------------|---|
| cuBLAS | Reference | — | 263.4 | — |
| 1 | v1 | Baseline | 45.9 | 16.9% | — |
| 2 | v2 | + vector loads | 58.4 | 22.2% | +26% |
| 3 | v3 | + SMEM padding | 134.5 | 50.5% | **+131%** |
| 4 | v4 | + `Swizzle<3,3,3>` | 115.3 | 42.9% | −14% |
| 5 | v5 | + `cp.async` CACHEALWAYS | 170.8 | 64.1% | **+48%** |
| 6 | v6 | swizzle, single-stage | 180.2 | 68.0% | +5% |
| 7 | v7 | + 2-stage smem, pipelined K-loop | 172.9 | 65.0% | −4% |
| 8 | v8 | + 3-stage smem | 200.4 | 75.8% | **+16%** |
| 9 | ptx_gemm | + inline PTX | **211.0** | 79.4% | +7% |

### H100 — CuTe (WGMMA / TMA)

Problem: `M = N = K = 16384`, bf16

| Kernel | Description | TFLOPS | % of cuBLAS |
|--------|-------------|--------|-------------|
| cuBLAS | Reference | 741 | — |
| matmul_v1 | Baseline WGMMA | 356.4 | ~49% |
| matmul_v2 | WGMMA with prefetch | 365.8 | ~50% |
| matmul_v3 | WGMMA with cluster sync | 365.6 | ~50% |
| matmul_v4 | WGMMA with TMA barriers | 365.7 | ~50% |

> **Note:** H100 peak = ~988 TFLOPS (bf16). cuBLAS achieves ~75% of peak at this shape. Kernel times: 24.0–24.7 ms.

```bash
modal run scripts/cute/run.py::main --task kernels/cute/H100/matmul_v1.cu --gpu H100
```

### H100 — CuTe DSL

Shapes: M = N = K in {128, 256, 512, 1K, 2K, 4K, 5K, 8K}, FP16

| Shape | cuBLAS | K1 | K2 | K3 | K4 | K5 | K6 |
|-------|--------|----|----|----|----|----|----|
| 128 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| 512 | 20.1 | 20.2 | 17.7 | 20.6 | 21.3 | 15.9 | 18.1 |
| 1K | 162.1 | 113.5 | 175.0 | 142.2 | 149.7 | 121.8 | 111.6 |
| 4K | 755.0 | 501.7 | 542.2 | 578.6 | 573.5 | **674.9** | 655.5 |
| 8K | 745.5 | 494.4 | 526.7 | 522.2 | 557.3 | 672.0 | **728.8** |

**K6 reaches 98% of cuBLAS at 8192. K5 hits 89% at 4K.**

| Kernel | Tile | Stages | Key Optimization |
|--------|------|--------|------------------|
| K1 | 128×128×128 | 1 | Single-CTA TMA + WGMMA baseline |
| K2 | 128×128×128 | 3 | `PipelineTmaAsync` multi-stage pipeline |
| K3 | 128×128×128 | 3+4 | TMA store epilogue with 4-stage pipeline |
| K4 | 128×128×128 | 3+3 | Warp-specialized TMA producer / MMA consumer |
| K5 | 128×256×64 | 4 | Asymmetric tile + 4-stage pipeline |
| K6 | 128×256×64 | 4 | 2×1 TMA multicast cluster |

```bash
# Benchmark all kernels
modal run scripts/cute_dsl/run.py::main --task H100/scripts/benchmark_all.py --gpu H100

# Generate charts
python kernels/cute_dsl/H100/scripts/plot_results.py
```

### B200 — CuTe DSL

Problem: `M = N = K = 8192`, bf16

| Version | Kernel time (us) | TFLOPS | % of cuBLAS | Speedup vs v1 |
|---------|------------------|--------|-------------|---------------|
| cuBLAS | Reference | ~1478 | — | — |
| v1 | 2400.18 | 458.10 | 31% | 1.00x |
| v2 | 1229.82 | 894.04 | 60% | 1.95x |
| v3 | 762.88 | 1441.26 | 97% | 3.15x |
| v4 | 652.78 | 1684.34 | 114% | 3.68x |
| v5 | 597.49 | 1840.22 | 125% | 4.02x |
| v6 | 617.95 | 1779.28 | 120% | 3.89x |

All B200 versions pass numerical verification against a PyTorch `einsum` reference.  
**CuTe DSL v5 reaches ~125% of cuBLAS peak.**

> **Perfetto traces** for every B200 version (v1–v6 + cuBLAS) live in the
> `iiserkbikram/cute-dsl-b200` Hugging Face bucket — one click per version, with
> the per-launch kernel time / SMEM / register evidence behind each optimization
> step. See `kernels/cute_dsl/B200/README.md#profiling-traces`.

## Optimizations

### H100 — Key Techniques

| Technique | Description |
|-----------|-------------|
| WGMMA / TMA | Warp-group MMA with Tensor Memory Accelerator async loads |
| cp.async | 128-bit async gmem→smem copies bypassing L1 |
| Swizzle `<3,3,3>` | 128-bit shared-memory swizzle to eliminate bank conflicts |
| 3-stage pipeline | Prefetched K-tile software pipeline with `cp.async_fence` |
| LDSM `x4` | 128-bit smem→register loads feeding tensor cores |
| Warpgroup MMA | `warpgroup_arrive/commit/wait` for async warp-group execution |
| Cluster barriers | `ClusterTransactionBarrier` + `ClusterBarrier` for producer/consumer sync |
| TMA barriers | Barrier-annotated TMA copies for pipelined gmem→smem |

### B200 — Key Techniques

| Technique | Description |
|-----------|-------------|
| TMA | Tensor Memory Accelerator async loads with cp.async |
| Software pipelining | K-tile overlap via `prefetch_stages=ab_stages-2` |
| 2-CTA MMA | `use_2cta_instrs=True`, 2×1 cluster for 2× throughput |
| Warp specialization | Dedicated TMA warp + MMA warp group + epilogue warps |
| Dynamic shapes | `assumed_align=32`, `mark_layout_dynamic`, `mark_compact_shape_dynamic` |
| SMEM swizzle | Structured swizzle for bank-conflict-free shared memory |
| TMEM | Tensor Memory for accumulator staging and epilogue |

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
