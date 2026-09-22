# fast-gpu-kernels

Hand-optimized BF16 GEMM kernels for NVIDIA A100 / H100 / B200, benchmarked on Modal.

| GPU | Architecture | Hardware | Peak (BF16 dense) |
|-----|--------------|----------|-------------------|
| B200 | sm_100a (Blackwell) | B200 | ~2,250 TFLOPS |
| H100 | sm_90 (Hopper) | H100 80GB HBM3 | ~988 TFLOPS |
| A100 | sm_80 (Ampere) | A100-SXM4-40GB | 312 TFLOPS |

Default problem size is `M = N = K = 16384` (bf16), except B200 which uses `M = N = K = 8192`.

## Results at a glance

Best kernel per GPU / approach at the largest benchmarked shape.

| GPU | Approach | Best kernel | TFLOPS | vs cuBLAS |
|-----|----------|-------------|--------|-----------|
| B200 | CuTe DSL (8192³) | v5 | 1840.2 | 125% |
| B200 | Hand-written CUDA (4096³) | v7 | 1062 | 74% |
| H100 | CuTe DSL, sweep (8192) | K6 | 728.8 | 98% |
| H100 | CuTe WGMMA / TMA (16384³) | matmul_v2 | 365.8 | ~50% |
| A100 | Hand-written CUDA (16384³) | v11a | 258.7 | 86.1% |
| A100 | CuTe (16384³) | ptx_gemm | 211.0 | 79.4% |

See [Benchmarks](#benchmarks) for full iteration history.

---

## Benchmarks

### B200 — Blackwell (sm_100a)

#### CuTe DSL

Problem: `M = N = K = 8192`, bf16.

**Best: v5 at 1840 TFLOPS (~125% of cuBLAS).** All versions verified against a PyTorch `einsum` reference.

| Version | Kernel time (us) | TFLOPS | % of cuBLAS | Speedup vs v1 |
|---------|------------------|--------|-------------|---------------|
| cuBLAS | Reference | ~1478 | — | — |
| v1 | 2400.18 | 458.10 | 31% | 1.00x |
| v2 | 1229.82 | 894.04 | 60% | 1.95x |
| v3 | 762.88 | 1441.26 | 97% | 3.15x |
| v4 | 652.78 | 1684.34 | 114% | 3.68x |
| v5 | 597.49 | 1840.22 | 125% | 4.02x |
| v6 | 617.95 | 1779.28 | 120% | 3.89x |

#### Hand-written CUDA (TMA + tcgen05)

Problem: `M = N = K = 8192`, bf16. Sources in `kernels/cuda/B200/GEMM/bf16/` (staged epilogue).

**Best: v7 at 1062 TFLOPS on 4096³ (74% of cuBLAS).**

| Version | Tile | Stages | Technique | TFLOPS (8192³) | TFLOPS (4096³) | % cuBLAS (4096³) |
|---------|------|--------|-----------|----------------|----------------|------------------|
| v1 | 128×256×256 | 1 | TMA + tcgen05 baseline | 294 | 251 | 17% |
| v2 | 128×256×256 | 1 | + smem-staged coalesced store | 296 | 254 | 18% |
| v3 | 128×256×128 | 2 | + 2-stage pipeline | 360 | 306 | 22% |
| v4 | 128×256×128 | 2 | + 128B swizzle (64-wide TMA) | 970 | 965 | 71% |
| v5 | 128×128×64 | 7 | + warp-specialized TMA/MMA (2*N+1 mbar) | 884 | 1092 | 80% |
| v6 | 128×128×64 | 6+32KB | + dedicated epilogue (fixes 48.6% race) | 831 | 1024 | 76% |
| v7 | 128×128×64 | 6+32KB | + TMA store GMEM after staging | 842 | **1062** | **74%** |

> `v5` `128,64,7` (`224 KB`, depth=6) is the `1200 TF` warp-spec config (`gau-nernst.github.io/tcgen05` §5). `v6` cuts `7→6` (`192 KB+32 KB=224 KB`) for a dedicated epilogue (B200 limit `228 KB`). `v7` keeps the `v6` tile/stages and replaces phase-2 stores with a `TMA 2D store`. Full story in `kernels/cuda/B200/GEMM/bf16/README.md:Act 7-8`.

<details>
<summary>Run B200 benchmarks</summary>

```bash
modal run scripts/cuda/run_torch.py::main --task B200/GEMM/bf16/matmul_v4/test.py --gpu B200
modal run scripts/cuda/run_torch.py::main --task B200/GEMM/bf16/matmul_v6/test.py --gpu B200
modal run scripts/cuda/run_torch.py::main --task B200/GEMM/bf16/matmul_v7/test.py --gpu B200
modal run scripts/cuda/run_torch.py::main --task B200/GEMM/bf16/matmul_v7/benchmark.py --gpu B200
modal run scripts/cute_dsl/run.py::main --task B200/matmul_v6.py --gpu B200
```

</details>

---

### H100 — Hopper (WGMMA / TMA)

#### CuTe DSL

Shapes: `M = N = K` in {128, 256, 512, 1K, 2K, 4K, 5K, 8K}, FP16.

**Best: K6 at 728.8 TFLOPS (98% of cuBLAS at 8192); K5 at 674.9 TFLOPS (89% at 4K).**

| Shape | cuBLAS | K1 | K2 | K3 | K4 | K5 | K6 |
|-------|--------|----|----|----|----|----|----|
| 128 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| 512 | 20.1 | 20.2 | 17.7 | 20.6 | 21.3 | 15.9 | 18.1 |
| 1K | 162.1 | 113.5 | 175.0 | 142.2 | 149.7 | 121.8 | 111.6 |
| 4K | 755.0 | 501.7 | 542.2 | 578.6 | 573.5 | **674.9** | 655.5 |
| 8K | 745.5 | 494.4 | 526.7 | 522.2 | 557.3 | 672.0 | **728.8** |

<details>
<summary>Kernel configs (K1–K6)</summary>

| Kernel | Tile | Stages | Key Optimization |
|--------|------|--------|------------------|
| K1 | 128×128×128 | 1 | Single-CTA TMA + WGMMA baseline |
| K2 | 128×128×128 | 3 | `PipelineTmaAsync` multi-stage pipeline |
| K3 | 128×128×128 | 3+4 | TMA store epilogue with 4-stage pipeline |
| K4 | 128×128×128 | 3+3 | Warp-specialized TMA producer / MMA consumer |
| K5 | 128×256×64 | 4 | Asymmetric tile + 4-stage pipeline |
| K6 | 128×256×64 | 4 | 2×1 TMA multicast cluster |

</details>

#### CuTe (WGMMA / TMA)

Problem: `M = N = K = 16384`, bf16. Kernel times: 24.0–24.7 ms.

**Best: matmul_v2 at 365.8 TFLOPS (~50% of cuBLAS; cuBLAS is ~75% of peak here).**

| Kernel | Description | TFLOPS | % of cuBLAS |
|--------|-------------|--------|-------------|
| cuBLAS | Reference | 741 | — |
| matmul_v1 | Baseline WGMMA | 356.4 | ~49% |
| matmul_v2 | WGMMA with prefetch | 365.8 | ~50% |
| matmul_v3 | WGMMA with cluster sync | 365.6 | ~50% |
| matmul_v4 | WGMMA with TMA barriers | 365.7 | ~50% |

<details>
<summary>Run H100 benchmarks</summary>

```bash
modal run scripts/cute/run.py::main --task kernels/cute/H100/matmul_v1.cu --gpu H100
modal run scripts/cute_dsl/run.py::main --task H100/scripts/benchmark_all.py --gpu H100
modal run scripts/cute_dsl/run.py::main --task H100/matmul_v2.py --gpu H100
modal run scripts/benchmark_modal.py::main --gpu H100

# Generate charts
python kernels/cute_dsl/H100/scripts/plot_results.py
```

</details>

---

### A100 — Ampere

#### Hand-written CUDA

Problem: `M = N = K = 16384`, bf16.

**Best: v11a at 258.7 TFLOPS (86.1% of cuBLAS).**

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

#### CuTe

Problem: `M = N = K = 16384`, bf16.

**Best: ptx_gemm at 211.0 TFLOPS (79.4% of cuBLAS).**

| # | Kernel | Key Optimisation | TFLOPS | % of cuBLAS | Δ |
|---|--------|------------------|--------|-------------|---|
| cuBLAS | Reference | — | 263.4 | — | — |
| 1 | v1 | Baseline | 45.9 | 16.9% | — |
| 2 | v2 | + vector loads | 58.4 | 22.2% | +26% |
| 3 | v3 | + SMEM padding | 134.5 | 50.5% | **+131%** |
| 4 | v4 | + `Swizzle<3,3,3>` | 115.3 | 42.9% | −14% |
| 5 | v5 | + `cp.async` CACHEALWAYS | 170.8 | 64.1% | **+48%** |
| 6 | v6 | swizzle, single-stage | 180.2 | 68.0% | +5% |
| 7 | v7 | + 2-stage smem, pipelined K-loop | 172.9 | 65.0% | −4% |
| 8 | v8 | + 3-stage smem | 200.4 | 75.8% | **+16%** |
| 9 | ptx_gemm | + inline PTX | **211.0** | 79.4% | +7% |

<details>
<summary>Run A100 benchmarks</summary>

```bash
modal run scripts/run.py::main --task kernels/cuda/A100/benchmark.cu --gpu A100
modal run scripts/cute/run.py::main --task kernels/cute/A100/benchmark.cu --gpu A100
```

</details>

---

## Key techniques

- **B200:** TMA async loads, K-tile software pipelining (`prefetch_stages=ab_stages-2`), 2-CTA MMA (`use_2cta_instrs=True`, 2×1 cluster), warp-specialized TMA / MMA / epilogue, SMEM swizzle, TMEM accumulator staging, dynamic shapes (`assumed_align=32`).
- **H100:** WGMMA + TMA, `cp.async` 128-bit gmem→smem, `Swizzle<3,3,3>` for bank conflicts, 3-stage K-loop pipeline (`cp.async_fence`), `ldmatrix.x4` smem→register, warpgroup `arrive/commit/wait`, cluster and TMA barriers.
- **A100:** `cp.async` 2-stage copies, SMEM padding (+8) and XOR swizzle, `ldmatrix.x4`, 3-stage pipelined K-loop, warp tiling (4×2 warps / 256 threads), vectorized loads.

---

## Quickstart

Requirements: Python 3.12+, a [Modal](https://modal.com) account with `modal setup`, and Git (for CUTLASS).

```bash
git clone https://github.com/bikrammajhi/fast-gpu-kernels.git
cd fast-gpu-kernels
pip install -e ".[dev]"
modal setup
```

Run any kernel with `--gpu` (`B200`, `H200`, `H100`, `A100-80GB`, `A100-40GB`, `L40S`, `A10`, `L4`, `T4`, `RTXPRO6000`). Default GPU is set in `scripts/run.py:108`.

```bash
modal run scripts/run.py::main --task kernels/cuda/A100/benchmark.cu --gpu A100
modal run scripts/cute/run.py::main --task kernels/cute/A100/benchmark.cu --gpu A100
modal run scripts/benchmark_modal.py::main --gpu H100
modal run scripts/cute_dsl/run.py::main --task H100/matmul_v2.py --gpu H100
modal run scripts/cute_dsl/run.py::main --task B200/matmul_v6.py --gpu B200
```

---

## Profiling with Nsight Compute (ncu)

Capture `.ncu-rep` reports for any kernel and open them locally in the Nsight Compute GUI — no local GPU needed. Every capture uses the full metric set:

```bash
ncu --set full --warp-sampling-interval auto --clock-control base \
    --launch-skip 1 --launch-count 1 -o <path>/<name>.ncu-rep <application>
```

<details>
<summary>Capture on a machine with the GPU</summary>

```bash
# hand-written CUDA (A100) — compile, then capture
nvcc -O3 -arch=sm_80 -lcublas -o /tmp/bench kernels/cuda/A100/benchmark.cu
ncu --set full --warp-sampling-interval auto --clock-control base \
    --launch-skip 1 --launch-count 1 -o out/matmul_v11a.ncu-rep /tmp/bench

# CuTe (H100) — add your cutlass -I flags to the nvcc line
ncu --set full --warp-sampling-interval auto --clock-control base \
    --launch-skip 1 --launch-count 1 -o out/matmul_v4.ncu-rep /tmp/bench

# CuTe DSL Python (B200) — no compile step
ncu --set full --warp-sampling-interval auto --clock-control base \
    --launch-skip 1 --launch-count 1 \
    -o out/matmul_v5.ncu-rep python3 kernels/cute_dsl/B200/matmul_v5.py
```

</details>

<details>
<summary>Capture on Modal (no local GPU)</summary>

```bash
modal run scripts/ncu_capture.py --src kernels/cute_dsl/B200/matmul_v5.py --gpu B200
#   → writes captures/matmul_v5.ncu-rep on the gpulab-cute-dsl-traces volume

modal volume get gpulab-cute-dsl-traces captures/matmul_v5.ncu-rep ./matmul_v5.ncu-rep
```

The runner compiles `.cu` sources with the repo's arch map (`sm_80` A100, `sm_90` H100, `sm_100` B200) and sets `CUTE_DSL_ARCH` for DSL kernels. `--clock-control boost|none`, `--launch-skip N`, and `--launch-count N` map straight onto ncu.

```bash
ncu-ui matmul_v5.ncu-rep   # or File → Open in the Nsight Compute GUI
```

An `.ncu-rep` is self-contained — ncu only needs the GPU at capture time. A golden capture is committed at `kernels/cute_dsl/B200/results/golden/matmul_v1.ncu-rep`.

</details>
