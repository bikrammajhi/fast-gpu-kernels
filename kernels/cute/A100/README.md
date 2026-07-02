# A100 BF16 GEMM Kernel Lab

Hand-written BF16 GEMM kernels for NVIDIA A100-SXM4-40GB, implemented with NVIDIA CuTe. This repository documents a step-by-step optimization progression from a baseline synchronous kernel through to a hand-written PTX kernel, with measured performance at each stage.

**Target problem:** M = N = K = 8192, bf16  
**Hardware:** A100-SXM4-40GB (108 SMs, SM80)  
**Peak tensor core throughput:** 312 TFLOPS (bf16)  
**Best kernel:** 211 TFLOPS (79.4% of cuBLAS)

---

## Optimization Ladder

The following nine kernels represent the canonical optimization path. Each kernel is self-contained in `matmul_v<N>.cu` and includes its own `main()` for standalone benchmarking, or can be swapped into `benchmark.cu` for multi-size sweeps.

```
v1  Baseline ............................  46 TFLOPS (17.3%)
v2  + 128-bit vectorized gmem loads .....  58 TFLOPS (22.2%)   +26%
v3  + smem bank-conflict padding ........ 134 TFLOPS (50.5%)  +131%
v4  + XOR swizzle (regresses at 8192) ... 115 TFLOPS (43.0%)   −14%
v5  + cp.async CACHEALWAYS .............. 171 TFLOPS (64.1%)   +48%
v6  + swizzle replaces padding .......... 180 TFLOPS (68.0%)    +5%
v7  + 2-stage smem + prefetch loop ...... 173 TFLOPS (65.0%)    −4%
v8  + 3-stage smem (sweet spot) ......... 200 TFLOPS (75.8%)   +16%
v37 + hand-written PTX inline-asm ....... 211 TFLOPS (80.1%)    +7%
```

**cuBLAS reference:** ~266 TFLOPS (100%)

---

## Measured Performance (8192×8192×8192, bf16)

| # | Kernel | Key Optimization | CuTe TFLOPS | cuBLAS TFLOPS | % of cuBLAS | Δ vs Previous |
|---|--------|-----------------|---:|---:|---:|---:|
| 1 | v1 | Baseline: sync gmem→smem, row-major smem | 45.9 | 271.2 | 16.9% | — |
| 2 | v2 | + `UniversalCopy<uint128>` (128-bit vector loads) | 58.4 | 263.7 | 22.2% | **+26%** |
| 3 | v3 | + smem padding `kPad=8` (eliminates bank conflicts) | 134.5 | 266.5 | 50.5% | **+131%** |
| 4 | v4 | + `Swizzle<3,3,3>` replaces padding | 115.3 | 269.0 | 42.9% | −14% |
| 5 | v5 | + `cp.async` CACHEALWAYS, single-stage sync K-loop | 170.8 | 266.5 | 64.1% | **+48%** |
| 6 | v6 | swizzle replaces padding, single-stage | 180.2 | 264.9 | 68.0% | +5% |
| 7 | v7 | + 2-stage smem (`bP=2`) + pipelined `while` prefetch | 172.9 | 265.9 | 65.0% | −4% |
| 8 | v8 | + 3-stage smem (`bP=3`) | 200.4 | 264.2 | 75.8% | **+16%** |
| 9 | v37 | hand-written PTX (`ldmatrix.x4`, `mma.sync`) | 211.0 | 263.4 | **80.1%** | +7% |

> **Note on v7 regression:** The 2-stage prefetch loop introduces dynamic `while`-loop pointer-swing overhead (`tXsA(_,_,_,smem_pipe_read)` index math) that exceeds the latency-hiding benefit at this tile size on A100. The 3-stage buffer in v8 is the correct sweet spot.

---

## What Each Optimization Does

### v1 → v2: Vectorized global loads
Switch from scalar `DefaultCopy` to `UniversalCopy<uint128_t>`. Each thread issues one 128-bit gmem load per logical copy call, matching the memory transaction width. Zero kernel-structure changes.

### v2 → v3: Eliminate SMEM bank conflicts
With `stride = bK = 64`, consecutive threads in a warp access addresses 4 bytes apart. A100 has 32 banks with 4-way interleaving at 4B granularity, so all 32 threads hit the same bank. Fix: `kPad = 8`, `stride = bK + kPad = 72`. Zero runtime cost.

### v3 → v4: XOR swizzle (regresses at 8192)
Replace padding with `Swizzle<3,3,3>` to structurally avoid bank conflicts while saving 16 KB smem per CTA. At 8192, the XOR address arithmetic in LDSM adds ~5 cycles per access; the conflict savings do not offset the overhead. Swizzle wins for tiles ≥ 16384.

### v4 → v5: `cp.async` CACHEALWAYS
`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` issues non-blocking global loads that bypass L1 and write directly to SMEM, pinned in L2 for LDSM. This is the second-biggest single win in the ladder.

**Critical correctness requirement:** `__syncthreads()` after `gemm()` is mandatory. Without it, the next K-step's `cp.async` DMA can overwrite the current tile while another warp is still reading it via LDSM — a silent data race.

### v5 → v6: Swizzle replaces padding (single-stage)
v5 uses padded row-major smem. v6 swaps in `Swizzle<3,3,3>` while keeping cp.async, single-stage smem, and the synchronous K-loop. At 8192, swizzle outperforms padding by 5%.

### v6 → v7: 2-stage smem + pipelined loop
Add a `stages` dimension (`bP = Int<2>{}`) and restructure the K-loop into a `while` loop that rotates `smem_pipe_read`/`smem_pipe_write` pointers, issuing `cp.async` for the next tile while the current tile is in MMA. Regresses at 8192 due to pointer-swing overhead.

### v7 → v8: 3-stage smem (sweet spot)
Increase to `bP = Int<3>{}` (3 stages = two tiles buffered + one in-flight). The `cp.async` round-trip fully hides behind the current tile's MMA. 4+ stages over-subscribes registers for 128×128×64 tiles on A100.

### v8 → v37: Hand-written PTX
v37 abandons CuTe's high-level API entirely for inline PTX with:
- `ldmatrix.x4` for B (loads two B matrices per instruction)
- `mma.sync.aligned.m16n8k16`
- Explicit double-buffered register files for A/B fragments

This breaks through the ~200 TFLOPS ceiling of the CuTe abstraction.

---

## Experiments (Did Not Work)

The following kernels are preserved in `experiments/` for reference. All were benchmarked at 8192×8192×8192 bf16.

| Kernel | Change | Result | Diagnosis |
|--------|--------|--------|-----------|
| `v9.cu` | grid swizzle factor=8, 2-stage | 198 TFLOPS | Marginal gain; folded into v8 tuning |
| `v21.cu` | bN=256, bP=4 | 196 TFLOPS | Tile-to-thread ratio too wide for 128 threads |
| `v22.cu` | bN=256, 128 threads | 78 TFLOPS | Output-tile-to-thread ratio 16:1 → severe under-utilization |
| `v23.cu` | v22 + CACHEALWAYS | 31 TFLOPS | Compounded by wide tile |
| `v24.cu` | bK=128, padded 4-stage | 203 TFLOPS | Marginal gain over v8 |
| `v25.cu` | bK=128, 2-stage | 179 TFLOPS | Wider K-tile needs 3+ stages |
| `v26.cu` | MMA 2×4 atoms (256 threads) | compile error | LDSM TiledCopy val layout mismatch: 16 values vs 32 required |
| `v27.cu` | swizzle factor=4 | 198 TFLOPS | Neutral vs v8 |
| `v28.cu` | swizzle factor=16 | 203 TFLOPS | Neutral vs v8 |
| `v29.cu` | no grid swizzle | 197 TFLOPS | Neutral vs v8 |
| `v30.cu` | bP=5 stages | 142 TFLOPS | Over-subscribes registers |
| `v31.cu` | padded row-major, bP=4 | 141 TFLOPS | Bank conflicts re-emerge |
| `v32.cu` | swizzle factor=4, bP=4 | 205 TFLOPS | Marginal gain |
| `v33.cu` | swizzle factor=16, bP=4 | 202 TFLOPS | Neutral |
| `v34.cu` | bM=256, bN=128, 128 threads | 29 TFLOPS | 1.6% compute occupancy |
| `v35.cu` | swizzle factor=8, bP=4 | 203 TFLOPS | Marginal gain |
| `v36.cu` | tuned scheduling | 198 TFLOPS | Neutral vs v8 |

---

## Directory Layout

```
kernels/cute/A100/
├── README.md              ← you are here
├── benchmark.cu           Multi-size benchmark vs cuBLAS
├── bench_all.sh           Modal runner: v1–v9
├── bench_all_8192.sh      Modal runner: v1–v37 at 8192
├── Notes.md               Copy atom / LDSM / bank-conflict notes
├── matmul_v1.cu           Baseline
├── matmul_v2.cu           + 128-bit vector loads
├── matmul_v3.cu           + smem padding
├── matmul_v4.cu           + swizzle (regresses at 8192)
├── matmul_v5.cu           + cp.async CACHEALWAYS
├── matmul_v6.cu           + swizzle replaces padding
├── matmul_v7.cu           + 2-stage smem + prefetch loop
├── matmul_v8.cu           + 3-stage smem (sweet spot)
├── matmul_v37.cu          + hand-written PTX
├── experiments/           Regressed / broken kernels
│   ├── v9.cu
│   ├── v21.cu
│   ├── v22.cu … v36.cu
│   └── ...
└── good_versions/         Mirrored copies of confirmed high-performing kernels
    ├── v1.cu … v8.cu
    └── v37.cu
```

---

## Building & Running

### Prerequisites
- NVIDIA CUDA toolkit (`nvcc`)
- CUTLASS headers (`$CUTLASS/include`, `$CUTLASS/tools/util/include`)
- cuBLAS
- Optional: [Modal](https://modal.com) for remote A100 execution

### Local build

```bash
# Compile a single kernel
nvcc -O3 -arch=sm_80 \
  -I$CUTLASS/include -I$CUTLASS/tools/util/include \
  -lcublas matmul_v8.cu -o /tmp/matmul_v8
/tmp/matmul_v8

# Compile and run multi-size benchmark
nvcc -O3 -arch=sm_80 \
  -I$CUTLASS/include -I$CUTLASS/tools/util/include \
  -lcublas benchmark.cu -o /tmp/benchmark
/tmp/matmul_v8
```

### Remote benchmark via Modal

```bash
# Run all 9 main kernels (v1–v9)
bash kernels/cute/A100/bench_all.sh

# Run full 8192 sweep including experiments (v1–v37)
bash kernels/cute/A100/bench_all_8192.sh
```

### Benchmark harness

`benchmark.cu` is parameterized by the included kernel header. To benchmark a different version:

```bash
# Temporarily swap the include
sed -i 's|#include "matmul_v1.cu"|#include "matmul_v37.cu"|' benchmark.cu
nvcc -O3 -arch=sm_80 -I$CUTLASS/include -I$CUTLASS/tools/util/include \
  -lcublas benchmark.cu -o /tmp/bench_v37 && /tmp/bench_v37
```

---

## Hardware Reference

| Property | Value |
|----------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| Tensor Core peak (bf16) | 312 TFLOPS |
| Architecture | Ampere (`sm_80`) |
| SMs | 108 |
| Shared memory / SM | 164 KB (configurable) |
| L2 cache | 40 MB |
| HBM bandwidth | 1.5 TB/s |

---

## Key Takeaways

| Optimization | 8192 TFLOPS | % of cuBLAS | Effort |
|---:|---:|---:|---|
| smem bank-conflict padding | 134 | 50% | Trivial |
| `cp.async` CACHEALWAYS | 171 | 64% | Low |
| 3-stage swizzled pipeline | 200 | 75% | Medium |
| Hand-written PTX | 211 | 79% | High |
| **Remaining gap to cuBLAS** | **55 TFLOPS** | **21%** | — |

1. **Bank conflicts are the silent killer.** One constant (`kPad=8`) gave +131%. Always profile first.
2. **`cp.async` is mandatory for >150 TFLOPS.** Pair with `cp_async_fence()` + `cp_async_wait<0>()` + `__syncthreads()` after `gemm()`.
3. **3 pipeline stages is the SMEM sweet spot** for 128×128×64 CTA tiles on A100.
4. **CuTe abstraction ceiling:** ~200 TFLOPS (75%). Inline PTX adds the final +11 TFLOPS.
