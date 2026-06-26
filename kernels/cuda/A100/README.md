# A100 BF16 GEMM Kernel Lab

From 20% to 81% of cuBLAS — hand-written BF16 GEMM kernels on A100-SXM4-40GB.

## The Optimization Map

```
                        THE ROAD TO cuBLAS
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   v1: Baseline (64 TFLOPS, 20.5% SOL)                  │
    │   │  sync loads, 2x2 warps, single-stage               │
    │   │                                                     │
    │   ▼                                                     │
    │   v2: + cp.async + 2-stage pipeline (72 TFLOPS, 23.3%) │
    │   │  overlap global loads with compute                  │
    │   │  ───────────────────────────── +13% ──────────────  │
    │   │                                                     │
    │   ▼                                                     │
    │   v3: + smem padding +8 (162 TFLOPS, 52.1%)            │
    │   │  ELIMINATES BANK CONFLICTS                          │
    │   │  ───────────────────────────── +124% ─────────────  │
    │   │                                                     │
    │   ▼                                                     │
    │   v4: + swizzle replaces padding (214 TFLOPS, 68.5%)   │
    │   │  same perf, saves 16KB smem, XOR address remap     │
    │   │  ───────────────────────────── +32% ──────────────  │
    │   │                                                     │
    │   ▼                                                     │
    │   v7s3: + swizzle_better + ldmatrix_x4 + 3-stage       │
    │   │      (215 TFLOPS, 69.0%)                            │
    │   │  clean rewrite, multi-stage pipeline                │
    │   │  ───────────────────────────── +1% ───────────────  │
    │   │                                                     │
    │   ▼                                                     │
    │   v10: + lambda-local register declarations             │
    │   │     (247 TFLOPS, 79.2%)                             │
    │   │  compiler sees regs better, better scheduling      │
    │   │  ───────────────────────────── +15% ──────────────  │
    │   │                                                     │
    │   ▼                                                     │
    │   v11a: + 4x2 warps, 256 threads (254 TFLOPS, 81.5%)  │
    │   │  more warps = more ILP + latency hiding            │
    │   │  ───────────────────────────── +3% ───────────────  │
    │   │                                                     │
    │   ▼                                                     │
    │   v13d: + double-buffered regs + GROUP_M=16             │
    │   │      (254 TFLOPS, 81.5%)  ← BEST                   │
    │   │  combines v11a + v12 insights                       │
    │   │  ───────────────────────────── +0% ───────────────  │
    │   │                                                     │
    │   ▼                                                     │
    │   ▓▓▓▓▓▓▓▓▓▓▓  cuBLAS: 296 TFLOPS (94.9%)  ▓▓▓▓▓▓▓▓  │
    │                                                         │
    │   ~14% gap remaining                                    │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    SIDE EXPERIMENTS:
    ┌─────────────────────────────────────────────────────────┐
    │  v14:      smem epilogue → 243 TFLOPS (-4% regression) │
    │  stream-k: atomicAdd decomposition → 93 TFLOPS (30%)   │
    └─────────────────────────────────────────────────────────┘
```

## Benchmark Results (N=8192, A100-SXM4-40GB)

| # | Kernel | New Technique | TFLOPS | % SOL | Δ from prev |
|---|--------|--------------|--------|-------|-------------|
| 1 | v1 | Baseline: sync loads, 2x2 warps, 1-stage | 64.2 | 20.6% | — |
| 2 | v2 | + `cp.async` + 2-stage pipeline | 72.6 | 23.3% | +13% |
| 3 | v3 | + shared memory padding (+8) | 162.4 | 52.1% | **+124%** |
| 4 | v4 | + XOR swizzle replaces padding | 213.8 | 68.5% | +32% |
| 5 | v7s3 | + `swizzle_better` + `ldmatrix_x4` + 3-stage | 215.2 | 69.0% | +1% |
| 6 | v10 | + lambda-local register declarations | 247.3 | 79.2% | +15% |
| 7 | v11a | + 4x2 warps (256 threads) | 254.1 | 81.5% | +3% |
| 8 | v13d | + double-buffered regs + GROUP_M=16 | 254.2 | 81.5% | +0% |
| 9 | v14 | smem epilogue (failed experiment) | 243.0 | 77.9% | -4% |
| 10 | stream-k | Stream-K decomposition | 93.5 | 30.0% | — |
| — | cuBLAS | Reference | 296.1 | 94.9% | — |

## Full Sweep

```
N        v1       v2       v3       v4       v7s3     v10      v11a     v13d     cuBLAS
────── ──────── ──────── ──────── ──────── ──────── ──────── ──────── ──────── ────────
  128     0.2      0.3      0.4      0.4      0.4      0.4      0.4      0.4      0.5
  256     1.2      1.3      2.3      2.3      2.5      2.5      2.6      2.5      4.0
  512     5.5      6.1     12.1     13.0     14.2     14.3     14.5     14.4     26.7
 1024    23.5     26.3     57.7     62.0     71.1     70.5     71.8     71.1     86.1
 2048    37.3     41.6    102.8    113.0    114.2    129.5    130.9    130.5    123.8
 4096    54.5     61.4    180.7    196.1    195.8    227.0    229.7    229.9    270.5
 8192    64.2     72.6    162.4    213.8    215.2    247.3    254.1    254.2    296.1
```

## Blog: From 20% to 81% of cuBLAS

### The Starting Point

We start with a naive BF16 GEMM on A100. The GPU has 312 TFLOPS of BF16 tensor core throughput. Our v1 kernel hits 64 TFLOPS — 20.5% of peak. The gap is massive. Here's how we closed it.

### Step 1: Async Pipeline (+13%)

**v1 → v2: 64 → 73 TFLOPS**

v1 uses synchronous `__syncthreads()` barriers between every load and compute phase. The GPU sits idle during global memory loads. We replace this with `cp.async` — an Ampere instruction that copies global→shared memory without going through registers. Combined with 2-stage double buffering (load stage N+1 while computing stage N), we overlap memory latency with compute.

**Takeaway**: Always pipeline. The hardware supports it, use it.

### Step 2: Kill the Bank Conflicts (+124%)

**v2 → v3: 73 → 162 TFLOPS** — the single biggest win.

v2 achieves only 73 TFLOPS despite pipelining. The culprit: **shared memory bank conflicts**. When 32 threads in a warp access shared memory addresses that map to the same bank, accesses serialize. With BLOCK_K=64 (64 bf16 elements = 128 bytes per row), consecutive threads access consecutive 4-byte words, creating massive conflicts.

The fix is trivially simple: **pad each row by 8 elements** (`SMEM_STRIDE = BLOCK_K + 8 = 72`). This shifts the bank mapping so threads no longer collide. One constant changes. Performance doubles.

**Takeaway**: Bank conflicts are the silent killer. Profile them. Fix them.

### Step 3: Swizzle Over Padding (+32%)

**v3 → v4: 162 → 214 TFLOPS**

Padding works but wastes shared memory (+16KB per CTA). The better approach: **XOR swizzle**. Instead of shifting addresses with padding, we remap them with `address ^ (row * 8)`. This achieves the same bank-conflict-free access pattern without wasting memory. The swizzle is applied at two points:
1. **cp.async writes**: where each thread stores into shared memory
2. **ldmatrix reads**: where each thread loads from shared memory

Same performance, less memory, more room for multi-stage pipelines.

**Takeaway**: Swizzle > padding. Same effect, no memory overhead.

### Step 4: Clean Rewrite (+1%)

**v4 → v7s3: 214 → 215 TFLOPS**

v7 is a complete rewrite incorporating three improvements:
- **`swizzle_better`**: a cleaner XOR swizzle at 16-byte granularity
- **`ldmatrix.x4` for B**: loads two B matrices in one instruction instead of two separate `ldmatrix.x2`
- **Multi-stage pipeline**: `cp_async_commit_group/wait_group` replaces `wait_all`, allowing N-1 in-flight prefetches

The performance gain is small because the 128×128 tile is already well-utilized. The real benefit is code cleanliness for future optimizations.

### Step 5: Help the Compiler (+15%)

**v7s3 → v10: 215 → 247 TFLOPS**

This one surprised us. The compute logic is identical — same ldmatrix, same MMA, same pipeline. The difference: **register variable scope**.

In v7, accumulator registers (`A_buf`, `B_buf`) are declared at kernel scope. In v10, they're declared inside the compute lambda. This gives the CUDA compiler better visibility into register lifetimes, enabling more aggressive instruction scheduling and register allocation. A 15% jump from a code reorganization.

**Takeaway**: Compiler hints matter. Local variables give better optimization scope.

### Step 6: More Warps (+3%)

**v10 → v11a: 247 → 254 TFLOPS**

We double the warp count from 2×2 (4 warps, 128 threads) to 4×2 (8 warps, 256 threads). Each warp now computes a smaller 32×64 sub-tile instead of 64×64. Benefits:
- More warps to hide memory latency (instruction-level parallelism)
- Lower per-warp register pressure (smaller sub-tile)
- Better occupancy

**Takeaway**: More warps help, but with diminishing returns.

### Step 7: Double-Buffer the Registers (+0%)

**v11a → v13d: 254 → 254 TFLOPS**

We combine two techniques:
- **Double-buffered register loads**: while MMA executes for k-slice N, ldmatrix loads registers for k-slice N+1
- **GROUP_M=16**: Triton-style block swizzle grouping 16 M-rows for L2 cache reuse of B

At this point we're hitting the ceiling of what this tile configuration can achieve. The register file and instruction throughput are saturated.

### What Didn't Work

**v14 (smem epilogue)**: We tried staging the output through shared memory for coalesced 128-bit int4 stores. The extra `__syncthreads()` barrier and shared memory round-trip cost more than the store coalescing saves. **-4% regression**.

**Stream-K**: We implemented the Stream-K parallel decomposition (split K-dimension across CTAs with atomicAdd accumulation). The 67M atomicAdd operations to global memory destroy performance. **93 TFLOPS (30%)** — not competitive for this problem size.

### The Remaining 14% Gap

cuBLAS achieves 296 TFLOPS (94.9%). Our best is 254 TFLOPS (81.5%). The ~14% gap likely comes from:

1. **L2 cache tiling**: cuBLAS uses sophisticated tile scheduling for L2 reuse
2. **Warp specialization**: dedicated load warps vs compute warps
3. **Epilogue optimization**: better global memory access patterns
4. **Instruction scheduling**: hand-tuned PTX scheduling

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| Tensor Core Peak | 312 TFLOPS (BF16, SM80) |
| Architecture | Ampere (`sm_80`) |
| SMs | 108 |
| Shared Memory / SM | 164 KB (configurable) |
| HBM Bandwidth | 1.5 TB/s |

## Files

```
kernels/cuda/A100/
├── common.h            PTX helpers: cp_async, LDMATRIX_X4, MMA_M16N8K16, swizzle
├── matmul_v1.cu        Baseline: sync loads, 2x2 warps, single-stage
├── matmul_v2.cu        + cp.async + 2-stage pipeline
├── matmul_v3.cu        + shared memory padding (bank conflicts)
├── matmul_v4.cu        + XOR swizzle replaces padding
├── matmul_v5.cu        v3 variant (threadblock swizzle exploration)
├── matmul_v6.cu        v4 variant (swizzle on store + load)
├── matmul_v7.cu        + swizzle_better + ldmatrix_x4 + multi-stage
├── matmul_v8.cu        + tunable GROUP_M parameter
├── matmul_v9.cu        + tile/warp exploration (256x128, 4x4)
├── matmul_v10.cu       + lambda-local register declarations
├── matmul_v11.cu       + 4x2 warps (256 threads)
├── matmul_v12.cu       + double-buffered register loads
├── matmul_v13.cu       v11a + v12 combined (best: v13d)
├── matmul_v14.cu       smem epilogue experiment
├── matmul_streamk.cu   Stream-K decomposition
└── benchmark.cu        Harness: timing, correctness, cuBLAS comparison
```

## Building & Running

```bash
# On Modal (A100 GPU)
modal run scripts/run.py --task kernels/cuda/A100/benchmark.cu

# Local compile
nvcc -O3 -arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
    -lcublas -o benchmark benchmark.cu
./benchmark
```

## Adding a New Kernel

1. Create `matmul_vN.cu` with `matmul_vN_launch(A, B, C, M, N, K)`
2. In `benchmark.cu`: `#include "matmul_vN.cu"` + add to `kernels[]` array
3. Run: `modal run scripts/run.py --task kernels/cuda/A100/benchmark.cu`

## Key Takeaways

| Optimization | Impact | Difficulty |
|-------------|--------|------------|
| `cp.async` pipeline | +13% | Low |
| Bank conflict fix (padding) | **+124%** | Trivial |
| XOR swizzle (replaces padding) | +32% | Medium |
| Lambda-local registers | +15% | Trivial |
| More warps (4x2) | +3% | Low |
| Double-buffered regs | +0% | High |

**The lesson**: Profile before optimizing. Bank conflicts gave us a 2x speedup with a one-line change. Meanwhile, complex techniques like double-buffered registers gave us nothing at the ceiling.
