# Chasing cuBLAS: A BF16 Tensor Core GEMM on A100

Seven hand-optimized BF16 GEMM kernels on A100-SXM4-40GB. From 20.6% to 83.1% of cuBLAS peak — each kernel adds exactly one optimization technique.

| Kernel | TFLOPS | % Peak | Δ | Technique |
|--------|--------|--------|---|-----------|
| v1 | 64.2 | 20.6% | — | Baseline: sync loads, 2×2 warps |
| v2 | 73.2 | 23.5% | +14% | `cp.async` pipeline |
| v3 | 152.1 | 48.7% | **+108%** | Bank conflict fix |
| v4 | 153.9 | 49.3% | +1% | XOR swizzle |
| v7s3 | 219.5 | 70.3% | +43% | Multi-stage + ldmatrix.x4 |
| v10 | 252.6 | 81.0% | +15% | Lambda-scoped registers |
| v11a | 258.7 | 82.9% | +2% | 4×2 warps (256 threads) |
| **cuBLAS** | **300.4** | **96.3%** | — | Reference |

```
Performance climb (N=16384, A100-SXM4-40GB)
TFLOPS
300 ┤                                            ▓ cuBLAS 300
250 ┤                              ▓ v11a 259
   ┤                          ▓ v10 253
200 ┤              ▓ v7s3 220
   ┤
150 ┤          ▓ v3 152  ▓ v4 154
100 ┤
 50 ┤ ▓ v1 64  ▓ v2 73
   └───┬────┬────┬────┬────┬────┬────┬────
       v1   v2   v3   v4  v7s3 v10 v11a cuBLAS
```

Hardware: A100-SXM4-40GB, 108 SMs, 312 TFLOPS BF16 peak, 1.5 TB/s HBM2e.

---

## The MMA Atom

Everything builds on one instruction:

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

It computes **D[16×8] += A[16×16] × B[16×8]** — 256 MACs (512 FLOPs) per warp, per cycle.

But tensor cores don't read memory. They read **registers**. The pipeline:

```
GMEM ──cp.async (16B)──▸ SHMEM ──ldmatrix (8×8)──▸ REGS ──mma──▸ ACC
 ~500 cyc                    ~30 cyc                1 cyc issue
```

The entire game: keep this pipeline full.

---

## v1 — Baseline (64 TFLOPS, 20.6%)

### Tile map

One CTA = 128×128 output tile. Four warps (2×2), each owns 64×64.

```
CTA (128×128):
┌──────────────┬──────────────┐
│ warp(0,0)    │ warp(0,1)    │
│ 64×64        │ 64×64        │
├──────────────┼──────────────┤
│ warp(1,0)    │ warp(1,1)    │
│ 64×64        │ 64×64        │
└──────────────┴──────────────┘
```

Each warp splits 64×64 into 4×8 = 32 MMA tiles of 16×8.

### K-loop

K processed in chunks of BLOCK_K=64. For each chunk: load A (128×64) and B (128×64) to shared memory, then 4 inner steps of MMA_K=16.

```cpp
for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    gmem2smem(A);       // sync copy 128×64
    gmem2smem(B);       // sync copy 128×64
    __syncthreads();

    for (int k = 0; k < BLOCK_K; k += MMA_K) {
        ldmatrix(A);    // smem → regs
        ldmatrix(B);    // smem → regs
        mma(32 tiles);  // compute
    }
    __syncthreads();
    advance K pointer;
}
```

### Why it's slow

```
Timeline per k-chunk (synchronous):

gmem load A ████████░░░░░░░░░░░░░░░░   (~200 cyc)
gmem load B ░░░░████████░░░░░░░░░░░░   (~200 cyc)
__syncthreads ░░░░░░░░████░░░░░░░░░░░░   (~40 cyc)
compute     ░░░░░░░░░░░░████████░░░░   (~50 cyc)
__syncthreads ░░░░░░░░░░░░░░░░░░████   (~40 cyc)

~500 cycles total, ~50 cycles compute = 10% utilization
```

Nsight Compute: ~80% stall cycles (mostly "long scoreboard" = waiting for global memory).

---

## v2 — Async Pipeline (73 TFLOPS, +14%)

`cp.async` copies global → shared memory through a dedicated copy engine, bypassing registers. Returns immediately.

### Double-buffered timeline

```
v1 (synchronous):
     load ████████ compute ████████ load ████████ compute ████████

v2 (2-stage pipeline):
  load ████████░░░░░░████████░░░░░░
  comp ░░░░████████░░░░░░████████░░
             ↑ overlap!
```

Two shared memory buffers: compute on stage 0 while loading stage 1.

```cpp
// Prologue: kick off first load
cp_async(to_smem(buf[0]), A + 0, 16);
cp_async(to_smem(buf[0]), B + 0, 16);
cp_async_commit_group();

// Main loop
for (int k = 1; k < num_k_iters; k++) {
    int prev = (k - 1) % 2;
    int next = k % 2;

    cp_async_wait_group<1>();        // wait for prev load
    __syncthreads();
    compute(prev);                    // compute prev stage
    __syncthreads();

    cp_async(to_smem(buf[next]), A + k * BLOCK_K);
    cp_async(to_smem(buf[next]), B + k * BLOCK_K);
    cp_async_commit_group();
}
```

Result: 73 TFLOPS. Better, but still terrible. The pipeline hides gmem latency, revealing a **much bigger** bottleneck.

Warps now stall on `ldmatrix`, not gmem. Why?

---

## v3 — Bank Conflicts (152 TFLOPS, +108%)

### The problem

Shared memory = 32 banks, 4 bytes wide. Consecutive 4-byte words map to consecutive banks. When two threads in a warp access the same bank, accesses serialize.

v2 config: `SMEM_STRIDE = BLOCK_K = 64` elements = 128 bytes/row.

```
Thread accessing row r, col c:
  addr = r × 128 + c × 2  (bytes)
  bank = (addr / 4) % 32
       = (r × 32 + c/2) % 32
       = (c/2) % 32              ← ONLY depends on column!

32 threads reading consecutive rows at same column → SAME BANK → 32-way conflict
```

Every `ldmatrix` was serializing 32×.

### The fix: one constant

```cpp
// v2: SMEM_STRIDE = 64
// v3: SMEM_STRIDE = 72   (+8 elements = 16 bytes padding)
```

```
v2 (no padding, stride=64):
┌────────────────────────────┐
│ row 0: [0 ... 63]         │  all start at bank 0
│ row 1: [0 ... 63]         │  all start at bank 0  ← conflict!
│ row 2: [0 ... 63]         │  all start at bank 0
└────────────────────────────┘

v3 (padded, stride=72):
┌──────────────────────────────────────┐
│ row 0: [0 ... 63] ░░░░░░░░░░        │  16 bytes padding
│ row 1: [0 ... 63] ░░░░░░░░░░        │  starts at bank 4
│ row 2: [0 ... 63] ░░░░░░░░░░        │  starts at bank 8
└──────────────────────────────────────┘
  bank = (r × 144 + c × 2) / 4 % 32
       = (r × 4 + c/2) % 32           ← depends on BOTH row and col
  → conflict-free
```

### The result

```
v2:  73 TFLOPS  ░░░░░░░░░░░░░░░░░░░░░░░░░░
v3: 152 TFLOPS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
              ↑ +108% from changing ONE CONSTANT
```

Nsight Compute: bank conflict replay ratio drops from ~95% to ~0%. **Highest ROI optimization ever.**

---

## v4 — XOR Swizzle (154 TFLOPS, +1%)

Padding wastes 12.5% of shared memory (+16 KB/CTA). Better: XOR the address to spread rows across banks **without** wasting space.

```cpp
template <int STRIDE_BYTES>
__device__ uint32_t swizzle(uint32_t index) {
    uint32_t row = (index / STRIDE_BYTES) % 8;
    uint32_t xor_bits = row / (128 / STRIDE_BYTES > 1 ? 128 / STRIDE_BYTES : 1);
    return index ^ (xor_bits << 4);   // XOR bit 4 (bank group selector)
}
```

For BLOCK_K=64, STRIDE_BYTES=128:

```
row 0:  index ^ 0x00  (no change)
row 1:  index ^ 0x10  (flip bit 4 → different bank group)
row 2:  index ^ 0x20  (flip bit 5)
...
row 7:  index ^ 0x70
row 8:  index ^ 0x00  (pattern repeats every 8 rows)
```

Bit 4 selects the bank group within a 16-byte chunk. XORing it with the row index spreads consecutive rows across different groups → no conflicts.

**Must apply at both ends:**

```
cp.async ──write──▸ SHMEM ──ldmatrix──▸ REGS
            ↑                   ↑
        swizzle addr        swizzle addr
```

```
| Technique   | Smem/CTA | TFLOPS | % Peak |
|-------------|----------|--------|--------|
| No fix      | 32 KB    | 73     | 23.5%  |
| Padding     | 36 KB    | 152    | 48.7%  |
| XOR swizzle | 32 KB    | 154    | 49.3%  |
```

Same perf as padding, zero memory waste. The saved KB enables deeper pipelines and larger tiles.

---

## v7s3 — Multi-Stage + ldmatrix.x4 (220 TFLOPS, +43%)

A complete rewrite incorporating three improvements:

### `swizzle_better`
A cleaner XOR swizzle at 16-byte granularity, simpler to understand and apply.

### `ldmatrix.x4` for B
Loads two B matrices in one instruction instead of two separate `ldmatrix.x2`. Reduces instruction count by 25% for B loads.

### Multi-stage pipeline
`cp_async_commit_group/wait_group` replaces `wait_all`, allowing N-1 in-flight prefetches instead of just 1.

```
v4 (2-stage):    load ████████ compute ████████ load
v7s3 (3-stage):  load ████████ load ░░░░░░░░ compute ████████
                                  ↑ 2 loads ahead, more overlap
```

The combination unlocks significantly better compute utilization.

---

## v10 — Lambda-Scoped Registers (253 TFLOPS, +15%)

**Same algorithm. Same instructions. Only change: variable scope.**

```cpp
// v7: kernel-scoped
__global__ void matmul(...) {
    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};
    uint32_t A_buf[4], B_buf[2];

    for (k_iter ...) {
        for (k ...) {
            ldmatrix(A_buf, ...);
            ldmatrix(B_buf, ...);
            mma(A_buf, B_buf, acc[m][n]);
        }
    }
}

// v10: lambda-scoped
__global__ void matmul(...) {
    auto compute = [&](int k_iter) {
        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};
        uint32_t A_buf[4], B_buf[2];
        for (k ...) {
            ldmatrix(A_buf, ...);
            ldmatrix(B_buf, ...);
            mma(A_buf, B_buf, acc[m][n]);
        }
    };

    for (k_iter ...) {
        g2s_swizzled(...);
        compute(k_iter);
    }
}
```

### Why does this matter?

Kernel scope: `acc`, `A_buf`, `B_buf` all in same scope. Compiler **can't prove** `ldmatrix` doesn't touch `acc`. It **must** spill `acc` registers across `ldmatrix` calls.

Lambda scope: closed scope. Compiler **proves** `ldmatrix` doesn't touch `acc`. Result:
- No spilling between `ldmatrix` and `mma`
- Better instruction scheduling
- Higher register allocation quality

```
v7s3: 220 TFLOPS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
v10:  253 TFLOPS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
              ↑ +15%, zero instruction changes
```

PTX confirms: `st.local` (spill) instructions disappear in v10.

---

## v11a — More Warps (259 TFLOPS, +2%)

Double warps: 4 (128 threads, 2×2) → 8 (256 threads, 4×2).

```
4 warps (128 threads):
┌──────────────┬──────────────┐
│ warp_0       │ warp_1       │
│ 64×64        │ 64×64        │
├──────────────┼──────────────┤
│ warp_2       │ warp_3       │
│ 64×64        │ 64×64        │
└──────────────┴──────────────┘

8 warps (256 threads):
┌──────────────┬──────────────┐
│ warp_0       │ warp_1       │  rows 0-31
├──────────────┼──────────────┤
│ warp_2       │ warp_3       │  rows 32-63
├──────────────┼──────────────┤
│ warp_4       │ warp_5       │  rows 64-95
├──────────────┼──────────────┤
│ warp_6       │ warp_7       │  rows 96-127
└──────────────┴──────────────┘

Each warp: 32×64 (half size!) → 16 MMA tiles instead of 32
```

Benefits:
- **More ILP**: 8 independent instruction streams vs 4
- **Lower reg pressure**: 64 acc regs/warp vs 128 → less spill
- **Better latency hiding**: when warp 0 stalls on ldmatrix, warps 1-7 issue mma

```
A100 SM registers: 65,536 × 32-bit

128 threads/CTA: 65,536 / 128 = 512 regs/thread max
256 threads/CTA: 65,536 / 256 = 256 regs/thread max (sweet spot)
512 threads/CTA: 65,536 / 512 = 128 regs/thread max (would spill)
```

---

## The Remaining ~14% Gap

cuBLAS achieves 300 TFLOPS (96.3%). Our best is 259 TFLOPS (82.9%). The ~14% gap likely comes from:

| Technique | Est. impact | Notes |
|-----------|------------|-------|
| L2 cache tiling | 5-6% | cuBLAS uses Morton/Hilbert CTA scheduling |
| Warp specialization | 3-4% | Dedicated load warps vs compute warps |
| Epilogue fusion | 2-3% | cuBLAS fuses cast+store with optimized access patterns |
| Autotuning | 2-3% | Hundreds of hand-tuned kernels for different sizes |
| Instruction scheduling | 1-2% | Hand-scheduled PTX vs compiler output |

---

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

---

## What Matters Most

```
Optimization           Impact    Effort
────────────────────────────────────────
Bank conflict fix      +108%     1 constexpr change
Multi-stage + ldmatrix +43%      moderate
Lambda-scoped regs      +15%     move 2 lines
Async pipeline          +14%     moderate
More warps (4×2)        +2%      low
```

**The lesson**: Profile before optimizing. The bank conflict fix — one `constexpr` from 64 to 72 — gave 2.4× the performance of every other optimization combined.

---

```bash
nvcc -O3 -arch=sm_80 --std=c++17 --expt-relaxed-constexpr \
    -lcublas -o benchmark benchmark.cu
./benchmark
```
