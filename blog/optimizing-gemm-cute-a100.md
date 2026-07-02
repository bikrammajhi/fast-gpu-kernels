# Optimizing bf16 GEMM on A100 with CuTe: 17% → 79% of cuBLAS

**Hardware**: NVIDIA A100-SXM4-40GB (108 SMs, SM80)
**Problem**: M=N=K=8192, bf16
**Reference**: cuBLAS ≈ 266 TFLOPS
**Result**: v37 hand-written PTX kernel → 211 TFLOPS (79.4% of cuBLAS)

---

## 1. Baseline: Synchronous gmem→smem copy (v1) — 46 TFLOPS (17.3%)

A 128×128×64 CTA tile with synchronous `UniversalCopy` for gmem→smem, row-major smem layout, and 4 warps (2×2 MMA atoms). Two bottlenecks:

1. **Warp stall on every gmem→smem transfer.** `UniversalCopy` is synchronous; the warp waits for the load to complete before issuing the next instruction.
2. **8-way SMEM bank conflicts.** With stride=`bK=64`, 32 threads in a warp access consecutive 4-byte bf16 values, all mapping to the same bank on A100's 32-bank interleaved scheme.

```cpp
auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<64>{};
auto sA = make_layout(make_shape(bM, bK), make_stride(bK, Int<1>{}));
TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                               Layout<Shape<_2,_2>>{}, Tile<_32,_32,_16>{});
```

---

## 2. Vectorized global loads (v2) — 58 TFLOPS (+26%)

Switch gmem copy atom to `UniversalCopy<uint128_t>`. Each thread issues one 128-bit load per logical copy call, matching the memory transaction width. Zero kernel-structure changes.

```cpp
using GmemCopyAtom = UniversalCopy<uint128_t>;
```

---

## 3. SMEM padding (v3) — 134 TFLOPS (+131%)

**Single biggest win in the entire ladder.**

Pad each smem row by 8 bf16 elements (`stride = bK + 8 = 72`). This shifts the bank mapping so threads no longer collide. Zero runtime cost — compiled into the layout stride.

```cpp
constexpr int kPad = 8;
auto sA = make_layout(make_shape(bM, bK), make_stride(bK + kPad, Int<1>{}));
```

> **Takeaway**: Profile bank conflicts first. One constant can double throughput.

---

## 4. Swizzle layout (v4) — 115 TFLOPS (regression, −14%)

Replace padding with `Swizzle<3,3,3>` to remap addresses structurally, saving 16 KB smem per CTA. **Regressed at 8192**: XOR address arithmetic runs through the register pipeline, adding ~5 cycles per LDSM access. The conflict savings don't offset the XOR cost at this tile size. Padding wins for ≤ 8192; swizzle wins at ≥ 16384.

---

## 5. `cp.async` CACHEALWAYS (v5) — 171 TFLOPS (+48%)

`SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` issues non-blocking global loads bypassing L1, writing directly to SMEM and pinning in L2 for LDSM.

```cpp
using GmemCopyAtom = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
// K-loop:
copy(copy_a, tAgA(_,_,_,k_tile), tAsA);
cp_async_fence();
cp_async_wait<0>();
__syncthreads();
// LDSM + MMA...
gemm(mma, tCrA, tCrB, tCrC);
__syncthreads();   // ← prevents next DMA from clobbering current tile
```

### Critical correctness bug

The first cp.async implementation **omitted `__syncthreads()` after `gemm()`**. Because `cp.async` DMA completes hundreds of cycles after the issuing instruction retires, a later tile's `cp.async` could overwrite the current smem tile while another warp was still reading it via LDSM — a silent data race.

Fix: `__syncthreads()` after `gemm()`. Cost: ~2.4% throughput loss vs the momentarily-buggy version.

---

## 6. cp.async + swizzled smem (v6) — 180 TFLOPS (+5% over v5)

v5 uses padded row-major smem with no stages dimension and a synchronous K-loop. v6 swaps padding for `Swizzle<3,3,3>` while keeping everything else identical: cp.async, single-stage smem, same synchronous K-loop. The XOR cost of swizzle is amortized at 8192 over the bank-conflict savings.

```cpp
// v6 — swizzle, but still single-stage (no stages dimension)
auto sA = tile_to_shape(swizzled_128B_atom, make_shape(bM, bK));
```

No pipeline yet. The +5% is purely from swizzle outperforming padding at this tile size.

---

## 7. 2-stage pipeline + pipelined loop (v7) — 173 TFLOPS (−4% vs v6)

v7 adds a `stages` dimension and replaces the synchronous K-loop with an explicit prefetch/rotate `while` loop:

```cpp
// v7 — 2-stage pipelined prefetch loop
auto bP = Int<2>{};
auto sA = tile_to_shape(swizzled_128B_atom, make_shape(bM, bK, Int<bP>{}));
auto sB = tile_to_shape(swizzled_128B_atom, make_shape(bN, bK, Int<bP>{}));
```

The `while` loop issues `cp.async` for the next K-tile while the current tile is in MMA, rotating `smem_pipe_read`/`smem_pipe_write` pointers. Despite adding a software pipeline, v7 is **marginally slower than v6** at 8192. The dynamic `while`-loop pointer-swing (`tXsA(_,_,_,smem_pipe_read)` index math per iteration) adds overhead that exceeds the latency-hiding benefit for this tile size. The 2-stage buffer is too shallow to absorb the full `cp.async` round-trip on A100.

---

## 8. 3-stage sweet spot (v8) — 200 TFLOPS (+16% over v7)

v7 uses `bP = Int<2>{}` (2 stages = one double-buffer). v8 increases to `bP = Int<3>{}` (3 stages = two tiles buffered + one in-flight):

```cpp
// v8 — the only change from v7
auto bP = Int<3>{};
auto sA = tile_to_shape(swizzled_128B_atom, make_shape(bM, bK, Int<bP>{}));
auto sB = tile_to_shape(swizzled_128B_atom, make_shape(bN, bK, Int<bP>{}));
```

3 stages lets the `cp.async` round-trip fully hide behind the current tile's MMA. 4+ stages over-subscribes registers for 128×128×64 tiles. This is the SMEM sweet spot: ~49 KB raw per tile × 3 stages = ~294 KB, within the A100 practical limit for the swizzled layout.

---

## 9. Plateau and PTX escape (v37) — 211 TFLOPS (+7%)

From v8 onward, tuning grid swizzle factors, K-tile sizes, and scheduling moves the needle by <5%. The final gain comes from v37: abandoning CUTE's high-level API entirely for hand-written PTX inline-asm with `ldmatrix.x4`, `mma.sync.aligned.m16n8k16`, and explicit double-buffered register files.

---

## Key Takeaways

| Optimization | 8192 TFLOPS | % cuBLAS | Difficulty |
|---|---:|---:|---|
| cp.async + correct sync | 171 | 64% | Low |
| **Bank conflict fix (padding)** | **134** | **50%** | **Trivial** |
| 3-stage swizzled pipe | 200 | 75% | Medium |
| Hand-written PTX | 211 | 79% | High |
| **Remaining gap to cuBLAS** | **55 TFLOPS** | **21%** | — |

| Rule | Insight |
|---|---|
| Profile first | Bank conflicts gave +131% with one constant |
| `cp.async` is mandatory | Required for >150 TFLOPS |
| `__syncthreads()` after `gemm()` | Prevents silent DMA races |
| 3 stages is the sweet spot | SMEM budget vs register pressure |
| CUTE has a ceiling | ~200 TFLOPS (75%); inline PTX adds +11 |

---

## Files

```
kernels/cute/A100/
├── matmul_v1.cu … matmul_v9.cu      Optimization ladder
├── matmul_v21.cu … matmul_v37.cu    Later tuning attempts
├── experiments/                      Regressed / broken kernels
├── good_versions/                    Confirmed high-performing copies
├── benchmark.cu                      Multi-size benchmark vs cuBLAS
├── bench_all.sh                      Modal runner v1–v6
├── bench_all_8192.sh                 8192 sweep v1–v37
├── README.md                         ← you are here
├── Notes.md                          Copy atom / LDSM / bank-conflict notes
└── blog/
    └── optimizing-gemm-cute-a100.md  This document
```

---

## Building & Running

```bash
# Single kernel locally
nvcc -O3 -arch=sm_80 \
  -I$CUTLASS/include -I$CUTLASS/tools/util/include \
  -lcublas matmul_v8.cu -o /tmp/matmul_v8 && /tmp/matmul_v8

# Multi-size benchmark via Modal
bash kernels/cute/A100/bench_all.sh

# 8192×8192×8192 sweep (v1–v37 + cuBLAS baseline)
bash kernels/cute/A100/bench_all_8192.sh
```

---

## References

- `kernels/cute/A100/Notes.md` — inline rationale for copy atoms, LDSM, and bank-conflict strategies
- `kernels/cute/A100/experiments/` — regressed kernels with diagnosis
- `kernels/cute/A100/good_versions/` — mirrored copies of confirmed high-performing kernels
