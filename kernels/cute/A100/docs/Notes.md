# Tiling & Layout Rationale for `matmul_v1.cu`

This document explains why specific shapes and layouts were chosen for the bf16 GEMM kernel targeting A100 (SM80).

---

## 1. Gmem → Smem Copy Tiling (cp.async)

```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
    Layout<Shape<_16,_8>, Stride<_8,_1>>{},   // Thr layout
    Layout<Shape< _1,_8>>{}                    // Val layout
);
```

### Copy Atom: `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>`

- Each `cp.async` instruction moves **128 bits = 16 bytes = 8 bf16 values**.
- This is the widest L1/SMEM transaction on A100, giving peak bandwidth.
- `CACHEALWAYS` pins the data in L1/L2 for the tensor core's LDSM read.

### Val layout `(1, 8)`

- Each thread handles **1 element along M × 8 contiguous elements along K**.
- Since one instruction moves 8 values, every thread issues exactly **one `cp.async.128`** per logical copy call. No wasted throughput.

### Thr layout `(16, 8)` with stride `(8, 1)`

- **16 × 8 = 128 threads**, matching `size(mmaC)` (no idle threads).
- **16 threads along M** with stride 8 → thread `i` handles every 8th M-position. This interleaved pattern ensures **coalesced global memory accesses** — adjacent threads hit adjacent 128B-aligned cache lines.
- **8 threads along K** with stride 1 → cover the K dimension contiguously.

### Coverage of a (128, 64) tile

| Dimension | Threads × Values/thread | Coverage |
|-----------|------------------------|----------|
| M         | 16 × 1                 | 16       |
| K         | 8 × 8                  | 64       |
| **Total** |                        | **1024 elements per call** |

The CTA tile is **128×64 = 8192 elements** → each thread needs **8192 / 1024 = 8 calls** to fill the tile. The extra `CPY_M` inner dimension handles this: each thread does 8 `cp.async` calls per K-tile step, moving 64 values total.

---

## 2. MMA Tiling

```cpp
TiledMMA mmaC = make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_2,_2>>{},    // Atom replication
    Tile<_32,_32,_16>{}        // Logical MMA tile
);
```

### MMA Atom: `SM80_16x8x16_F32BF16BF16F32_TN`

The A100 tensor core instruction — one warp (32 threads) computes:

```
C(16 × 8 f32) += A(16 × 16 bf16) × B(8 × 16 bf16)
```

- `_TN` — A is row-major (T = Transposed from col-major), B is column-major (N = Not transposed), matching `C = A × B` convention.
- Each thread holds fragments of A, B, and C in registers.

### Atom Replication: `Layout<Shape<_2,_2>>`

Arranges **4 MMA atoms = 4 warps** in a 2×2 grid:

| Warp | C output region |
|------|----------------|
| `(0,0)` | Top-left 16×8 |
| `(0,1)` | Top-right 16×8 |
| `(1,0)` | Bottom-left 16×8 |
| `(1,1)` | Bottom-right 16×8 |

Total = 4 warps × 32 threads = **128 threads**, matching the copy tiling.

### Logical Tile: `Tile<_32,_32,_16>`

This is the **accumulator tile per `gemm()` call** within the kernel loop. It tells CuTe how to partition the A/B register fragments that feed each MMA instruction.

With CTA tile = 128×128×64:

```
Per K-tile (size 64):
  MMA tiles along M = 128 / 32 = 4
  MMA tiles along N = 128 / 32 = 4
  MMA tiles along K =  64 / 16 = 4
Total MMA calls per CTA = 4 × 4 × 4 = 64  (16 per warp)
```

Each MMA call is one warp-level tensor core instruction. The 4 K-tiles along the reduction dimension mean each warp accumulates 4 partial products before writing out the final C fragment.

---

## 3. LDSM (Smem → Registers)

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_A;
```

- `SM75_U32x4_LDSM_N` loads **4 × uint32 = 128 bits = 8 bf16 values** per thread from shared memory into registers via a single LDSM instruction.
- This is the **exact inverse** of the cp.async store (also 128b per thread) — the data layout maps 1:1 without wasted bandwidth.
- The `_N` suffix means "normal" (column) access — matching the K-major layout of each thread's 8-value chunk written by cp.async.

When combined via `make_tiled_copy_A(s2r_atom_a, mma)`, CuTe creates a `TiledCopy` that routes the correct 8-element segment from smem into each MMA thread's A-fragment registers — matching the MMA atom's 16×16×16 input tile.

---

## Summary

| Parameter | Value | Rationale |
|---|---|---|
| Copy atom | `uint128_t` (8 vals) | Widest SMEM transaction = peak BW |
| Copy threads | 128 (`16×8`) | Matches MMA thread count (no idle threads) |
| Thr stride (M) | 8 | Coalesced global memory access pattern |
| MMA atom | 16×8×16 | A100's native tensor core shape |
| MMA replication | 2×2 = 4 warps | Fills all 128 threads |
| MMA tile | 32×32×16 | Balances register pressure vs. data reuse |
| LDSM | U32x4 (8 vals) | Matches cp.async stride 1:1 |

---

## 4. Smem Conflict Mitigation: Padding vs Swizzle

Two approaches to reduce bank conflicts in the row-major smem layout:

### Padding (`kPad=8`)
```cpp
auto sA = make_layout(make_shape(bM, bK), make_stride(bK + kPad, Int<1>{}));
```
Adds extra elements per row so adjacent rows don't map to the same banks. No XOR overhead — the offset is baked into the stride address calculation at zero extra cost.

### Swizzle (`Swizzle<3,3,3>`)
```cpp
auto swizzled_atom = composition(Swizzle<3,3,3>{}, make_layout(...));
auto sA = tile_to_shape(swizzled_atom, make_shape(bM, bK));
```
XOR transformation that permutes bits `[6:8]→[3:5]` on every smem address, guaranteeing no bank conflicts regardless of access pattern. Costs ~3-5 cycles of ALU per address calculation.

### Performance comparison

| Copy strategy | Size | Padding | Swizzle | Winner |
|---------------|------|:-------:|:-------:|:------:|
| **UniversalCopy** | 5120×4096 | 105.4 | 97.6 | padding |
| cp.async | 1024 | 47.8 | 44.0 | padding |
| cp.async | 2048 | 111.2 | 110.0 | padding |
| cp.async | 4096 | 151.8 | 147.6 | padding |
| cp.async | 5120×4096 | 198.8 | 189.1 | padding |
| cp.async | **8192** | 171.8 | **172.4** | **swizzle** |
| cp.async | **16384** | 155.3 | **156.1** | **swizzle** |
| cp.async | **32768** | 145.7 | **148.3** | **swizzle** |

### Key insights

1. **cp.async dominates** — switching from UniversalCopy to cp.async gives a bigger jump (40→162 TFLOPS at 5120) than the choice of padding vs swizzle.

2. **Padding wins at medium sizes** (≤5120) — the XOR overhead of swizzle costs more than the remaining bank conflicts from padding's 1-bank offset. At 5120×4096, padding leads by ~5%.

3. **Swizzle wins at large sizes** (≥8192) — as the problem grows, more tiles run and the bank conflict avoidance compounds. The XOR cost is fixed per access, so it's amortized. At 32768, swizzle leads by ~1.8%.

4. **Cross-over** happens between 5120 and 8192 — this is where the accumulated bank conflict replays exceed the fixed XOR overhead.

5. **With sync copies** (UniversalCopy), padding wins at all sizes — the XOR address computation goes through the register pipeline, adding latency that isn't hidden. With cp.async, the address computation overlaps with the DMA setup, reducing the XOR cost.
