# Deep Dive: Tensor Core MMA Swizzle Layout

This document explains how swizzle layouts enable bank-conflict-free shared memory access for tensor core MMA instructions on Ampere GPUs. Based on the [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-swizzling-modes) and the excellent [blog by Yifan Yang](https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html).

---

## 1. Why Swizzle Exists

Tensor cores don't read shared memory directly. Warps use `ldmatrix` to load **8×16B subtiles** from shared memory into registers. The `ldmatrix.m8n8` instruction must load this subtile at **full SMEM read bandwidth (128B/cycle)** — meaning all 32 threads must access different banks simultaneously.

**Swizzle layout** is the exact solution that guarantees this property. It's not optional — it's a prerequisite for writing any functional MMA kernel.

---

## 2. Shared Memory Bank Organization

Each SM has **32 banks**, 4 bytes wide. Consecutive 4-byte words map to consecutive banks.

```
Byte address:     0    4    8   12   16   20   24   28   32   36  ...
Bank number:      0    1    2    3    4    5    6    7    8    9  ...
Formula: bank = (byte_address / 4) % 32
```

**128 bytes** = 32 banks × 4 bytes = one full SMEM read cycle.

---

## 3. The 8×16B Subtile: ldmatrix's Unit of Work

The `ldmatrix.m8n8` instruction loads an **8×16B** subtile (M=8 rows, K=16 bytes = 8 bf16 elements). This is the fundamental unit that must be bank-conflict-free.

```
8×16B subtile = 128 bytes = 1 full SMEM read cycle

Each of 32 threads reads 4 bytes (2 bf16 elements) from a different bank.
All 32 threads complete in 1 cycle if and only if no two threads hit the same bank.
```

---

## 4. The 8 Legal Swizzle Layouts

There are 8 legal SMEM swizzle layouts the tensor core understands. They differ in two dimensions:

### Major-ness
- **K-major**: K dimension is contiguous (used for row-major A, column-major B)
- **MN-major**: M/N dimension is contiguous (used when transpose happens in ldmatrix)

### Swizzle Atom Size
- **None**: 8×16B atom (1 ldmatrix subtile per atom)
- **32B**: 8×32B atom (2 ldmatrix subtiles per atom)
- **64B**: 8×64B atom (4 ldmatrix subtiles per atom)
- **128B**: 8×128B atom (8 ldmatrix subtiles per atom)

The naming convention: `K-Major Swizzle 64B` means a K-major layout with 8×64B swizzle atoms.

---

## 5. Swizzle Atom Layouts Visualized

### Swizzle None (8×16B atom)

The simplest layout. One swizzle atom = one ldmatrix subtile.

```
8×16B atom stored contiguously in SMEM:
┌─────────────────────────────────────────────────┐
│ Chunk 0  │ Chunk 1  │ ... │ Chunk 7  │         │
│ (16B)    │ (16B)    │     │ (16B)    │         │
│ banks 0-3│ banks 4-7│     │ banks 28-31│       │
└─────────────────────────────────────────────────┘
128 bytes total = 32 banks. Perfect 1-cycle load.
```

### Swizzle 32B (8×32B atom)

Two ldmatrix subtiles per atom, interleaved to avoid bank conflicts.

```
8×32B atom (16 chunks, 256 bytes):
Chunk order in SMEM: 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15

Grey subtile (ldmatrix 0): chunks 0, 4, 8, 12, 2, 6, 10, 14
Red subtile (ldmatrix 1):  chunks 1, 5, 9, 13, 3, 7, 11, 15

Each subtile accesses 8 different banks → 1 cycle per load.
```

### Swizzle 64B (8×64B atom)

Four ldmatrix subtiles per atom. Used in our v4 kernel.

```
8×64B atom (32 chunks, 512 bytes):
4 subtiles interleaved:
  Grey:    chunks 0, 8, 16, 24, 4, 12, 20, 28
  Red:     chunks 1, 9, 17, 25, 5, 13, 21, 29
  Purple:  chunks 2, 10, 18, 26, 6, 14, 22, 30
  Blue:    chunks 3, 11, 19, 27, 7, 15, 23, 31

Each subtile accesses 8 different banks → 1 cycle per load.
```

### Swizzle 128B (8×128B atom)

Eight ldmatrix subtiles per atom. **The most commonly used layout** — maximizes GMEM access efficiency.

```
8×128B atom (64 chunks, 1024 bytes):
8 subtiles interleaved. Best GMEM read efficiency:
- 128B contiguous loads match GPU cacheline size
- Used when tile K >= 128 bytes (64 bf16 elements)
```

---

## 6. Why Swizzle ≠ Padding

### Padding (our v3 approach)

Physically space rows apart in SMEM so different rows land in different banks:

```
Stride = 72 elements (144 bytes):
Row 0: bytes   0..143  → starts at bank 0
Row 1: bytes 144..287  → starts at bank 4
Row 2: bytes 288..431  → starts at bank 8
Each row shifts by 4 banks → no conflicts

Cost: +16 bytes per row = 2 KB per matrix tile
```

### Swizzle (our v4 approach)

Keep the same stride, but **remap addresses** using XOR so threads *think* they're accessing consecutive rows, but hardware routes them to different banks:

```
swizzle(byte_addr) = byte_addr ^ (row_index << 4)

Row 0: addr ^ 0x00  (no change)    → bank 0
Row 1: addr ^ 0x10  (flip bit 4)   → bank 4
Row 2: addr ^ 0x20  (flip bit 5)   → bank 8
...
Same bank distribution as padding, zero wasted memory.
```

**Key insight from PTX spec**: Swizzle doesn't change the major-ness of the input tile. A K-major tile in GMEM stays K-major in SMEM after swizzling. No transpose happens during gmem→smem copy. Transpose happens during smem→rf copy via `ldmatrix.trans`.

---

## 7. The XOR Swizzle Formula

From CUTLASS/Triton, the general swizzle function:

```cpp
template <int STRIDE_BYTES>
__device__ uint32_t swizzle(uint32_t index) {
    uint32_t row = (index / STRIDE_BYTES) % 8;
    uint32_t divisor = 128 / STRIDE_BYTES;
    if (divisor < 1) divisor = 1;
    uint32_t xor_bits = row / divisor;
    return index ^ (xor_bits << 4);
}
```

| BLOCK_K | STRIDE_BYTES | 128/STRIDE | divisor | XOR frequency |
|---------|--------------|------------|---------|---------------|
| 32      | 64           | 2          | 2       | Every 2 rows  |
| 64      | 128          | 1          | 1       | Every row     |
| 128     | 256          | 0.5→1      | 1       | Every row     |

For our kernel (BLOCK_K=64, STRIDE_BYTES=128):

```
Row 0: index ^ 0x00  → bank 0
Row 1: index ^ 0x10  → bank 4  (XOR bit 4 = 16)
Row 2: index ^ 0x20  → bank 8  (XOR bit 5 = 32)
Row 3: index ^ 0x30  → bank 12
...
Row 7: index ^ 0x70  → bank 28
Row 8: index ^ 0x00  → pattern repeats every 8 rows
```

---

## 8. Full Trace: 8 Rows at Column 0

| Row | Linear index | row_in_cycle | XOR value | Swizzled addr | Bank |
|-----|-------------|--------------|-----------|---------------|------|
| 0   | 0           | 0            | 0x00      | 0             | 0    |
| 1   | 128         | 1            | 0x10      | 144           | 4    |
| 2   | 256         | 2            | 0x20      | 288           | 8    |
| 3   | 384         | 3            | 0x30      | 416           | 12   |
| 4   | 512         | 4            | 0x40      | 544           | 16   |
| 5   | 640         | 5            | 0x50      | 672           | 20   |
| 6   | 768         | 6            | 0x60      | 800           | 24   |
| 7   | 896         | 7            | 0x70      | 928           | 28   |

**Result**: 8 threads → 8 different banks (0,4,8,12,16,20,24,28). Zero conflicts.

---

## 9. Apply at Both Write AND Read

Swizzle must be applied at **two points**:

1. **cp.async writes**: where each thread stores into shared memory
2. **ldmatrix reads**: where each thread loads from shared memory

Both sides must use the **same permutation**, otherwise reads land in wrong locations.

```
GMEM ──cp.async (swizzle addr)──▸ SHMEM ──ldmatrix (swizzle addr)──▸ REGS
                ↑                                    ↑
           swizzle on write                   swizzle on read
```

---

## 10. 16B Atomicity: Why Swizzle Preserves Major-ness

A key property: **16B chunks remain contiguous in both GMEM and SMEM after swizzling.**

- A 16B chunk = 8 bf16 elements along the K dimension (for K-major)
- Swizzle only reorders **between** 16B chunks, never **within** a chunk
- This means a K-major tile stays K-major after swizzling

This is why we say swizzle doesn't change the major-ness of the input tile. The transpose happens during `ldmatrix.trans`, not during swizzle.

---

## 11. Which Swizzle Atom to Choose?

**Rule**: Use the largest swizzle atom that fits your tile size. This maximizes GMEM access efficiency.

| Tile K size | Best swizzle | GMEM read granularity |
|-------------|-------------|----------------------|
| 16B (8 bf16) | Swizzle None | 16B |
| 32B (16 bf16) | Swizzle 32B | 32B |
| 64B (32 bf16) | Swizzle 64B | 64B |
| ≥128B (≥64 bf16) | Swizzle 128B | 128B |

**GPU cacheline = 128B.** Larger swizzle atoms → larger contiguous reads → better bandwidth utilization.

For our kernel with BLOCK_K=64 bf16 = 128 bytes: **Swizzle 128B** is ideal (but we use 64B because our tile is only 64 bf16 wide).

---

## 12. Performance Impact

```
                     Smem/CTA    TFLOPS    GMEM efficiency
No fix (v2)           32 KB      73        16B reads
Padding (v3)          36 KB     153        16B reads (+16 bytes/row wasted)
XOR Swizzle (v4)      32 KB     155        64B reads (no waste)
```

Swizzle achieves same bank-conflict elimination as padding with:
- **Zero memory overhead** (saves 4 KB per CTA)
- **Better GMEM access** (64B reads vs 16B reads)
- **More room for pipelines** (deeper multi-stage, larger tiles)

---

## 13. Transpose: Handled by ldmatrix, Not Swizzle

If the input is MN-major (transposed) in GMEM:

```
K-major A (no transpose):
  ldmatrix.m8n8         → loads 8×16B subtile → correct K-major layout in RF

MN-major A (needs transpose):
  ldmatrix.m8n8.trans   → loads 16B×8 subtile → transposes to K-major layout in RF
```

The `.trans` suffix on `ldmatrix` handles the transpose during smem→rf copy. Swizzle layout is irrelevant to transpose — it only ensures bank-conflict-free access to both shapes.

---

## 14. Our Kernel's Swizzle Implementation

In `common.h`, the swizzle is applied at two points:

```cpp
// 1. During cp.async write (storing to shared memory)
uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr + swizzle<STRIDE_BYTES>(idx));

// 2. During ldmatrix read (loading from shared memory)
uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr + swizzle<STRIDE_BYTES>(idx));
```

Both use the same `swizzle<128>` function (STRIDE_BYTES=128 for BLOCK_K=64 bf16).

---

## 15. Summary

| Concept | What it does |
|---------|-------------|
| Swizzle layout | Ensures ldmatrix loads 8×16B at 128B/cycle without bank conflicts |
| 16B atomicity | 16B chunks stay contiguous; major-ness preserved |
| Swizzle atom | Building block: None/32B/64B/128B determine chunk interleaving |
| XOR formula | `addr ^ (row_index << 4)` remaps addresses to different bank groups |
| Both ends | Must apply swizzle at both write (cp.async) and read (ldmatrix) |
| vs Padding | Same bank-conflict fix, zero overhead, better GMEM efficiency |
| Transpose | Handled by `ldmatrix.trans`, not by swizzle layout |

**The lesson**: Swizzle is not optional for tensor core kernels. It's the mechanism that makes `ldmatrix` work at full bandwidth. Our v4 kernel's XOR swizzle achieves this with zero memory overhead — the same bank-conflict elimination as padding, but without wasting 16KB of shared memory per CTA.
