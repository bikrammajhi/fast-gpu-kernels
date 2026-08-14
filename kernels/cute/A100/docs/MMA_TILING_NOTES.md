# MMA Tiling Analysis: CTA 128×128×64

## Current Config

```
MMA atom:           SM80_16x8x16  → Shape_MNK = (16, 8, 16), 32 thr/atom = 1 warp
Atom layout:        Layout<Shape<_2,_2>>{}  → 2×2 grid over (M,N) = 4 atoms = 4 warps = 128 thr
Permutation (Tile): Tile<_32,_32,_16>{}  → value tiling (affects fragment partition, not output size)
```

## CTA partition summary

| Tensor | Shape | Perm tile | M_rest | N_rest | K_rest |
|--------|-------|-----------|--------|--------|--------|
| sA (M,K) | (128, 64) | (32, 16) | 4 | — | 4 |
| sB (N,K) | (128, 64) | (32, 16) | — | 4 | 4 |
| gC (M,N) | (128, 128) | (32, 32) | 4 | 4 | — |

## gemm() internal dispatch chain

`gemm(mma, tCrA, tCrB, tCrC)` dispatches via fragment ranks:

```
Dispatch [5]: (V, M_rest, K_rest) × (V, N_rest, K_rest) → (V, M_rest, N_rest)
  for k in 0..K_rest-1:                                    // K_rest = 4
    Dispatch [4]: (V, M_rest) × (V, N_rest) → (V, M_rest, N_rest)
      for m in 0..M_rest-1, n in 0..N_rest-1 (serpentine): // M_rest × N_rest = 4×4 = 16
        Dispatch [1]: mma.call(D, A, B, C)                  // ← actual MMA instruction
```

## Work per warp per gemm() call

| Level | Loop | Count |
|-------|------|-------|
| K_rest (D5) | for k | 4 |
| M_rest × N_rest (D4) | for m,n | 4 × 4 = 16 |
| **Inner MMA calls per gemm()** | | **64** |

### Each MMA instruction (Dispatch [1])
- All 32 threads in a warp participate
- Computes one 16×8×16 product
- Updates one 16×8 piece of the accumulator

### Mapping 4 warps to the output

Within each (m_rest, n_rest) tile (each 32×32 of the full output):

| Warp | Atom idx | M range | N range |
|------|----------|---------|---------|
| 0 | (0,0) | M[0:16] | N[0:8] |
| 1 | (0,1) | M[0:16] | N[8:16] |
| 2 | (1,0) | M[16:32] | N[0:8] |
| 3 | (1,1) | M[16:32] | N[8:16] |

Each warp iterates over all 16 (m_rest,n_rest) positions, contributing its
16×8 piece to each → covers full 128×128 output.

### Per warp totals

```
MMA instr per gemm():  64
Elements per warp per gemm():  64 × (16×8) = 8192  (half of 128×128 = 16384)
```

## Full GEMM (K=4096, bK=64)

```
K-loop iterations:  4096 / 64 = 64
MMA per warp per CTA:  64 K-loop × 64 MMA/gemm = 4096
MMA per CTA:  4096 × 4 warps = 16384
```

## Key insight: Tile<32,32,16>

The permutation `Tile<32,32,16>` determines the **logical tile size** used by
`logical_divide` when partitioning fragments. It sets M_rest=4 and N_rest=4,
K_rest=4. It does NOT mean the output tile is 32×32 — the actual output per
`gemm()` call is 32×16 (from 2×2 atoms × 16×8 atom).

## For reference: MMA atom internals

```
SM80_16x8x16_F32BF16BF16F32_TN
├── DRegisters = float[4]       (4 accumulator regs per thread)
├── ARegisters = uint32_t[4]    (4 A regs per thread)
├── BRegisters = uint32_t[2]    (2 B regs per thread)
├── CRegisters = float[4]       (4 initial C regs per thread)
├── Shape_MNK  = (_16, _8, _16) => (M=16, N=8, K=16)
├── ThrID      = Layout<_32>    (32 threads per atom = 1 warp)
├── ALayout    = complex 2-level layout mapping 32 thr → 16×16 A tile
├── BLayout    = complex 2-level layout mapping 32 thr → 16×8 B tile
└── CLayout    = SM80_16x8_Row  (16×8 C tile, row-major)
```
