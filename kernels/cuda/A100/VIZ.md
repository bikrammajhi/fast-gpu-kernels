# A100 GEMM Kernel — Detailed Visual Reference

All visuals correspond 1:1 to the code in `matmul_v1.cu` and `common.h`.

---

## 1. Global Matrix Layout & Block Mapping

Code reference: `matmul_v1.cu:34-38`

```cpp
const int num_blocks_n = cdiv(N, BLOCK_N);   // = cdiv(N, 128)
const int block_m = bid / num_blocks_n;       // block row
const int block_n = bid % num_blocks_n;       // block col
const int offset_m = block_m * BLOCK_M;       // = block_m * 128
const int offset_n = block_n * BLOCK_N;       // = block_n * 128
```

Example: N=512, num_blocks_n = 4

```
bid=0  bid=1  bid=2  bid=3
  ┌──────┬──────┬──────┬──────┐
  │(0,0) │(0,1) │(0,2) │(0,3) │  bid = block_m * num_blocks_n + block_n
  ├──────┼──────┼──────┼──────┤
  │(1,0) │(1,1) │(1,2) │(1,3) │  block_m = bid / 4
  ├──────┼──────┼──────┼──────┤  block_n = bid % 4
  │(2,0) │(2,1) │(2,2) │(2,3) │
  ├──────┼──────┼──────┼──────┤  Each block: 128×128 elements of C
  │(3,0) │(3,1) │(3,2) │(3,3) │  offset_m = block_m * 128
  └──────┴──────┴──────┴──────┘  offset_n = block_n * 128

    M=512
    └────────────────────────┘
              N=512
```

---

## 2. Global Pointer Offsets

Code reference: `matmul_v1.cu:43-45`

```cpp
A += offset_m * K;                    // A is M×K, row-major
B += offset_n * K;                    // B is K×N, row-major (transposed storage)
C += (offset_m + warp_m * WARP_M) * N + (offset_n + warp_n * WARP_N);
```

```
A (M×K, row-major):
┌───────────────────────────────────────────────┐
│ row 0                                        │
│ ...                                          │
│ row offset_m  ◄── A += offset_m * K          │
│ ┌────────────┐                               │
│ │ 128 rows   │ (BLOCK_M = 128)               │
│ │ × BLOCK_K  │ (BLOCK_K = 64, per k-tile)    │
│ └────────────┘                               │
│ ...                                          │
│ row M-1                                       │
└───────────────────────────────────────────────┘

B (K×N, row-major — note: B is stored transposed):
┌───────────────────────────────────────────────┐
│ row 0                                        │
│ ...                                          │
│ row 0     ◄── B += offset_n * K              │
│ ┌────────────┐                               │
│ │ BLOCK_K    │ (64, per k-tile)               │
│ │ × 128 cols │ (BLOCK_N = 128)               │
│ └────────────┘                               │
│ ...                                          │
└───────────────────────────────────────────────┘

C (M×N, row-major):
┌───────────────────────────────────────────────┐
│ row 0                                        │
│ ...                                          │
│ row (offset_m + warp_m * WARP_M)  ◄── C ptr  │
│         col (offset_n + warp_n * WARP_N)     │
│ ┌────────────┐                               │
│ │ WARP_M=64  │ rows this warp writes         │
│ │ × WARP_N=64│ cols this warp writes         │
│ └────────────┘                               │
│ ...                                          │
└───────────────────────────────────────────────┘
```

---

## 3. Warp Mapping Within CTA

Code reference: `matmul_v1.cu:31-32, 40-41`

```cpp
const int warp_id = tid / WARP_SIZE;           // 0..3
const int lane_id = tid % WARP_SIZE;           // 0..31

const int warp_m = warp_id / NUM_WARP_N;       // warp_id / 2
const int warp_n = warp_id % NUM_WARP_N;       // warp_id % 2
```

```
128 threads = 4 warps, 2×2 layout:

         WARP_N=0      WARP_N=1
       ┌────────────┬────────────┐
WARP_M │  warp_0    │  warp_1    │
  =0   │  tid 0-31  │  tid 32-63 │
       ├────────────┼────────────┤
WARP_M │  warp_2    │  warp_3    │
  =1   │  tid 64-95 │  tid 96-127│
       └────────────┴────────────┘

  warp_id = tid / 32
  warp_m  = warp_id / 2     (0 or 1)
  warp_n  = warp_id % 2     (0 or 1)

  Each warp owns WARP_M × WARP_N = 64 × 64 of C
  = (64/16) × (64/8) = 4 × 8 = 32 MMA tiles of 16×8
```

---

## 4. Shared Memory Layout

Code reference: `matmul_v1.cu:48-49`

```cpp
__nv_bfloat16* A_smem = smem;
__nv_bfloat16* B_smem = smem + BLOCK_M * SMEM_STRIDE;
//                     = smem + 128 * 64 = smem + 8192
```

```
Dynamic shared memory (32 KB total):
┌─────────────────────────────────────────────────┐
│  A_smem                                          │
│  offset: 0                                        │
│  size: BLOCK_M × SMEM_STRIDE = 128 × 64 = 8192  │
│  bytes: 8192 × 2 = 16384 (16 KB)                │
│                                                   │
│  ┌─────────────────────────────────────┐         │
│  │ row 0   [col 0 ............. col 63]│         │
│  │ row 1   [col 0 ............. col 63]│         │
│  │ ...                                 │         │
│  │ row 127 [col 0 ............. col 63]│         │
│  └─────────────────────────────────────┘         │
├─────────────────────────────────────────────────┤
│  B_smem                                          │
│  offset: BLOCK_M * SMEM_STRIDE = 8192            │
│  size: BLOCK_N × SMEM_STRIDE = 128 × 64 = 8192  │
│  bytes: 8192 × 2 = 16384 (16 KB)                │
│                                                   │
│  ┌─────────────────────────────────────┐         │
│  │ row 0   [col 0 ............. col 63]│         │
│  │ row 1   [col 0 ............. col 63]│         │
│  │ ...                                 │         │
│  │ row 127 [col 0 ............. col 63]│         │
│  └─────────────────────────────────────┘         │
└─────────────────────────────────────────────────┘

  SMEM_STRIDE = BLOCK_K = 64  (no padding)
  A_smem address = smem + 0
  B_smem address = smem + 8192 (bf16 elements)
```

---

## 5. Global → Shared Memory Copy

Code reference: `common.h:51-63` (`gmem2smem`) called at `matmul_v1.cu:58-59`

```cpp
// For A: gmem2smem<CTA_SIZE=128, HEIGHT=128, WIDTH=64, SMEM_STRIDE=64>
// For B: gmem2smem<CTA_SIZE=128, HEIGHT=128, WIDTH=64, SMEM_STRIDE=64>

constexpr int ne = sizeof(uint4) / sizeof(__nv_bfloat16);   // = 16/2 = 8
constexpr int ni = (HEIGHT * WIDTH) / (CTA_SIZE * ne);      // = (128*64)/(128*8) = 8

// Each thread does 8 iterations, loading 8 bf16 elements (16 bytes) per iteration
// Total: 128 threads × 8 iter × 8 elem = 8192 elements = 128 × 64 tile
```

```
Thread mapping for gmem2smem (A tile, 128×64):

Iteration 0:
  tid=0:   idx=0    → row=0/64=0,   col=0%64=0   → load elements [0..7]
  tid=1:   idx=8    → row=8/64=0,   col=8%64=8   → load elements [8..15]
  tid=2:   idx=16   → row=16/64=0,  col=16%64=16 → load elements [16..23]
  ...
  tid=7:   idx=56   → row=56/64=0,  col=56%64=56 → load elements [56..63]
  tid=8:   idx=64   → row=64/64=1,  col=64%64=0  → load elements [0..7] of row 1
  ...
  tid=127: idx=1016 → row=1016/64=15, col=1016%64=56

Iteration 1:
  tid=0:   idx=1024 → row=1024/64=16, col=0    → row 16
  ...

Iteration i:
  idx = (i * 128 + tid) * 8
  row = idx / 64
  col = idx % 64

  Vectorized: reinterpret_cast<uint4*> loads 16 bytes = 8 bf16 elements
```

```
Visual: how iterations map to rows of A_smem:

  Iteration 0:  rows 0-15   (128 threads × 8 elem = 1024 elem = 16 rows × 64 cols)
  Iteration 1:  rows 16-31
  Iteration 2:  rows 32-47
  Iteration 3:  rows 48-63
  Iteration 4:  rows 64-79
  Iteration 5:  rows 80-95
  Iteration 6:  rows 96-111
  Iteration 7:  rows 112-127

  A_smem:
  ┌──────────────────────────────┐ ◄── iter 0: rows 0-15
  │ iter 0: 16 rows × 64 cols   │
  ├──────────────────────────────┤ ◄── iter 1: rows 16-31
  │ iter 1: 16 rows × 64 cols   │
  ├──────────────────────────────┤
  │ iter 2                      │
  ├──────────────────────────────┤
  │ iter 3                      │
  ├──────────────────────────────┤
  │ iter 4                      │
  ├──────────────────────────────┤
  │ iter 5                      │
  ├──────────────────────────────┤
  │ iter 6                      │
  ├──────────────────────────────┤
  │ iter 7: rows 112-127        │
  └──────────────────────────────┘
```

---

## 6. Warp's View of Shared Memory (K-Loop)

Code reference: `matmul_v1.cu:63-64`

```cpp
const __nv_bfloat16* A_smem_warp = A_smem + warp_m * WARP_M * SMEM_STRIDE + k;
const __nv_bfloat16* B_smem_warp = B_smem + warp_n * WARP_N * SMEM_STRIDE + k;
```

```
A_smem (128 × 64):
┌──────────────────────────────────────────────────────┐
│                                                      │
│   warp_m=0, WARP_M=64:  rows 0-63                   │
│   ┌──────────────────────────────────────┐           │
│   │  A_smem_warp (at k=0):               │ ◄── + 0  │
│   │  64 rows × 64 cols (current k-tile)  │           │
│   └──────────────────────────────────────┘           │
│                                                      │
│   warp_m=1, WARP_M=64:  rows 64-127                 │
│   ┌──────────────────────────────────────┐           │
│   │  A_smem_warp (at k=0):               │ ◄── +64*64│
│   │  64 rows × 64 cols                   │           │
│   └──────────────────────────────────────┘           │
│                                                      │
└──────────────────────────────────────────────────────┘

  A_smem_warp = A_smem + warp_m * 64 * 64 + k

  When k advances: +k shifts the column pointer
  (we process K in steps of BLOCK_K=64, then within
   that in steps of MMA_K=16)
```

```
B_smem (128 × 64):
┌──────────────────────────────────────────────────────┐
│                                                      │
│   warp_n=0, WARP_N=64:  rows 0-63                   │
│   ┌──────────────────────────────────────┐           │
│   │  B_smem_warp (at k=0):               │ ◄── + 0  │
│   │  64 rows × 64 cols                   │           │
│   └──────────────────────────────────────┘           │
│                                                      │
│   warp_n=1, WARP_N=64:  rows 64-127                 │
│   ┌──────────────────────────────────────┐           │
│   │  B_smem_warp (at k=0):               │ ◄── +64*64│
│   │  64 rows × 64 cols                   │           │
│   └──────────────────────────────────────┘           │
│                                                      │
└──────────────────────────────────────────────────────┘

  B_smem_warp = B_smem + warp_n * 64 * 64 + k
```

---

## 7. LDMATRIX Address Calculation — B

Code reference: `matmul_v1.cu:69-71`

```cpp
B_smem_ptr = B_smem_warp
    + (n * MMA_N + (lane_id % 8)) * SMEM_STRIDE    // row offset
    + (lane_id / 8) * MMA_K;                        // col offset
```

```
B_smem_warp points to this warp's 64×64 tile.
Within it, we iterate n = 0..NUM_MMA_N-1 (0..7),
each covering MMA_N=8 rows.

For one MMA tile (n=0, 8 rows × 16 cols):

  lane_id % 8  →  row within the 8-row tile  (0..7)
  lane_id / 8  →  which 16-col chunk         (0,1,2,3)
                  = col offset = (lane_id/8) * 16

  SMEM_STRIDE = 64 (stride between rows in smem)

  Example: lane_id = 13
    row = n * 8 + (13 % 8) = 0 + 5 = 5
    col = (13 / 8) * 16    = 1 * 16 = 16

  Full address from B_smem_warp:
    + (row) * 64 + col
    = (n*8 + lane_id%8) * 64 + (lane_id/8) * 16

  Visual for n=0, all 32 lanes:
  ┌────────────────────────────────────────────────────────┐
  │ B_smem_warp (this warp's B tile)                       │
  │                                                        │
  │ row 0:  [lane 0/8 cols 0-15] [lane 1/8 cols 16-31]   │
  │         [lane 2/8 cols 32-47] [lane 3/8 cols 48-63]   │
  │ row 1:  ...                                            │
  │ ...                                                    │
  │ row 7:  [lane 24/8 cols 0-15] ...                     │
  │                                                        │
  │ LDMATRIX_X2 loads 2 tiles of 8×8:                      │
  │   tile 0: rows 0-7, cols 0-7    ← regs[0]            │
  │   tile 1: rows 0-7, cols 8-15   ← regs[1]            │
  │                                                        │
  │   (covers MMA_K=16 columns for 8 rows)                │
  └────────────────────────────────────────────────────────┘

  After LDMATRIX_X2:
    B_regs[n][0] = 8×8 tile (rows 0-7, cols 0-7)
    B_regs[n][1] = 8×8 tile (rows 0-7, cols 8-15)
```

---

## 8. LDMATRIX Address Calculation — A

Code reference: `matmul_v1.cu:78-80`

```cpp
A_smem_ptr = A_smem_warp
    + (m * MMA_M + lane_id % 16) * SMEM_STRIDE    // row offset
    + (lane_id / 16) * MMA_K;                      // col offset
```

```
A_smem_warp points to this warp's 64×64 tile.
Within it, we iterate m = 0..NUM_MMA_M-1 (0..3),
each covering MMA_M=16 rows.

For one MMA tile (m=0, 16 rows × 16 cols):

  lane_id % 16  →  row within the 16-row tile  (0..15)
  lane_id / 16  →  which 16-col chunk          (0 or 1)
                   = col offset = (lane_id/16) * 16

  SMEM_STRIDE = 64

  Example: lane_id = 25
    row = m * 16 + (25 % 16) = 0 + 9 = 9
    col = (25 / 16) * 16     = 1 * 16 = 16

  Full address from A_smem_warp:
    + (m*16 + lane_id%16) * 64 + (lane_id/16) * 16

  Visual for m=0, all 32 lanes:
  ┌────────────────────────────────────────────────────────┐
  │ A_smem_warp (this warp's A tile)                       │
  │                                                        │
  │ row 0:   [lane 0/16 cols 0-15]  [lane 1/16 cols 16-31]│
  │ row 1:   [lane 1/16 cols 0-15]  [lane 0/16 cols 16-31]│
  │ ...                                                    │
  │ row 15:  [lane 15/16 cols 0-15] [lane 14/16 cols ...] │
  │                                                        │
  │ LDMATRIX_X4 loads 4 tiles of 8×8:                      │
  │   tile 0: rows 0-7,   cols 0-7    ← regs[0]          │
  │   tile 1: rows 0-7,   cols 8-15   ← regs[1]          │
  │   tile 2: rows 8-15,  cols 0-7    ← regs[2]          │
  │   tile 3: rows 8-15,  cols 8-15   ← regs[3]          │
  │                                                        │
  │   (covers 16×16 = MMA_M × MMA_K)                      │
  └────────────────────────────────────────────────────────┘

  After LDMATRIX_X4:
    A_regs[0] = 8×8 tile (rows 0-7,   cols 0-7)
    A_regs[1] = 8×8 tile (rows 0-7,   cols 8-15)
    A_regs[2] = 8×8 tile (rows 8-15,  cols 0-7)
    A_regs[3] = 8×8 tile (rows 8-15,  cols 8-15)
```

---

## 9. Accumulator Register Layout

Code reference: `matmul_v1.cu:51-55`

```cpp
constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;   // = 16*8/32 = 4
float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs];            // [4][8][4]
```

```
Each thread holds 4 floats for one MMA tile:

  acc[m][n][0..3] maps to a 16×8 output tile:

  ┌───────────────────────────┐ 16 rows
  │ tid%4==0  tid%4==1        │
  │ [0] [1]   [0] [1]        │
  │ [2] [3]   [2] [3]        │
  │                           │
  │ tid%4==2  tid%4==3        │
  │ [0] [1]   [0] [1]        │
  │ [2] [3]   [2] [3]        │
  └───────────────────────────┘
              8 cols

  Per thread:
    regs[0] → row = lane_id/4 + 0, col = (lane_id%4)*2 + 0
    regs[1] → row = lane_id/4 + 0, col = (lane_id%4)*2 + 1
    regs[2] → row = lane_id/4 + 1, col = (lane_id%4)*2 + 0
    regs[3] → row = lane_id/4 + 1, col = (lane_id%4)*2 + 1

  Full accumulator for one warp (NUM_MMA_M=4 × NUM_MMA_N=8):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │(0,0) │(0,1) │(0,2) │(0,3) │(0,4) │(0,5) │(0,6) │(0,7) │  m=0
  ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
  │(1,0) │(1,1) │(1,2) │(1,3) │(1,4) │(1,5) │(1,6) │(1,7) │  m=1
  ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
  │(2,0) │(2,1) │(2,2) │(2,3) │(2,4) │(2,5) │(2,6) │(2,7) │  m=2
  ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
  │(3,0) │(3,1) │(3,2) │(3,3) │(3,4) │(3,5) │(3,6) │(3,7) │  m=3
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

  Each cell: 16×8 tile, 4 regs per thread
  Total per thread: 4 × 8 × 4 = 128 float registers
```

---

## 10. MMA Instruction: MMA_M16N8K16

Code reference: `common.h:43-48`

```
PTX instruction:
  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32

  A: 16×16 bf16 (4 regs × 32 bits = 128 bits per thread)
  B: 16×8 bf16  (2 regs × 32 bits = 64 bits per thread)
  C/D: 16×8 f32 (4 regs per thread)

  A register layout (each thread holds 4 × 32-bit regs):
  ┌──────────────────────────────────────┐
  │ reg[0]: rows 0-3,   cols 0-7        │
  │ reg[1]: rows 4-7,   cols 0-7        │
  │ reg[2]: rows 8-11,  cols 0-7        │
  │ reg[3]: rows 12-15, cols 0-7        │
  └──────────────────────────────────────┘

  B register layout (each thread holds 2 × 32-bit regs):
  ┌──────────────────────────────────────┐
  │ reg[0]: rows 0-7,   cols 0-7        │
  │ reg[1]: rows 8-15,  cols 0-7        │
  └──────────────────────────────────────┘

  D (accumulator) layout (each thread holds 4 × float regs):
  ┌──────────────────────────────────────┐
  │ reg[0]: row = lane/4,     col = (lane%4)*2     │
  │ reg[1]: row = lane/4,     col = (lane%4)*2 + 1 │
  │ reg[2]: row = lane/4 + 1, col = (lane%4)*2     │
  │ reg[3]: row = lane/4 + 1, col = (lane%4)*2 + 1 │
  └──────────────────────────────────────┘
```

---

## 11. Epilogue — Writing C

Code reference: `matmul_v1.cu:95-104`

```cpp
const int c_row = m * MMA_M + (lane_id / 4);        // m*16 + lane/4
const int c_col = n * MMA_N + (lane_id % 4) * 2;    // n*8 + (lane%4)*2

// Store 2 bf16 values at a time (bf162):
C[c_row * N + c_col]       = tmp.x    // regs[0]
C[c_row * N + c_col + 1]   = tmp.y    // regs[1]
C[(c_row+1) * N + c_col]   = tmp.x    // regs[2]
C[(c_row+1) * N + c_col+1] = tmp.y    // regs[3]
```

```
Thread → C element mapping for one MMA tile (m, n):

  c_row = m * 16 + lane_id / 4       (0..15 within tile)
  c_col = n * 8  + (lane_id % 4) * 2 (0..7  within tile)

  lane_id:  0  1  2  3  4  5  6  7  ... 28 29 30 31
  c_row:    0  0  0  0  1  1  1  1  ...  7  7  7  7
  c_col:    0  2  4  6  0  2  4  6  ...  0  2  4  6

  Each thread writes 2×2 = 4 elements (bf162 vectorized):

  ┌──────────────────────────────────────────────┐
  │ lane 0: C[row 0, col 0-1]   + C[row 1, col 0-1] │
  │ lane 1: C[row 0, col 2-3]   + C[row 1, col 2-3] │
  │ lane 2: C[row 0, col 4-5]   + C[row 1, col 4-5] │
  │ lane 3: C[row 0, col 6-7]   + C[row 1, col 6-7] │
  │ lane 4: C[row 1, col 0-1]   + C[row 2, col 0-1] │
  │ ...                                               │
  │ lane 31: C[row 7, col 6-7]  + C[row 8, col 6-7]  │
  └──────────────────────────────────────────────┘

  Full C tile (64×64 per warp, showing how warps tile it):
  ┌────────────────────────────────────────────────────────────┐
  │  Warp (0,0) writes:  C[offset_m + 0..63, offset_n + 0..63]│
  │  Warp (0,1) writes:  C[offset_m + 0..63, offset_n + 64..127]│
  │  Warp (1,0) writes:  C[offset_m + 64..127, offset_n + 0..63]│
  │  Warp (1,1) writes:  C[offset_m + 64..127, offset_n + 64..127]│
  └────────────────────────────────────────────────────────────┘
```

---

## 12. Full K-Iteration Timeline

Code reference: `matmul_v1.cu:57-90`

```
K = 8192, BLOCK_K = 64, MMA_K = 16
Outer loop: block_k = 0, 64, 128, ..., 8128  (128 iterations)
Inner loop: k = 0, 16, 32, 48                 (4 iterations per block_k)

Timeline for ONE block_k iteration:

  Time ──────────────────────────────────────────────────────►

  ┌──────────────────────────────────────────────────────────┐
  │ 1. gmem2smem(A)    ──────────────────────────────►      │
  │ 2. gmem2smem(B)    ──────────────────────────────►      │
  │ 3. __syncthreads()                                    │
  │ 4. k=0: ldmatrix(B) → ldmatrix(A) → MMA × 4×8 tiles   │
  │ 5. k=16: ldmatrix(B) → ldmatrix(A) → MMA × 4×8 tiles  │
  │ 6. k=32: ldmatrix(B) → ldmatrix(A) → MMA × 4×8 tiles  │
  │ 7. k=48: ldmatrix(B) → ldmatrix(A) → MMA × 4×8 tiles  │
  │ 8. __syncthreads()                                    │
  │ 9. A += 64, B += 64                                   │
  └──────────────────────────────────────────────────────────┘

  Per block_k: load 2×32KB → sync → 16 MMA instructions → sync
  Total: 128 block_k iterations × (load + 16 MMA)
  Grand total: 128 × 4 = 512 MMA instructions per thread
```

---

## 13. Concrete Example: N=256

```
N=256, num_blocks_n = cdiv(256, 128) = 2

Grid: 2×2 = 4 blocks
  bid=0: block_m=0, block_n=0  → A[0:128, :], B[:, 0:128]
  bid=1: block_m=0, block_n=1  → A[0:128, :], B[:, 128:256]
  bid=2: block_m=1, block_n=0  → A[128:256, :], B[:, 0:128]
  bid=3: block_m=1, block_n=1  → A[128:256, :], B[:, 128:256]

Each block: 128 threads, 32 KB smem
K-loop: 256/64 = 4 outer iterations × 4 inner = 16 MMA steps
```
