# Debugging Log — SM100a FlashAttention Baseline Kernel

## Overview

Getting `1_baseline.cu` to compile and run on a B200 GPU required fixing **five distinct classes of errors**: two compilation issues, one linker error, one PTX assembly error, and two runtime memory errors. This document covers each in detail, with emphasis on the memory-related bugs.

---

## Error 1: PTX Assembly — Wrong Architecture Target

### Symptom

```
ptxas error: Instruction 'tcgen05.ld' not supported on .target 'sm_100'
ptxas error: Feature '.32x32b' not supported on .target 'sm_100'
ptxas error: Instruction 'tcgen05.mma' not supported on .target 'sm_100'
```

Every `tcgen05.*`, `mbarrier.*`, and `elect.sync` instruction was rejected.

### Root Cause

Blackwell's async TMEM/TMA instructions (`tcgen05.mma`, `tcgen05.ld`, `tcgen05.st`, `tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.commit`, `tcgen05.fence`, `tcgen05.wait`) require the `sm_100a` target, **not** `sm_100`. The `a` suffix designates "all" Blackwell sub-architectures and enables the async feature set. Without it, ptxas treats the target as vanilla sm_100 and rejects all async instructions.

The original `GPU_ARCH` mapping had:
```python
"B200": ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"]
```

### Fix

Changed to `sm_100a`:
```python
"B200": ["-arch=sm_100a", "-gencode", "arch=compute_100a,code=sm_100a",
         "-gencode", "arch=compute_100a,code=compute_100a"]
```

Also switched the Modal Docker image from `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` to `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` because CUDA 12.8 was the first toolkit version with full `sm_100a` ptxas support.

### Lesson

The `a` suffix on `sm_100a` is not cosmetic — it gates the entire async/TMEM/UMMA instruction set. The PTX file name changes from `compute_100.ptx` to `compute_100a.ptx`, confirming the correct target is selected.

---

## Error 2: Linker — Undefined Reference to `cuTensorMapEncodeTiled`

### Symptom

```
/usr/bin/ld: undefined reference to `cuTensorMapEncodeTiled'
/usr/bin/ld: undefined reference to `cuGetErrorString'
```

### Root Cause

`cuTensorMapEncodeTiled` and `cuGetErrorString` live in `libcuda.so` (the CUDA Driver API library). The nvcc command only linked `-lcublas`, which does not pull in the driver library.

### Fix

Added `-lcuda` to the link flags:
```python
cmd = ["nvcc", "-O3", *includes, *arch, "-lcublas", "-lcuda", *flags, *extra_flags, "-o", binary, src]
```

### Lesson

Any code using the CUDA Driver API (`cu*` functions) requires `-lcuda`. The Runtime API (`cuda*` functions) is linked automatically by nvcc, but the Driver API is not.

---

## Error 3: PTX Assembly — `elect.sync` Missing Predicate Output

### Symptom

```
ptxas error: Predicate output expected for instruction 'elect'
```

Happened at every `elect.sync` call site (5 locations in the generated PTX).

### Root Cause

The PTX `elect.sync` instruction **requires** a predicate register (`%p`) output. It does not accept a general-purpose register as the sole output. The initial implementation was:

```cuda
asm volatile("elect.sync %0, %1;" : "=r"(ret) : "r"(-1));
```

This tells ptxas to write the result into a `.b32` register, but `elect.sync` only writes to a `.pred` predicate register. ptxas rejects this.

The correct PTX form is:

```ptx
elect.sync _ | %p, 0xFFFFFFFF;
```

where `_` discards the lane ID output and `%p` captures the predicate (true for lane 0, false for all others).

### Fix

Adopted the CUTLASS pattern from `cute/arch/cluster_sm90.hpp`:

```cuda
__device__ __forceinline__
int elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}
```

Key points:
- `_` in `elect.sync _|%%px` means "don't store the lane ID"
- `@%%px mov.s32 %0, 1` conditionally writes 1 only if the predicate is true (lane 0)
- Returns 0 or 1, not a lane ID

### Lesson

`elect.sync` is a warp-level predicate instruction, not a general-purpose register output. Always pair it with a `.pred` output and a conditional move to convert to an integer.

---

## Error 4: Runtime — Illegal Memory Access from Wrong SMEM Descriptor Encoding

### Symptom

```
CUDA error 1_baseline.cu:419: an illegal memory access was encountered
XID: NVRM: Xid 13, Graphics SM Warp Exception: Out Of Range Address
```

Multiple SMs (GPC 0 TPC 4, GPC 0 TPC 7, GPC 1 TPC 5, GPC 1 TPC 7) all reported "Out Of Range Address" exceptions. The kernel launched 4 blocks (batch=4), and every block crashed.

### Root Cause — Deep Analysis

The `desc_encode` function that encodes shared memory addresses into UMMA (Unified MMA) descriptors was fundamentally wrong:

```cuda
// OURS (WRONG):
__device__ __forceinline__
uint32_t desc_encode(uint32_t offset) {
    return offset & 0x3FFF;    // 14-bit mask, no shift
}
```

```cuda
// REFERENCE (CORRECT):
__device__ __forceinline__
uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;   // 18-bit mask, right-shift by 4
}
```

This single function is used to encode three fields into every 64-bit SMEM descriptor for the UMMA instruction:

```
Bits [13:0]  = desc_encode(smem_address)     — base SMEM address
Bits [29:16] = desc_encode(LBO)              — leading byte offset (row stride)
Bits [45:32] = desc_encode(SBO)              — striding byte offset (atom stride)
Bit  [46]    = valid bit (always 1)
```

**What went wrong numerically:**

With a shared memory base address of, say, 65536 bytes:
- Our encoding: `65536 & 0x3FFF = 0` (only keeps bottom 14 bits → **address truncated to 0**)
- Correct encoding: `(65536 & 0x3FFFF) >> 4 = 65536 >> 4 = 4096` (proper 14-bit encoded value)

Every SMEM descriptor — for Q, K, P, and V operands — had its address field set to 0 or a truncated value. The UMMA instruction would then:

1. Read the **LBO and SBO fields** (which were also incorrectly encoded, since `desc_encode` applied to them too) → wrong stride values
2. Use the **base address** field as-is → pointing to offset 0 in SMEM or to a completely wrong location
3. Issue TMA-like async reads through the shared memory bus to **addresses that don't belong to this CTA's shared memory allocation**

The GPU's shared memory interconnect detected that the computed addresses fell outside the valid range for this CTA's SMEM partition, triggering the "Out Of Range Address" exception on every SM that ran the kernel.

**Why this was particularly insidious:**
- The descriptor structurally looked correct (64-bit value, valid bit set, fields in the right positions)
- The kernel didn't crash at descriptor construction time — it only crashed when the UMMA instruction tried to use the wrong descriptors
- All four SMs crashed simultaneously because all four blocks had the same corrupted descriptors
- The error manifested as a hardware-level "Out Of Range Address" exception, not a software error, because the async memory proxy directly accesses shared memory without bounds checking in software

### Fix

```cuda
__device__ __forceinline__
uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}
```

The SMEM descriptor format uses 14-bit encoded fields, but the encoding is **not** a simple truncation. The hardware expects the value to be right-shifted by 4 bits, effectively encoding a 14-bit field from an 18-bit input (with the bottom 4 bits discarded). This aligns with the PTX ISA documentation's definition of the SMEM descriptor format for `tcgen05.mma`.

**Additional type change:** The return type and parameter changed from `uint32_t` to `uint64_t` because the descriptor is 64-bit and the shift operation can produce values that need the full width during intermediate computation.

### Lesson

SMEM descriptor encoding for UMMA is not a simple bit-mask. The PTX ISA defines a specific encoding scheme where values are right-shifted by 4. Getting this wrong produces descriptors that look plausible but point to completely wrong shared memory locations, causing silent memory corruption that only manifests as hardware exceptions when the UMMA instruction executes.

---

## Error 5: Runtime — PV MMA Writes to Wrong TMEM Region

### Symptom

After fixing the `desc_encode` bug, the kernel still crashed with the same "Out Of Range Address" XID error.

### Root Cause — Deep Analysis

The kernel allocates 256 columns of TMEM:
- Columns 0–127 (`taddr_s`): for the QK score matrix S (BLOCK_M × BLOCK_N = 128×128)
- Columns 128–255 (`taddr_o`): for the output matrix O (BLOCK_M × HEAD_DIM = 128×128)

The PV MMA (P × V → O) was writing its result to `taddr_s` instead of `taddr_o`:

```cuda
// WRONG — overwrites S in TMEM:
tcgen05_mma_f16(taddr_s, a_desc, b_desc, idesc_PV, enable);

// CORRECT — writes to O region:
tcgen05_mma_f16(taddr_o, a_desc, b_desc, idesc_PV, enable);
```

**Why this caused a memory error:**

The `tcgen05.mma` instruction writes its result to a **specific TMEM address** that encodes both a row and column offset. The address `taddr_s` tells the hardware to write into the S region (columns 0–127). Writing O values there corrupted S.

But the more immediate cause of the crash was the **feedback loop across KV tiles**:

1. **Tile 0:** QK writes S to `taddr_s`. Softmax reads S. PV writes O to `taddr_s` (WRONG — overwrites S with O).
2. **Tile 1:** QK tries to write new S to `taddr_s` with `enable_input_d=0` (zero accumulator). This works. Softmax reads S. Then the code tries to **rescale O** by reading from `taddr_o` (columns 128–255), but O was never written there — it was written to columns 0–127 in the previous tile.
3. The `tcgen05.ld` instruction reads from `taddr_o + trow + n8*8`, which accesses TMEM addresses that haven't been properly initialized by any MMA. The hardware reads garbage or unmapped TMEM locations, causing the "Out Of Range Address" exception.

**TMEM addressing details:**

TMEM uses a `(row << 16) | column` addressing scheme. Each warp owns a 32-row slice:
- Warp 0: `trow = 0 << 16 = 0` → accesses columns 0–127 (S) or 128–255 (O)
- Warp 1: `trow = 32 << 16 = 0x200000` → accesses the next 32 rows
- etc.

The `tcgen05.ld.sync.aligned.32x32b.x8.b32` instruction reads 32 rows × 8 columns of 32-bit values starting at the given TMEM address. When the address points to the O region (columns 128–255) but no MMA has ever written there, the read accesses unallocated TMEM space → hardware exception.

### Fix

```cuda
// PV → O (TMEM): write to taddr_o, not taddr_s
tcgen05_mma_f16(taddr_o, a_desc, b_desc, idesc_PV, enable);
```

### Lesson

TMEM is a **fixed-address** storage — the MMA instruction writes to the address you give it, and the `tcgen05.ld/st` instructions read/write from the address you give them. Unlike SMEM where you compute offsets from a base, TMEM has no implicit base pointer. The programmer must manually ensure that:
1. The MMA destination address and the ld/st source address refer to the **same TMEM region**
2. Different logical matrices (S vs O) occupy **non-overlapping** TMEM regions
3. The `enable_input_d` flag (0 = zero-init, 1 = accumulate) is used correctly for accumulation across tiles

---

## Summary Table

| # | Error Type | Root Cause | Fix |
|---|-----------|-----------|-----|
| 1 | PTX assembly | Wrong arch target (`sm_100` vs `sm_100a`) | Use `-arch=sm_100a` and CUDA 12.8 toolkit |
| 2 | Linker | Missing `-lcuda` for Driver API symbols | Add `-lcuda` to nvcc flags |
| 3 | PTX assembly | `elect.sync` requires `.pred` output, not `.b32` | Use `elect.sync _\|%px` with conditional move |
| 4 | **Runtime memory** | `desc_encode` used wrong bit-mask/shift, corrupted all SMEM descriptors | Use `(x & 0x3'FFFF) >> 4` |
| 5 | **Runtime memory** | PV MMA wrote O to `taddr_s` (S region); rescale read uninitialized `taddr_o` | Write PV to `taddr_o` |

Errors 4 and 5 are the memory-critical bugs. Error 4 corrupted the shared memory address space used by every UMMA instruction, causing hardware-level out-of-range address exceptions across all SMs. Error 5 corrupted the TMEM address space, causing reads from uninitialized TMEM locations during the O rescaling pass.
