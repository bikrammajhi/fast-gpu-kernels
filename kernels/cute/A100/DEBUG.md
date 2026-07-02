# DEBUG.md — CuTe matmul_v1.cu Fixes

## Original Status

The original `matmul_v1.cu` had **48 compilation errors**. It was adapted from the CUTLASS CuTe tutorial `sgemm_sm80.cu` but was incomplete/corrupted (missing template params, wrong tensor slicing, wrong CuTe API calls, undefined symbols, and type mismatches with `__nv_bfloat16`).

---

## Errors & Fixes

### 1. Undefined template parameters in `gemm()` host function

**Error:** `identifier "TA" is undefined`, `identifier "TB" is undefined`, `identifier "TC" is undefined`

The `gemm()` function used `TA`, `TB`, `TC` as type names without declaring them as template parameters.

**Fix:** Added template parameter list:
```cpp
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, ...)
```

---

### 2. Missing `using namespace cute;` in `gemm()`

**Error:** `identifier "make_shape" is undefined`, `identifier "Swizzle" is undefined`, `identifier "TiledCopy" is undefined`, etc.

CuTe types like `make_shape`, `Swizzle`, `TiledCopy`, `Int`, `_8`, etc. are in the `cute` namespace.

**Fix:** Added `using namespace cute;` inside `gemm()` function body.

---

### 3. Missing `#include "cutlass/util/GPU_Clock.hpp"`

**Error:** `identifier "GPU_Clock" is undefined`

`GPU_Clock` is defined in `cutlass/util/GPU_Clock.hpp`, not in `cute/tensor.hpp`.

**Fix:** Added:
```cpp
#include "cutlass/util/GPU_Clock.hpp"
```

---

### 4. Missing `cudaStream_t stream` parameter

**Error:** `identifier "stream" is undefined` in kernel launch `<<<... , stream>>>`

The `gemm()` function launched a kernel with a `stream` parameter that was never defined.

**Fix:** Added `cudaStream_t stream = 0` as the last parameter of `gemm()`.

---

### 5. Undefined `transA`, `transB` in `main()`

**Error:** `identifier "transA" is undefined`, `identifier "transB" is undefined`

The `main()` function referenced `transA` and `transB` in a print statement but never defined them.

**Fix:** Replaced with:
```cpp
std::cout << "C = A * B" << std::endl;
```

---

### 6. Wrong leading dimensions (`ldA`, `ldB`, `ldC`)

**Error:** No compilation error (runtime correctness issue), but `ldA = 0, ldB = 0, ldC = m` makes all rows alias.

Original: `int ldA = 0, ldB = 0, ldC = m;`

For row-major MxK, NxK, MxN matrices:
- A has `M*K` elements, stride = K → `ldA = K`
- B has `N*K` elements, stride = K → `ldB = K`
- C has `M*N` elements, stride = N → `ldC = N`

**Fix:** `int ldA = k, ldB = k, ldC = n;`

---

### 7. Wrong tensor slicing — `tAgA(_,_,k)` instead of `tAgA(_,_,_,k_tile)`

**Error:** `function "copy(const ThrCopy<...> &, ...)" cannot be referenced -- it is a deleted function` + cascading template errors.

`tAgA = thr_copy_a.partition_S(gA)` produces shape `(CPY, CPY_M, CPY_K, k)` — **rank 4**, not rank 3. The original `tAgA(_,_,k)` indexed mode 2 (`CPY_K`) with the tile count `k`, which type-mismatched and produced an invalid layout with `C<0>` strides, causing `copy()` to fail.

Similarly, `tBgB(_,_,k)` → `tBgB(_,_,_,k_tile)`.

**Fix:**
```cpp
copy(copy_a, tAgA(_,_,_,k_tile), tAsA);
copy(copy_b, tBgB(_,_,_,k_tile), tBsB);
```

Loop variable added:
```cpp
int k_tile_count = size<3>(tAgA);
for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) { ... }
```

---

### 8. Wrong fragment initialization — `sA(_,_,0)` instead of `sA(_,_)`

**Error:** Cascading template errors from invalid tensor slicing.

`sA` has shape `(BLK_M, BLK_K)` — **rank 2**, no pipe dimension. The original `sA(_,_,0)` tried to index a non-existent 3rd mode.

**Fix:**
```cpp
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_));
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_));
```

---

### 9. Deleted `ThrCopy::copy()` — use `Copy_Atom` directly

**Error:** `function "cute::copy(const ThrCopy<...> &, ...)" cannot be referenced -- it is a deleted function`

The `ThrCopy` overload of `copy()` is deleted in this CUTLASS version. The `s2r` copy used `copy(s2r_thr_copy_a, ...)` where `s2r_thr_copy_a` is a `ThrCopy`.

**Fix:** Use the `Copy_Atom` directly:
```cpp
copy(s2r_atom_a, tXsA, tXrA);
copy(s2r_atom_b, tXsB, tXrB);
```

---

### 10. `mma()` → `gemm()`

**Error:** `call of an object of a class type without appropriate operator() or conversion functions to pointer-to-function type`

TiledMMA cannot be called with `operator()`. Must use the `gemm()` free function.

**Fix:**
```cpp
gemm(mma, tCrA, tCrB, tCrC);
```

---

### 11. `__nv_bfloat16` → `cute::bfloat16_t`

**Error:** `more than one operator * matches these operands: const float * __nv_bfloat16` (ambiguous operator* due to many implicit conversion operators on `__nv_bfloat16`).

CuTe has proper type trait support for `cute::bfloat16_t` (which is `cutlass::bfloat16_t`), but not for raw `__nv_bfloat16`.

**Fix:** Replaced all `__nv_bfloat16` with `cute::bfloat16_t` (aliased as `bf16`):
```cpp
using bf16 = cute::bfloat16_t;
```

---

### 12. `axpby` incompatible with `cutlass::bfloat16_t`

**Error:** `no operator "=" matches these operands: cutlass::bfloat16_t = float`

`cutlass::bfloat16_t` has an `explicit` constructor from `float` but no `operator=(float)`. The `axpby()` function does `y(i) = alpha * x(i) + beta * y(i)` which requires `bfloat16_t = float` assignment.

**Fix:** Replaced `axpby(alpha, tCrC, beta, tCgC)` with an explicit loop:
```cpp
CUTE_UNROLL
for (int i = 0; i < size(tCrC); ++i) {
    tCgC(i) = cute::bfloat16_t(alpha * tCrC(i) + beta * static_cast<float>(tCgC(i)));
}
```

---

### 13. Wrong C stride

**Error:** Runtime correctness issue, no compilation error.

Original: `auto dC = make_stride(ldC, N);` — creates stride `(ldC, n)`.
For row-major C[M][N], we need `(N, 1)`.

**Fix:** Changed to `auto dC = make_stride(ldC, Int<1>{});` — stride `(n, 1)`.

---

## Fixed file

See `matmul_v1.cu` in this directory.

## Benchmark results (M=N=5120, K=4096, A100-SXM4-40GB)

### cp_async fence/wait/sync variants

| Variant | TFLOPS | Time (ms) | Notes |
|---------|:------:|:---------:|-------|
| No fence, no wait, no sync | **174.7** | 1.2296 | Fastest; lacks `cp_async_wait` — correctness not guaranteed |
| `cp_async_fence` + `cp_async_wait<0>` | 164.5 | 1.3056 | Waits for copies but no cross-warp barrier |
| `cp_async_fence` + `cp_async_wait<0>` + `__syncthreads` (current) | 152.6 | 1.4069 | Correct single-buffer pattern |

The reference tutorial (`sgemm_sm80.cu`) always pairs `__syncthreads()` with `cp_async_wait` before reading smem — it never uses `cp_async_fence` alone. The `__syncthreads()` is required because multiple warps share the same shared memory buffer.

---

## cp.async Race Condition (matmul_v5.cu)

### Symptom

After replacing `UniversalCopy<uint128_t>` with `SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>` in v5, we got **162.3 TFLOPS** but `max_rel_err=5.36` (v3 with UniversalCopy gave 0.00). Only 100/26M entries were wrong, suggesting a rare data race rather than a systematic bug.

### Investigation

We verified both source and destination addresses are 16-byte aligned (required by cp.async 128-bit):
- Global memory: each thread reads at offset `i*65536 + j*16` from base — always 16‑byte aligned
- Shared memory (padded layout stride=72 bf16=144B): each thread writes at offset `1152*i + 16*j` bytes — always 16‑byte aligned

The cp.async dispatch chain was traced through `copy(TiledCopy, src, dst)` → `copy(CopyAtom, src, dst)` → `copy_unpack()` → `SM80_CP_ASYNC_CACHEALWAYS::copy()` and verified correct.

### Root Cause

The K-loop structure was:
```cpp
for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    copy(cp_async_A, ..., k_tile);  // issue async DMA to smem
    copy(cp_async_B, ..., k_tile);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();   // (A) ensures tile(k)'s DMA visible to all warps

    copy(s2r_atom_a, tXsA, tXrA);  // LDSM: smem → regs
    copy(s2r_atom_b, tXsB, tXrB);
    gemm(mma, tCrA, tCrB, tCrC);
    // ← missing __syncthreads() here
}
```

The top sync (A) ensures tile(k)'s gmem→smem DMA is visible. The gap at the bottom lets the next iteration's DMA trample the current tile's data while other warps are still reading it.

**Timeline of the race (warp 0 fast, warp 3 slow):**

```
k=0:
warp0: | cp.async(k=0) | fence/wait | ╔══sync══╗ | LDSM | gemm | cp.async(k=1) ← overwrites smem |
warp3: | cp.async(k=0) | fence/wait | ╚══sync══╝ |                 LDSM ← still reading smem     |
```

After the sync, both warps proceed independently. Warp 0 finishes gemm quickly and starts `cp.async` for k=1 — this issues **asynchronous DMA writes** to the exact same smem addresses. Warp 3's LDSM (loading tile(k=0) into registers) hasn't completed yet, so it reads a mix of tile-0 and tile-1 data → corrupted register values → wrong MMA.

**The sync at (A) ensures tile(k)'s DMA is done. The missing sync after gemm lets tile(k+1)'s DMA overlap tile(k)'s LDSM reads from other warps.**

UniversalCopy (v3) avoided this in practice because regular store instructions are synchronous — a store must retire before the next instruction issues, making the race window extremely narrow (though still theoretically possible). cp.async is different: the instruction returns immediately and the actual DMA can complete hundreds of cycles later, widening the conflict window.

### Fix

Added `__syncthreads()` after `gemm()`, matching the pattern in CuTe's `sgemm_1.cu` tutorial:

```cpp
for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    copy(cp_async_A, ..., k_tile);
    copy(cp_async_B, ..., k_tile);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(s2r_atom_a, tXsA, tXrA);
    copy(s2r_atom_b, tXsB, tXrB);
    gemm(mma, tCrA, tCrB, tCrC);

    __syncthreads();   // ← ensures all threads finish reading smem before next write
}
```

### Result

| Config | TFLOPS | max_rel_err |
|--------|:------:|:-----------:|
| v3 (UniversalCopy) | 105.4 | 0.00 |
| v5 (cp.async) — **buggy** | 162.3 | 5.36 |
| v5 (cp.async) — **fixed** | **158.4** | **0.00** |

The `__syncthreads()` cost ~2.4% performance (162.3→158.4 TFLOPS) but is mandatory for correctness. This is a common pitfall when moving from synchronous stores to async DMA — the cross-thread race window widens from negligible to observable.
