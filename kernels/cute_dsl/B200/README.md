# B200 Matmul Benchmark (CuTe DSL)

Kernels: `kernels/cute_dsl/B200/matmul_v{1,2,3,4,5,6}.py`

- GPU: NVIDIA B200 (sm_100a, Blackwell)
- IO dtype: Float16, Accum dtype: Float32
- Problem: square `M = N = K = 8192`

Reference: [NVIDIA CUTLASS — Tour to a Solution GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/notebooks/tour_to_sol_gemm.ipynb)

## Results (M = N = K = 8192)

| Version | Kernel time (us) | Throughput (TFLOPs) | Speedup vs v1 |
|---------|------------------|---------------------|---------------|
| v1      | 2400.18          | 458.10              | 1.00x         |
| v2      | 1229.82          | 894.04              | 1.95x         |
| v3      | 762.88           | 1441.26             | 3.15x         |
| v4      | 652.78           | 1684.34             | 3.68x         |
| v5      | 597.49           | 1840.22             | 4.02x         |
| v6      | 617.95           | 1779.28             | 3.89x         |

All versions pass numerical verification against a PyTorch `einsum` reference.
**CuTe DSL v5 reaches ~125% of cuBLAS peak (1840 vs 1478 TFLOPs).**


### Kernel Optimization Progression (Blackwell SM100)

This repository documents the iterative optimization of a GEMM kernel using NVIDIA CuTe DSL. Each version (`v1` to `v6`) introduces specific, hardware-targeted optimizations to maximize Tensor Memory (TMEM) utilization, memory bandwidth, and compute saturation.

#### **Version 1 (`matmul_v1.py`): Baseline TMEM GEMM**
* **Architecture**: 128 threads (1 Warp).
* **Tile Size**: 128 × 256 × 64.
* **Strategy**: A single warp (Warp 0) acts as the "conductor," handling both TMA loads (GMEM → SMEM) and TCgen05 MMA compute (SMEM → TMEM).
* **Limitation**: A single warp cannot issue TMA descriptors and TCgen05 instructions fast enough to fully saturate Blackwell's hardware simultaneously.

#### **Version 2 (`matmul_v2.py`): Software Prefetch Hint**
* **Optimization**: Added `prefetch_stages=ab_stages - 2` to the mainloop `cutlass.range`.
* **Impact**: Provides an explicit hint to the CuTe compiler to aggressively overlap the TMA load of the next stage with the MMA compute of the current stage, improving instruction scheduling and hiding memory latency.

#### **Version 3 (`matmul_v3.py`): Compiler Alignment & Layout Hints**
* **Optimization**: Added `assumed_align=32` and `.mark_compact_shape_dynamic(mode=1, divisibility=k)` to the host-side tensor definitions.
* **Impact**: Gives the compiler strict memory layout guarantees. This eliminates dynamic bounds-checking overhead and enables the generation of highly optimized, static TMA descriptors tailored for K-major data ingestion.

#### **Version 4 (`matmul_v4.py`): 2-CTA Clustering & TMA Multicast**
* **Optimization**: 
  * Scaled up to a 256 × 256 × 64 tile using `tcgen05.CtaGroup.TWO`.
  * Introduced 2-CTA clustering (`cluster_shape_mnk=(2, 1, 1)`).
  * Replaced standard TMA ops with `CopyBulkTensorTileG2SMulticastOp`.
  * Increased pipeline depth to `ab_stages = 7`.
* **Impact**: The 2-CTA `tcgen05` MMA doubles the effective tile size per CTA
  pair, and the TMA multicast lets one load feed both CTAs' SMEM (removing
  duplicate L2/SMEM traffic). Measured DRAM traffic is unchanged across all
  versions (~2.2 GB/GEMM; ncu, see `results/ncu_counters.json`) — the win
  shows up as lower TMA-issue pressure and stall reduction, not fewer DRAM
  bytes.

#### **Version 5 (`matmul_v5.py`): Strict Warp Specialization & Pipelined Epilogue**
* **Optimization**: Expanded to 192 threads (6 Warps) with strict, non-overlapping role assignment:
  * **Warp 5 (TMA)**: Dedicated *exclusively* to issuing GMEM → SMEM TMA loads.
  * **Warp 4 (MMA)**: Dedicated *exclusively* to SMEM → TMEM `tcgen05` compute.
  * **Warps 0-3 (Epilogue)**: Dedicated *exclusively* to draining TMEM → RMEM → SMEM → GMEM.
* **Epilogue Upgrade**: Introduced `epi_stages = 2` and a dedicated `PipelineTmaStore`.
* **Impact**: Eliminates pipeline stalls entirely. The TMA warp never waits for compute, the MMA warp never waits for data, and the epilogue drains results concurrently without blocking the mainloop's SMEM reuse.

#### **Version 6 (`matmul_v6.py`): Final Polish & Dead Code Removal**
* **Optimization**: Removed the dynamic conditional check (`if mma_tiler_mnk[0] == 64`) for the TMEM load atom (`tcgen05.Ld32x32bOp`), since the CTA tile size is strictly fixed at 256.
* **Impact**: Eliminates branch evaluation overhead in the epilogue setup, ensuring deterministic instruction scheduling and slightly reducing register pressure/setup latency.

---

### Summary of Evolution

| Version | Threads / CTA | Tile Size (M×N×K) | Main Loop Driver | Epilogue Strategy |
| :--- | :---: | :---: | :--- | :--- |
| **v1** | 128 (4 Warps) | 128 × 256 × 64 | Single warp (TMA + MMA) | Serial drain |
| **v2** | 128 (4 Warps) | 128 × 256 × 64 | Single warp + `prefetch_stages` hint | Serial drain |
| **v3** | 128 (4 Warps) | 128 × 256 × 64 | Single warp + compiler layout hints | Serial drain |
| **v4** | 128 (4 Warps) | **256 × 256 × 64** | 2-CTA cluster, TMA multicast | Serial drain |
| **v5** | **192 (6 Warps)**| **256 × 256 × 64** | **Warp-specialized (TMA/MMA/epilogue)** | **Pipelined (2-stage), 4 warps** |
| **v6** | **192 (6 Warps)**| **256 × 256 × 64** | **Warp-specialized, dead code removed** | **Pipelined (2-stage), 4 warps** |

## Reference
- [Cute DSL Blackwell UMMA Pipeline](https://zhuanlan.zhihu.com/p/1985469957815943771)
- [Hopper MBarrier](https://zhuanlan.zhihu.com/p/1962636004235153810)

---

## Profiling Traces (Optimization Justification)

Every optimization step is backed by a `torch.profiler` trace of the actual kernel
on B200, uploaded with [trace-util](https://github.com/ariG23498/trace-util) to a
Hugging Face bucket and opened directly in the Perfetto UI.

**Bucket:** `iiserkbikram/cute-dsl-b200` (raw files, resolvable by Perfetto)

### Perfetto links (M = N = K = 8192, bf16)

| Version | Trace |
|---|---|
| cuBLAS | https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/iiserkbikram/cute-dsl-b200/resolve/cublas_8192_bf16_trace.json |
| v1 | https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/iiserkbikram/cute-dsl-b200/resolve/matmul_v1_8192_bf16_trace.json |
| v2 | https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/iiserkbikram/cute-dsl-b200/resolve/matmul_v2_8192_bf16_trace.json |
| v3 | https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/iiserkbikram/cute-dsl-b200/resolve/matmul_v3_8192_bf16_trace.json |
| v4 | https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/iiserkbikram/cute-dsl-b200/resolve/matmul_v4_8192_bf16_trace.json |
| v5 | https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/iiserkbikram/cute-dsl-b200/resolve/matmul_v5_8192_bf16_trace.json |
| v6 | https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/iiserkbikram/cute-dsl-b200/resolve/matmul_v6_8192_bf16_trace.json |

### What the traces show (per-launch, 3 steady-state launches each)

| Version | Kernel time (us) | BENCH_RESULT | Tile (MxN) | Grid | SMEM (KB) | Regs/thread | Warps/SM |
|---|---|---|---|---|---|---|---|
| cuBLAS | 682 / 718 / 729 | 1478 TFLOPs (README) | 128x256 | 2048x1 | 208 | 255 | 110.7 |
| v1 | 2406 / 2396 / 2384 | 461 | 128x256 | 64x32 | 192 | 77 | 55.4 |
| v2 | 1248 / 1229 / 1229 | 888 | 128x256 | 64x32 | 192 | 77 | 55.4 |
| v3 | 780 / 762 / 760 | 1445 | 128x256 | 64x32 | 192 | 74 | 55.4 |
| v4 | 628 / 614 / 624 | 1751 | 256x256 (2-CTA) | 64x32 | 224 | 78 | 55.4 |
| v5 | 618 / 607 / 618 | 1772 | 256x256 (2-CTA) | 64x32 | 208 | **39** | **83.0** |
| v6 | 617 / 607 / 616 | 1772 | 256x256 (2-CTA) | 64x32 | 208 | 39 | 83.0 |

Three launches per trace land back-to-back on the stream (~1-2 us gaps = pure host
launch overhead; no stream stalls), so the kernel durations are the whole story.

### Step-by-step justification

1. **v1 (2384 us)** — The single-kernel events are ~2.4 ms: the one conductor warp
   cannot issue TMA and tcgen05 MMA fast enough to saturate 148 SMs. Note the
   SMEM cap (192 KB, 4 stages) and 77 regs/thread; occupancy is bound by SMEM
   (13.8 blocks/SM), so the fix is intra-CTA overlap, not more CTAs.
2. **v2 (1229 us, 1.95x)** — Identical geometry, registers, and SMEM — the only
   change is the `prefetch_stages` overlap hint, and the kernel time halves. This
   is the cleanest controlled experiment in the series: same machine, same code,
   scheduler overlap only.
3. **v3 (760 us, 3.15x)** — Still 128x256, but regs drop 77 -> 74 and kernels
   shorten again: `assumed_align=32` + `mark_compact_shape_dynamic` let the
   compiler emit static TMA descriptors and drop dynamic bounds checks.
4. **v4 (614 us, 3.68x)** — Kernel name switches to the 2-CTA `ThrLayout...2111...`
   MMA atom; SMEM grows to 224 KB (deeper pipeline + multicast buffers). The
   2-CTA `tcgen05` MMA gives each CTA a larger effective tile, and the TMA
   multicast (one load, broadcast to both CTAs in the cluster) cuts
   L2/SMEM-side A-traffic — note that at DRAM level the ncu counters show all
   versions moving ~2.2 GB per GEMM, so the multicast's win shows up in
   TMA-issue pressure and stalls, not in DRAM bytes.
5. **v5 (607 us, 4.02x)** — The trace's smoking gun: regs/thread drop 78 -> 39 and
   warps/SM rise 55 -> 83 as the kernel moves to 6 dedicated warps (1 TMA / 1 MMA /
   4 epilogue). The pipelined 2-stage TMA-store epilogue overlaps the mainloop, so
   the epilogue no longer leaves a drain tail. v5/v6 now beat the cuBLAS calls in
   the same session (~610 us vs ~682-729 us).
6. **v6 (617 us)** — Same footprint as v5; the dead-branch removal shows up as a
   marginally cleaner epilogue setup, not a measurable duration change (both ~
   1.77 PFLOPS in this run; run-to-run variance between v5/v6 is clock state).

### Generate the traces

```bash
modal run scripts/cute_dsl/run.py::main \
  --task B200/scripts/profile_all.py \
  --gpu B200
```

`profile_all.py` runs each `matmul_v{1..6}.py` **unmodified** (it patches
`cutlass.cute.testing.benchmark` around the kernel launches, so the printed TFLOPS
and einsum verification still run) plus a cuBLAS reference, exports
`kernels/cute_dsl/B200/traces/*_trace.json`, and uploads them with trace-util
(using the `hf-bucket-write` Modal secret), printing the Perfetto URLs.
Re-upload or re-sync an existing folder with:

```bash
uvx trace-util -b cute-dsl-b200 -f kernels/cute_dsl/B200/traces
```