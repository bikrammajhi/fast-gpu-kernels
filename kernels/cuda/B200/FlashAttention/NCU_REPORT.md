# NCU Report — 1_baseline.cu on B200 (SM100a)

**Kernel:** `SM100a_FA_V1` | **Config:** batch=4, Q=[128,128], KV=[128,128], HEAD_DIM=128  
**Build:** `nvcc -arch=sm_100a -gencode arch=compute_100a,code=sm_100a` + CUDA 12.8.0, `-lcublas -lcuda`  
**Run:** `modal run scripts/ncu_capture.py --src kernels/cuda/B200/FlashAttention/1_baseline.cu --gpu B200 --launch-skip 3 --launch-count 1` (39 passes, `--set full --clock-control base`)  
**Report:** `captures/1_baseline.ncu-rep` (volume `gpulab-cute-dsl-traces`) — open with `ncu-ui`

---

## 1. Speed-of-Light & Throughput

| Metric | Value |
|---|---|
| Duration | **19.46 µs** |
| Elapsed Cycles | 22,340 |
| SM Frequency | 1.13 GHz |
| DRAM Frequency | 3.98 GHz |
| SM Active Cycles | 439 |
| Compute (SM) Throughput | **0.25 %** |
| Memory Throughput | **1.05 %** |
| DRAM Throughput | 0.34 % |
| L1/TEX Throughput | **31.94 %** |
| L2 Throughput | 1.05 % |
| Waves Per SM | **0.01** |

**Rule `SOLBottleneck`:** _Grid is too small — only 0.0 full waves across all SMs._

**Takeaway:** With `Grid=(4,1,1)` on a B200 (148 SMs) the GPU is 97% idle. Benchmark is correct for functional verification, but useless for perf. Use larger batch / longer sequence to measure real throughput (see §7).

---

## 2. Launch Statistics

| | Value |
|---|---|
| Block Size | 128 (4 warps) |
| Grid Size | 4 |
| Registers / Thread | **85** |
| Static SMEM / Block | 1.02 KB |
| Dynamic SMEM / Block | **98.30 KB** (Q+K+V tiles = 3× 32 KB) |
| Driver SMEM / Block | 1.02 KB |
| Configured SMEM | 200.70 KB |

---

## 3. Occupancy (148 SMs × 64 warps/SM on B200)

| | Value |
|---|---|
| Theoretical Warps / SM | 8 |
| Theoretical Occupancy | **12.5 %** |
| Achieved Warps / SM | 4.0 |
| Achieved Occupancy | **6.25 %** |
| Theoretical Warps / Scheduler | 2 / 16 max |
| Block Limit (Registers) | 5 |
| Block Limit (Shared Mem) | **2** ← limiter |
| Block Limit (Warps/SM/Barriers) | 16 / 32 / 32 |

**Rule `TheoreticalOccupancy`:** _Limited by shared memory_ (98 KB/block → max 2 blocks/SM; Blackwell max ≈ 228 KB/SM). Register pressure (85 regs/thread → 5 blocks) is secondary.

**Optimization levers:**
- Reduce SMEM: e.g. double-buffer Q/K tiles or use warp-specialized pipeline instead of 3 full tiles resident.
- Reduce registers: `__launch_bounds__(128, 2)` already set, but 85 regs is high — move softmax `float rowmax/rowsum` to SMEM or use `__restrict__` + `-maxrregcount 64` to trade spills for occupancy.
- Larger grid (bigger batch) hides the 0.01 waves/SM problem first.

---

## 4. Compute Workload Breakdown

```
SM: Inst Executed              0.25%
SM: Issue Active                0.25%
  ├─ Tensor (tcgen05.mma)       0.12%
  ├─ Tc                         0.14%
  ├─ Alu                        0.12%
  ├─ Fma                        0.11%
  ├─ Shared                     0.12%
  ├─ Xu                         0.13%
  └─ Tma                        0.07%
```

Tensor/TMA are barely active — kernel is dominated by small-tile overhead and softmax epilogue, not math.

---

## 5. Memory Workload

- Global stores: only **16/32 bytes per sector utilized** (Rule `MemoryCacheAccessPattern`) — epilogue does `int4` (16 B) stores of `bf162` pairs but with misaligned `out_ptr + row*HEAD_DIM` pattern. Coalesce to 32 B or use `st.global.v4`.
- L2 sectors: **8192 actual vs 4096 ideal** (2× overhead) — `SourceCounters: L2 Theoretical Sectors Global Excessive 4.10 KB`.
- L1 conflicts: **15 shared N-way bank conflicts** (K-major 8×8 atom layout helps but softmax P writes still conflict).
- `smsp__sass_inst_executed_op_tma_ld = 1152`, `op_global_st = 256` — TMA is the only global load path, which is correct.

---

## 6. Warp & Scheduler Stalls

**Top stall (WarpStateStats):**
> 4.4 cycles avg per warp waiting for **L1TEX scoreboard** (57% of 7.7 cycles between issues).

Caused by `tcgen05.ld` (TMEM→RF) latency and TMA `mbarrier.wait` gaps. Solutions:
- Overlap TMA load of next KV tile with current QK/PV MMA (pipeline depth 2).
- Use `tcgen05.commit` + `mbarrier` double-buffer so LSU and tensor core overlap.
- Group `tcgen05.ld.32x32b.x8` into fewer, larger loads.

Other stalls: `LG_THROTTLE`, `MIO_THROTTLE` < 1 cycle — not critical.

**L2 imbalance:** min slice 48.9% below avg, max 32.3% above — 4-block grid causes uneven L2 slice utilization.

---

## 7. What to Fix First (future kernels)

1. **Scale the problem:** `batch=32, len_q=2048, len_kv=2048` → Grid=512, Waves/SM ≈ 3.5. Re-run NCU; current numbers are fabric-limited, not kernel-limited.
2. **SMEM occupancy:** Shrink to 2 tiles (Q + K/P) + V streamed, or use `cudaFuncAttributeMaxDynamicSharedMemorySize` + `__shared__` aliasing already done but still 98 KB.
3. **Register budget:** Target ≤64 regs/thread (try `__launch_bounds__(128,2)` + `#pragma unroll` tuning). Check `ncu --section Occupancy --metrics sm__maximum_warps_per_active_cycle_pct`.
4. **Coalescing:** Fix epilogue store to 32 B transactions; verify with `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`.
5. **Pipeline:** Add TMA → MMA overlap (producer warps vs consumer warpgroup) — baseline is fully synchronous per tile.
6. **Keep `bench.h` harness:** ensures consistent `launch-skip` for NCU (warmup=3, skip=3).

---

## 8. Reproduce

```bash
# Capture (B200, fix in scripts/ncu_capture.py: GPU_ARCH B200 = sm_100a, -lcuda, image 12.8.0)
modal run scripts/ncu_capture.py --src kernels/cuda/B200/FlashAttention/1_baseline.cu --gpu B200 --launch-skip 3 --launch-count 1

# Download & open
modal volume get gpulab-cute-dsl-traces captures/1_baseline.ncu-rep .
ncu-ui 1_baseline.ncu-rep   # or ncu --import captures/1_baseline.ncu-rep --section Occupancy --csv
```

Report generated 2026-09-03 from NCU 2025.1.1 (CUDA 12.8) on B200, driver 580.95.05.
