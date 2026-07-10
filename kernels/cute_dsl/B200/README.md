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

## Optimizations

### v1 → v2: K-tile software pipelining
Added `prefetch_stages=ab_stages - 2` to overlap TMA loads with MMA:
```python
for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
```
Throughput doubled: 455 → 895 TFLOPs.

### v2 → v3: Aligned, compact dynamic shapes
```python
a_tensor_ = (
    from_dlpack(a, assumed_align=32)
    .mark_layout_dynamic(leading_dim=1)
    .mark_compact_shape_dynamic(mode=1, divisibility=k)
)
```
`assumed_align=32` ensures TMA alignment. `mark_compact_shape_dynamic` enables efficient codegen.
Another 1.6x speedup: 895 → 1438 TFLOPs.

### v3 → v4: 2-CTA MMA
```python
use_2cta_instrs = True
cluster_shape_mnk = (2, 1, 1)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
ab_stages = 7
```

Benefits:
- **Reduced B SMEM size**: For 1CTA, B = 256x64x2B = 32KB (limits stages to 4). For 2CTA, B SMEM is halved to 16KB, allowing 7 stages.
- **Better latency hiding**: 512*(7-1) = 3K cycles vs 1.5K cycles for 1CTA.
- **Reduced L2 traffic via TMA multicast**: For 2x1 cluster, L2 traffic = 16KB/1 + 32KB/2 = 24KB vs 48KB without multicast.

### v4 → v5: Warp-specialized design
Dedicated warps: TMA (ID 5), MMA (ID 4), Epilogue (IDs 0-3). Warps work in parallel, hiding DRAM latency.

### v5 → v6: Code cleanup
Cleaned version maintaining identical optimizations.
