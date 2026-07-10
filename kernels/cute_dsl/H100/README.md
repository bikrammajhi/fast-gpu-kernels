# H100 CuTe DSL Kernels

Hopper (SM90) kernels targeting H100 80GB HBM3.

- GPU: NVIDIA H100 80GB HBM3
- IO dtype: Float16, Accum dtype: Float32
- Problem: square `M = N = K = 4096`

Reference: [NVIDIA CUTLASS — Tour to a Solution GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/notebooks/tour_to_sol_gemm.ipynb)

## Results (M = N = K = 4096)

| Kernel | Duration | TFLOPS | % of cuBLAS |
|--------|----------|--------|-------------|
| cuBLAS | Reference | 776.0 | — |
| DSL v2 | 200.91 µs | 684.07 | ~88% |

All versions pass numerical verification against a PyTorch `einsum` reference.

## Run

```bash
modal run scripts/cute_dsl/run.py::main --task H100/matmul_v2.py --gpu H100
```
