# H100 CuTe Kernels

Hopper (SM90) kernels targeting H100 80GB HBM3.

## Results (M=N=K=16384, bf16)

| Kernel | Status | Duration | TFLOPS |
|--------|--------|----------|--------|
| v1 | PASS | 24.6800 ms | 356.4 |
| v2 | PASS | 24.0442 ms | 365.8 |
| v3 | PASS | 24.0581 ms | 365.6 |
| v4 | PASS | 24.0498 ms | 365.7 |

## Run

```bash
modal run scripts/cute/run.py::main --task H100/matmul_v1.cu --gpu H100
modal run scripts/cute/run.py::main --task H100/matmul_v4.cu --gpu H100
```
