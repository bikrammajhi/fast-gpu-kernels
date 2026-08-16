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

---

## Profiling with Nsight Compute (ncu)

Capture a full-detail NVIDIA Nsight Compute report (`.ncu-rep`) and open it
in your local Nsight Compute GUI — no local GPU needed. Only the report file
is written, to the path you give `-o`:

```bash
nvcc -O3 -arch=sm_90 -lcublas -I<your cutlass>/include -I<your cutlass>/tools/util/include \
    -o /tmp/bench benchmark.cu
ncu --set full --warp-sampling-interval auto --clock-control base \
    --launch-skip 1 --launch-count 1 -o out/matmul_v4.ncu-rep /tmp/bench
```

On Modal (no local GPU) — download only the `.ncu-rep` (the runner compiles
with the repo's cutlass clone and arch map):

```bash
modal run scripts/ncu_capture.py --src kernels/cute/H100/benchmark.cu --gpu H100
modal volume get gpulab-cute-dsl-traces captures/benchmark.ncu-rep ./benchmark.ncu-rep
```

Open it locally: `ncu-ui benchmark.ncu-rep` (or File → Open in the Nsight
Compute GUI) — ncu only needs the GPU at capture time.
