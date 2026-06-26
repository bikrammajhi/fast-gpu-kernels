// =============================================================================
// benchmark.cu — GEMM benchmark harness for A100
//
// To add a new kernel version:
//   1. Create matmul_vN.cu with your kernel
//   2. #include it below
//   3. Add { "vN", matmul_vN_launch } to the kernels[] array
// =============================================================================

#include "common.h"

// --- include kernel versions ---
#include "matmul_v1.cu"
#include "matmul_v2.cu"
#include "matmul_v3.cu"
#include "matmul_v4.cu"
#include "matmul_v5.cu"
#include "matmul_v6.cu"
#include "matmul_v7.cu"
#include "matmul_v8.cu"
#include "matmul_v9.cu"
#include "matmul_v10.cu"
#include "matmul_v11.cu"
#include "matmul_v12.cu"
#include "matmul_v13.cu"
#include "matmul_v14.cu"
#include "matmul_streamk.cu"

// =============================================================================
// Kernel registry
// =============================================================================

struct Kernel {
    const char* name;
    void (*launch)(const __nv_bfloat16* A, const __nv_bfloat16* B,
                   __nv_bfloat16* C, int M, int N, int K);
};

Kernel kernels[] = {
    // =====================================================================
    // MAIN OPTIMIZATION PROGRESSION (each adds ONE new technique)
    // =====================================================================
    // 1. v1    — Baseline: sync gmem->smem, 2x2 warps, single-stage
    // 2. v2    — + cp.async + 2-stage pipeline (overlap load/compute)
    // 3. v3    — + shared memory padding to eliminate bank conflicts
    // 4. v4    — + swizzle replaces padding (same perf, saves smem)
    // 5. v7s3  — + swizzle_better + ldmatrix_x4 for B + multi-stage
    // 6. v10   — + lambda-local register declarations (better compiler opts)
    // 7. v11a  — + 4x2 warps (256 threads, more ILP + latency hiding)
    // 8. v13d  — + double-buffered register loads + GROUP_M=16
    //
    // SIDE EXPERIMENTS (different approaches, included for completeness)
    // 9. v14      — shared memory epilogue attempt (regression)
    // 10. stream-k — Stream-K decomposition (different parallelization)
    // =====================================================================
    { "v1",       matmul_v1_launch },
    { "v2",       matmul_v2_launch },
    { "v3",       matmul_v3_launch },
    { "v4",       matmul_v4_launch },
    { "v7s3",     matmul_v7s3_launch },
    { "v10",      matmul_v10_launch },
    { "v11a",     matmul_v11a_launch },
    { "v13d",     matmul_v13d_launch },
    { "v14",      matmul_v14_launch },
    { "stream-k", matmul_streamk_launch },
};

// =============================================================================
// Timing
// =============================================================================

struct Result {
    float ms;
    double tflops;
    double pct;
};

template<typename Fn>
static Result bench(Fn fn, int warmup, int iters, double flops, double peak)
{
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    double tf = flops / (ms / iters * 1e-3) / 1e12;
    return { ms / iters, tf, tf / peak * 100.0 };
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int n_kern = sizeof(kernels) / sizeof(kernels[0]);
    const int sizes[] = { 128, 256, 512, 1024, 2048, 4096, 8192 };
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int warmup = 5, iters = 20;
    const double peak = 312.0;

    // --- header ---
    printf("\n%-8s ", "N");
    for (int k = 0; k < n_kern; k++)
        printf(" %-18s", kernels[k].name);
    printf(" %-18s\n", "cuBLAS");

    printf("-------");
    for (int k = 0; k < n_kern; k++) printf(" ------------------");
    printf(" ------------------\n");

    // --- sweep sizes ---
    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t sz_a = (size_t)M * K * sizeof(__nv_bfloat16);
        size_t sz_b = (size_t)K * N * sizeof(__nv_bfloat16);
        size_t sz_c = (size_t)M * N * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, sz_a));
        CUDA_CHECK(cudaMalloc(&d_B, sz_b));
        CUDA_CHECK(cudaMalloc(&d_C, sz_c));

        {
            __nv_bfloat16 *h_a = (__nv_bfloat16*)malloc(sz_a);
            __nv_bfloat16 *h_b = (__nv_bfloat16*)malloc(sz_b);
            for (size_t i = 0; i < (size_t)M * K; i++) h_a[i] = __float2bfloat16(1.0f);
            for (size_t i = 0; i < (size_t)K * N; i++) h_b[i] = __float2bfloat16(1.0f);
            CUDA_CHECK(cudaMemcpy(d_A, h_a, sz_a, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_b, sz_b, cudaMemcpyHostToDevice));
            free(h_a); free(h_b);
        }

        double flops = 2.0 * (double)M * N * K;

        printf("%-8d ", M);

        for (int k = 0; k < n_kern; k++) {
            // Correctness check on first iteration
            CUDA_CHECK(cudaMemset(d_C, 0, sz_c));
            kernels[k].launch(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            {
                __nv_bfloat16 *h_c = (__nv_bfloat16*)malloc(sz_c);
                CUDA_CHECK(cudaMemcpy(h_c, d_C, sz_c, cudaMemcpyDeviceToHost));
                auto bf16_to_float = [](__nv_bfloat16 v) -> float {
                    unsigned short bits;
                    memcpy(&bits, &v, sizeof(bits));
                    unsigned int fbits = ((unsigned int)bits) << 16;
                    float result;
                    memcpy(&result, &fbits, sizeof(result));
                    return result;
                };
                bool ok = true;
                for (int i = 0; i < M * N; i++) {
                    float val = bf16_to_float(h_c[i]);
                    float expected = (float)K;
                    float rel_err = fabsf(val - expected) / expected;
                    if (rel_err > 0.01f) {
                        int row = i / N, col = i % N;
                        printf("\n    WRONG at [%d,%d] val=%.4f expected=%.4f\n", row, col, val, expected);
                        ok = false;
                        break;
                    }
                }
                free(h_c);
                if (!ok) { printf(" %5s  WRONG    ", kernels[k].name); continue; }
            }

            auto r = bench([&]() { kernels[k].launch(d_A, d_B, d_C, M, N, K); },
                           warmup, iters, flops, peak);
            printf(" %5.1f%% %7.1f TF", r.pct, r.tflops);
        }

        auto cb = bench([&]() { cublas_gemm(handle, d_A, d_B, d_C, M, N, K); },
                        warmup, iters, flops, peak);
        printf(" %5.1f%% %7.1f TF\n", cb.pct, cb.tflops);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    cublasDestroy(handle);
    printf("\n");
    return 0;
}
