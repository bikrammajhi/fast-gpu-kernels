#pragma once
// ============================================================================
// bench.h — Shared benchmark harness for FlashAttention kernels.
//
// Usage: each kernel file defines its config constants, kernel, launch
// function, then ends with:
//     BENCH_MAIN(kernel_name, launch_fn)
// ============================================================================

#define BENCH_MAIN(kernel_name, launch_fn)                                     \
int main() {                                                                   \
    const int batch_size = 4;                                                  \
    const int len_q      = 128;                                               \
    const int len_kv     = 128;                                               \
                                                                               \
    const size_t Q_elems = (size_t)batch_size * len_q * HEAD_DIM;             \
    const size_t K_elems = (size_t)batch_size * len_kv * HEAD_DIM;            \
    const size_t V_elems = (size_t)batch_size * len_kv * HEAD_DIM;            \
    const size_t O_elems = (size_t)batch_size * len_q * HEAD_DIM;             \
                                                                               \
    const size_t Q_bytes = Q_elems * sizeof(nv_bfloat16);                     \
    const size_t K_bytes = K_elems * sizeof(nv_bfloat16);                     \
    const size_t V_bytes = V_elems * sizeof(nv_bfloat16);                     \
    const size_t O_bytes = O_elems * sizeof(nv_bfloat16);                     \
                                                                               \
    nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;                                       \
    cuda_check(cudaMalloc(&d_Q, Q_bytes));                                     \
    cuda_check(cudaMalloc(&d_K, K_bytes));                                     \
    cuda_check(cudaMalloc(&d_V, V_bytes));                                     \
    cuda_check(cudaMalloc(&d_O, O_bytes));                                     \
                                                                               \
    {                                                                          \
        nv_bfloat16* h_Q = (nv_bfloat16*)malloc(Q_bytes);                     \
        nv_bfloat16* h_K = (nv_bfloat16*)malloc(K_bytes);                     \
        nv_bfloat16* h_V = (nv_bfloat16*)malloc(V_bytes);                     \
                                                                               \
        for (size_t i = 0; i < Q_elems; ++i)                                  \
            h_Q[i] = __float2bfloat16(1e-3f * (float)(i % 64));              \
        for (size_t i = 0; i < K_elems; ++i)                                  \
            h_K[i] = __float2bfloat16(1e-3f * (float)(i % 64));              \
        for (size_t i = 0; i < V_elems; ++i)                                  \
            h_V[i] = __float2bfloat16(1e-3f * (float)(i % 64));              \
                                                                               \
        cuda_check(cudaMemcpy(d_Q, h_Q, Q_bytes, cudaMemcpyHostToDevice));    \
        cuda_check(cudaMemcpy(d_K, h_K, K_bytes, cudaMemcpyHostToDevice));    \
        cuda_check(cudaMemcpy(d_V, h_V, V_bytes, cudaMemcpyHostToDevice));    \
                                                                               \
        free(h_Q);                                                             \
        free(h_K);                                                             \
        free(h_V);                                                             \
    }                                                                          \
                                                                               \
    printf(#kernel_name "  batch=%d  Q=[%d,%d]  KV=[%d,%d]\n",               \
           batch_size, len_q, HEAD_DIM, len_kv, HEAD_DIM);                    \
                                                                               \
    for (int i = 0; i < 3; ++i)                                               \
        launch_fn(d_Q, d_K, d_V, d_O, batch_size, len_q, len_kv, 0);         \
    cuda_check(cudaDeviceSynchronize());                                       \
                                                                               \
    cudaEvent_t t0, t1;                                                        \
    cudaEventCreate(&t0);                                                      \
    cudaEventCreate(&t1);                                                      \
                                                                               \
    const int iters = 20;                                                      \
    cudaEventRecord(t0);                                                       \
    for (int i = 0; i < iters; ++i)                                           \
        launch_fn(d_Q, d_K, d_V, d_O, batch_size, len_q, len_kv, 0);         \
    cudaEventRecord(t1);                                                       \
    cudaEventSynchronize(t1);                                                  \
                                                                               \
    float ms = 0;                                                              \
    cudaEventElapsedTime(&ms, t0, t1);                                        \
    const float avg_ms = ms / iters;                                          \
                                                                               \
    const double flops  = 2.0 * batch_size * len_q * len_kv * HEAD_DIM;       \
    const double tflops = flops / (avg_ms * 1e-3) / 1e12;                     \
                                                                               \
    printf("Time:   %.3f ms (avg over %d iters)\n", avg_ms, iters);          \
    printf("FLOPS:  %.1f TFLOPS\n", tflops);                                 \
                                                                               \
    {                                                                          \
        nv_bfloat16* h_O = (nv_bfloat16*)malloc(O_bytes);                     \
        cuda_check(cudaMemcpy(h_O, d_O, O_bytes, cudaMemcpyDeviceToHost));    \
                                                                               \
        printf("O[0..7] = ");                                                 \
        for (int i = 0; i < 8; ++i) {                                         \
            unsigned short bits;                                               \
            memcpy(&bits, &h_O[i], sizeof(bits));                             \
            unsigned int fbits = ((unsigned int)bits) << 16;                  \
            float val;                                                         \
            memcpy(&val, &fbits, sizeof(val));                                \
            printf("%.6f ", val);                                             \
        }                                                                      \
        printf("\n");                                                          \
        free(h_O);                                                             \
    }                                                                          \
                                                                               \
    cudaEventDestroy(t0);                                                      \
    cudaEventDestroy(t1);                                                      \
    cudaFree(d_Q);                                                             \
    cudaFree(d_K);                                                             \
    cudaFree(d_V);                                                             \
    cudaFree(d_O);                                                             \
                                                                               \
    return 0;                                                                  \
}
