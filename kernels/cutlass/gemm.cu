#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cute;

__global__ void gemm_kernel(
    float* A, float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    auto tA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, 1));
    auto tB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, 1));
    auto tC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, 1));

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += tA(row, k) * tB(k, col);
    }
    tC(row, col) = acc;
}

int main() {
    const int M = 256, N = 256, K = 256;

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // verify: every element should be K (sum of K ones * ones)
    bool pass = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - K) > 1e-3) { pass = false; break; }
    }

    printf("cutlass gemm (cute tensors): M=%d N=%d K=%d  %s\n",
           M, N, K, pass ? "PASSED" : "FAILED");

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
