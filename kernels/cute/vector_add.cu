#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cute;

__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    auto tA = make_tensor(make_gmem_ptr(a), make_shape(n));
    auto tB = make_tensor(make_gmem_ptr(b), make_shape(n));
    auto tC = make_tensor(make_gmem_ptr(c), make_shape(n));

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tC(i) = tA(i) + tB(i);
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("cute vector_add: n=%d  c[0]=%.1f  c[n-1]=%.1f\n", n, h_c[0], h_c[n-1]);

    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
