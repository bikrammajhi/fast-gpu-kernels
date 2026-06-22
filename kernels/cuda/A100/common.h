#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

__host__ __device__ inline constexpr int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;
static constexpr int WARP_SIZE = 32;

__device__ inline uint32_t to_smem(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ inline void LDMATRIX_X2(uint32_t reg[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
}

__device__ inline void LDMATRIX_X4(uint32_t reg[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
}

__device__ inline void MMA_M16N8K16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]));
}

template<int CTA_SIZE, int HEIGHT, int WIDTH, int DST_STRIDE>
__device__ void gmem2smem(const __nv_bfloat16* src, int src_stride,
                           __nv_bfloat16* dst, int tid)
{
    constexpr int ne = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int ni = (HEIGHT * WIDTH) / (CTA_SIZE * ne);
    #pragma unroll
    for (int i = 0; i < ni; ++i) {
        const int idx = (i * CTA_SIZE + tid) * ne;
        const int row = idx / WIDTH, col = idx % WIDTH;
        reinterpret_cast<uint4*>(dst + row * DST_STRIDE + col)[0] =
            reinterpret_cast<const uint4*>(src + row * src_stride + col)[0];
    }
}

template<typename Kern, typename... Args>
void launch_safe(Kern* k, int grid, int block, int smem, Args... args) {
    if (smem > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(
            k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    k<<<grid, block, smem>>>(args...);
    CUDA_CHECK(cudaGetLastError());
}

inline void cublas_gemm(cublasHandle_t h,
                         const __nv_bfloat16* A, const __nv_bfloat16* B,
                         __nv_bfloat16* C, int M, int N, int K)
{
    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    __nv_bfloat16 beta  = __float2bfloat16(0.0f);
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K,
        &beta,  C, CUDA_R_16BF, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}
