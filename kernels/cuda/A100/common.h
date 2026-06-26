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

__host__ __device__ inline int swizzle_block_idx(int bid, int grid_n, int grid_m) {
    constexpr int GROUP_SIZE = 8;
    int num_groups_m = cdiv(grid_m, GROUP_SIZE);
    int group_id = bid / GROUP_SIZE;
    int group_idx = bid % GROUP_SIZE;
    int block_m = group_id % num_groups_m;
    int block_n = group_idx + (group_id / num_groups_m) * GROUP_SIZE;
    if (block_n >= grid_n) return -1;
    return block_m * grid_n + block_n;
}

// Triton-style threadblock swizzle
__host__ __device__ inline void swizzle_block_idx_triton(int bid, int grid_m, int grid_n, int &bid_m, int &bid_n) {
    constexpr int GROUP_M = 8;
    if constexpr (GROUP_M == 0) {
        bid_m = bid / grid_n;
        bid_n = bid % grid_n;
    } else {
        const int group_size = GROUP_M * grid_n;
        const int group_id = bid / group_size;
        const int group_off_m = group_id * GROUP_M;
        const int group_m = (grid_m - group_off_m) < GROUP_M ? (grid_m - group_off_m) : GROUP_M;
        bid_m = group_off_m + ((bid % group_size) % group_m);
        bid_n = (bid % group_size) / group_m;
    }
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

// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy
__device__ inline
void cp_async(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
};

__device__ inline
void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); };

template <int N>
__device__ inline
void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); };

__device__ inline
void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); };

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / ((128 / STRIDE) > 1 ? (128 / STRIDE) : 1);
  return index ^ (bits_to_xor << 4);
}

template<int CTA_SIZE, int HEIGHT, int WIDTH, int DST_STRIDE, bool use_swizzle>
__device__ void gmem2smem_async(const __nv_bfloat16* src, int src_stride,
                           __nv_bfloat16* smem_ptr, int tid)
{
    constexpr int ne = 16 / sizeof(__nv_bfloat16);  // cp.async requires 16B aligned address
    constexpr int ni = (HEIGHT * WIDTH) / (CTA_SIZE * ne);
    constexpr int STRIDE_BYTES = DST_STRIDE * sizeof(__nv_bfloat16);
    uint32_t dst_addr_base = to_smem(smem_ptr);
    #pragma unroll
    for (int i = 0; i < ni; ++i) {
        const int idx = (i * CTA_SIZE + tid) * ne;
        const int row = idx / WIDTH, col = idx % WIDTH;
        uint32_t dst_addr = dst_addr_base + (row * DST_STRIDE + col) * sizeof(__nv_bfloat16);
        if constexpr (use_swizzle)
            dst_addr = swizzle<STRIDE_BYTES>(dst_addr);
        cp_async(dst_addr, src + row * src_stride + col);
    }
}

// STRIDE in bytes, col in units of 16 bytes
template <int STRIDE>
__device__ static uint32_t swizzle_better(uint32_t row, uint32_t col) {
  if constexpr (STRIDE >= 128)
    col ^= (row % 8) / ((128 / STRIDE) > 1 ? (128 / STRIDE) : 1);
  return row * STRIDE + col * 16;
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
