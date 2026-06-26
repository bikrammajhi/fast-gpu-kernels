#pragma once
#include "common.h"

// =============================================================================
// matmul_v1 — basic smem + ldmatrix, no pipelining
// =============================================================================

template <
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int NUM_WARP_M,
    int NUM_WARP_N,
    int SMEM_STRIDE
>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void matmul_v1_kern(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
    constexpr int CTA_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_blocks_n = cdiv(N, BLOCK_N);
    const int block_m = bid / num_blocks_n;
    const int block_n = bid % num_blocks_n;
    const int offset_m = block_m * BLOCK_M;
    const int offset_n = block_n * BLOCK_N;

    const int warp_m = warp_id / NUM_WARP_N;
    const int warp_n = warp_id % NUM_WARP_N;

    A += offset_m * K;
    B += offset_n * K;
    C += (offset_m + warp_m * WARP_M) * N + (offset_n + warp_n * WARP_N);

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* A_smem = smem;
    __nv_bfloat16* B_smem = smem + BLOCK_M * SMEM_STRIDE;

    constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
    constexpr int num_A_regs = MMA_M * MMA_K * sizeof(__nv_bfloat16) / 4 / WARP_SIZE;
    constexpr int num_B_regs = MMA_K * MMA_N * sizeof(__nv_bfloat16) / 4 / WARP_SIZE;

    float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};

    for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
        gmem2smem<CTA_SIZE, BLOCK_M, BLOCK_K, SMEM_STRIDE>(A, K, A_smem, tid);
        gmem2smem<CTA_SIZE, BLOCK_N, BLOCK_K, SMEM_STRIDE>(B, K, B_smem, tid);
        __syncthreads();

        for (int k = 0; k < BLOCK_K; k += MMA_K) {
            const __nv_bfloat16* A_smem_warp = A_smem + warp_m * WARP_M * SMEM_STRIDE + k;
            const __nv_bfloat16* B_smem_warp = B_smem + warp_n * WARP_N * SMEM_STRIDE + k;

            uint32_t B_regs[NUM_MMA_N][num_B_regs];
            #pragma unroll
            for (int n = 0; n < NUM_MMA_N; ++n) {
                const __nv_bfloat16* B_smem_ptr = B_smem_warp
                    + (n * MMA_N + (lane_id % 8)) * SMEM_STRIDE
                    + (lane_id / 8) * 8;
                LDMATRIX_X2(B_regs[n], to_smem(B_smem_ptr));
            }

            #pragma unroll
            for (int m = 0; m < NUM_MMA_M; ++m) {
                uint32_t A_regs[num_A_regs];
                const __nv_bfloat16* A_smem_ptr = A_smem_warp
                    + (m * MMA_M + lane_id % 16) * SMEM_STRIDE
                    + (lane_id / 16) * 8;
                LDMATRIX_X4(A_regs, to_smem(A_smem_ptr));
                #pragma unroll
                for (int n = 0; n < NUM_MMA_N; ++n)
                    MMA_M16N8K16(A_regs, B_regs[n], acc[m][n]);
            }
        }
        __syncthreads();
        A += BLOCK_K;
        B += BLOCK_K;
    }

    #pragma unroll
    for (int m = 0; m < NUM_MMA_M; ++m) {
        for (int n = 0; n < NUM_MMA_N; ++n) {
            const int c_row = m * MMA_M + (lane_id / 4);
            const int c_col = n * MMA_N + (lane_id % 4) * 2;
            float* regs = acc[m][n];
            __nv_bfloat162 tmp;
            tmp.x = __float2bfloat16_rn(regs[0]);
            tmp.y = __float2bfloat16_rn(regs[1]);
            reinterpret_cast<__nv_bfloat162*>(C + (c_row + 0) * N + c_col)[0] = tmp;
            tmp.x = __float2bfloat16_rn(regs[2]);
            tmp.y = __float2bfloat16_rn(regs[3]);
            reinterpret_cast<__nv_bfloat162*>(C + (c_row + 8) * N + c_col)[0] = tmp;
        }
    }
}

inline void matmul_v1_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              __nv_bfloat16* C, int M, int N, int K)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int NUM_WARP_M = 2;
    constexpr int NUM_WARP_N = 2;
    constexpr int SMEM_STRIDE = BLOCK_K;

    launch_safe(
        matmul_v1_kern<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SMEM_STRIDE>,
        cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),
        NUM_WARP_M * NUM_WARP_N * WARP_SIZE,
        (BLOCK_M + BLOCK_N) * SMEM_STRIDE * (int)sizeof(__nv_bfloat16),
        A, B, C, M, N, K);
}
