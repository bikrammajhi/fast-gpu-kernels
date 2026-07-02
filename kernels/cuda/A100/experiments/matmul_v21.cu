#pragma once
#include "common.h"

// =============================================================================
// matmul_v21 — v11a baseline + cp.async.ca + 3-stage pipeline
//   Builds directly on v11a (128x128x64, 4x2 warps, 256 threads).
//   Two targeted changes from v11a:
//     1. cp.async.ca.shared.global.L2::128B  — keep prefetched tiles sticky in L2
//     2. 3-stage pipeline                     — deeper in-flight prefetch window
// =============================================================================

template <
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int NUM_WARP_M, int NUM_WARP_N,
    int NUM_STAGES
>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void matmul_v21_kern(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    constexpr int MMA_K = 16;
    constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
    constexpr int CTA_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;

    const int grid_m = cdiv(M, BLOCK_M);
    const int grid_n = cdiv(N, BLOCK_N);
    int bid_m, bid_n;
    swizzle_block_idx_triton(bid, grid_m, grid_n, bid_m, bid_n);

    A += bid_m * BLOCK_M * K;
    B += bid_n * BLOCK_N * K;
    C += (bid_m * BLOCK_M + warp_id_m * WARP_M) * N + (bid_n * BLOCK_N + warp_id_n * WARP_N);

    constexpr int A_size = BLOCK_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int B_size = BLOCK_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int AB_size = A_size + B_size;

    extern __shared__ __nv_bfloat16 shm[];
    const uint32_t shm_u32 = to_smem(shm);
    const uint32_t A_shm_base = shm_u32;
    const uint32_t B_shm_base = A_shm_base + A_size;

    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

    constexpr int STRIDE_BYTES = BLOCK_K * (int)sizeof(__nv_bfloat16);
    const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
    const uint32_t A_shm_thread = A_shm_base + swizzle_better<STRIDE_BYTES>(A_offm, lane_id / 16);
    const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
    const uint32_t B_shm_thread = B_shm_base + swizzle_better<STRIDE_BYTES>(B_offn, (lane_id % 16) / 8);

    const int num_k_iters = cdiv(K, BLOCK_K);
    const __nv_bfloat16 *A_ptr = A;
    const __nv_bfloat16 *B_ptr = B;

    auto load_AB = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;
        g2s_swizzled_ca<CTA_SIZE, BLOCK_M, BLOCK_K>(A_ptr, K, A_shm_base + stage_id * AB_size, tid);
        g2s_swizzled_ca<CTA_SIZE, BLOCK_N, BLOCK_K>(B_ptr, K, B_shm_base + stage_id * AB_size, tid);
        A_ptr += BLOCK_K;
        B_ptr += BLOCK_K;
        cp_async_commit_group();
    };

    auto compute = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;
        const uint32_t stage_offset = stage_id * AB_size;

        uint32_t A_regs[NUM_MMA_K][NUM_MMA_M][4];
        uint32_t B_regs[NUM_MMA_K][NUM_MMA_N][2];

        #pragma unroll
        for (int k = 0; k < NUM_MMA_K; k++) {
            #pragma unroll
            for (int m = 0; m < NUM_MMA_M; m++) {
                uint32_t A_addr = A_shm_thread + stage_offset;
                A_addr += m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
                LDMATRIX_X4(A_regs[k][m], A_addr ^ (k * 32));
            }
            #pragma unroll
            for (int n = 0; n < NUM_MMA_N; n += 2) {
                uint32_t B_addr = B_shm_thread + stage_offset;
                B_addr += n * MMA_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
                LDMATRIX_X4(B_regs[k][n], B_addr ^ (k * 32));
            }
        }

        #pragma unroll
        for (int k = 0; k < NUM_MMA_K; k++)
            #pragma unroll
            for (int m = 0; m < NUM_MMA_M; m++)
                #pragma unroll
                for (int n = 0; n < NUM_MMA_N; n++)
                    MMA_M16N8K16(A_regs[k][m], B_regs[k][n], acc[m][n]);
    };

    for (int stage = 0; stage < NUM_STAGES - 1 && stage < num_k_iters; stage++)
        load_AB(stage);

    constexpr int PD = NUM_STAGES - 1;
    for (int k_iter = 0; k_iter < num_k_iters - PD; k_iter++) {
        __syncthreads();
        load_AB(k_iter + PD);
        cp_async_wait_group<PD>();
        __syncthreads();
        compute(k_iter);
    }

    for (int k_iter = num_k_iters - PD; k_iter < num_k_iters; k_iter++) {
        __syncthreads();
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();
        compute(k_iter);
    }

    epilogue_store_bf16<NUM_MMA_M, NUM_MMA_N>(acc, C, N, lane_id, warp_id_m, WARP_M, warp_id_n, WARP_N);
}

// =============================================================================
// v21: 128x128x64, 4x2 warps (256 threads), 3-stage, cp.async.ca
// =============================================================================
inline void matmul_v21_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              __nv_bfloat16* C, int M, int N, int K) {
    constexpr int BM=128, BN=128, BK=64, WM=4, WN=2, ST=3;
    constexpr int smem = (BM*BK + BN*BK) * (int)sizeof(__nv_bfloat16) * ST;
    launch_safe(matmul_v21_kern<BM,BN,BK,WM,WN,ST>,
        cdiv(M,BM)*cdiv(N,BN), WM*WN*WARP_SIZE, smem, A,B,C,M,N,K);
}
