#pragma once
#include "common.h"

// =============================================================================
// matmul_v9 — High-performance kernel with interleaved load/compute
//   - Interleaves ldmatrix with MMA to reduce register lifetime
//   - 256x128x64 tile with 4x2 warps (256 threads) for better occupancy
//   - 128x128x64 tile with 2x2 warps (128 threads) variant
//   - Custom swizzle for bank-conflict-free shared memory access
// =============================================================================

// =============================================================================
// Core kernel: interleaved ldmatrix + MMA
// =============================================================================

template <
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int NUM_WARP_M, int NUM_WARP_N,
    int NUM_STAGES, int GROUP_M
>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void matmul_v9_kern(
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
    swizzle_block_idx_triton(bid, grid_m, grid_n, bid_m, bid_n, GROUP_M);

    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;
    A += off_m * K;
    B += off_n * K;
    C += (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

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

    auto load_AB = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;
        g2s_swizzled<CTA_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shm_base + stage_id * AB_size, tid);
        g2s_swizzled<CTA_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shm_base + stage_id * AB_size, tid);
        A += BLOCK_K;
        B += BLOCK_K;
        cp_async_commit_group();
    };

    // Interleaved: load regs + compute in same pass to reduce register lifetime
    auto compute_interleaved = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;
        const uint32_t stage_offset = stage_id * AB_size;

        #pragma unroll
        for (int k = 0; k < NUM_MMA_K; k++) {
            // Load A regs for this k-slice
            #pragma unroll
            for (int m = 0; m < NUM_MMA_M; m++) {
                uint32_t A_addr = A_shm_thread + stage_offset;
                A_addr += m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
                uint32_t A_reg[4];
                LDMATRIX_X4(A_reg, A_addr ^ (k * 32));
                // Immediately feed into MMA below
            }

            // Load B regs for this k-slice
            #pragma unroll
            for (int n = 0; n < NUM_MMA_N; n += 2) {
                uint32_t B_addr = B_shm_thread + stage_offset;
                B_addr += n * MMA_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
                uint32_t B_reg[2];
                LDMATRIX_X4(B_reg, B_addr ^ (k * 32));

                // MMA for all m rows with this B
                #pragma unroll
                for (int m = 0; m < NUM_MMA_M; m++) {
                    uint32_t A_addr2 = A_shm_thread + stage_offset;
                    A_addr2 += m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
                    uint32_t A_r[4];
                    LDMATRIX_X4(A_r, A_addr2 ^ (k * 32));
                    MMA_M16N8K16(A_r, B_reg, acc[m][n]);
                }
            }
        }
    };

    // Simple compute (no interleaving) for comparison
    auto compute_simple = [&](int k_iter) {
        uint32_t A_regs[NUM_MMA_K][NUM_MMA_M][4];
        uint32_t B_regs[NUM_MMA_K][NUM_MMA_N][2];
        const int stage_id = k_iter % NUM_STAGES;
        const uint32_t stage_offset = stage_id * AB_size;

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

    // Prologue
    for (int stage = 0; stage < NUM_STAGES - 1 && stage < num_k_iters; stage++)
        load_AB(stage);

    // Main loop
    constexpr int PIPELINE_DEPTH = NUM_STAGES - 1;
    for (int k_iter = 0; k_iter < num_k_iters - PIPELINE_DEPTH; k_iter++) {
        __syncthreads();
        load_AB(k_iter + PIPELINE_DEPTH);
        cp_async_wait_group<PIPELINE_DEPTH>();
        __syncthreads();
        compute_simple(k_iter);
    }

    // Epilogue
    for (int k_iter = num_k_iters - PIPELINE_DEPTH; k_iter < num_k_iters; k_iter++) {
        __syncthreads();
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();
        compute_simple(k_iter);
    }

    epilogue_store_bf16<NUM_MMA_M, NUM_MMA_N>(acc, C, N, lane_id, warp_id_m, WARP_M, warp_id_n, WARP_N);
}

// =============================================================================
// Launch configurations
// =============================================================================

// v9: 128x128x64, 2x2, 2-stage, GROUP_M=8 (baseline, matches v7s2)
inline void matmul_v9_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              __nv_bfloat16* C, int M, int N, int K) {
    constexpr int BM=128, BN=128, BK=64, WM=2, WN=2, ST=2, GM=8;
    constexpr int smem = (BM*BK + BN*BK) * (int)sizeof(__nv_bfloat16) * ST;
    launch_safe(matmul_v9_kern<BM,BN,BK,WM,WN,ST,GM>,
        cdiv(M,BM)*cdiv(N,BN), WM*WN*WARP_SIZE, smem, A,B,C,M,N,K);
}

// v9b: 256x128x64, 4x2, 2-stage, GROUP_M=8
inline void matmul_v9b_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                               __nv_bfloat16* C, int M, int N, int K) {
    constexpr int BM=256, BN=128, BK=64, WM=4, WN=2, ST=2, GM=8;
    constexpr int smem = (BM*BK + BN*BK) * (int)sizeof(__nv_bfloat16) * ST;
    launch_safe(matmul_v9_kern<BM,BN,BK,WM,WN,ST,GM>,
        cdiv(M,BM)*cdiv(N,BN), WM*WN*WARP_SIZE, smem, A,B,C,M,N,K);
}

// v9c: 128x256x64, 2x4, 2-stage, GROUP_M=8
inline void matmul_v9c_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                               __nv_bfloat16* C, int M, int N, int K) {
    constexpr int BM=128, BN=256, BK=64, WM=2, WN=4, ST=2, GM=8;
    constexpr int smem = (BM*BK + BN*BK) * (int)sizeof(__nv_bfloat16) * ST;
    launch_safe(matmul_v9_kern<BM,BN,BK,WM,WN,ST,GM>,
        cdiv(M,BM)*cdiv(N,BN), WM*WN*WARP_SIZE, smem, A,B,C,M,N,K);
}

// v9d: 128x128x64, 2x2, 3-stage, GROUP_M=4
inline void matmul_v9d_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                               __nv_bfloat16* C, int M, int N, int K) {
    constexpr int BM=128, BN=128, BK=64, WM=2, WN=2, ST=3, GM=4;
    constexpr int smem = (BM*BK + BN*BK) * (int)sizeof(__nv_bfloat16) * ST;
    launch_safe(matmul_v9_kern<BM,BN,BK,WM,WN,ST,GM>,
        cdiv(M,BM)*cdiv(N,BN), WM*WN*WARP_SIZE, smem, A,B,C,M,N,K);
}
