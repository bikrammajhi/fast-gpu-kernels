#pragma once
#include "common.h"

// =============================================================================
// matmul_v7 — High-performance kernel matching reference v6
//   - swizzle_better for both cp.async writes and ldmatrix reads
//   - Pre-computed per-thread base addresses
//   - XOR for k-step offset in ldmatrix
//   - ldmatrix_x4 for B (loads 2 B matrices at once)
//   - Multi-stage pipeline with cp_async_commit_group / cp_async_wait_group
//   - Triton-style threadblock swizzle
// =============================================================================

template <
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int NUM_WARP_M,
    int NUM_WARP_N,
    int NUM_STAGES
>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void matmul_v7_kern(
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

    uint32_t A_regs[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_regs[NUM_MMA_K][NUM_MMA_N][2];
    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

    // Pre-compute per-thread base addresses with swizzle
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

    auto compute = [&](int k_iter) {
        // A: shared -> regs
        for (int k = 0; k < NUM_MMA_K; k++)
            for (int m = 0; m < NUM_MMA_M; m++) {
                uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                A_addr += m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
                LDMATRIX_X4(A_regs[k][m], A_addr ^ (k * 32));
            }

        // B: shared -> regs (ldmatrix_x4 loads 2 B matrices at once)
        for (int k = 0; k < NUM_MMA_K; k++)
            for (int n = 0; n < NUM_MMA_N; n += 2) {
                uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                B_addr += n * MMA_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
                LDMATRIX_X4(B_regs[k][n], B_addr ^ (k * 32));
            }

        // MMA
        for (int k = 0; k < NUM_MMA_K; k++)
            for (int m = 0; m < NUM_MMA_M; m++)
                for (int n = 0; n < NUM_MMA_N; n++)
                    MMA_M16N8K16(A_regs[k][m], B_regs[k][n], acc[m][n]);
    };

    // Prologue: initiate NUM_STAGES-1 prefetch stages
    for (int stage = 0; stage < NUM_STAGES - 1 && stage < num_k_iters; stage++)
        load_AB(stage);

    // Main loop: maintain NUM_STAGES-1 in-flight prefetches
    for (int k_iter = 0; k_iter < num_k_iters - (NUM_STAGES - 1); k_iter++) {
        __syncthreads();
        load_AB(k_iter + NUM_STAGES - 1);
        cp_async_wait_group<NUM_STAGES - 1>();
        __syncthreads();
        compute(k_iter);
    }

    // Epilogue: drain remaining stages
    for (int k_iter = num_k_iters - (NUM_STAGES - 1); k_iter < num_k_iters; k_iter++) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();
        compute(k_iter);
    }

    epilogue_store_bf16<NUM_MMA_M, NUM_MMA_N>(acc, C, N, lane_id, warp_id_m, WARP_M, warp_id_n, WARP_N);
}

inline void matmul_v7_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              __nv_bfloat16* C, int M, int N, int K)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int NUM_WARP_M = 2;
    constexpr int NUM_WARP_N = 2;
    constexpr int NUM_STAGES = 2;

    constexpr int A_size = BLOCK_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int B_size = BLOCK_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int smem_total = (A_size + B_size) * NUM_STAGES;

    launch_safe(
        matmul_v7_kern<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>,
        cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),
        NUM_WARP_M * NUM_WARP_N * WARP_SIZE,
        smem_total,
        A, B, C, M, N, K);
}

inline void matmul_v7s3_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                               __nv_bfloat16* C, int M, int N, int K)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int NUM_WARP_M = 2;
    constexpr int NUM_WARP_N = 2;
    constexpr int NUM_STAGES = 3;

    constexpr int A_size = BLOCK_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int B_size = BLOCK_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int smem_total = (A_size + B_size) * NUM_STAGES;

    launch_safe(
        matmul_v7_kern<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>,
        cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),
        NUM_WARP_M * NUM_WARP_N * WARP_SIZE,
        smem_total,
        A, B, C, M, N, K);
}
