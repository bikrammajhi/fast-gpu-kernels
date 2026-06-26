#pragma once
#include "common.h"

// =============================================================================
// matmul_v14 — v13d + epilogue via shared memory (coalesced int4 stores)
// Same main loop as v13d, but the C store goes:
//   registers -> shared memory -> coalesced int4 global stores
// =============================================================================

template <int CTA_SIZE, int HEIGHT, int WIDTH>
__device__ static void g2s_v14(
    const __nv_bfloat16 *in, int in_stride, uint32_t out, int tid)
{
  constexpr int num_elems = 16 / sizeof(__nv_bfloat16);
  constexpr int num_iters = (HEIGHT * WIDTH) / (CTA_SIZE * num_elems);
  #pragma unroll
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * CTA_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    uint32_t dst_addr = out + swizzle_better<WIDTH * sizeof(__nv_bfloat16)>(row, col / num_elems);
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int NUM_WARP_M, int NUM_WARP_N,
    int NUM_STAGES, int GROUP_M = 8
>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void matmul_v14_kern(
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
    swizzle_triton_v13(bid, grid_m, grid_n, bid_m, bid_n, GROUP_M);

    A += bid_m * BLOCK_M * K;
    B += bid_n * BLOCK_N * K;

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
        g2s_v14<CTA_SIZE, BLOCK_M, BLOCK_K>(A_ptr, K, A_shm_base + stage_id * AB_size, tid);
        g2s_v14<CTA_SIZE, BLOCK_N, BLOCK_K>(B_ptr, K, B_shm_base + stage_id * AB_size, tid);
        A_ptr += BLOCK_K;
        B_ptr += BLOCK_K;
        cp_async_commit_group();
    };

    auto compute_db = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;
        const uint32_t stage_offset = stage_id * AB_size;
        uint32_t A_buf[2][NUM_MMA_M][4];
        uint32_t B_buf[2][NUM_MMA_N][2];

        #pragma unroll
        for (int m = 0; m < NUM_MMA_M; m++) {
            uint32_t A_addr = A_shm_thread + stage_offset + m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
            LDMATRIX_X4(A_buf[0][m], A_addr);
        }
        #pragma unroll
        for (int n = 0; n < NUM_MMA_N; n += 2) {
            uint32_t B_addr = B_shm_thread + stage_offset + n * MMA_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
            LDMATRIX_X4(B_buf[0][n], B_addr);
        }

        #pragma unroll
        for (int k = 0; k < NUM_MMA_K; k++) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;

            if (k + 1 < NUM_MMA_K) {
                #pragma unroll
                for (int m = 0; m < NUM_MMA_M; m++) {
                    uint32_t A_addr = A_shm_thread + stage_offset
                        + m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
                    LDMATRIX_X4(A_buf[nxt][m], A_addr ^ ((k+1) * 32));
                }
                #pragma unroll
                for (int n = 0; n < NUM_MMA_N; n += 2) {
                    uint32_t B_addr = B_shm_thread + stage_offset
                        + n * MMA_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
                    LDMATRIX_X4(B_buf[nxt][n], B_addr ^ ((k+1) * 32));
                }
            }

            #pragma unroll
            for (int m = 0; m < NUM_MMA_M; m++)
                #pragma unroll
                for (int n = 0; n < NUM_MMA_N; n++)
                    MMA_M16N8K16(A_buf[cur][m], B_buf[cur][n], acc[m][n]);
        }
    };

    // Prologue
    for (int stage = 0; stage < NUM_STAGES - 1 && stage < num_k_iters; stage++)
        load_AB(stage);

    // Main loop
    constexpr int PD = NUM_STAGES - 1;
    for (int k_iter = 0; k_iter < num_k_iters - PD; k_iter++) {
        __syncthreads();
        load_AB(k_iter + PD);
        cp_async_wait_group<PD>();
        __syncthreads();
        compute_db(k_iter);
    }

    // Epilogue (drain pipeline)
    for (int k_iter = num_k_iters - PD; k_iter < num_k_iters; k_iter++) {
        __syncthreads();
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();
        compute_db(k_iter);
    }

    // ============================================================
    // STORE EPILOGUE: registers -> shared memory -> coalesced global
    // ============================================================
    // Reuse AB shared memory for C output staging.
    // Layout: shm[128][128] where stride = BLOCK_N = 128
    // Each warp writes its sub-tile, then we stream out via int4.
    constexpr int C_STRIDE = BLOCK_N;  // 128

    // Phase 1: Write accumulator to shared memory
    #pragma unroll
    for (int m = 0; m < NUM_MMA_M; m++) {
        #pragma unroll
        for (int n = 0; n < NUM_MMA_N; n++) {
            const int row = m * MMA_M + warp_id_m * WARP_M + (lane_id / 4);
            const int col = n * MMA_N + warp_id_n * WARP_N + (lane_id % 4) * 2;

            // Convert float acc to bf16 and store to shared memory
            __nv_bfloat162 v0 = __float22bfloat162_rn({acc[m][n][0], acc[m][n][1]});
            __nv_bfloat162 v1 = __float22bfloat162_rn({acc[m][n][2], acc[m][n][3]});

            reinterpret_cast<__nv_bfloat162*>(&shm[row * C_STRIDE + col])[0] = v0;
            reinterpret_cast<__nv_bfloat162*>(&shm[(row + 8) * C_STRIDE + col])[0] = v1;
        }
    }

    __syncthreads();

    // Phase 2: Stream out via coalesced 128-bit (int4) stores
    // Per warp: WARP_M=32 rows, WARP_N=64 cols -> 32*64=2048 bf16 elements = 256 int4s
    // 32 threads per warp -> 8 int4s per thread
    constexpr int INT4_PER_ROW = WARP_N / 8;  // 8
    constexpr int INT4S_PER_THREAD = (WARP_M * INT4_PER_ROW) / WARP_SIZE;  // 32*8/32 = 8

    const int gmem_row_base = bid_m * BLOCK_M + warp_id_m * WARP_M;
    const int gmem_col_base = bid_n * BLOCK_N + warp_id_n * WARP_N;

    #pragma unroll
    for (int i = 0; i < INT4S_PER_THREAD; i++) {
        const int flat = lane_id * INT4S_PER_THREAD + i;
        const int r = flat / INT4_PER_ROW;
        const int c_int4 = flat % INT4_PER_ROW;

        if (r < WARP_M) {
            const int smem_row = warp_id_m * WARP_M + r;
            const int smem_col = warp_id_n * WARP_N + c_int4 * 8;
            const int gmem_row = gmem_row_base + r;
            const int gmem_col = gmem_col_base + c_int4 * 8;

            if (gmem_row < M && gmem_col + 7 < N) {
                reinterpret_cast<int4*>(&C[gmem_row * N + gmem_col])[0] =
                    reinterpret_cast<const int4*>(&shm[smem_row * C_STRIDE + smem_col])[0];
            }
        }
    }
}

// =============================================================================
// Launch
// =============================================================================
inline void matmul_v14_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                               __nv_bfloat16* C, int M, int N, int K) {
    constexpr int BM=128, BN=128, BK=64, WM=4, WN=2, ST=2, GM=16;
    // AB pipeline smem + C staging smem. C staging needs 128*128*2=32KB.
    // AB pipeline needs (128*64 + 128*64)*2*2 = 65536 = 64KB for 2-stage.
    constexpr int ab_smem = (BM*BK + BN*BK) * (int)sizeof(__nv_bfloat16) * ST;
    constexpr int c_smem = BM * BN * (int)sizeof(__nv_bfloat16);
    constexpr int smem = (ab_smem > c_smem) ? ab_smem : c_smem;
    launch_safe(matmul_v14_kern<BM,BN,BK,WM,WN,ST,GM>,
        cdiv(M,BM)*cdiv(N,BN), WM*WN*WARP_SIZE, smem, A,B,C,M,N,K);
}
