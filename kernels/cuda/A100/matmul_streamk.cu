#pragma once
#include "common.h"

// =============================================================================
// matmul_streamk — Stream-K GEMM for A100
//
// Based on "Stream-K: Work-centric Parallel Decomposition for Dense
// Matrix-Matrix Multiplication on the GPU" (arXiv:2301.03598)
//
// Uses a cached float workspace to avoid repeated allocation.
// Partial results accumulated via atomicAdd, then epilogue converts to bf16.
// =============================================================================

__global__ void streamk_epilogue_kernel(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int grid_m, int grid_n)
{
    constexpr int BLOCK = 128;
    const int tid = threadIdx.x;
    const int tile_idx = blockIdx.x;
    const int bid_m = tile_idx / grid_n;
    const int bid_n = tile_idx % grid_n;
    if (bid_m >= grid_m || bid_n >= grid_n) return;

    const int off_m = bid_m * BLOCK;
    const int off_n = bid_n * BLOCK;
    const float* tile_ws = workspace + (size_t)tile_idx * BLOCK * BLOCK;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int idx = (tid * 32 + i) * 2;
        int row = idx / BLOCK;
        int col = idx % BLOCK;
        float v0 = tile_ws[row * BLOCK + col];
        float v1 = tile_ws[row * BLOCK + col + 1];
        reinterpret_cast<__nv_bfloat162*>(&C[(off_m + row) * N + (off_n + col)])[0] =
            __float22bfloat162_rn({v0, v1});
    }
}

template <
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int NUM_WARP_M, int NUM_WARP_N
>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void matmul_streamk_kern(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ workspace,
    int M, int N, int K,
    int total_work_units,
    int grid_m, int grid_n, int k_slices
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

    constexpr int A_size = BLOCK_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int B_size = BLOCK_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
    constexpr int AB_size = A_size + B_size;

    extern __shared__ __nv_bfloat16 shm[];
    const uint32_t shm_u32 = to_smem(shm);
    const uint32_t A_shm_base = shm_u32;
    const uint32_t B_shm_base = A_shm_base + A_size;

    constexpr int STRIDE_BYTES = BLOCK_K * (int)sizeof(__nv_bfloat16);
    const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
    const uint32_t A_shm_thread = A_shm_base + swizzle_better<STRIDE_BYTES>(A_offm, lane_id / 16);
    const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
    const uint32_t B_shm_thread = B_shm_base + swizzle_better<STRIDE_BYTES>(B_offn, (lane_id % 16) / 8);

    const int work_per_cta = (total_work_units + gridDim.x - 1) / gridDim.x;
    const int work_start = bid * work_per_cta;
    const int work_end = min(work_start + work_per_cta, total_work_units);

    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};
    int prev_tile = -1;

    auto load_AB = [&](int tile_idx, int k_step) {
        const int bm = tile_idx / grid_n;
        const int bn = tile_idx % grid_n;
        const __nv_bfloat16* A_ptr = A + (size_t)bm * BLOCK_M * K + k_step * BLOCK_K;
        const __nv_bfloat16* B_ptr = B + (size_t)bn * BLOCK_N * K + k_step * BLOCK_K;
        g2s_swizzled<CTA_SIZE, BLOCK_M, BLOCK_K>(A_ptr, K, A_shm_base, tid);
        g2s_swizzled<CTA_SIZE, BLOCK_N, BLOCK_K>(B_ptr, K, B_shm_base, tid);
        cp_async_commit_group();
    };

    auto do_compute = [&]() {
        uint32_t A_regs[NUM_MMA_K][NUM_MMA_M][4];
        uint32_t B_regs[NUM_MMA_K][NUM_MMA_N][2];

        #pragma unroll
        for (int k = 0; k < NUM_MMA_K; k++) {
            #pragma unroll
            for (int m = 0; m < NUM_MMA_M; m++) {
                uint32_t A_addr = A_shm_thread;
                A_addr += m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
                LDMATRIX_X4(A_regs[k][m], A_addr ^ (k * 32));
            }
            #pragma unroll
            for (int n = 0; n < NUM_MMA_N; n += 2) {
                uint32_t B_addr = B_shm_thread;
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

    auto flush_acc = [&](int tile_idx) {
        if (tile_idx < 0) return;
        float* ws_ptr = workspace + (size_t)tile_idx * BLOCK_M * BLOCK_N;

        #pragma unroll
        for (int m = 0; m < NUM_MMA_M; m++)
            #pragma unroll
            for (int n = 0; n < NUM_MMA_N; n++) {
                const int row = m * MMA_M + warp_id_m * WARP_M + (lane_id / 4);
                const int col = n * MMA_N + warp_id_n * WARP_N + (lane_id % 4) * 2;
                float *regs = acc[m][n];
                atomicAdd(ws_ptr + (size_t)row * BLOCK_N + col + 0, regs[0]);
                atomicAdd(ws_ptr + (size_t)row * BLOCK_N + col + 1, regs[1]);
                atomicAdd(ws_ptr + (size_t)(row + 8) * BLOCK_N + col + 0, regs[2]);
                atomicAdd(ws_ptr + (size_t)(row + 8) * BLOCK_N + col + 1, regs[3]);
            }
    };

    for (int w = work_start; w < work_end; w++) {
        int tile_idx = w / k_slices;
        int k_step = w % k_slices;

        if (tile_idx != prev_tile) {
            flush_acc(prev_tile);
            #pragma unroll
            for (int m = 0; m < NUM_MMA_M; m++)
                #pragma unroll
                for (int n = 0; n < NUM_MMA_N; n++)
                    #pragma unroll
                    for (int v = 0; v < 4; v++)
                        acc[m][n][v] = 0.0f;
            prev_tile = tile_idx;
        }

        __syncthreads();
        load_AB(tile_idx, k_step);
        cp_async_wait_group<0>();
        __syncthreads();
        do_compute();
    }

    flush_acc(prev_tile);
}

// =============================================================================
// Host launch with cached workspace
// =============================================================================

inline void matmul_streamk_launch(
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    __nv_bfloat16* C, int M, int N, int K)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int NUM_WARP_M = 4;
    constexpr int NUM_WARP_N = 2;

    const int grid_m = cdiv(M, BLOCK_M);
    const int grid_n = cdiv(N, BLOCK_N);
    const int k_slices = cdiv(K, BLOCK_K);
    const int num_tiles = grid_m * grid_n;
    const int total_work = k_slices * num_tiles;

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int num_ctas = min(total_work, num_sms * 2);

    constexpr int smem = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * (int)sizeof(__nv_bfloat16);

    // Cached workspace — avoid repeated cudaMalloc/cudaFree in benchmark loop
    static float* d_workspace = nullptr;
    static size_t ws_capacity = 0;
    size_t ws_bytes = (size_t)num_tiles * BLOCK_M * BLOCK_N * sizeof(float);

    if (ws_bytes > ws_capacity) {
        if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
        CUDA_CHECK(cudaMalloc(&d_workspace, ws_bytes));
        ws_capacity = ws_bytes;
    }
    CUDA_CHECK(cudaMemset(d_workspace, 0, ws_bytes));

    launch_safe(
        matmul_streamk_kern<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N>,
        num_ctas, NUM_WARP_M * NUM_WARP_N * WARP_SIZE, smem,
        A, B, d_workspace, M, N, K,
        total_work, grid_m, grid_n, k_slices);

    streamk_epilogue_kernel<<<num_tiles, 256>>>(
        d_workspace, C, M, N, grid_m, grid_n);
    CUDA_CHECK(cudaGetLastError());
}
