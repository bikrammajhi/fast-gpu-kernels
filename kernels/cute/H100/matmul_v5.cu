/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Hopper WGMMA GEMM with TMA warp-specialization (CuTe C++ port of CuTe DSL c2_wgmma_tma_specialized_pipeline.py)
 *
 * WARP SPECIALIZATION:
 *   - Warps 0-3 (threads 0-127): MMA consumer (128 threads = 1 warp group)
 *   - Warp 4 (threads 128-159): TMA producer (32 threads, but only 1 does TMA)
 *
 * Tile: 128x128 with 4-stage pipeline
 *************************************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template<class ProblemShape, class CtaTiler,
         class TA, class SmemLayoutA, class TmaA,
         class TB, class SmemLayoutB, class TmaB,
         class TC, class CStride, class TiledMMA>

__global__ static
__launch_bounds__(160)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                 TC      * C, CStride dC, TiledMMA mma)
{
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
    Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                       group_modes<0,2>(sA), group_modes<0,2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                       group_modes<0,2>(sB), group_modes<0,2>(gB));

    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                            + sizeof(make_tensor_like(tensor<0>(tBsB)));

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    auto K_PIPE_MAX = size<1>(tAsA);
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if (lane_predicate) {
            ProducerBarType::init(&producer_mbar[pipe],   1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }
    __syncthreads();

    int pipeline_state_phase = 0;

// ========================
    // TMA PRODUCER (warp 4)
    // ========================
    if (warp_idx == 4) {
        auto write_state = cutlass::PipelineState<K_PIPE_MAX>(); // Start at phase 0
        int k_tile_count = size<1>(tAgA);
        int k_tile = 0;

        // Prefetch: fill all pipeline stages
        // TMA arrives at producer barriers (full) at phase 0
        // We don't wait on consumer barriers during prefetch - just signal ready for TMA
        CUTE_UNROLL
        for (int pipe = 0; pipe < min(K_PIPE_MAX, k_tile_count); ++pipe) {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
            ++write_state;
            --k_tile_count;
            ++k_tile;
        }

        // Advance to phase 1 - consumer will arrive on phase 0 barriers
        // After consumer arrives on phase 0, barrier completes and flips to phase 1
        write_state += K_PIPE_MAX; // Now at (index 0, phase 1)

        // Mainloop: wait on consumer barrier, then issue TMA
        while (k_tile_count > -K_PIPE_MAX) {
            int pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
            ++write_state;
            --k_tile_count;
            ++k_tile;
        }
    }
    // ========================
    // MMA CONSUMER (warps 0-3)
    // ========================
    else {
        ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_B(sB);
        Tensor tCgC = thr_mma.partition_C(gC);

        Tensor tCrC = thr_mma.make_fragment_C(tCgC);
        clear(tCrC);

        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        auto read_state = cutlass::PipelineState<K_PIPE_MAX>(); // Start at phase 0
        int k_tile_count = size<1>(tAgA);

        // Mainloop: consume prefetched data then continue
        CUTE_NO_UNROLL
        while (k_tile_count > -K_PIPE_MAX) {
            int read_pipe = read_state.index();
            // Wait for TMA to complete (prefetch data at phase 0)
            ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

            warpgroup_arrive();
            gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<0>();

            ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
            ++read_state;
            --k_tile_count;
        }

        axpby(half_t(1.0f), tCrC, half_t(0.0f), tCgC);
    }
}

template <class TA, class TB, class TC>
void gemm(int m, int n, int k,
          TA const* A, int ldA,
          TB const* B, int ldB,
          TC* C, int ldC,
          cudaStream_t stream = 0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto bP = Int<4>{};

    auto cta_tiler = make_shape(bM, bN, bK);
    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

    TiledMMA tiled_mma = make_tiled_mma(
        SM90_64x128x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{}
    );

    Tensor mA = make_tensor(A, make_shape(M,K), dA);
    Tensor mB = make_tensor(B, make_shape(N,K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

    int smemBytes = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    dim3 dimBlock(160);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(M, bM)), dimCluster.x),
                 round_up(size(ceil_div(N, bN)), dimCluster.y));

    auto* kernel_ptr = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                    TA, decltype(sA), decltype(tmaA),
                                    TB, decltype(sB), decltype(tmaB),
                                    TC, decltype(dC), decltype(tiled_mma)>;

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smemBytes));

    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                               prob_shape, cta_tiler,
                                                               A, tmaA,
                                                               B, tmaB,
                                                               C, dC, tiled_mma);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Kernel launch failed: " << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#ifndef BENCHMARK_SUITE
int main(int argc, char** argv)
{
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);
    if (props.major != 9) {
        std::cout << "This example is only supported on H100 GPUs (compute capability 9.0)" << std::endl;
        return 0;
    }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    int m = 5120;
    if (argc >= 2) sscanf(argv[1], "%d", &m);
    int n = 5120;
    if (argc >= 3) sscanf(argv[2], "%d", &n);
    int k = 4096;
    if (argc >= 4) sscanf(argv[3], "%d", &k);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    for (int j = 0; j < m * k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n * k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < m * n; ++j) h_C[j] = TC(0);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    double gflops = 2.0 * m * n * k * 1e-9;
    const int timing_iters = 100;
    GPU_Clock timer;

    gemm(m, n, k, d_A.data().get(), m, d_B.data().get(), n, d_C.data().get(), m);
    CUTE_CHECK_LAST();

    timer.start();
    for (int i = 0; i < timing_iters; ++i) {
        gemm(m, n, k, d_A.data().get(), m, d_B.data().get(), n, d_C.data().get(), m);
    }
    double cute_time = timer.seconds() / timing_iters;
    CUTE_CHECK_LAST();
    printf("CUTE_WARP_SPEC: M=%d N=%d K=%d | %8.4f ms | %8.1f TFLOPs\n", m, n, k, cute_time * 1e3, gflops / cute_time / 1000.0);

#else
    std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled, but it's not. Test skipped." << std::endl;
#endif

    return 0;
}
#endif