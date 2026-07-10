#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,        // (M, K, P)
          class SmemLayoutB>        // (N, K, P)
struct SharedStorage
{
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
    
    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
    
};

template<class ProblemShape, class CtaTiler,
         class TA, class SmemLayoutA, class TmaA,
         class TB, class SmemLayoutB, class TmaB,
         class TC, class CStride, class TiledMMA,
         class Alpha, class Beta>

__global__ static
__launch_bounds__(decltype(size(TiledMMA{}))::value)
void gmem_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                 TC      * C, CStride dC, TiledMMA mma,
                 Alpha alpha, Beta beta)
{
    // Represent full tensors
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                      // (M, K) TMA Tensor
    Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                      // (N, K) TMA Tensor
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);  // (M, N) 

    // Get the appropriate tile for this CTA
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});    // (BLK_M, BLK_K, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});    // (BLK_N, BLK_K, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});    // (BLK_M, BLK_N, k)

    // Shared memory tensors
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});      // (BLK_M, BLK_K, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});      // (BLK_N, BLK_K, PIPE)

    // Partition the copying of A and B tiles
    auto [tAgA,tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},         // Indicates not Mulitcast
                                     group_modes<0,2>(sA), group_modes<0,2>(gA));   // (TMA, k) and (TMA, PIPE)
    auto [tBgB,tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                     group_modes<0,2>(sB), group_modes<0,2>(gB));   // (TMA, k) and (TMA, PIPE)
    
    // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

    // Define A/B partitioning and C accumulators
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);              // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);              // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);              // (MMA, MMA_M, MMA_N)

    // Allocate register for pipelining
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);       // MMA Descriptors constructed as view of SMEM
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);       // MMA Descriptors constructed as view of SMEM
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    // Clear the accumulators
    clear(tCrC);


    ///////////////////////      PREFETCH    //////////////////////////
    
    // Total number of k_tiles
    auto k_tile_count = size<1>(tAgA);
    // Number of pipeline stages
    auto K_PIPE_MAX = size<1>(tAsA);
    // Current tile index in gmem to read from 
    int k_tile = 0;
    
    // Initialize Barriers
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;
    
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;   // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier;              // MMA
    
    CUTE_UNROLL
    for (int pipe=0; pipe<K_PIPE_MAX; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }
    // Ensure barrie init is complete on all CTAs
    cluster_sync();
    
    // Start async loads for all pipes
    CUTE_UNROLL
    for (int pipe=0; pipe<K_PIPE_MAX; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            // Set expected Tx Bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe)); 
        }
        --k_tile_count;
        ++k_tile;
    }

    ///////////////////////      MAINLOOP    //////////////////////////
    
    // Rather than interleaving the stages and instruction like SM70, SM80, 
    // the SM90 mainloops rely on explicit producer-consumer synchronization
    // on purely asyc instruction TMA and MMA. 
    
    // PipelineState is a circular pipe index [.index()] and a pipe phase[.phase()]
    // that flips each cycle through K_PIPE_MAX.
    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();    // TMA writes
    auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();    // MMA reads

    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
        // Wait for Producer to complete
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
        
        // MMAs to cover 1 K_TILE
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);  // (V,M) X (V,N) --> (V,M,N)
        warpgroup_commit_batch();
        
        // Wait for all MMAs in a K_TILE to complete
        warpgroup_wait<0>();
        
        // Notify that consumption is done
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;
        
        // Only issue new TMA copies if there are more tiles to fetch
        if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0))
        {
            int pipe = write_state.index();
            // Wait for consumer to complete consumption
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            // Set the expected Tx Bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }

    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);

}

// Setup params for GEMM
template <class TA, class TB, class TC, 
          class Alpha, class Beta>
void gemm(int m, int n, int k,
          Alpha alpha,
          TA const* A, int ldA,
          TB const* B, int ldB,
          Beta beta,
          TC* C, int ldC,
          cudaStream_t stream = 0)
{
    // Define shapes (dyanamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    // Define strides
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    // Define CTA Tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    // Define the smem layouts (static)
    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x128x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});
    
    // Define the TMAs
    // Create Global memory tensors for TMA inspection
    Tensor mA = make_tensor(A, make_shape(M,K), dA);
    Tensor mB = make_tensor(B, make_shape(N,K), dB);
    
    // Create TMA Atoms with the desired copy operations on the source and destination
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

    // Launch params
    dim3 dimBlock(size(tiled_mma));
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
                 round_up(size(ceil_div(n, bN)), dimCluster.y));
    int smemBytes = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));

    auto* kernel_ptr = &gmem_device<decltype(prob_shape), decltype(cta_tiler),
                                 TA, decltype(sA), decltype(tmaA),
                                 TB, decltype(sB), decltype(tmaB),
                                 TC, decltype(dC), decltype(tiled_mma),
                                 decltype(alpha), decltype(beta)>;

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smemBytes));

    // Kernel Launch
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                               prob_shape, cta_tiler,
                                                               A, tmaA,
                                                               B, tmaB,
                                                               C, dC, tiled_mma,
                                                               alpha, beta);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Kernel launch failed with error: " << cutlassGetStatusString(status) << std::endl;
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

    using TA = cute::half_t; using TC = cute::half_t;
    using TB = cute::half_t; using TI = cute::half_t;

    TI alpha = TI(1.0f); TI beta = TI(0.0f);

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    // Initialize the tensors
    for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n*k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    int ldA = m;
    int ldB = n;
    int ldC = m;

    double gflops = 2.0 * m * n * k * 1e-9;
    const int timing_iters = 100;
    GPU_Clock timer;

    // Warmup
    d_C = h_C;
    gemm(m, n, k, alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    // Timing
    timer.start();
    for (int i = 0; i < timing_iters; ++i) {
        gemm(m, n, k, alpha,
             d_A.data().get(), ldA,
             d_B.data().get(), ldB,
             beta,
             d_C.data().get(), ldC);
    }
    double cute_time = timer.seconds() / timing_iters;
    CUTE_CHECK_LAST();
    printf("CUTE: M=%d N=%d K=%d | %8.4f ms | %8.1f TFLOPs\n", m, n, k, cute_time * 1e3, gflops / cute_time / 1000.0);

#else
    std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled, but it's not. Test skipped." << std::endl;
#endif

    return 0;
}

#endif
