#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "cutlass/util/GPU_Clock.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

using namespace cute;
using bf16 = cute::bfloat16_t;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma)
{

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);          // (BLK_N,BLK_K,PIPE)

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);                                   // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);                                   // (CPY,CPY_N,CPY_K,PIPE)

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                                        // (MMA,MMA_M,MMA_N)

    // Allocate registers (fragments) for A, B and C
    // sA/sB now have a PIPE dimension so slice with (_,_,0) to get (BLK_M,BLK_K)
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                        // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                        // (MMA,MMA_N,MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                                  // (MMA,MMA_M,MMA_N)

    // Clear the accumulators
    clear(tCrC);

    // Create smem->rmem copy atoms (LDSM) that feed the MMA
    // tXsA/tXsB have a PIPE dimension from sA/sB
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                                // (CPY,MMA_M,MMA_K,PIPE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                                 // (CPY,MMA_M,MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                                // (CPY,MMA_N,MMA_K,PIPE)   
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                                 // (CPY,MMA_N,MMA_K)

   //
   // PREFETCH: Prime the smem pipeline
   //

    auto K_PIPE_MAX = size<3>(tAsA);                                              // number of smem pipe buffers

    // Total count of tiles along K
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    // This fills K_PIPE_MAX-1 smem buffers so the main loop can overlap compute with copy
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    // Current pipe index in smem to read from (for LDSM + gemm)
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to (for cp.async)
    int smem_pipe_write = K_PIPE_MAX-1;

    //
    // PIPELINED MAIN LOOP
    // Loop runs while there are tiles to COMPUTE (including those already prefetched).
    // k_tile_count starts at total - (K_PIPE_MAX-1) after the prologue.
    // The condition extends past zero to drain the prefetched tiles from the pipe.
    //

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX-1))
    {
        // Wait for the smem_pipe_read tile to finish loading
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();

        // LDSM: smem -> registers for the MMA
        copy(s2r_atom_a, tXsA(_,_,_,smem_pipe_read), tXrA);
        copy(s2r_atom_b, tXsB(_,_,_,smem_pipe_read), tXrB);

        // Tensor core MMA on the register fragments
        gemm(mma, tCrA, tCrB, tCrC);

        // Kick off async copy for the next tile into the smem_pipe_write buffer
        // Only if there are tiles left to fetch
        if (k_tile_count > 0) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
            cp_async_fence();
        }

        // Advance the pipe indices (ping-pong)
        smem_pipe_read  = (smem_pipe_read  + 1) % K_PIPE_MAX;
        smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_MAX;

        // Advance the gmem tile index
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    // Epilogue: write accumulators back to global memory
    copy(tCrC, tCgC);
}

template <class TA, class TB, class TC>
void gemm(int m, int n, int k,
          TA const *A, int ldA,
          TB const *B, int ldB,
          TC *C, int ldC,
          cudaStream_t stream = 0)
{

    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(ldC, Int<1>{});

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<2>{};  // Pipeline (double-buffer)

    auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});

    auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
    auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
    auto sC = make_layout(make_shape(bM, bN));

    TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
                                        Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                        Layout<Shape< _1,_8>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{},
                                        Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                        Layout<Shape< _1,_8>>{});

    TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                    Layout<Shape<_2,_2>>{},
                                    Tile<_32,_32,_16>{});
    
    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_A;
    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_B;

    int smem_size = int(sizeof(SharedStorage<bf16, bf16, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                size(ceil_div(N, bN)));

    auto kernel_fptr = gemm_device<
        decltype(prob_shape), decltype(cta_tiler),
        bf16, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
        bf16, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
        bf16, decltype(dC), decltype(sC), decltype(mmaC)>;

    // Increase dynamic shared memory limit for pipelined buffers (96KB vs default 48KB)
    cudaError_t err = cudaFuncSetAttribute(kernel_fptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
        printf("cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(err));
    }

    kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
        (prob_shape, cta_tiler,
        A, dA, sA, copyA, s2r_atom_A,
        B, dB, sB, copyB, s2r_atom_B,
        C, dC, sC, mmaC);
}


#ifndef BENCHMARK_SUITE
int main(int argc, char** argv)
{
  int m = 5120, n = 5120, k = 4096;

  thrust::host_vector<bf16> h_A(m * k);
  thrust::host_vector<bf16> h_B(n * k);
  thrust::host_vector<bf16> h_C(m * n);
  for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<bf16>(2.0f * (rand() / double(RAND_MAX)) - 1.0f);
  for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<bf16>(2.0f * (rand() / double(RAND_MAX)) - 1.0f);
  for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<bf16>(-1.0f);

  thrust::device_vector<bf16> d_A = h_A;
  thrust::device_vector<bf16> d_B = h_B;
  thrust::device_vector<bf16> d_C = h_C;

  GPU_Clock timer;
  int iters = 100;

  gemm(m, n, k, d_A.data().get(), k, d_B.data().get(), k, d_C.data().get(), n);
  CUTE_CHECK_LAST();

  timer.start();
  for (int i = 0; i < iters; ++i)
    gemm(m, n, k, d_A.data().get(), k, d_B.data().get(), k, d_C.data().get(), n);
  double t = timer.seconds() / iters;
  CUTE_CHECK_LAST();

  double tflops = (2.0 * m * n * k) * 1e-9 / t / 1000.0;
  printf("CUTE_GEMM: [%6.1f] TFlop/s  (%6.4f ms)\n", tflops, t * 1000);

  // --- Correctness check against cuBLAS ---
  {
    // Copy our result to host
    thrust::host_vector<bf16> h_cute = d_C;

    // Run cuBLAS reference
    thrust::device_vector<bf16> d_ref(m * n);
    thrust::fill(d_ref.begin(), d_ref.end(), bf16(-1.0f));

    cublasHandle_t handle;
    cublasCreate(&handle);
    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    __nv_bfloat16 beta  = __float2bfloat16(0.0f);
    auto* raw_A = reinterpret_cast<const __nv_bfloat16*>(d_A.data().get());
    auto* raw_B = reinterpret_cast<const __nv_bfloat16*>(d_B.data().get());
    auto* raw_C = reinterpret_cast<__nv_bfloat16*>(d_ref.data().get());

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, raw_B, CUDA_R_16BF, n,
                raw_A, CUDA_R_16BF, k,
        &beta,  raw_C, CUDA_R_16BF, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<bf16> h_ref = d_ref;

    // Compare
    double max_rel_err = 0.0;
    int bad_count = 0;
    for (size_t i = 0; i < h_cute.size(); ++i) {
      float f_cute = static_cast<float>(h_cute[i]);
      float f_ref  = static_cast<float>(h_ref[i]);
      float denom = fmaxf(1.0f, fabsf(f_ref));
      float rel_err = fabsf(f_cute - f_ref) / denom;
      if (rel_err > max_rel_err) max_rel_err = rel_err;
      if (rel_err > 0.05f) ++bad_count;
    }

    printf("CORRECTNESS: max_rel_err=%.2e  entries_with_>5%%_error=%d/%zu\n",
           max_rel_err, bad_count, h_cute.size());

    if (max_rel_err > 0.1f) {
      printf("  >>> FAIL: relative error too large\n");
    } else {
      printf("  >>> PASS\n");
    }

    cublasDestroy(handle);
  }

  return 0;
}
#endif
