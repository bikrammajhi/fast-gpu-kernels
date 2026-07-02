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

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Grid swizzling for better L2 locality
    int m_idx = blockIdx.x;
    int n_idx = blockIdx.y;
    constexpr int swizzle_factor = 8;
    if (gridDim.x >= swizzle_factor) {
        int tid = blockIdx.x + blockIdx.y * gridDim.x;
        int idx_outer = tid / swizzle_factor;
        int idx_inner = tid % swizzle_factor;
        int m_grid = gridDim.x;
        int n_swizzled = idx_outer / ((m_grid + swizzle_factor - 1) / swizzle_factor);
        int m_swizzled = (idx_outer % ((m_grid + swizzle_factor - 1) / swizzle_factor)) * swizzle_factor + idx_inner;
        if (m_swizzled < m_grid) {
            m_idx = m_swizzled;
            n_idx = n_swizzled;
        }
    }

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(m_idx, n_idx, _);                        // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k_tiles)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k_tiles)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory (layout includes stages dimension)
    extern __shared__ char shared_memory[];
    bf16* smem_base = reinterpret_cast<bf16*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem_base), sA_layout);                    // (BLK_M,BLK_K,2)
    Tensor sB = make_tensor(make_smem_ptr(smem_base + cosize_v<ASmemLayout>), sB_layout); // (BLK_N,BLK_K,2)

    // Partition the copying of A and B tiles across the threads
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k_tiles)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,2)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k_tiles)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,2)

    //
    // PREFETCH: Prime the smem pipeline (before MMA/LDSM setup)
    //
    auto K_PIPE_MAX = size<3>(tAsA);

    int k_tile_count = size<3>(tAgA);
    int k_tile_next = 0;

    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) ++k_tile_next;
    }

    //
    // Define A/B partitioning and C accumulators
    //
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

    clear(tCrC);

    //
    // Copy Atom retiling (LDSM setup)
    //
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

    //
    // Register-level prefetch (preload first K-block from smem)
    //
    int smem_pipe_read  = 0;
    int smem_pipe_write = K_PIPE_MAX-1;

    Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
    Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

    auto K_BLOCK_MAX = size<2>(tCrA);

    if (K_BLOCK_MAX > 1) {
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();

        copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
        copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
    }

    //
    // PIPELINED MAIN LOOP
    //
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX-1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                tXsA_p = tXsA(_,_,_,smem_pipe_read);
                tXsB_p = tXsB(_,_,_,smem_pipe_read);

                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
            copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

            if (k_block == 0)
            {
                copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                cp_async_fence();

                --k_tile_count;
                if (k_tile_count > 0) ++k_tile_next;

                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
            }

            gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
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

    // Define problem shape (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                   // (M, N, K)

    // Define strides (column-major: ld is stride for first mode)
    auto dA = make_stride(ldA, Int<1>{});                    // (dM, dK)  -- TN layout (A non-transposed)
    auto dB = make_stride(ldB, Int<1>{});                    // (dN, dK)
    auto dC = make_stride(ldC, Int<1>{});                    // (dC0, dC1)

    // Define CTA tile sizes (static)
    auto bM = Int<256>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);                 // (BLK_M, BLK_N, BLK_K)

    // Swizzled shared memory layout (128B swizzle, with 3-stage pipe)
    auto bP = Int<3>{};  // Pipeline
    auto swizzled_128B_atom = composition(
                    Swizzle<3,3,3>{},
                    make_layout(
                        make_shape(Int<8>{}, make_shape(Int<8>{}, Int<8>{})),
                        make_stride(Int<8>{}, make_stride(Int<1>{}, Int<64>{})))
                    );
    auto sA = tile_to_shape(swizzled_128B_atom, make_shape(bM, bK, Int<bP>{}));
    auto sB = tile_to_shape(swizzled_128B_atom, make_shape(bN, bK, Int<bP>{}));
    auto sC = make_layout(make_shape(bM, bN));

    // Gmem -> smem copy via cp.async (128-bit async copy, bypasses L1)
    using GmemCopyAtom = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
    TiledCopy copyA = make_tiled_copy(Copy_Atom<GmemCopyAtom, bf16>{},
                                        Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout: 16x8 = 128 threads
                                        Layout<Shape< _1,_8>>{});               // Val layout: 1x8 k-major (8 vals/thr)
    TiledCopy copyB = make_tiled_copy(Copy_Atom<GmemCopyAtom, bf16>{},
                                        Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout: 16x8 = 128 threads
                                        Layout<Shape< _1,_8>>{});               // Val layout: 1x8 k-major

    // MMA tiling: SM80_16x8x16 (16×8×16 atom) × 2×2 atoms → 32×16 per gemm(), Tile<32,32,16> value perm
    TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                    Layout<Shape<_2,_2>>{},    // 2×2 atoms over (M,N) = 4 warps
                                    Tile<_32,_32,_16>{});      // Value permutation (not output size)

    // LDSM atoms for smem -> registers (U32x4 = 128b LDSM, 4 registers per thread)
    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_A;
    Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_B;

    // Smem size: both A and B buffers, each with STAGES
    int smem_elems_A = int(cosize(sA));
    int smem_elems_B = int(cosize(sB));
    int smem_size = (smem_elems_A + smem_elems_B) * sizeof(bf16);

    dim3 dimBlock(size(mmaC));                                // 128 threads (4 warps)
    dim3 dimGrid(size(ceil_div(M, bM)),                       // M-tiles
                size(ceil_div(N, bN)));                       // N-tiles

    auto kernel_fptr = gemm_device<
        decltype(prob_shape), decltype(cta_tiler),
        bf16, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
        bf16, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
        bf16, decltype(dC), decltype(sC), decltype(mmaC)>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
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
    thrust::host_vector<bf16> h_cute = d_C;
    cublasHandle_t handle;
    cublasCreate(&handle);
    auto* raw_A = reinterpret_cast<const __nv_bfloat16*>(d_A.data().get());
    auto* raw_B = reinterpret_cast<const __nv_bfloat16*>(d_B.data().get());
    thrust::device_vector<bf16> d_ref(m * n);
    auto* raw_C = reinterpret_cast<__nv_bfloat16*>(d_ref.data().get());

    float alpha = 1.0f, beta = 0.0f;

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
        &alpha, raw_A, CUDA_R_16BF, k,
                raw_B, CUDA_R_16BF, k,
        &beta,  raw_C, CUDA_R_16BF, m,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<bf16> h_ref = d_ref;

    double max_rel_err = 0.0;
    int bad = 0;
    for (int i = 0; i < m && bad < 100; ++i) {
      for (int j = 0; j < n && bad < 100; ++j) {
        float v_our = static_cast<float>(h_cute[i * n + j]);
        float v_ref = static_cast<float>(h_ref[i + j * m]);
        float denom = fmaxf(1.0f, fabsf(v_ref));
        float rel = fabsf(v_our - v_ref) / denom;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 0.05f) ++bad;
      }
    }
    printf("CORRECTNESS vs cuBLAS: max_rel_err=%.2e  >5%%_err=%d/%d  %s\n",
           max_rel_err, bad, m * n, bad < m * n / 2 ? "PASS" : "FAIL");

    cublasDestroy(handle);
  }

  return 0;
}
#endif
