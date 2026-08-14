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

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k_tiles)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k_tiles)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory buffers
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K)

    // Partition the copying of A and B tiles across the threads
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k_tiles)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k_tiles)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)

    // Partition the C tile across MMA threads
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate registers (fragments) for A, B and C
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_));                 // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_));                 // (MMA,MMA_N,MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

    // Clear the accumulators
    clear(tCrC);

    // Create smem->rmem copy atoms (LDSM) that feed the MMA
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

    // Main K-tile loop
    int k_tile_count = size<3>(tAgA);
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        // Synchronous copy gmem -> smem 
        copy(copy_a, tAgA(_,_,_,k_tile), tAsA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBsB);
        __syncthreads();

        // LDSM: smem -> registers for the MMA
        copy(s2r_atom_a, tXsA, tXrA);
        copy(s2r_atom_b, tXsB, tXrB);

        // Tensor core MMA on the register fragments
        gemm(mma, tCrA, tCrB, tCrC);
    }

    // Epilogue: write accumulators back to global memory
    __syncthreads();
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
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);                 // (BLK_M, BLK_N, BLK_K)

    // Padded row-major shared memory layouts to reduce bank conflicts
    // Pad K-stride so (bK + kPad) * sizeof(bf16) is 16-byte aligned for vector stores
    // 64*2=128 (aligned). (64+kPad)*2%16=0 → kPad%8=0. Use 8 for 1-bank offset.
    constexpr int kPad = 8;
    auto sA = make_layout(make_shape(bM, bK), make_stride(bK + Int<kPad>{}, Int<1>{}));
    auto sB = make_layout(make_shape(bN, bK), make_stride(bK + Int<kPad>{}, Int<1>{}));
    auto sC = make_layout(make_shape(bM, bN));

    // Gmem -> smem copy (128-bit vector loads)
    using GmemCopyAtom = UniversalCopy<uint128_t>;
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

    // Compute smem size and grid dimensions
    int smem_size = int(sizeof(SharedStorage<bf16, bf16, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(mmaC));                                // 128 threads (4 warps)
    dim3 dimGrid(size(ceil_div(M, bM)),                       // M-tiles
                size(ceil_div(N, bN)));                       // N-tiles

    auto kernel_fptr = gemm_device<
        decltype(prob_shape), decltype(cta_tiler),
        bf16, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
        bf16, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
        bf16, decltype(dC), decltype(sC), decltype(mmaC)>;

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
  // Reference: https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/
  //
  // CuTe kernel computes: C(M,N) = sum_k A(i,k) * B_stored(j,k)
  //                         = A(M,K) * B_std(K,N)  where B_std = B_stored^T
  //
  // Our data is all row-major. The blog's table for "all row-major" gives:
  //   transa=N, transb=N, m=n', n=m', k=k', A=B', B=A', C=C'
  //   (swaps A↔B and m↔n, writes C as n'×m' col-major = C_row^T)
  //
  // Instead we use a T,N approach that avoids the swap by exploiting that:
  //   A(M,K) row   = K×M col  (ld=K); op(A)=A^T → A_row(M,K)    ✓
  //   B(N,K) row   = K×N col  (ld=K); op(B)=B   → B_std(K,N)    ✓
  //   C(M,N) row             = M×N col  (ld=M) — compare transposed
  {
    thrust::host_vector<bf16> h_cute = d_C;
    cublasHandle_t handle;
    cublasCreate(&handle);
    auto* raw_A = reinterpret_cast<const __nv_bfloat16*>(d_A.data().get());
    auto* raw_B = reinterpret_cast<const __nv_bfloat16*>(d_B.data().get());
    thrust::device_vector<bf16> d_ref(m * n);
    auto* raw_C = reinterpret_cast<__nv_bfloat16*>(d_ref.data().get());

    // scale type CUDA_R_32F → alpha/beta must be float* (NOT __nv_bfloat16*!)
    float alpha = 1.0f, beta = 0.0f;

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
        &alpha, raw_A, CUDA_R_16BF, k,   // A: K×M col, op(A)=A^T → A_row(M,K) ✓
                raw_B, CUDA_R_16BF, k,   // B: K×N col, op(B)=B   → B_std(K,N) ✓
        &beta,  raw_C, CUDA_R_16BF, m,   // C: M×N col-major, ldc=m
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<bf16> h_ref = d_ref;

    // C_col(i,j) at i + j*m. C_row(i,j) at i*n + j.
    double max_rel_err = 0.0;
    int bad = 0;
    for (int i = 0; i < m && bad < 100; ++i) {
      for (int j = 0; j < n && bad < 100; ++j) {
        float v_our = static_cast<float>(h_cute[i * n + j]);  // row-major
        float v_ref = static_cast<float>(h_ref[i + j * m]);   // col-major
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
