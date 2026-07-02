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

// =============================================================================
// Low-level helpers (ported from cuda/A100/common.h)
// =============================================================================

__host__ __device__ inline constexpr int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// Triton-style threadblock swizzle with runtime GROUP_M
__host__ __device__ inline void swizzle_block_idx_triton(
    int bid, int grid_m, int grid_n, int &bid_m, int &bid_n, int GROUP_M)
{
    if (GROUP_M == 0) {
        bid_m = bid / grid_n;
        bid_n = bid % grid_n;
    } else {
        const int group_size = GROUP_M * grid_n;
        const int group_id = bid / group_size;
        const int group_off_m = group_id * GROUP_M;
        const int group_m = (grid_m - group_off_m) < GROUP_M
            ? (grid_m - group_off_m) : GROUP_M;
        bid_m = group_off_m + ((bid % group_size) % group_m);
        bid_n = (bid % group_size) / group_m;
    }
}

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;
static constexpr int WARP_SIZE = 32;

__device__ inline uint32_t to_smem(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ inline void LDMATRIX_X4(uint32_t reg[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
}

__device__ inline void MMA_M16N8K16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]));
}

__device__ inline void cp_async(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}

__device__ inline void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ inline void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ inline void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// STRIDE in bytes, col in units of 16 bytes
template <int STRIDE>
__device__ static uint32_t swizzle_better(uint32_t row, uint32_t col) {
  if constexpr (STRIDE >= 128)
    col ^= (row % 8) / ((128 / STRIDE) > 1 ? (128 / STRIDE) : 1);
  return row * STRIDE + col * 16;
}

// Async global-to-shared copy with swizzle_better
template <int CTA_SIZE, int HEIGHT, int WIDTH>
__device__ static void g2s_swizzled(
    const __nv_bfloat16 *in, int in_stride, uint32_t out, int tid)
{
  constexpr int num_elems = 16 / sizeof(__nv_bfloat16);  // 8
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

// Epilogue: store MMA accumulators to global memory as bf16
template <int NUM_MMA_M, int NUM_MMA_N>
__device__ inline void epilogue_store_bf16(
    float acc[][NUM_MMA_N][4], __nv_bfloat16* C, int N, int lane_id,
    int warp_id_m, int WARP_M, int warp_id_n, int WARP_N)
{
  #pragma unroll
  for (int m = 0; m < NUM_MMA_M; m++)
    #pragma unroll
    for (int n = 0; n < NUM_MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;
      float *regs = acc[m][n];
      reinterpret_cast<__nv_bfloat162*>(C + (row + 0) * N + col)[0] =
          __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<__nv_bfloat162*>(C + (row + 8) * N + col)[0] =
          __float22bfloat162_rn({regs[2], regs[3]});
    }
}

template<typename Kern, typename... Args>
void launch_safe(Kern* k, int grid, int block, int smem, Args... args) {
    if (smem > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(
            k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    k<<<grid, block, smem>>>(args...);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// v37 kernel — v13d approach (8 warps, double-buffered regs, raw PTX MMA)
// =============================================================================

template <
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int NUM_WARP_M, int NUM_WARP_N,
    int NUM_STAGES, int GROUP_M = 8
>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void matmul_v37_kern(
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

    A += bid_m * BLOCK_M * K;
    B += bid_n * BLOCK_N * K;
    C += (bid_m * BLOCK_M + warp_id_m * WARP_M) * N
       + (bid_n * BLOCK_N + warp_id_n * WARP_N);

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
    const uint32_t A_shm_thread = A_shm_base
        + swizzle_better<STRIDE_BYTES>(A_offm, lane_id / 16);
    const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
    const uint32_t B_shm_thread = B_shm_base
        + swizzle_better<STRIDE_BYTES>(B_offn, (lane_id % 16) / 8);

    const int num_k_iters = cdiv(K, BLOCK_K);
    const __nv_bfloat16 *A_ptr = A;
    const __nv_bfloat16 *B_ptr = B;

    auto load_AB = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;
        g2s_swizzled<CTA_SIZE, BLOCK_M, BLOCK_K>(
            A_ptr, K, A_shm_base + stage_id * AB_size, tid);
        g2s_swizzled<CTA_SIZE, BLOCK_N, BLOCK_K>(
            B_ptr, K, B_shm_base + stage_id * AB_size, tid);
        A_ptr += BLOCK_K;
        B_ptr += BLOCK_K;
        cp_async_commit_group();
    };

    // Double-buffered register compute
    auto compute_db = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;
        const uint32_t stage_offset = stage_id * AB_size;
        uint32_t A_buf[2][NUM_MMA_M][4];
        uint32_t B_buf[2][NUM_MMA_N][2];

        // Load first k-slice
        #pragma unroll
        for (int m = 0; m < NUM_MMA_M; m++) {
            uint32_t A_addr = A_shm_thread + stage_offset
                + m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
            LDMATRIX_X4(A_buf[0][m], A_addr);
        }
        #pragma unroll
        for (int n = 0; n < NUM_MMA_N; n += 2) {
            uint32_t B_addr = B_shm_thread + stage_offset
                + n * MMA_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
            LDMATRIX_X4(B_buf[0][n], B_addr);
        }

        #pragma unroll
        for (int k = 0; k < NUM_MMA_K; k++) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;

            // Load next k-slice regs (overlaps with compute below)
            if (k + 1 < NUM_MMA_K) {
                #pragma unroll
                for (int m = 0; m < NUM_MMA_M; m++) {
                    uint32_t A_addr = A_shm_thread + stage_offset;
                    A_addr += m * MMA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);
                    LDMATRIX_X4(A_buf[nxt][m], A_addr ^ ((k+1) * 32));
                }
                #pragma unroll
                for (int n = 0; n < NUM_MMA_N; n += 2) {
                    uint32_t B_addr = B_shm_thread + stage_offset;
                    B_addr += n * MMA_N * BLOCK_K * (int)sizeof(__nv_bfloat16);
                    LDMATRIX_X4(B_buf[nxt][n], B_addr ^ ((k+1) * 32));
                }
            }

            // MMA for current k-slice
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

    // Epilogue
    for (int k_iter = num_k_iters - PD; k_iter < num_k_iters; k_iter++) {
        __syncthreads();
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();
        compute_db(k_iter);
    }

    epilogue_store_bf16<NUM_MMA_M, NUM_MMA_N>(
        acc, C, N, lane_id, warp_id_m, WARP_M, warp_id_n, WARP_N);
}

// =============================================================================
// gemm() host function — matches CuTe benchmark interface
// =============================================================================

template <class TA, class TB, class TC>
void gemm(int m, int n, int k,
          TA const *A, int ldA,
          TB const *B, int ldB,
          TC *C, int ldC,
          cudaStream_t stream = 0)
{
    // v13d config: 128x128x64, 4x2 warps, 2-stage, GROUP_M=16
    constexpr int BM=128, BN=128, BK=64, WM=4, WN=2, ST=2, GM=16;
    constexpr int smem = (BM*BK + BN*BK) * (int)sizeof(__nv_bfloat16) * ST;
    constexpr int CTA_SIZE = WM * WN * WARP_SIZE;

    auto* raw_A = reinterpret_cast<const __nv_bfloat16*>(A);
    auto* raw_B = reinterpret_cast<const __nv_bfloat16*>(B);
    auto* raw_C = reinterpret_cast<__nv_bfloat16*>(C);

    launch_safe(matmul_v37_kern<BM,BN,BK,WM,WN,ST,GM>,
        cdiv(m, BM) * cdiv(n, BN), CTA_SIZE, smem,
        raw_A, raw_B, raw_C, m, n, k);
}


#ifndef BENCHMARK_SUITE
int main(int argc, char** argv)
{
  int m = 16384, n = 16384, k = 16384;

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
  int iters = 50;

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
