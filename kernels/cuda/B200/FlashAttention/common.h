#pragma once
// ============================================================================
// common.h — SM100a (Blackwell B200) FlashAttention PTX helpers
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define cuda_check(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        abort();                                                            \
    }                                                                       \
} while (0)

#define check_cuda(call) cuda_check(call)

// ---------------------------------------------------------------------------
// Warp-level elect sync
// ---------------------------------------------------------------------------

__device__ __forceinline__
int elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

// ---------------------------------------------------------------------------
// mbarrier helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__
void mbarrier_init(int mbarrier_address, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(mbarrier_address), "r"(count));
}

__device__ __forceinline__
void cluster_fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

__device__ __forceinline__
void mbarrier_arrive_expect(const int mbarrier_address, const int copy_size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mbarrier_address), "r"(copy_size) : "memory");
}

__device__ __forceinline__
void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

// ---------------------------------------------------------------------------
// Tensor Memory (TMEM) helpers
// ---------------------------------------------------------------------------

template <int CTA_GROUP = 1>
__device__ __forceinline__
void alloc_tmem(const int tmem_addr, int width) {
    asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(tmem_addr), "r"(width), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ __forceinline__
void dealloc_tmem(int taddr, int width) {
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(width), "n"(CTA_GROUP));
}

// ---------------------------------------------------------------------------
// Shared memory descriptor encoding (14-bit fields)
// ---------------------------------------------------------------------------

__device__ __forceinline__
uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

// ---------------------------------------------------------------------------
// TMA tensor map init (cuTensorMapEncodeTiled for cp.async.bulk.tensor)
// ---------------------------------------------------------------------------

__host__ inline void init_tmap_2d_simple(
    CUtensorMap* tmap,
    const nv_bfloat16* ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank]       = {global_width, global_height};
    uint64_t globalStrides[rank-1] = {global_width * sizeof(nv_bfloat16)};
    uint32_t boxDim[rank]          = {shared_width, shared_height};
    uint32_t elementStrides[rank]  = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void*)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (err != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", msg);
        abort();
    }
}

// ---------------------------------------------------------------------------
// TMA load helpers (cp.async.bulk.tensor)
// ---------------------------------------------------------------------------

// Load Q/K into canonical no-swizzle K-major width-8 slices.
// Used for QK matmul where GEMM-K = HEAD_DIM.
__device__ __forceinline__
void tma_load_qk(
    int dst_smem_base,
    const CUtensorMap* tmap,
    int tile_height,
    int global_row0,
    int mbar_addr)
{
    for (int k = 0; k < HEAD_DIM / ATOM_COLS; ++k) {
        const int off_k = k * ATOM_COLS;
        const int slice_bytes = tile_height * ATOM_COLS * BF16_BYTES;
        const int dst = dst_smem_base + k * slice_bytes;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(dst), "l"(tmap), "r"(off_k),
               "r"(global_row0), "r"(mbar_addr) : "memory");
    }
}

// Load V into canonical no-swizzle MN-major order for PV.
__device__ __forceinline__
void tma_load_v(
    int dst_smem_base,
    const CUtensorMap* tmap,
    int global_row0,
    int mbar_addr)
{
    const int n8_stride = ATOM_ROWS * ATOM_COLS * BF16_BYTES;
    const int k8_stride = (HEAD_DIM / ATOM_COLS) * n8_stride;

    for (int k8 = 0; k8 < BLOCK_N / ATOM_ROWS; ++k8) {
        const int off_k = k8 * ATOM_ROWS;
        for (int n8 = 0; n8 < HEAD_DIM / ATOM_COLS; ++n8) {
            const int k_slice_offset = k8 * k8_stride;
            const int n_tile_offset = n8 * n8_stride;
            const int dst = dst_smem_base + k_slice_offset + n_tile_offset;
            const int off_n = n8 * ATOM_COLS;
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :: "r"(dst), "l"(tmap), "r"(off_n),
                   "r"(global_row0 + off_k), "r"(mbar_addr) : "memory");
        }
    }
}

// SMEM descriptor builder (for UMMA tcgen05.mma operands)
// Single function handling both K-major and MN-major canonical no-swizzle layouts.
// K-major: LBO = height * ATOM_COLS * BF16,  uses height (rows) - Q/K/P [M,K] / [M,N]
// MN-major: LBO = ATOM_ROWS * width * BF16, uses width (cols) - V [K,N] as MN-major B
__device__ __forceinline__
uint64_t make_smem_desc(int smem_addr_bytes, int dim, bool mn_major = false) {
    const int LBO = mn_major ? (ATOM_ROWS * dim * BF16_BYTES)
                             : (dim * ATOM_COLS * BF16_BYTES);
    const int SBO = ATOM_ROWS * ATOM_COLS * BF16_BYTES;
    return desc_encode(smem_addr_bytes)
         | (uint64_t(desc_encode(LBO)) << 16ULL)
         | (uint64_t(desc_encode(SBO)) << 32ULL)
         | (1ULL << 46ULL);
}

// ---------------------------------------------------------------------------
// tcgen05 UMMA (Unified Matrix Multiply-Accumulate)
// ---------------------------------------------------------------------------

__device__ __forceinline__
void tcgen05_mma_f16(
    int taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int enable_input_d)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
        : "memory"
    );
}

template <int CTA_GROUP = 1>
__device__ __forceinline__
void tcgen05_commit(const int mbarrier_address) {
    asm volatile(
        "tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mbarrier_address), "n"(CTA_GROUP) : "memory");
}

// ---------------------------------------------------------------------------
// tcgen05 TMEM load / store (32x32b x8)
// ---------------------------------------------------------------------------

__device__ __forceinline__
void tcgen05_ld(int taddr, float (&out)[8]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__
void tcgen05_st(int taddr, const uint32_t (&in)[8]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 "
        "[%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
        :
        : "r"(taddr), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]),
          "r"(in[4]), "r"(in[5]), "r"(in[6]), "r"(in[7])
        : "memory");
}

__device__ __forceinline__
void tcgen05_st(int taddr, const float (&in)[8]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 "
        "[%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
        :
        : "r"(taddr), "f"(in[0]), "f"(in[1]), "f"(in[2]), "f"(in[3]),
          "f"(in[4]), "f"(in[5]), "f"(in[6]), "f"(in[7])
        : "memory");
}

__device__ __forceinline__
void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;");
}
