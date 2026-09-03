// ============================================================================
// 1_baseline.cu — SM100a FlashAttention V1 (Blackwell B200)
//
// Online softmax, tiled QK → P → PV, using TMEM + UMMA (tcgen05).
// Tile sizes: BLOCK_M = BLOCK_N = HEAD_DIM = 128, MMA_K = 16.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_WARPS    = 4;
constexpr int TB_SIZE      = NUM_WARPS * 32;
constexpr int BLOCK_M      = 128;
constexpr int BLOCK_N      = 128;
constexpr int HEAD_DIM     = 128;
constexpr int MMA_K        = 16;
constexpr int BF16_BYTES   = int(sizeof(nv_bfloat16));
constexpr int MMA_K_BYTES  = MMA_K * BF16_BYTES;
constexpr int ATOM_ROWS    = 8;
constexpr int ATOM_COLS    = 8;
constexpr int TMEM_S_COLS  = BLOCK_N;
constexpr int TMEM_O_COLS  = HEAD_DIM;
constexpr int Q_TILE_BYTES = BLOCK_M * HEAD_DIM * BF16_BYTES;
constexpr int K_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;
constexpr int V_TILE_BYTES = BLOCK_M * BLOCK_N * BF16_BYTES;

#include "common.h"

// ---------------------------------------------------------------------------
// Shared memory address: canonical K-major width-8 slice [slice][row][in8]
// ---------------------------------------------------------------------------

__device__ __forceinline__
int smem_offset_k(int rows, int row, int col) {
    const int atom_idx    = col / ATOM_COLS;
    const int col_in_atom = col % ATOM_COLS;
    return atom_idx * (rows * ATOM_COLS) + row * ATOM_COLS + col_in_atom;
}

// ---------------------------------------------------------------------------
// Attention kernel
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(TB_SIZE)
void SM100a_FA_V1(
    const __grid_constant__ CUtensorMap Q_tmap,
    const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap,
    nv_bfloat16* __restrict__ O_ptr,
    int len_q,
    int len_kv)
{
    const int tid     = threadIdx.x;
    const int bid     = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int row_base = warp_id * WARP_SIZE;
    const int row      = row_base + lane_id;
    const int tmem_row = row_base << 16;

    // Tile and batch mapping
    const int q_tiles_per_batch = len_q / BLOCK_M;
    const int batch_id          = bid / q_tiles_per_batch;
    const int q_tile_id         = bid % q_tiles_per_batch;
    const int q_row0            = batch_id * len_q + q_tile_id * BLOCK_M;

    // Shared memory layout
    extern __shared__ __align__(1024) char smem_raw[];

    nv_bfloat16* Q_smem_ptr = reinterpret_cast<nv_bfloat16*>(smem_raw);
    nv_bfloat16* K_smem_ptr = Q_smem_ptr + BLOCK_M * HEAD_DIM;
    nv_bfloat16* P_smem_ptr = K_smem_ptr;   // P reuses K buffer after QK

    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
    const int Q_smem    = smem_base;
    const int K_smem    = Q_smem + Q_TILE_BYTES;
    const int V_smem    = K_smem + K_TILE_BYTES;

    // mbarrier + TMEM allocation
    __shared__ uint64_t mbar[1];
    __shared__ int      tmem_addr[1];
    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbar));
    int phase = 0;

    if (warp_id == 0 && elect_sync()) {
        mbarrier_init(mbar_addr, 1);
        cluster_fence_mbarrier_init();
    } else if (warp_id == 1) {
        const int tmem_smem_addr = static_cast<int>(
            __cvta_generic_to_shared(tmem_addr));
        alloc_tmem(tmem_smem_addr, TMEM_S_COLS + TMEM_O_COLS);
    }
    __syncthreads();

    const int tmem_addr_s = tmem_addr[0];
    const int tmem_addr_o = tmem_addr_s + TMEM_S_COLS;

    // UMMA descriptor IDs
    constexpr uint32_t idesc_QK = (1U << 4)                // fp32 accumulator
                                | (1U << 7)                // A: bf16
                                | (1U << 10)               // B: bf16
                                | ((BLOCK_N >> 3) << 17)   // MMA_N / 8
                                | ((BLOCK_N >> 4) << 24);  // MMA_N / 16
    constexpr uint32_t idesc_PV = idesc_QK | (1U << 16);   // MN-major B

    // ── Load Q into shared memory ────────────────────────────────────────
    if (warp_id == 0 && elect_sync()) {
        tma_load_qk(Q_smem, &Q_tmap, BLOCK_M, q_row0, mbar_addr);
        mbarrier_arrive_expect(mbar_addr, Q_TILE_BYTES);
    }

    mbarrier_wait(mbar_addr, phase);
    asm volatile("tcgen05.fence::after_thread_sync;");
    phase ^= 1;

    // Online softmax state
    float rowmax = -FLT_MAX;
    float rowsum = 0.0f;
    const float softmax_scale = rsqrt(float(HEAD_DIM));

    const int kv_tiles = len_kv / BLOCK_N;

    // ── KV loop ──────────────────────────────────────────────────────────
    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const int  kv_row0  = batch_id * len_kv + kv_tile * BLOCK_N;
        const bool first_kv = (kv_tile == 0);

        // ── Load K and V into shared memory ──────────────────────────────
        if (warp_id == 0 && elect_sync()) {
            tma_load_qk(K_smem, &K_tmap, BLOCK_N, kv_row0, mbar_addr);
            tma_load_v(V_smem, &V_tmap, kv_row0, mbar_addr);
            mbarrier_arrive_expect(mbar_addr, K_TILE_BYTES + V_TILE_BYTES);
        }

        mbarrier_wait(mbar_addr, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;

        // ── QK → S (TMEM) ───────────────────────────────────────────────
        if (warp_id == 0 && elect_sync()) {
            constexpr uint64_t a_step = (BLOCK_M * MMA_K_BYTES) / 16;
            constexpr uint64_t b_step = (BLOCK_N * MMA_K_BYTES) / 16;
            uint64_t a_desc = make_smem_desc(Q_smem, BLOCK_M);
            uint64_t b_desc = make_smem_desc(K_smem, BLOCK_N);

            for (int k = 0; k < HEAD_DIM / MMA_K; ++k) {
                tcgen05_mma_f16(tmem_addr_s, a_desc, b_desc, idesc_QK, (k == 0 ? 0 : 1));
                a_desc += a_step;
                b_desc += b_step;
            }
            tcgen05_commit(mbar_addr);
        }

        mbarrier_wait(mbar_addr, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;

        // ── Pass 1: compute row-max of S ─────────────────────────────────
        float tile_rowmax = -FLT_MAX;
        for (int n8 = 0; n8 < BLOCK_N / 8; ++n8) {
            float s8[8];
            tcgen05_ld(tmem_addr_s + tmem_row + n8 * 8, s8);
            for (int i = 0; i < 8; ++i)
                tile_rowmax = fmaxf(tile_rowmax, s8[i]);
        }

        // Online softmax update
        tile_rowmax *= softmax_scale;
        const float new_rowmax = fmaxf(rowmax, tile_rowmax);
        const float rescale    = __expf(rowmax - new_rowmax);
        rowmax = new_rowmax;
        rowsum *= rescale;

        // Rescale O (skip on first tile — O is initialised by PV's first micro-step)
        if (!first_kv) {
            float o8[8];
            for (int n8 = 0; n8 < HEAD_DIM / 8; ++n8) {
                const int taddr = tmem_addr_o + tmem_row + n8 * 8;
                tcgen05_ld(taddr, o8);

                for (int i = 0; i < 8; ++i)
                    o8[i] *= rescale;

                uint32_t o8_u32[8];
                for (int i = 0; i < 4; ++i) {
                    nv_bfloat16 lo = __float2bfloat16_rn(o8[2 * i]);
                    nv_bfloat16 hi = __float2bfloat16_rn(o8[2 * i + 1]);
                    unsigned short p[2] = {
                        *reinterpret_cast<unsigned short*>(&lo),
                        *reinterpret_cast<unsigned short*>(&hi)};
                    o8_u32[i] = (unsigned int)p[0] | ((unsigned int)p[1] << 16);
                }
                tcgen05_st(taddr, o8_u32);
            }
            tcgen05_wait_st();
        }

        // Ensure all threads finished rescaling before PV reads O as input-D
        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        // ── Pass 2: P = softmax(S), write P to SMEM, accumulate rowsum ───
        float tile_rowsum = 0.0f;
        for (int n8 = 0; n8 < BLOCK_N / 8; ++n8) {
            float s8[8];
            tcgen05_ld(tmem_addr_s + tmem_row + n8 * 8, s8);

            for (int i = 0; i < 8; ++i) {
                const int   n = n8 * 8 + i;
                const float s = s8[i] * softmax_scale;
                const float p = __expf(s - rowmax);
                tile_rowsum += p;

                P_smem_ptr[smem_offset_k(BLOCK_M, row, n)] = __float2bfloat16_rn(p);
            }
        }
        rowsum += tile_rowsum;

        // Fence P stores before MMA reads P through the async proxy
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        // ── PV → O (TMEM): A=P, B=V ─────────────────────────────────────
        if (warp_id == 0 && elect_sync()) {
            constexpr uint64_t a_step = (BLOCK_M  * MMA_K_BYTES) / 16;
            constexpr uint64_t b_step = (HEAD_DIM * MMA_K_BYTES) / 16;
            uint64_t a_desc = make_smem_desc(K_smem, BLOCK_M);
            uint64_t b_desc = make_smem_desc(V_smem, HEAD_DIM);

            for (int k = 0; k < BLOCK_M / MMA_K; ++k) {
                const int enable = (first_kv && k == 0) ? 0 : 1;
                tcgen05_mma_f16(tmem_addr_o, a_desc, b_desc, idesc_PV, enable);
                a_desc += a_step;
                b_desc += b_step;
            }
            tcgen05_commit(mbar_addr);
        }

        mbarrier_wait(mbar_addr, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;
    }

    // ── Epilogue: normalise O and write to global memory ─────────────────
    const float inv_denom = 1.0f / rowsum;
    for (int n8 = 0; n8 < HEAD_DIM / 8; ++n8) {
        float o8[8];
        tcgen05_ld(tmem_addr_o + tmem_row + n8 * 8, o8);

        nv_bfloat162 out_bf16x2[4];
        for (int i = 0; i < 4; ++i) {
            const float2 v = make_float2(
                o8[2 * i]     * inv_denom,
                o8[2 * i + 1] * inv_denom);
            out_bf16x2[i] = __float22bfloat162_rn(v);
        }

        nv_bfloat16* out_ptr = O_ptr + (q_row0 + row) * HEAD_DIM + n8 * 8;
        reinterpret_cast<int4*>(out_ptr)[0] =
            reinterpret_cast<int4*>(out_bf16x2)[0];
    }

    __syncthreads();

    // Deallocate TMEM (single warp, same as allocator)
    if (warp_id == 1) {
        dealloc_tmem(tmem_addr[0], TMEM_S_COLS + TMEM_O_COLS);
    }
}

// ---------------------------------------------------------------------------
// Launch helper
// ---------------------------------------------------------------------------

void SM100a_FA_V1_launch(
    const nv_bfloat16* Q_ptr,
    const nv_bfloat16* K_ptr,
    const nv_bfloat16* V_ptr,
    nv_bfloat16* O_ptr,
    int batch_size,
    int len_q,
    int len_kv,
    cudaStream_t stream)
{
    cuda_check(cudaGetLastError());
    if (batch_size <= 0)          { fprintf(stderr, "batch_size must be > 0\n"); abort(); }
    if (len_q < BLOCK_M)         { fprintf(stderr, "len_q must be >= %d\n", BLOCK_M); abort(); }
    if (len_kv < BLOCK_N)        { fprintf(stderr, "len_kv must be >= %d\n", BLOCK_N); abort(); }
    if (len_q % BLOCK_M != 0)    { fprintf(stderr, "len_q must be a multiple of %d\n", BLOCK_M); abort(); }
    if (len_kv % BLOCK_N != 0)   { fprintf(stderr, "len_kv must be a multiple of %d\n", BLOCK_N); abort(); }

    const int Q_height  = batch_size * len_q;
    const int KV_height = batch_size * len_kv;

    // Build TMA tensor maps
    CUtensorMap Q_tmap{};
    CUtensorMap K_tmap{};
    CUtensorMap V_tmap{};

    init_tmap_2d_simple(&Q_tmap, Q_ptr,
                        Q_height, HEAD_DIM,
                        BLOCK_M, ATOM_COLS,
                        CU_TENSOR_MAP_SWIZZLE_NONE);

    init_tmap_2d_simple(&K_tmap, K_ptr,
                        KV_height, HEAD_DIM,
                        BLOCK_N, ATOM_COLS,
                        CU_TENSOR_MAP_SWIZZLE_NONE);

    init_tmap_2d_simple(&V_tmap, V_ptr,
                        KV_height, HEAD_DIM,
                        ATOM_ROWS, ATOM_COLS,
                        CU_TENSOR_MAP_SWIZZLE_NONE);

    const int   grid_size  = batch_size * (len_q / BLOCK_M);
    const size_t smem_bytes = size_t(Q_TILE_BYTES + K_TILE_BYTES + V_TILE_BYTES);

    if (smem_bytes > 48 * 1024) {
        cuda_check(cudaFuncSetAttribute(
            SM100a_FA_V1,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(smem_bytes)));
    }

    SM100a_FA_V1<<<grid_size, TB_SIZE, smem_bytes, stream>>>(
        Q_tmap, K_tmap, V_tmap, O_ptr, len_q, len_kv);

    check_cuda(cudaGetLastError());
}

#include "bench.h"
BENCH_MAIN(SM100a_FA_V1, SM100a_FA_V1_launch)
