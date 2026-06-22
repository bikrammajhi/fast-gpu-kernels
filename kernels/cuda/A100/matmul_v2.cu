#pragma once
#include "common.h"

// =============================================================================
// matmul_v2 — placeholder for your next optimized version
//
// TODO: implement your kernel here, then:
//   1. define __global__ void matmul_v2_kern<...>(...)
//   2. define inline void matmul_v2_launch(A, B, C, M, N, K)
//   3. add to kernels[] in benchmark.cu
// =============================================================================

/*
template <int BM, int BN, int BK, int WM, int WN, int SS>
__launch_bounds__(WM * WN * WARP_SIZE)
__global__ void matmul_v2_kern(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    // TODO
}

inline void matmul_v2_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              __nv_bfloat16* C, int M, int N, int K)
{
    constexpr int BM = 128, BN = 128, BK = 64, WM = 2, WN = 2, SS = BK;
    launch_safe(matmul_v2_kern<BM, BN, BK, WM, WN, SS>,
        cdiv(M, BM) * cdiv(N, BN), WM * WN * WARP_SIZE,
        (BM + BN) * SS * (int)sizeof(__nv_bfloat16), A, B, C, M, N, K);
}
*/
