#ifndef SME2_REARRANGE_H
#define SME2_REARRANGE_H

/*

==============================================================================
REARRANGE MEMORY FORMAT
==============================================================================

Transpose and densely pack matrices to make computations on them more cache
efficient.

==============================================================================

*/


#include <cstring>

#include "sme2_za_ops.h"



// pack rxc section of src (assuming src is row-major) into dst
// 

// ISSUE: This may be inefficient becase calls to std::memcpy cause a switch out of streaming mode which invokes context saving
//        of ZA which is a large overhead.
template<typename DTYPE>
static void inline pack_normal(const DTYPE* src, DTYPE* dst, int r, int c, int ldim) __arm_streaming_compatible __arm_preserves("za") {
    for (int i = 0; i < r; ++i) {
        // __arm_sc_memcpy((void *)(dst+i*ldim), (const void *)(src+i*c), (size_t)(sizeof(DTYPE)*c));
        std::memcpy( (void *)(dst+i*ldim), (const void *)(src+i*c), (size_t)(sizeof(DTYPE)*c) );
    }
}

#ifdef __ARM_FEATURE_SME

// dst is assumed to be rxc and ldim is leading dimension of src
template<typename DTYPE, int SVL, int R, int C, bool transpose>
static void inline pack_sme(const DTYPE* src, DTYPE* dst, int r, int c, int ldim) __arm_streaming __arm_inout("za") {
    constexpr int ELEMS_PER_VEC = SVL / (sizeof(DTYPE));
    constexpr int R_TILE_SIZE = ELEMS_PER_VEC * R;
    constexpr int C_TILE_SIZE = ELEMS_PER_VEC * C;

    for (int r_tile = 0; r_tile < r; r_tile += R_TILE_SIZE) {
        int r_remaining = r - r_tile;
        for (int c_tile = 0; c_tile < c; c_tile += C_TILE_SIZE) {
            int c_remaining = c - c_tile;
            // Load from src horizontally into ZA
            gemm_sme2_load_za<DTYPE, BetaScaleCase::ONE, R, C, ELEMS_PER_VEC>(src+r_tile*ldim+c_tile, r_remaining, c_remaining, 1.0, ldim);

            // Store into dst vertically from ZA
            if constexpr (transpose) {
                gemm_sme2_store_za<DTYPE, R, C, ELEMS_PER_VEC, ZA_Dim::V>(dst+c_tile*r+r_tile, r_remaining, c_remaining, r);
            }
            else {
                gemm_sme2_store_za<DTYPE, R, C, ELEMS_PER_VEC, ZA_Dim::H>(dst+r_tile*c+c_tile, r_remaining, c_remaining, c);
            }
        }
    }
}

#endif // __ARM_FEATURE_SME
#endif // SME2_REARRANGE_H