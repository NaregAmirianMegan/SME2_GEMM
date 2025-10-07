
#ifndef SME2_ZA_OPS_H
#define SME2_ZA_OPS_H

/*

==============================================================================
ZA OPS
==============================================================================

Functions to load and store directly to/from memory and ZA.

==============================================================================

*/


#include <arm_sme.h>
#include <utility>

#include "sme2_gemm_config.h"

#ifdef __ARM_FEATURE_SME



// ZA load function implementation
template<typename DTYPE, int C, int ELEMS_PER_VEC, int TILE_ROW, int VEC_IN_TILE, int TILE_COL>
static void inline gemm_sme2_load_za_vec_noscale(const DTYPE* row_ptr, int n_remaining) __arm_streaming __arm_inout("za") {
    
    constexpr uint32_t TILE_ID = TILE_ROW * C + TILE_COL;
    const DTYPE* col_ptr = row_ptr + TILE_COL * ELEMS_PER_VEC;

    // Load horizontal vector into ZA
    svbool_t n_pred_bool = svwhilelt_b32_u64(ELEMS_PER_VEC*TILE_COL, n_remaining);
    svld1_hor_za32(TILE_ID, VEC_IN_TILE, n_pred_bool, col_ptr);
}

template<typename DTYPE, int C, int ELEMS_PER_VEC, int TILE_ROW, int VEC_IN_TILE, int TILE_COL>
static void inline gemm_sme2_load_za_vec_scale(const DTYPE* row_ptr, int n_remaining, DTYPE beta) __arm_streaming __arm_inout("za") {
    
    constexpr uint32_t TILE_ID = TILE_ROW * C + TILE_COL;
    const DTYPE* col_ptr = row_ptr + TILE_COL * ELEMS_PER_VEC;

    // Load data into vector register
    svbool_t n_pred_bool = svwhilelt_b32_u64(ELEMS_PER_VEC*TILE_COL, n_remaining);
    svfloat32_t vec_data = svld1_f32(n_pred_bool, col_ptr);

    // Scale by beta
    svfloat32_t beta_vec = svdup_f32(beta);
    svfloat32_t scaled_data = svmul_f32_x(n_pred_bool, vec_data, beta_vec);

    // Store scaled data to correct ZA slice
    svwrite_hor_za32_f32_m(TILE_ID, VEC_IN_TILE, n_pred_bool, scaled_data);
}

template<typename DTYPE, BetaScaleCase BETA_CASE, int R, int C, int ELEMS_PER_VEC, int TILE_ROW, int VEC_IN_TILE, int... TileCols>
static void inline gemm_sme2_load_za_vec(const DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc, [[maybe_unused]] DTYPE beta,
                                            std::integer_sequence<int, TileCols...>)
    __arm_streaming __arm_inout("za") {
    
    constexpr int row_idx = TILE_ROW * ELEMS_PER_VEC + VEC_IN_TILE;

    if constexpr (BETA_CASE == BetaScaleCase::ONE) {
        // Valid row - load data from matrix C
        if (row_idx < m_remaining) {
            const DTYPE* row_ptr = c_ptr + row_idx * ldc;

            ((gemm_sme2_load_za_vec_noscale<DTYPE, C, ELEMS_PER_VEC, TILE_ROW, VEC_IN_TILE, TileCols>(row_ptr, n_remaining)), ...);
        }
        // NOTE: Don't need to zero unused rows because they won't be stored back
    }
    else if constexpr (BETA_CASE == BetaScaleCase::OTHER) {
        // Valid row - load data from matrix C
        if (row_idx < m_remaining) {
            const DTYPE* row_ptr = c_ptr + row_idx * ldc;
        
            ((gemm_sme2_load_za_vec_scale<DTYPE, C, ELEMS_PER_VEC, TILE_ROW, VEC_IN_TILE, TileCols>(row_ptr, n_remaining, beta)), ...);
        }
        // NOTE: Don't need to zero unused rows because they won't be stored back
    }
    else {
        static_assert(0, "Something went wrong with BetaScaleCase, compilation shouldn't reach this path!");
    }
}

template<typename DTYPE, BetaScaleCase BETA_CASE, int R, int C, int ELEMS_PER_VEC, int TILE_ROW, int... VecIndices, int... TileCols>
static void inline gemm_sme2_load_za_row(const DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc, [[maybe_unused]] DTYPE beta,
                                            std::integer_sequence<int, VecIndices...>,
                                            std::integer_sequence<int, TileCols...> cols)
    __arm_streaming __arm_inout("za") {
    
    ((gemm_sme2_load_za_vec<DTYPE, BETA_CASE, R, C, ELEMS_PER_VEC, TILE_ROW, VecIndices>(
        c_ptr, m_remaining, n_remaining, ldc, beta, cols)), ...);
}

template<typename DTYPE, BetaScaleCase BETA_CASE, int R, int C, int ELEMS_PER_VEC, int... TileRows, int... VecIndices, int... TileCols>
static void inline gemm_sme2_load_za_impl(const DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc, [[maybe_unused]] DTYPE beta,
                                             std::integer_sequence<int, TileRows...>,
                                             std::integer_sequence<int, VecIndices...> vecs,
                                             std::integer_sequence<int, TileCols...> cols)
    __arm_streaming __arm_inout("za") {
    
    ((gemm_sme2_load_za_row<DTYPE, BETA_CASE, R, C, ELEMS_PER_VEC, TileRows>(
        c_ptr, m_remaining, n_remaining, ldc, beta, vecs, cols)), ...);
}

template<typename DTYPE, BetaScaleCase BETA_CASE, int R, int C, int ELEMS_PER_VEC>
static void inline gemm_sme2_load_za(const DTYPE* c_ptr, int m_remaining, int n_remaining, [[maybe_unused]] DTYPE beta, 
                                     int ldc) 
    __arm_streaming __arm_inout("za") {
    if constexpr (BETA_CASE == BetaScaleCase::ZERO) {
        // C is just empty memory so we can zero ZA
        svzero_za();
    }
    else {
        gemm_sme2_load_za_impl<DTYPE, BETA_CASE, R, C, ELEMS_PER_VEC>(
            c_ptr, m_remaining, n_remaining, ldc, beta,
            std::make_integer_sequence<int, R>{},
            std::make_integer_sequence<int, ELEMS_PER_VEC>{},
            std::make_integer_sequence<int, C>{}
        );
    }
}


// ZA store function implementation  
template<typename DTYPE, int C, int ELEMS_PER_VEC, int TILE_ROW, int VEC_IN_TILE, int TILE_COL>
static void inline gemm_sme2_store_za_vec_hor(DTYPE* row_ptr, int n_remaining) __arm_streaming __arm_inout("za") {
    constexpr uint32_t TILE_ID = TILE_ROW * C + TILE_COL;
    DTYPE* col_ptr = row_ptr + TILE_COL * ELEMS_PER_VEC;
    svbool_t n_pred_bool = svwhilelt_b32_u64(ELEMS_PER_VEC*TILE_COL, n_remaining);
    svst1_hor_za32(TILE_ID, VEC_IN_TILE, n_pred_bool, col_ptr);
}

template<typename DTYPE, int R, int ELEMS_PER_VEC, int TILE_ROW, int VEC_IN_TILE, int TILE_COL>
static void inline gemm_sme2_store_za_vec_ver(DTYPE* col_ptr, int m_remaining) __arm_streaming __arm_inout("za") {
    constexpr uint32_t TILE_ID = TILE_ROW * R + TILE_COL;
    DTYPE* row_ptr = col_ptr + TILE_COL * ELEMS_PER_VEC;
    svbool_t m_pred_bool = svwhilelt_b32_u64(ELEMS_PER_VEC*TILE_COL, m_remaining);
    svst1_ver_za32(TILE_ID, VEC_IN_TILE, m_pred_bool, row_ptr);
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC, ZA_Dim DIM, int TILE_ROW, int VEC_IN_TILE, int... TileCols>
static void inline gemm_sme2_store_za_vec(DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc, 
                                            std::integer_sequence<int, TileCols...>)
    __arm_streaming __arm_inout("za") {

    if constexpr (DIM == ZA_Dim::H) {
        constexpr int row_idx = TILE_ROW * ELEMS_PER_VEC + VEC_IN_TILE;
    
        if (row_idx < m_remaining) {
            DTYPE* row_ptr = c_ptr + row_idx * ldc;

            ((gemm_sme2_store_za_vec_hor<DTYPE, C, ELEMS_PER_VEC, TILE_ROW, VEC_IN_TILE, TileCols>(row_ptr, n_remaining)), ...);
        }
    }
    else {
        // Assumed to be in ZA_Dim::V case where columns are now rows
        constexpr int col_idx = TILE_ROW * ELEMS_PER_VEC + VEC_IN_TILE;

        if (col_idx < n_remaining) {
            DTYPE* col_ptr = c_ptr + col_idx * ldc; // ldc should actually be number of rows in c_ptr matrix

            ((gemm_sme2_store_za_vec_ver<DTYPE, R, ELEMS_PER_VEC, TILE_ROW, VEC_IN_TILE, TileCols>(col_ptr, m_remaining)), ...);
        }
    }
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC, ZA_Dim DIM, int TILE_ROW, int... VecIndices, int... TileCols>
static void inline gemm_sme2_store_za_row(DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc,
                                            std::integer_sequence<int, VecIndices...>,
                                            std::integer_sequence<int, TileCols...> cols)
    __arm_streaming __arm_inout("za") {
    
    ((gemm_sme2_store_za_vec<DTYPE, R, C, ELEMS_PER_VEC, DIM, TILE_ROW, VecIndices>(
        c_ptr, m_remaining, n_remaining, ldc, cols)), ...);
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC, ZA_Dim DIM, int... TileRows, int... VecIndices, int... TileCols>
static void inline gemm_sme2_store_za_impl(DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc,
                                             std::integer_sequence<int, TileRows...>,
                                             std::integer_sequence<int, VecIndices...> vecs,
                                             std::integer_sequence<int, TileCols...> cols)
    __arm_streaming __arm_inout("za") {
    
    ((gemm_sme2_store_za_row<DTYPE, R, C, ELEMS_PER_VEC, DIM, TileRows>(
        c_ptr, m_remaining, n_remaining, ldc, vecs, cols)), ...);
}

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC, ZA_Dim DIM>
static void inline gemm_sme2_store_za(DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc)
    __arm_streaming __arm_inout("za") {
    if constexpr (DIM == ZA_Dim::H) {
        gemm_sme2_store_za_impl<DTYPE, R, C, ELEMS_PER_VEC, DIM>(
            c_ptr, m_remaining, n_remaining, ldc,
            std::make_integer_sequence<int, R>{},
            std::make_integer_sequence<int, ELEMS_PER_VEC>{},
            std::make_integer_sequence<int, C>{}
        );
    }
    else if constexpr (DIM == ZA_Dim::V) {
        gemm_sme2_store_za_impl<DTYPE, R, C, ELEMS_PER_VEC, DIM>(
            c_ptr, m_remaining, n_remaining, ldc,
            std::make_integer_sequence<int, C>{},
            std::make_integer_sequence<int, ELEMS_PER_VEC>{},
            std::make_integer_sequence<int, R>{}
        );
    }
    else {
        static_assert(false, "Invalid ZA_Dim parameter passed to gemm_sme2_store_za!");
    }
}

#endif // __ARM_FEATURE_SME
#endif // SME2_ZA_OPS_H
