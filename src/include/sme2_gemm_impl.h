#ifndef SME2_GEMM_IMPL_H
#define SME2_GEMM_IMPL_H

#include <arm_sme.h>
#include <stdlib.h>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

#include "debug_util.h"

#ifdef __ARM_FEATURE_SME

#define DEBUG 0

enum class AlphaScaleCase {
    ONE,
    OTHER
};

enum class BetaScaleCase {
    ZERO,
    ONE,
    OTHER
};

enum class ZA_Dim {
    H,
    V,
};

//==============================================================================
// FORWARD DECLARATIONS
//==============================================================================

template<typename DTYPE, BetaScaleCase BETA_CASE, int R, int C, int ELEMS_PER_VEC>
static void inline gemm_sme2_load_za(const DTYPE* c_ptr, int m_remaining, int n_remaining, DTYPE beta, int ldc) 
    __arm_streaming __arm_inout("za");

template<typename DTYPE, int R, int C, int ELEMS_PER_VEC>
static void inline gemm_sme2_store_za(DTYPE* c_ptr, int m_remaining, int n_remaining, int ldc) 
    __arm_streaming __arm_inout("za");

template<typename DTYPE, int R, int C, int UNROLL, AlphaScaleCase ALPHA_CASE>
static void inline gemm_sme2_microkernel(const DTYPE* a_ptr, const DTYPE* b_ptr, 
                                            svcount_t m_pred, svcount_t n_pred, [[maybe_unused]] DTYPE alpha, [[maybe_unused]] int lda, [[maybe_unused]] int ldb) 
    __arm_streaming __arm_inout("za");

template<typename DTYPE, int MR, int NR, int SVL, int UNROLL, int R, int C, 
         AlphaScaleCase ALPHA_CASE, BetaScaleCase BETA_CASE>
static void inline gemm_sme2_kernel(int mr, int nr, int kc, 
                                    const DTYPE* a_ptr, const DTYPE* b_ptr,
                                    DTYPE* c_ptr, DTYPE alpha, DTYPE beta,
                                    int ldb, int ldc) 
    __arm_streaming __arm_inout("za");

//==============================================================================
// FUNCTION IMPLEMENTATIONS
//==============================================================================


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


// Micro-kernel specializations
template<>
void inline gemm_sme2_microkernel<float, 2, 2, 1, AlphaScaleCase::ONE>(
                    const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
                    svcount_t n_pred, [[maybe_unused]] float alpha,
                    [[maybe_unused]] int lda, [[maybe_unused]] int ldb) 
        __arm_streaming __arm_inout("za") {

    svfloat32x2_t a_cols = svld1_f32_x2(m_pred, a_ptr);
    svfloat32x2_t b_rows = svld1_f32_x2(n_pred, b_ptr);
    svfloat32_t a_col_0 = svget2_f32(a_cols, 0);
    svfloat32_t a_col_1 = svget2_f32(a_cols, 1);
    svfloat32_t b_row_0 = svget2_f32(b_rows, 0);
    svfloat32_t b_row_1 = svget2_f32(b_rows, 1);

    // we don't need to predicate the multiplies so we can just set these to always
    // be all true since we already predicated the loads. This shouldn't cost us any
    // time since all the multiplication/accumulation is done in paralell (may cost some power :( )
    // Saves us the trouble of converting from predicate as counter to predicate as mask
    svbool_t ptrue = svptrue_b32();
    svmopa_za32_f32_m(0, ptrue, ptrue, a_col_0, b_row_0);
    svmopa_za32_f32_m(1, ptrue, ptrue, a_col_0, b_row_1);
    svmopa_za32_f32_m(2, ptrue, ptrue, a_col_1, b_row_0);
    svmopa_za32_f32_m(3, ptrue, ptrue, a_col_1, b_row_1);

}

// ISSUE: For now we use single vector scaling instructions; SME2 offers multi-vector scaling, but the clang intrinsics aren't available yet
template<>
void inline gemm_sme2_microkernel<float, 2, 2, 1, AlphaScaleCase::OTHER>(
                    const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
                    svcount_t n_pred, [[maybe_unused]] float alpha,
                    [[maybe_unused]] int lda, [[maybe_unused]] int ldb) 
    __arm_streaming __arm_inout("za") {

    svfloat32x2_t a_cols = svld1_f32_x2(m_pred, a_ptr);
    svfloat32x2_t b_rows = svld1_f32_x2(n_pred, b_ptr);
    svfloat32_t a_col_0 = svget2_f32(a_cols, 0);
    svfloat32_t a_col_1 = svget2_f32(a_cols, 1);
    svfloat32_t b_row_0 = svget2_f32(b_rows, 0);
    svfloat32_t b_row_1 = svget2_f32(b_rows, 1);

    // we don't need to predicate the multiplies so we can just set these to always
    // be all true since we already predicated the loads. This shouldn't cost us any
    // time since all the multiplication/accumulation is done in paralell (may cost some power :( )
    // Saves us the trouble of converting from predicate as counter to predicate as mask
    svbool_t ptrue = svptrue_b32();

    // scale A with alpha before doing matrix multiply
    a_col_0 = svmul_n_f32_m(ptrue, a_col_0, (float32_t)alpha);
    a_col_1 = svmul_n_f32_m(ptrue, a_col_1, (float32_t)alpha);

    svmopa_za32_f32_m(0, ptrue, ptrue, a_col_0, b_row_0);
    svmopa_za32_f32_m(1, ptrue, ptrue, a_col_0, b_row_1);
    svmopa_za32_f32_m(2, ptrue, ptrue, a_col_1, b_row_0);
    svmopa_za32_f32_m(3, ptrue, ptrue, a_col_1, b_row_1);

}

// template<>
// void inline gemm_sme2_microkernel<float, 2, 2, 4, AlphaScaleCase::ONE>(
//                     const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
//                     svcount_t n_pred, [[maybe_unused]] float alpha, 
//                     [[maybe_unused]] int lda, [[maybe_unused]] int ldb) 
//         __arm_streaming __arm_inout("za") {

//     svbool_t ptrue = svptrue_b32();

//     svfloat32x2_t a_cols_0 = svld1_f32_x2(m_pred, a_ptr);
//     svfloat32x2_t b_rows_0 = svld1_f32_x2(n_pred, b_ptr);
//     svfloat32_t a_col_0_0 = svget2_f32(a_cols_0, 0);
//     svfloat32_t a_col_0_1 = svget2_f32(a_cols_0, 1);
//     svfloat32_t b_row_0_0 = svget2_f32(b_rows_0, 0);
//     svfloat32_t b_row_0_1 = svget2_f32(b_rows_0, 1);

//     a_ptr += lda;
//     b_ptr += ldb;

//     svfloat32x2_t a_cols_1 = svld1_f32_x2(m_pred, a_ptr);
//     svfloat32x2_t b_rows_1 = svld1_f32_x2(n_pred, b_ptr);
//     svfloat32_t a_col_1_0 = svget2_f32(a_cols_1, 0);
//     svfloat32_t a_col_1_1 = svget2_f32(a_cols_1, 1);
//     svfloat32_t b_row_1_0 = svget2_f32(b_rows_1, 0);
//     svfloat32_t b_row_1_1 = svget2_f32(b_rows_1, 1);

//     a_ptr += lda;
//     b_ptr += ldb;

//     svmopa_za32_f32_m(0, ptrue, ptrue, a_col_0_0, b_row_0_0);
//     svmopa_za32_f32_m(1, ptrue, ptrue, a_col_0_0, b_row_0_1);
//     svmopa_za32_f32_m(2, ptrue, ptrue, a_col_0_1, b_row_0_0);
//     svmopa_za32_f32_m(3, ptrue, ptrue, a_col_0_1, b_row_0_1);

//     svfloat32x2_t a_cols_2 = svld1_f32_x2(m_pred, a_ptr);
//     svfloat32x2_t b_rows_2 = svld1_f32_x2(n_pred, b_ptr);
//     svfloat32_t a_col_2_0 = svget2_f32(a_cols_2, 0);
//     svfloat32_t a_col_2_1 = svget2_f32(a_cols_2, 1);
//     svfloat32_t b_row_2_0 = svget2_f32(b_rows_2, 0);
//     svfloat32_t b_row_2_1 = svget2_f32(b_rows_2, 1);

//     a_ptr += lda;
//     b_ptr += ldb;

//     svmopa_za32_f32_m(0, ptrue, ptrue, a_col_1_0, b_row_1_0);
//     svmopa_za32_f32_m(1, ptrue, ptrue, a_col_1_0, b_row_1_1);
//     svmopa_za32_f32_m(2, ptrue, ptrue, a_col_1_1, b_row_1_0);
//     svmopa_za32_f32_m(3, ptrue, ptrue, a_col_1_1, b_row_1_1);

//     svfloat32x2_t a_cols_3 = svld1_f32_x2(m_pred, a_ptr);
//     svfloat32x2_t b_rows_3 = svld1_f32_x2(n_pred, b_ptr);
//     svfloat32_t a_col_3_0 = svget2_f32(a_cols_3, 0);
//     svfloat32_t a_col_3_1 = svget2_f32(a_cols_3, 1);
//     svfloat32_t b_row_3_0 = svget2_f32(b_rows_3, 0);
//     svfloat32_t b_row_3_1 = svget2_f32(b_rows_3, 1);

//     svmopa_za32_f32_m(0, ptrue, ptrue, a_col_2_0, b_row_2_0);
//     svmopa_za32_f32_m(1, ptrue, ptrue, a_col_2_0, b_row_2_1);
//     svmopa_za32_f32_m(2, ptrue, ptrue, a_col_2_1, b_row_2_0);
//     svmopa_za32_f32_m(3, ptrue, ptrue, a_col_2_1, b_row_2_1);

//     svmopa_za32_f32_m(0, ptrue, ptrue, a_col_3_0, b_row_3_0);
//     svmopa_za32_f32_m(1, ptrue, ptrue, a_col_3_0, b_row_3_1);
//     svmopa_za32_f32_m(2, ptrue, ptrue, a_col_3_1, b_row_3_0);
//     svmopa_za32_f32_m(3, ptrue, ptrue, a_col_3_1, b_row_3_1);
// }

#define LOAD_AB_VECS(x, a_ptr, b_ptr, m_pred, n_pred) \
    svfloat32x2_t a_cols_##x = svld1_f32_x2(m_pred, a_ptr); \
    svfloat32x2_t b_rows_##x = svld1_f32_x2(n_pred, b_ptr); \
    svfloat32_t a_col_##x##_0 = svget2_f32(a_cols_##x, 0); \
    svfloat32_t a_col_##x##_1 = svget2_f32(a_cols_##x, 1); \
    svfloat32_t b_row_##x##_0 = svget2_f32(b_rows_##x, 0); \
    svfloat32_t b_row_##x##_1 = svget2_f32(b_rows_##x, 1);

#define MOPA_AB_VECS(x) \
    svmopa_za32_f32_m(0, ptrue, ptrue, a_col_##x##_0, b_row_##x##_0); \
    svmopa_za32_f32_m(1, ptrue, ptrue, a_col_##x##_0, b_row_##x##_1); \
    svmopa_za32_f32_m(2, ptrue, ptrue, a_col_##x##_1, b_row_##x##_0); \
    svmopa_za32_f32_m(3, ptrue, ptrue, a_col_##x##_1, b_row_##x##_1);

template<>
void inline gemm_sme2_microkernel<float, 2, 2, 4, AlphaScaleCase::ONE>(
                    const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
                    svcount_t n_pred, [[maybe_unused]] float alpha, 
                    int lda, int ldb) 
        __arm_streaming __arm_inout("za") {

    svbool_t ptrue = svptrue_b32();

    // Pre-compute all addresses - removes dependency chain
    const float* a_ptr_0 = a_ptr;
    const float* a_ptr_1 = a_ptr + lda;
    const float* a_ptr_2 = a_ptr + 2*lda;
    const float* a_ptr_3 = a_ptr + 3*lda;

    const float* b_ptr_0 = b_ptr;
    const float* b_ptr_1 = b_ptr + ldb;
    const float* b_ptr_2 = b_ptr + 2*ldb;
    const float* b_ptr_3 = b_ptr + 3*ldb;

    // Issue all loads first - hardware can prefetch aggressively
    LOAD_AB_VECS(0, a_ptr_0, b_ptr_0, m_pred, n_pred)
    LOAD_AB_VECS(1, a_ptr_1, b_ptr_1, m_pred, n_pred)
    LOAD_AB_VECS(2, a_ptr_2, b_ptr_2, m_pred, n_pred)
    LOAD_AB_VECS(3, a_ptr_3, b_ptr_3, m_pred, n_pred)

    // Then compute - by now loads should be complete or in flight
    MOPA_AB_VECS(0)
    MOPA_AB_VECS(1)
    MOPA_AB_VECS(2)
    MOPA_AB_VECS(3)
}

// template<>
// void inline gemm_sme2_microkernel<float, 2, 2, 8, AlphaScaleCase::ONE>(
//                     const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
//                     svcount_t n_pred, [[maybe_unused]] float alpha, 
//                     int lda, int ldb) 
//         __arm_streaming __arm_inout("za") {

//     svbool_t ptrue = svptrue_b32();

//     // Pre-compute all addresses - removes dependency chain
//     const float* a_ptr_0 = a_ptr;
//     const float* a_ptr_1 = a_ptr + lda;
//     const float* a_ptr_2 = a_ptr + 2*lda;
//     const float* a_ptr_3 = a_ptr + 3*lda;
//     const float* a_ptr_4 = a_ptr + 4*lda;
//     const float* a_ptr_5 = a_ptr + 5*lda;
//     const float* a_ptr_6 = a_ptr + 6*lda;
//     const float* a_ptr_7 = a_ptr + 7*lda;

//     const float* b_ptr_0 = b_ptr;
//     const float* b_ptr_1 = b_ptr + ldb;
//     const float* b_ptr_2 = b_ptr + 2*ldb;
//     const float* b_ptr_3 = b_ptr + 3*ldb;
//     const float* b_ptr_4 = b_ptr + 4*ldb;
//     const float* b_ptr_5 = b_ptr + 5*ldb;
//     const float* b_ptr_6 = b_ptr + 6*ldb;
//     const float* b_ptr_7 = b_ptr + 7*ldb;

//     // Issue all loads first - hardware can prefetch aggressively
//     LOAD_AB_VECS(0, a_ptr_0, b_ptr_0, m_pred, n_pred)
//     LOAD_AB_VECS(1, a_ptr_1, b_ptr_1, m_pred, n_pred)
//     LOAD_AB_VECS(2, a_ptr_2, b_ptr_2, m_pred, n_pred)
//     LOAD_AB_VECS(3, a_ptr_3, b_ptr_3, m_pred, n_pred)
//     LOAD_AB_VECS(4, a_ptr_4, b_ptr_4, m_pred, n_pred)
//     LOAD_AB_VECS(5, a_ptr_5, b_ptr_5, m_pred, n_pred)
//     LOAD_AB_VECS(6, a_ptr_6, b_ptr_6, m_pred, n_pred)
//     LOAD_AB_VECS(7, a_ptr_7, b_ptr_7, m_pred, n_pred)

//     // Then compute - by now loads should be complete or in flight
//     MOPA_AB_VECS(0)
//     MOPA_AB_VECS(1)
//     MOPA_AB_VECS(2)
//     MOPA_AB_VECS(3)
//     MOPA_AB_VECS(4)
//     MOPA_AB_VECS(5)
//     MOPA_AB_VECS(6)
//     MOPA_AB_VECS(7)
// }

#define locality 1
template<>
void inline gemm_sme2_microkernel<float, 2, 2, 8, AlphaScaleCase::ONE>(
                    const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
                    svcount_t n_pred, [[maybe_unused]] float alpha, 
                    int lda, int ldb) 
        __arm_streaming __arm_inout("za") {

    svbool_t ptrue = svptrue_b32();

    // Force compiler to compute all addresses up front
    // Mark as const and use uintptr_t to prevent aliasing assumptions
    const uintptr_t a_base = (uintptr_t)a_ptr;
    const uintptr_t b_base = (uintptr_t)b_ptr;
    const uintptr_t lda_bytes = lda * sizeof(float);
    const uintptr_t ldb_bytes = ldb * sizeof(float);

    const float* __restrict a0 = (const float*)(a_base);
    const float* __restrict a1 = (const float*)(a_base + lda_bytes);
    const float* __restrict a2 = (const float*)(a_base + 2*lda_bytes);
    const float* __restrict a3 = (const float*)(a_base + 3*lda_bytes);
    const float* __restrict a4 = (const float*)(a_base + 4*lda_bytes);
    const float* __restrict a5 = (const float*)(a_base + 5*lda_bytes);
    const float* __restrict a6 = (const float*)(a_base + 6*lda_bytes);
    const float* __restrict a7 = (const float*)(a_base + 7*lda_bytes);

    const float* __restrict b0 = (const float*)(b_base);
    const float* __restrict b1 = (const float*)(b_base + ldb_bytes);
    const float* __restrict b2 = (const float*)(b_base + 2*ldb_bytes);
    const float* __restrict b3 = (const float*)(b_base + 3*ldb_bytes);
    const float* __restrict b4 = (const float*)(b_base + 4*ldb_bytes);
    const float* __restrict b5 = (const float*)(b_base + 5*ldb_bytes);
    const float* __restrict b6 = (const float*)(b_base + 6*ldb_bytes);
    const float* __restrict b7 = (const float*)(b_base + 7*ldb_bytes);

    // After address computation, before loads
    __builtin_prefetch(a0, 0, locality);  // Read, high temporal locality
    __builtin_prefetch(a1, 0, locality);
    __builtin_prefetch(a2, 0, locality);
    __builtin_prefetch(a3, 0, locality);
    __builtin_prefetch(a4, 0, locality);
    __builtin_prefetch(a5, 0, locality);
    __builtin_prefetch(a6, 0, locality);
    __builtin_prefetch(a7, 0, locality);
    __builtin_prefetch(b0, 0, locality);
    __builtin_prefetch(b1, 0, locality);
    __builtin_prefetch(b2, 0, locality);
    __builtin_prefetch(b3, 0, locality);
    __builtin_prefetch(b4, 0, locality);
    __builtin_prefetch(b5, 0, locality);
    __builtin_prefetch(b6, 0, locality);
    __builtin_prefetch(b7, 0, locality);

    // All loads together
    LOAD_AB_VECS(0, a0, b0, m_pred, n_pred)
    LOAD_AB_VECS(1, a1, b1, m_pred, n_pred)
    LOAD_AB_VECS(2, a2, b2, m_pred, n_pred)
    LOAD_AB_VECS(3, a3, b3, m_pred, n_pred)
    LOAD_AB_VECS(4, a4, b4, m_pred, n_pred)
    LOAD_AB_VECS(5, a5, b5, m_pred, n_pred)
    LOAD_AB_VECS(6, a6, b6, m_pred, n_pred)
    LOAD_AB_VECS(7, a7, b7, m_pred, n_pred)

    // All computes together
    MOPA_AB_VECS(0)
    MOPA_AB_VECS(1)
    MOPA_AB_VECS(2)
    MOPA_AB_VECS(3)
    MOPA_AB_VECS(4)
    MOPA_AB_VECS(5)
    MOPA_AB_VECS(6)
    MOPA_AB_VECS(7)
}

template<>
void inline gemm_sme2_microkernel<float, 2, 2, 16, AlphaScaleCase::ONE>(
                    const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
                    svcount_t n_pred, [[maybe_unused]] float alpha, 
                    int lda, int ldb) 
        __arm_streaming __arm_inout("za") {

    svbool_t ptrue = svptrue_b32();

    // Force compiler to compute all addresses up front
    // Mark as const and use uintptr_t to prevent aliasing assumptions
    const uintptr_t a_base = (uintptr_t)a_ptr;
    const uintptr_t b_base = (uintptr_t)b_ptr;
    const uintptr_t lda_bytes = lda * sizeof(float);
    const uintptr_t ldb_bytes = ldb * sizeof(float);

    const float* __restrict a0 = (const float*)(a_base);
    const float* __restrict a1 = (const float*)(a_base + lda_bytes);
    const float* __restrict a2 = (const float*)(a_base + 2*lda_bytes);
    const float* __restrict a3 = (const float*)(a_base + 3*lda_bytes);
    const float* __restrict a4 = (const float*)(a_base + 4*lda_bytes);
    const float* __restrict a5 = (const float*)(a_base + 5*lda_bytes);
    const float* __restrict a6 = (const float*)(a_base + 6*lda_bytes);
    const float* __restrict a7 = (const float*)(a_base + 7*lda_bytes);
    const float* __restrict a8 = (const float*)(a_base + 8*lda_bytes);
    const float* __restrict a9 = (const float*)(a_base + 9*lda_bytes);
    const float* __restrict a10 = (const float*)(a_base + 10*lda_bytes);
    const float* __restrict a11 = (const float*)(a_base + 11*lda_bytes);
    const float* __restrict a12 = (const float*)(a_base + 12*lda_bytes);
    const float* __restrict a13 = (const float*)(a_base + 13*lda_bytes);
    const float* __restrict a14 = (const float*)(a_base + 14*lda_bytes);
    const float* __restrict a15 = (const float*)(a_base + 15*lda_bytes);

    const float* __restrict b0 = (const float*)(b_base);
    const float* __restrict b1 = (const float*)(b_base + ldb_bytes);
    const float* __restrict b2 = (const float*)(b_base + 2*ldb_bytes);
    const float* __restrict b3 = (const float*)(b_base + 3*ldb_bytes);
    const float* __restrict b4 = (const float*)(b_base + 4*ldb_bytes);
    const float* __restrict b5 = (const float*)(b_base + 5*ldb_bytes);
    const float* __restrict b6 = (const float*)(b_base + 6*ldb_bytes);
    const float* __restrict b7 = (const float*)(b_base + 7*ldb_bytes);
    const float* __restrict b8 = (const float*)(b_base + 8*ldb_bytes);
    const float* __restrict b9 = (const float*)(b_base + 9*ldb_bytes);
    const float* __restrict b10 = (const float*)(b_base + 10*ldb_bytes);
    const float* __restrict b11 = (const float*)(b_base + 11*ldb_bytes);
    const float* __restrict b12 = (const float*)(b_base + 12*ldb_bytes);
    const float* __restrict b13 = (const float*)(b_base + 13*ldb_bytes);
    const float* __restrict b14 = (const float*)(b_base + 14*ldb_bytes);
    const float* __restrict b15 = (const float*)(b_base + 15*ldb_bytes);

    // After address computation, before loads
    __builtin_prefetch(a0, 0, locality);  // Read, high temporal locality
    __builtin_prefetch(a1, 0, locality);
    __builtin_prefetch(a2, 0, locality);
    __builtin_prefetch(a3, 0, locality);
    __builtin_prefetch(a4, 0, locality);
    __builtin_prefetch(a5, 0, locality);
    __builtin_prefetch(a6, 0, locality);
    __builtin_prefetch(a7, 0, locality);
    
    __builtin_prefetch(b0, 0, locality);
    __builtin_prefetch(b1, 0, locality);
    __builtin_prefetch(b2, 0, locality);
    __builtin_prefetch(b3, 0, locality);
    __builtin_prefetch(b4, 0, locality);
    __builtin_prefetch(b5, 0, locality);
    __builtin_prefetch(b6, 0, locality);
    __builtin_prefetch(b7, 0, locality);

    // All loads together
    LOAD_AB_VECS(0, a0, b0, m_pred, n_pred)
    LOAD_AB_VECS(1, a1, b1, m_pred, n_pred)
    LOAD_AB_VECS(2, a2, b2, m_pred, n_pred)
    LOAD_AB_VECS(3, a3, b3, m_pred, n_pred)
    LOAD_AB_VECS(4, a4, b4, m_pred, n_pred)
    LOAD_AB_VECS(5, a5, b5, m_pred, n_pred)
    LOAD_AB_VECS(6, a6, b6, m_pred, n_pred)
    LOAD_AB_VECS(7, a7, b7, m_pred, n_pred)

    // All computes together
    MOPA_AB_VECS(0)
    MOPA_AB_VECS(1)
    MOPA_AB_VECS(2)
    MOPA_AB_VECS(3)
    MOPA_AB_VECS(4)
    MOPA_AB_VECS(5)
    MOPA_AB_VECS(6)
    MOPA_AB_VECS(7)

    __builtin_prefetch(a8, 0, locality); 
    __builtin_prefetch(a9, 0, locality);
    __builtin_prefetch(a10, 0, locality);
    __builtin_prefetch(a11, 0, locality);
    __builtin_prefetch(a12, 0, locality);
    __builtin_prefetch(a13, 0, locality);
    __builtin_prefetch(a14, 0, locality);
    __builtin_prefetch(a15, 0, locality);

    __builtin_prefetch(b8, 0, locality);
    __builtin_prefetch(b9, 0, locality);
    __builtin_prefetch(b10, 0, locality);
    __builtin_prefetch(b11, 0, locality);
    __builtin_prefetch(b12, 0, locality);
    __builtin_prefetch(b13, 0, locality);
    __builtin_prefetch(b14, 0, locality);
    __builtin_prefetch(b15, 0, locality);

    LOAD_AB_VECS(8, a8, b8, m_pred, n_pred)
    LOAD_AB_VECS(9, a9, b9, m_pred, n_pred)
    LOAD_AB_VECS(10, a10, b10, m_pred, n_pred)
    LOAD_AB_VECS(11, a11, b11, m_pred, n_pred)
    LOAD_AB_VECS(12, a12, b12, m_pred, n_pred)
    LOAD_AB_VECS(13, a13, b13, m_pred, n_pred)
    LOAD_AB_VECS(14, a14, b14, m_pred, n_pred)
    LOAD_AB_VECS(15, a15, b15, m_pred, n_pred)

    MOPA_AB_VECS(8)
    MOPA_AB_VECS(9)
    MOPA_AB_VECS(10)
    MOPA_AB_VECS(11)
    MOPA_AB_VECS(12)
    MOPA_AB_VECS(13)
    MOPA_AB_VECS(14)
    MOPA_AB_VECS(15)
}

// template<>
// void inline gemm_sme2_microkernel<float, 2, 2, 16, AlphaScaleCase::ONE>(
//                     const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
//                     svcount_t n_pred, [[maybe_unused]] float alpha, 
//                     int lda, int ldb) 
//         __arm_streaming __arm_inout("za") {

//     svbool_t ptrue = svptrue_b32();

//     // Pre-compute all addresses - removes dependency chain
//     const float* a_ptr_0 = a_ptr;
//     const float* a_ptr_1 = a_ptr + lda;
//     const float* a_ptr_2 = a_ptr + 2*lda;
//     const float* a_ptr_3 = a_ptr + 3*lda;
//     const float* a_ptr_4 = a_ptr + 4*lda;
//     const float* a_ptr_5 = a_ptr + 5*lda;
//     const float* a_ptr_6 = a_ptr + 6*lda;
//     const float* a_ptr_7 = a_ptr + 7*lda;
//     const float* a_ptr_8 = a_ptr + 8*lda;
//     const float* a_ptr_9 = a_ptr + 9*lda;
//     const float* a_ptr_10 = a_ptr + 10*lda;
//     const float* a_ptr_11 = a_ptr + 11*lda;
//     const float* a_ptr_12 = a_ptr + 12*lda;
//     const float* a_ptr_13 = a_ptr + 13*lda;
//     const float* a_ptr_14 = a_ptr + 14*lda;
//     const float* a_ptr_15 = a_ptr + 15*lda;

//     const float* b_ptr_0 = b_ptr;
//     const float* b_ptr_1 = b_ptr + ldb;
//     const float* b_ptr_2 = b_ptr + 2*ldb;
//     const float* b_ptr_3 = b_ptr + 3*ldb;
//     const float* b_ptr_4 = b_ptr + 4*ldb;
//     const float* b_ptr_5 = b_ptr + 5*ldb;
//     const float* b_ptr_6 = b_ptr + 6*ldb;
//     const float* b_ptr_7 = b_ptr + 7*ldb;
//     const float* b_ptr_8 = b_ptr + 8*ldb;
//     const float* b_ptr_9 = b_ptr + 9*ldb;
//     const float* b_ptr_10 = b_ptr + 10*ldb;
//     const float* b_ptr_11 = b_ptr + 11*ldb;
//     const float* b_ptr_12 = b_ptr + 12*ldb;
//     const float* b_ptr_13 = b_ptr + 13*ldb;
//     const float* b_ptr_14 = b_ptr + 14*ldb;
//     const float* b_ptr_15 = b_ptr + 15*ldb;

//     // Issue all loads first - hardware can prefetch aggressively
//     LOAD_AB_VECS(0, a_ptr_0, b_ptr_0, m_pred, n_pred)
//     LOAD_AB_VECS(1, a_ptr_1, b_ptr_1, m_pred, n_pred)
//     LOAD_AB_VECS(2, a_ptr_2, b_ptr_2, m_pred, n_pred)
//     LOAD_AB_VECS(3, a_ptr_3, b_ptr_3, m_pred, n_pred)
//     LOAD_AB_VECS(4, a_ptr_4, b_ptr_4, m_pred, n_pred)
//     LOAD_AB_VECS(5, a_ptr_5, b_ptr_5, m_pred, n_pred)
//     LOAD_AB_VECS(6, a_ptr_6, b_ptr_6, m_pred, n_pred)
//     LOAD_AB_VECS(7, a_ptr_7, b_ptr_7, m_pred, n_pred)
//     LOAD_AB_VECS(8, a_ptr_8, b_ptr_8, m_pred, n_pred)
//     LOAD_AB_VECS(9, a_ptr_9, b_ptr_9, m_pred, n_pred)
//     LOAD_AB_VECS(10, a_ptr_10, b_ptr_10, m_pred, n_pred)
//     LOAD_AB_VECS(11, a_ptr_11, b_ptr_11, m_pred, n_pred)
//     LOAD_AB_VECS(12, a_ptr_12, b_ptr_12, m_pred, n_pred)
//     LOAD_AB_VECS(13, a_ptr_13, b_ptr_13, m_pred, n_pred)
//     LOAD_AB_VECS(14, a_ptr_14, b_ptr_14, m_pred, n_pred)
//     LOAD_AB_VECS(15, a_ptr_15, b_ptr_15, m_pred, n_pred)

//     // Then compute - by now loads should be complete or in flight
//     MOPA_AB_VECS(0)
//     MOPA_AB_VECS(1)
//     MOPA_AB_VECS(2)
//     MOPA_AB_VECS(3)
//     MOPA_AB_VECS(4)
//     MOPA_AB_VECS(5)
//     MOPA_AB_VECS(6)
//     MOPA_AB_VECS(7)
//     MOPA_AB_VECS(8)
//     MOPA_AB_VECS(9)
//     MOPA_AB_VECS(10)
//     MOPA_AB_VECS(11)
//     MOPA_AB_VECS(12)
//     MOPA_AB_VECS(13)
//     MOPA_AB_VECS(14)
//     MOPA_AB_VECS(15)
// }

// template<>
// void inline gemm_sme2_microkernel<float, 2, 2, 16, AlphaScaleCase::ONE>(
//                     const float* a_ptr, const float* b_ptr, svcount_t m_pred, 
//                     svcount_t n_pred, [[maybe_unused]] float alpha, 
//                     [[maybe_unused]] int lda, [[maybe_unused]] int ldb) 
//         __arm_streaming __arm_inout("za") {

//     svbool_t ptrue = svptrue_b32();

//     LOAD_AB_VECS(0, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     LOAD_AB_VECS(1, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(0)

//     LOAD_AB_VECS(2, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(1)

//     LOAD_AB_VECS(3, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(2)

//     LOAD_AB_VECS(4, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(3)

//     LOAD_AB_VECS(5, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(4)

//     LOAD_AB_VECS(6, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(5)

//     LOAD_AB_VECS(7, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(6)

//     LOAD_AB_VECS(8, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(7)

//     LOAD_AB_VECS(9, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(8)

//     LOAD_AB_VECS(10, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(9)

//     LOAD_AB_VECS(11, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(10)

//     LOAD_AB_VECS(12, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(11)

//     LOAD_AB_VECS(13, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(12)

//     LOAD_AB_VECS(14, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(13)

//     LOAD_AB_VECS(15, a_ptr, b_ptr, m_pred, n_pred)
//     a_ptr += lda;
//     b_ptr += ldb;

//     MOPA_AB_VECS(14)

//     MOPA_AB_VECS(15)
// }
/* ... rest of micro-kernel implementations ... */


// Kernel function implementation

// Assumption: MRxNR is a square multiple of the tile size for the specified data type
// (i.e. MR is a multiple of elems_in_vec*R and NR is a multiple of elems_in_vec*C where
// RxC defines the tile-wise dimensions of the micro-kernel we will use)
// NOTE: We need leading dims for B, C matrices since these are optionally packed
template<typename DTYPE, int MR, int NR, int SVL, int UNROLL, int R, int C, AlphaScaleCase ALPHA_CASE, BetaScaleCase BETA_CASE>
static void inline gemm_sme2_kernel(int mr, int nr, int kc, const DTYPE* a_ptr, const DTYPE* b_ptr, 
                                            DTYPE* c_ptr, DTYPE alpha, DTYPE beta, int ldb, int ldc) 
    __arm_streaming __arm_inout("za") {

    constexpr int ELEMS_PER_VEC = SVL / (sizeof(DTYPE));
    constexpr int M_TILE_SIZE = ELEMS_PER_VEC * R;
    constexpr int N_TILE_SIZE = ELEMS_PER_VEC * C;
    static_assert(R * C <= sizeof(DTYPE), "Not enough ZA tiles for requested configuration");

    svcount_t all_true = svptrue_c32();
    svcount_t m_pred, n_pred;

    for (int m_tile = 0; m_tile < mr; m_tile += M_TILE_SIZE) {
        // define A block predicate as all true or set to partially true predicate if on last loop
        int m_remaining = mr - m_tile;
        if (m_remaining < M_TILE_SIZE) {
            m_pred = svwhilelt_c32_u64(0, m_remaining, 2);
        }
        else {
            m_pred = all_true;
        }

        for (int n_tile = 0; n_tile < nr; n_tile += N_TILE_SIZE) {
            // define B block predicate as all true or set to partially true predicate if on last loop
            int n_remaining = nr - n_tile;
            if (n_remaining < N_TILE_SIZE) {
                n_pred = svwhilelt_c32_u64(0, n_remaining, 2);
            }
            else {
                n_pred = all_true;
            }

            // load in C to the ZA tiles and scale if applicable
            gemm_sme2_load_za<DTYPE, BETA_CASE, R, C, ELEMS_PER_VEC>(c_ptr+m_tile*ldc+n_tile, m_remaining, n_remaining, beta, ldc);

#if DEBUG
            std::cout << "Load ZA: " << std::endl;
            std::cout << "------------------------------------------" << std::endl;
            dump_za<DTYPE, SVL, R, C>();
            std::cout << std::endl;
#endif // DEBUG

            // In the below K-loop we compute a M_TILE_SIZE x N_TILE_SIZE block to a depth of k
            int k = 0;
            for (; k+UNROLL < kc; k += UNROLL) {
                gemm_sme2_microkernel<DTYPE, R, C, UNROLL, ALPHA_CASE>(a_ptr+k*mr+m_tile, b_ptr+k*ldb+n_tile, m_pred, n_pred, alpha, mr, ldb);
#if DEBUG
                std::cout << "k: " << k << std::endl;
                std::cout << "------------------------------------------" << std::endl;
                dump_za<DTYPE, SVL, R, C>();
                std::cout << std::endl;
#endif // DEBUG
            }
            for (; k < kc; k++) {
                gemm_sme2_microkernel<DTYPE, R, C, 1, ALPHA_CASE>(a_ptr+k*mr+m_tile, b_ptr+k*ldb+n_tile, m_pred, n_pred, alpha, mr, ldb);
#if DEBUG
                std::cout << "k (tail): " << k << std::endl;
                std::cout << "------------------------------------------" << std::endl;
                dump_za<DTYPE, SVL, R, C>();
                std::cout << std::endl;
#endif // DEBUG
            }

            // store ZA storage back to the C block
            gemm_sme2_store_za<DTYPE, R, C, ELEMS_PER_VEC, ZA_Dim::H>(c_ptr+m_tile*ldc+n_tile, m_remaining, n_remaining, ldc);
#if DEBUG
            std::cout << "Resulting section of C after ZA store: " << std::endl;
            std::cout << "------------------------------------------" << std::endl;
            display_matrix<DTYPE>(c_ptr+m_tile*ldc+n_tile, std::min(m_remaining, M_TILE_SIZE), std::min(n_remaining, N_TILE_SIZE));
            std::cout << std::endl;
#endif // DEBUG
        }
    }
}

/*
    ----------------------------- SME2 GEMM -----------------------------
    This GEMM computes C <- {alpha}*AB + {beta}*C to a depth of KC.
    ---------------------------------------------------------------------

    Description:

    This function performs the computation outlined in the title and aims
    to make optimal use of memory and SME2 hardware. It does this by blocking
    the matrix multiplication, scaling, and accumulation in two stages. At
    the kernel stage we compute blocks of the matrices such that they fit in
    fast and local memory (such as the cache closest to the SME hardware).
    The kernel block is further broken down into micro-kernel blocks which
    compute sub-blocks of each kernel block such that each sub-block fits
    nicely inside of the SME hardware, fully leveraging it's ability to 
    parallelize outer-product accumlate operations on vectors. Ultimately,
    this structure aims to maximize compute ops per cycle as well as compute
    ops per time spent on memory ops.

    Supported Data-Types:
        * 32b Float

    Compile-Time Parameters:
        * DTYPE:    Data type being processed, all data inputs should be of 
                    the same underlying type

        * MR, NR:   Given in number of elements, MRxNR blocks of C are computed 
                    by the GEMM kernel (MR rows of A, NR cols of B)^[1][2]

        * SVL:      Streaming Vector Length in Bytes (this can technically 
                    change at runtime, but defining at compile time allows 
                    for more effective blocking)

        * UNROLL:   Number of inner-loops to unroll inside the micro-kernel

        * R, C:     Describes the shape of the tiles in the SME used by the
                    micro-kernel inside the kernel^[3]

        * ALPHA_CASE: {alpha} is a scalar value that is either 1 or some other
                        non-zero value^[4]

        * BETA_CASE: {beta} is a scalar value that is 0, 1, or some other
                        non-zero value^[4]

    Run-Time Parameters:
        * a: Matrix A data pointer, stored row-major
        * b: Matrix B data pointer, stored row-major
        * c: Matrix C data pointer, stored row-major
        * m, n, k: A is mxk, B is kxn, C is mxn
        * kc: depth of kernel blocks to process [2]
        * alpha, beta: scaling factors


    NOTES: 
        [1] MR, NR are required to be multiples of SVL / sizeof(DTYPE) in order
            to simplify kernel logic (avoiding branching slowdowns) and ensure
            maximal use of ZA per outer-product.

        [2] MR, NR, and kc should be chosen such that an MRxKC block of A, 
            KCxNR block of B, and MRxNR block of C fit into SME adjacent cache.

        [3] Viable RxC geometries are determined by what is available in the 
            micro-kernel API and further limited by what is possible in hardware. 
            The SME supports square tiles; let E = SVL/sizeof(DTYPE) then each tile
            is an ExE square of values and there are (SVL/8)/E squares. For example,
            an SVL of 512b using 32b floats would have 4 16x16 element tiles, so 
            possible micro-kernel shapes are 2x2, 1x4, 4x1, though only a subset may
            be implemented.

        [4] Knowing the value category of alpha and beta at compile time allows for
            more efficient load/store and computation routines within the GEMM kernel.
*/

// ISSUE: Add static asserts and requires () clauses to top level functions to ensure correct template parameterization
// ISSUE: Add a few runtime asserts at the top level to ensure the compile-time parameters line up with the run-time parameters
// ISSUE: We could create a wrapper around this function that automatically chooses the ScalingCase at runtime

// ISSUE: We could make the matrix packing more efficient by having a rotation of tile buffers and dispatching separate threads
//        to load the buffers for the next step while computing the current step


// pack rxc section of src (assuming src is row-major) into dst
// 
#include <cstring>
template<typename DTYPE>
static void inline pack_normal(const DTYPE* src, DTYPE* dst, int r, int c, int ldim) __arm_streaming_compatible __arm_preserves("za") {
    for (int i = 0; i < r; ++i) {
        // __arm_sc_memcpy((void *)(dst+i*ldim), (const void *)(src+i*c), (size_t)(sizeof(DTYPE)*c));
        std::memcpy( (void *)(dst+i*ldim), (const void *)(src+i*c), (size_t)(sizeof(DTYPE)*c) );
    }
}


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

template<typename DTYPE, int MR, int NR, int SVL, int UNROLL, int R, int C, 
            AlphaScaleCase ALPHA_CASE, BetaScaleCase BETA_CASE, bool PACK_B, bool PACK_C>
__arm_new("za") __arm_locally_streaming void gemm_sme2(const DTYPE* a, const DTYPE* b, DTYPE* c, 
                int m, int n, int k, int kc, DTYPE alpha, DTYPE beta, [[maybe_unused]] DTYPE* a_buf, 
                [[maybe_unused]] DTYPE* b_buffer, [[maybe_unused]] DTYPE* c_buffer) 
{
    // ISSUE: static assert that MR, NR are proper size w/ respect to SVL and sizeof(DTYPE)
    // ISSUE: assert that alpha and beta match ALPHA_CASE and BETA_CASE
    // Compile time compute proper R, C, UNROLL, or take in as compile timer parameters

    int ldb;
    std::conditional_t<PACK_B, DTYPE*, const DTYPE*> b_buf;
    if constexpr (PACK_B) {
        b_buf = b_buffer;
    }
    else {
        b_buf = b;
        ldb = n;
    }

    int ldc;
    DTYPE* c_buf;
    if constexpr (PACK_C) {
        c_buf = c_buffer;
    }
    else {
        c_buf = c;
        ldc = n;
    }

    for (int k_block = 0; k_block < k; k_block += kc) {
        int kr = std::min(kc, k - k_block);

        for (int m_block = 0; m_block < m; m_block += MR) {
            int mr = std::min(MR, m - m_block);
            // Pack MRxKC block of A column-major into a_buf
            // NOTE: We shouldn't worry about zeroing edge cases, these are not included in computation or when storing back to memory
            auto a_base = a + k*m_block + k_block;
            // pack<float>(a_base, a_buf, mr, kr, k);
            pack_sme<DTYPE, SVL, R, C, true>(a_base, a_buf, mr, kr, k);

            for (int n_block = 0; n_block < n; n_block += NR) {
                int nr = std::min(NR, n - n_block);
                auto b_base = b + n*k_block + n_block;
                if constexpr (PACK_B) {
                    // for (int i = 0; i < kr; ++i) {
                    //     for (int j = 0; j < nr; ++j) {
                    //         b_buf[i*nr + j] = b_base[i*n + j];
                    //     }
                    // }
                    // pack_normal<DTYPE>(b_base, b_buf, kr, nr, n);
                    pack_sme<DTYPE, SVL, R, C, false>(b_base, b_buf, kr, nr, n);
                    ldb = nr;
                }
                else {
                    b_buf = b_base;
                }

                // set c_ptr to upper left of target block of C
                DTYPE* c_base = c + m_block*n + n_block;
                if constexpr (PACK_C) {
                    for (int i = 0; i < mr; ++i) {
                        for (int j = 0; j < nr; ++j) {
                            c_buf[i*nr + j] = c_base[i*n + j];
                        }
                    }
                    ldc = nr;
                }
                else {
                    c_buf = c_base;
                }

                // ISSUE: Need to make sure these branches in ZERO and OTHER cases get optimized away or at least out of the inner loop
                if constexpr (BETA_CASE == BetaScaleCase::ZERO) {
                    if (k_block == 0) {
                        gemm_sme2_kernel<DTYPE, MR, NR, SVL, UNROLL, R, C, ALPHA_CASE, BetaScaleCase::ZERO>(mr, nr, kr, a_buf, b_buf, c_buf, alpha, beta, ldb, ldc);
                    }
                    else {
                        gemm_sme2_kernel<DTYPE, MR, NR, SVL, UNROLL, R, C, ALPHA_CASE, BetaScaleCase::ONE>(mr, nr, kr, a_buf, b_buf, c_buf, alpha, beta, ldb, ldc);
                    }
                } 
                else if constexpr (BETA_CASE == BetaScaleCase::OTHER) {
                    if (k_block == 0) {
                        gemm_sme2_kernel<DTYPE, MR, NR, SVL, UNROLL, R, C, ALPHA_CASE, BetaScaleCase::OTHER>(mr, nr, kr, a_buf, b_buf, c_buf, alpha, beta, ldb, ldc);
                    }
                    else {
                        gemm_sme2_kernel<DTYPE, MR, NR, SVL, UNROLL, R, C, ALPHA_CASE, BetaScaleCase::ONE>(mr, nr, kr, a_buf, b_buf, c_buf, alpha, beta, ldb, ldc);
                    }
                }
                else {
                    gemm_sme2_kernel<DTYPE, MR, NR, SVL, UNROLL, R, C, ALPHA_CASE, BETA_CASE>(mr, nr, kr, a_buf, b_buf, c_buf, alpha, beta, ldb, ldc);
                }

                // Need to unpack C results if packed
                if constexpr (PACK_C) {
                    for (int i = 0; i < mr; ++i) {
                        for (int j = 0; j < nr; ++j) {
                            c_base[i*n + j] = c_buf[i*NR + j];
                        }
                    }
                }

#if DEBUG
                // std::cout << "(n_block: " << n_block << ") (k_block: " << k_block << ") (m_block: " << m_block << ")" << std::endl;
                // std::cout << "------------------------------------------" << std::endl;
                // display_matrix<DTYPE>(c, m, n);
                // std::cout << std::endl;
#endif // DEBUG
            }
        }
    }
}

#endif // __ARM_FEATURE_SME
#endif // SME2_GEMM_IMPL_H