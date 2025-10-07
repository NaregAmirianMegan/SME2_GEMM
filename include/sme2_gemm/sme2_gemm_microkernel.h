#ifndef SME2_GEMM_MICROKERNEL_H
#define SME2_GEMM_MICROKERNEL_H

/*

==============================================================================
MICRO-KERNEL SPECIALIZATIONS
==============================================================================

This file contains micro-kernel implementations for various geometric 
arrangments of the ZA tiles, data types, and amount of loop unrolling.

The micro-kernel computes a part of a matrix-multiply that can be done 
entirely out of ZA-storage within the SME. 

==============================================================================

*/


#include <arm_sme.h>
#include <stdlib.h>
#include <algorithm>
#include <cassert>

#include "sme2_gemm_config.h"


#ifdef __ARM_FEATURE_SME

template<typename DTYPE, int R, int C, int UNROLL, AlphaScaleCase ALPHA_CASE>
static void inline gemm_sme2_microkernel(const DTYPE* a_ptr, const DTYPE* b_ptr, 
                                            svcount_t m_pred, svcount_t n_pred, [[maybe_unused]] DTYPE alpha, [[maybe_unused]] int lda, [[maybe_unused]] int ldb) 
    __arm_streaming __arm_inout("za");


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



#endif // __ARM_FEATURE_SME
#endif // SME2_GEMM_MICROKERNEL_H