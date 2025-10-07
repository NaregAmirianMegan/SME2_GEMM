#ifndef SME2_GEMM_H
#define SME2_GEMM_H

/*

==============================================================================
SME2 GEMM
==============================================================================

Top level implementation for the GEMM operation.

==============================================================================

*/


#include <type_traits>

#include "sme2_gemm_kernel.h"
#include "sme2_rearrange.h"
#include "sme2_gemm_config.h"


#ifdef __ARM_FEATURE_SME


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
#endif // SME2_GEMM_H
