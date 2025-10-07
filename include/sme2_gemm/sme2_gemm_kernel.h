#ifndef SME2_GEMM_KERNEL_H
#define SME2_GEMM_KERNEL_H

/*

==============================================================================
GEMM KERNEL
==============================================================================

Defines the GEMM kernel. Computes a tiled segment of a matrix multiply
meant to fit in compute adjacent cache.  

==============================================================================

*/



#include <arm_sme.h>

#include "sme2_za_ops.h"
#include "sme2_gemm_microkernel.h"
#include "sme2_gemm_config.h"
#include "debug_util.h"



#ifdef __ARM_FEATURE_SME


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



#endif // __ARM_FEATURE_SME
#endif // SME2_GEMM_KERNEL_H
