#ifndef SME2_GEMM_H
#define SME2_GEMM_H

#include "sme2_gemm_impl.h"

// Forward declaration
template<typename DTYPE, int MR, int NR, int SVL, int UNROLL, int R, int C, 
            AlphaScaleCase ALPHA_CASE, BetaScaleCase BETA_CASE, bool PACK_B, bool PACK_C>
__arm_locally_streaming void gemm_sme2(const DTYPE* a, const DTYPE* b, DTYPE* c, 
               int m, int n, int k, int kc, DTYPE alpha, DTYPE beta, DTYPE* a_buf);

#endif // SME2_GEMM_H
