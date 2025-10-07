#ifndef SME2_GEMM_CONFIG_H
#define SME2_GEMM_CONFIG_H

/*

==============================================================================
GEMM CONFIGURATION PARAMETERS FOR COMPILATION
==============================================================================

Defines useful enums, macros, ...

==============================================================================

*/


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

#endif // SME2_GEMM_CONFIG_H