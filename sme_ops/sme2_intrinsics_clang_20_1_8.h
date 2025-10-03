/* 
       Implementation Specific Macros:
       Compiler: Homebrew clang version 20.1.8
       Target: arm64-apple-darwin24.5.0
*/

#ifdef __cplusplus
extern "C" {
#endif


// ---- Feature guards --------------------------------------------------------
#if !defined(__ARM_FEATURE_SVE)
#  error "This file requires SVE for predication (compile with -msve)."
#endif
#if !defined(__ARM_FEATURE_SME2)
#  warning "__ARM_FEATURE_SME2 not defined. Ensure target supports SME2 (e.g., -march=armv9.2-a+sme2)."
#endif

#include <arm_sve.h>
#include <arm_sme.h>

// ---- User-mappable SME2 intrinsic hooks -----------------------------------
// Replace these with your real intrinsics from <arm_sme.h>. They are kept as macros so you
// can quickly port to differing spellings across Clang/LLVM releases.

#ifndef SME_START
#  define SME_START()            __asm__ __volatile__("smstart\n")
#endif
#ifndef SME_STOP
#  define SME_STOP()             __asm__ __volatile__("smstop\n")
#endif
#ifndef SME_ZERO_ZA32
#  define SME_ZERO_ZA32()        __asm__ __volatile__("zero {za}\n")
#endif

// Load from packed A (mr floats) and B (nr floats) into z-registers under predicate.
// Real code should load enough z-regs to cover MR and NR; here we model a single vector path.
#ifndef SME_LD_A_F32
#  define SME_LD_A_F32(z, pg, ptr)  z = svld1(pg, ptr)
#endif
#ifndef SME_LD_B_F32
#  define SME_LD_B_F32(z, pg, ptr)  z = svld1(pg, ptr)
#endif

// ZA outer-product accumulate (FP32) with predicate
#ifndef SME_FMOPA_F32
// Map FMOPA (FP32) to Apple Clang’s wrapper.
// slice=0 targets the default ZA tile slice; if you tile ZA, pass your slice id instead.
#  define SME_FMOPA_F32(p_rows, p_cols, a_vec, b_vec) \
       svmopa_za32_m(0u, (p_rows), (p_cols), (a_vec), (b_vec))
#endif


// ZA scale by scalar alpha (fp32). In practice, iterate ZA rows/cols and fmul.
#ifndef SME_SCALE_ZA_F32
#  define SME_SCALE_ZA_F32(alpha) __asm__ __volatile__("// za *= alpha placeholder\n")
#endif

// Accumulate beta*C into ZA (predicate-safe store/load path). Replace with ZA read + FMLA.
#ifndef SME_ZA_FMLA_WITH_C_F32
#  define SME_ZA_FMLA_WITH_C_F32(Cptr, ldc, beta, mr, nr, svl_bytes) __asm__ __volatile__("// ZA += beta*C placeholder\n")
#endif

// Store ZA tile to C with predicates for tails. Replace with ZA row extraction + svst1(pg,...).
#ifndef SME_STORE_ZA_TO_C_F32
#  define SME_STORE_ZA_TO_C_F32(Cptr, ldc, mr, nr, svl_bytes) __asm__ __volatile__("// store ZA->C placeholder\n")
#endif

// Optional: cache prefetch hints
#ifndef SME_PRFM_PLDL1
#  define SME_PRFM_PLDL1(ptr)    __asm__ __volatile__("prfm pldl1keep, [%[p]]\n" :: [p]"r"(ptr))
#endif
#ifndef SME_PRFM_PSTL1
#  define SME_PRFM_PSTL1(ptr)    __asm__ __volatile__("prfm pstl1keep, [%[p]]\n" :: [p]"r"(ptr))
#endif


#ifdef __cplusplus
} // extern "C"
#endif
