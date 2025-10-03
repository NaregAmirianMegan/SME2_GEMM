
# Project Structure:
- src contains BLAS implementations
- sme_ops contains compiler/target arch specific implementations for SME2 macros
- params contains hardware and algorithmic parameters for different hardware targets
- test contains correctness testing against verified BLAS
- benchmark contains benchmarking suite

# Build
- NOTE: Apple system Clang works best, LLVM Clang optimizations sometimes don't play well with Apple hardware.
- If using LLVM Clang you must use LLD (LLVM Linker), as the Apple system libraries don't define some optimizations in the way LLVM LLD expects (linkage may fail)
- TODO: Explicit build information and notes (require at least C++ 14 as some features like std::integer_sequence, std::conditional_t are critical)

# Notes
- This codebase pursposefully omits runtime checks and typical C++ standard practice with regards to memory safety in order to reduce slowdowns.

