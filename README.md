# SME2_GEMM

## Background
SME2 (Scalable Matrix Extension) and SSVE (Streaming Scalable Vector Extension) are features added to the ARM architecture in ARMv9-A (see ARM overview here: https://developer.arm.com/documentation/109246/0100/SME-Overview/SME-and-SME2). In short these two hardware features provide wide (length adustable) vector and matrix storage registers, and allow for 2D parellelized operations (like outer-product). This codebase is an attempt to implement BLAS operations tuned for SME2/SSVE using compiler intrinsics.

## Approach
Currently a two phase tiling approach is used. The first level of tiling is meant to allow cache residency of all data involved in further computations. The second level of tiling is aligned to the size and shape of the SME ZA block (2D computation matrix). 

        A (m × k)                                       B (k × n)
   ┌──────────────────────────┐           ┌──────────────┐
   │                          │           │              │
   │   ┌──────────────┐       │           │   ┌──────┐   │
   │   │   mr × kc    │       │    ×      │   │      │   │
   │   └──────────────┘       │           │   │ kc×nr│   │
   │                          │           │   │      │   │
   └──────────────────────────┘           │   └──────┘   │
                                          │              │
                                          └──────────────┘
                │                                           │
                └────────────────────►──────────────────────┘
                                     =
                               C (m × n)
                       ┌────────────────────┐
                       │                    │
                       │   ┌────────┐       │
                       │   │mr × nr │       │
                       │   └────────┘       │
                       │                    │
                       └────────────────────┘

Formula:  C_block(mr×nr) = A_block(mr×kc) × B_block(kc×nr)


## Status

This project is in early stages, and currently sits at about 60% utilization of the theoretical peak on an Apple M4 chip. Contributions or suggestions are welcome.

## Project Structure
- `include` contains much of the actual implementations, since this code is highly reliant on C++ templates
- `src` contains additional implementation outside of templates
- `sme_ops` contains compiler/target arch specific implementations for SME2 macros
- `params` contains hardware and algorithmic parameters for different hardware targets
- `test` contains correctness testing against verified BLAS
- `benchmark` contains a benchmarking suite using Google Benchmark

## Build
- Apple system Clang works best, LLVM Clang optimizations sometimes don't play well with Apple hardware.
- If using LLVM Clang you must use LLD (LLVM Linker), as the Apple system libraries don't define some optimizations in the way LLVM LLD expects (linkage may fail)
- TODO: Explicit build information and notes (require at least C++ 14 as some features like `std::integer_sequence`, `std::conditional_t` are critical)

## To Do
- Improve micro-kernel efficiency to achieve > `2.5 TFLOPS` for FP32 single thread performance. Currently the bottleneck on large matrices appears to be the micro-kernel, which implies we are not
  overlapping mem-ops and compute-ops as well as we could be.
- Expand implementation to other data types (FP64, BF16, INTs)
- Expand robustness of benchmarks and test suite

## Notes
- At times this codebase pursposefully omits runtime checks and typical C++ standard practice with regards to memory safety in order to reduce slowdowns.

