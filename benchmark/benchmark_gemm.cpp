/*

Statistics to track:
- icache/dcache stats
- mem ops, compute ops
- instr per cycle
- total execution time

*/
#include <random>
#include <benchmark/benchmark.h>
#include "sme2_gemm.h"
#include "gemm_truth.h"


#define KB (1 << 10)

static void fill_rand(float* a, float* b, float* c, int m, int k, int n) {
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			a[i*k+j] = dis(gen);
		}
	}

	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < n; ++j) {
			b[i*n+j] = dis(gen);
		}
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			c[i*n+j] = dis(gen);
		}
	}
}


// static void BM_GEMM_SQUARE_MATRIX_FP32_NAIVE(benchmark::State& state) {
// 	// SETUP
// 	int n = state.range(0);
// 	float* a = (float*) malloc(sizeof(float) * n * n); assert(nullptr != a);
// 	float* b = (float*) malloc(sizeof(float) * n * n); assert(nullptr != b);
// 	float* c = (float*) malloc(sizeof(float) * n * n); assert(nullptr != c);

// 	fill_rand(a, b, c, n, n, n);

// 	float alpha = 1.0;
// 	float beta = 1.0;

// 	// RUN
// 	for (auto _ : state) {
// 		gemm_truth<float>(a, b, c, n, n, n, alpha, beta);
// 	}

// 	// REPORT ADDITIONAL
//     // state.SetComplexityN(n);
//     // state.counters["FLOPS"] = benchmark::Counter(
//     //     2.0 * n * n * n * state.iterations(),
//     //     benchmark::Counter::kIsRate
//     // );

// 	// CLEAN UP
// 	free(a); free(b); free(c);
// }
// BENCHMARK(BM_GEMM_SQUARE_MATRIX_FP32_NAIVE)
// 			->Args({1*KB})
// 			->Args({2*KB})
// 			->Args({4*KB})
// 			->Args({8*KB});


#define MR 256
#define NR 256
#define SVL 64 // NOTE: Apple M4 supports only 512b=64B SVL
#define UNROLL 8
#define R 2
#define C 2
#define ALPHA_CASE AlphaScaleCase::ONE
#define BETA_CASE BetaScaleCase::ONE
#define PACK_B true
#define PACK_C false
static void BM_GEMM_SQUARE_MATRIX_FP32_SME2(benchmark::State& state) {
	// SETUP
	int n = state.range(0);
	float* a = (float*) malloc(sizeof(float) * n * n); assert(nullptr != a);
	float* b = (float*) malloc(sizeof(float) * n * n); assert(nullptr != b);
	float* c = (float*) malloc(sizeof(float) * n * n); assert(nullptr != c);

	fill_rand(a, b, c, n, n, n);

	int kc = 512;
	float alpha = 1.0;
	float beta = 1.0;

	float* a_buf = (float*) malloc(kc * MR * sizeof(float));
    assert(nullptr != a_buf);
    float* b_buf = (float*) malloc(kc * NR * sizeof(float));
    assert(nullptr != b_buf);
    // float* c_buf = (float*) malloc(MR * NR * sizeof(float));
    // assert(nullptr != c_buf);

	// RUN
	for (auto _ : state) {
		gemm_sme2<float, MR, NR, SVL, UNROLL, R, C, ALPHA_CASE, BETA_CASE, PACK_B, PACK_C>(a, b, c, n, n, n, kc, alpha, beta, a_buf, b_buf, nullptr);
	}

	// REPORT ADDITIONAL
    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(
        2.0 * n * n * n * state.iterations(),
        benchmark::Counter::kIsRate
    );

	// CLEAN UP
	free(a); free(b); free(c); free(a_buf); free(b_buf);
}
BENCHMARK(BM_GEMM_SQUARE_MATRIX_FP32_SME2)
			->Args({256})
			->Args({1*KB})
			->Args({2*KB})
			->Args({4*KB})
			->Args({8*KB})
			->Args({16*KB})
			->Args({32*KB});

BENCHMARK_MAIN();

