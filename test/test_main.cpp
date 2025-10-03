#include <stdlib.h>
#include "sme2_gemm.h"
#include <iostream>
#include <iomanip>
#include "gemm_truth.h"

#define KB (1 << 10)

#define MR 1024
#define NR 1024
#define SVL 64 // NOTE: Apple M2 supports only 512b=64B SVL
#define UNROLL 8
#define R 2
#define C 2
#define ALPHA_CASE AlphaScaleCase::ONE
#define BETA_CASE BetaScaleCase::ONE
#define PACK_B true
#define PACK_C false

int main() {
	int m = 8*KB, n = 8*KB, k = 8*KB;
	float* a = (float*) malloc(sizeof(float) * m * k);
	assert(nullptr != a);

	float* b = (float*) malloc(sizeof(float) * k * n);
	assert(nullptr != b);

	float* c = (float*) malloc(sizeof(float) * m * n);
	assert(nullptr != c);

	// float* a_truth = (float*) malloc(sizeof(float) * m * k);
	// assert(nullptr != a_truth);

	// float* b_truth = (float*) malloc(sizeof(float) * k * n);
	// assert(nullptr != b_truth);

	// float* c_truth = (float*) malloc(sizeof(float) * m * n);
	// assert(nullptr != c_truth);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			a[k*i+j] = 2;
			// a_truth[k*i+j] = 2;
		}
	}

	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < n; ++j) {
			b[n*i+j] = 0.5;
			// b_truth[n*i+j] = 0.5;
		}
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			c[n*i+j] = 1;
			// c_truth[n*i+j] = 1;
		}
	}

	int kc = 512;
	float alpha = 1.0;
	float beta = 1.0;

	float* a_buf = (float*) malloc(kc * MR * sizeof(float));
    assert(nullptr != a_buf);
    float* b_buf = (float*) malloc(kc * NR * sizeof(float));
    assert(nullptr != b_buf);
    // float* c_buf = (float*) malloc(MR * NR * sizeof(float));
    // assert(nullptr != c_buf);


 	gemm_sme2<float, MR, NR, SVL, UNROLL, R, C, ALPHA_CASE, BETA_CASE, PACK_B, PACK_C>(a, b, c, m, n, k, kc, alpha, beta, a_buf, b_buf, nullptr);

	// gemm_truth<float>(a_truth, b_truth, c_truth, m, n, k, alpha, beta);

	// std::cout << "SME2_GEMM OUTPUT:" << std::endl;
 	// for (int i = 0; i < m; ++i) {
	// 	for (int j = 0; j < n; ++j) {
	// 		std::cout << std::setw(2) << c[n*i+j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl;
	// std::cout << std::endl;

	// std::cout << "GEMM_TRUTH OUTPUT:" << std::endl;
 	// for (int i = 0; i < m; ++i) {
	// 	for (int j = 0; j < n; ++j) {
	// 		std::cout << std::setw(2) << c_truth[n*i+j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

 	free(a);
 	free(b);
 	free(c);

 	// free(a_truth);
 	// free(b_truth);
 	// free(c_truth);

 	free(a_buf);
 	free(b_buf);
 	// free(c_buf);

 	return 0;
}