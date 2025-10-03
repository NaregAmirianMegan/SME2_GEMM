#include <iostream>

template<typename DTYPE>
static void gemm_truth(const DTYPE* a, const DTYPE* b, DTYPE* c, int m, int n, int k, DTYPE alpha, DTYPE beta) {
	for (int row = 0; row < m; ++row) {
		for (int col = 0; col < n; ++col) {
			DTYPE val = c[row*n+col] * beta;
			for (int i = 0; i < k; ++i) {
				val += a[row*k+i] * b[i*n+col];
			}
			c[row*n+col] = alpha*val;
		}
	}
}