#pragma once
// SafeC Standard Library — single-precision matrix multiply on the CPU.
//
// sgemm is a cache-blocked GEMM: A and B are packed into contiguous panels
// and multiplied by a 4 x 16 register-tile kernel on vec<float, 8>, so it
// uses whatever vector unit the target has (SSE2/AVX on x86-64, NEON on
// AArch64, WebAssembly SIMD128, or scalar code elsewhere). Large products
// are split into tiles of C that run on std/sync/parallel.h's thread pool;
// small ones run on the calling thread.
//
// Matrices are row-major; 'ld' is the distance in elements between the
// starts of two rows (>= the row length).
#include <std/simd/simd.h>

namespace std {

#define SGEMM_NO_TRANS 0
#define SGEMM_TRANS    1

// C = alpha * op(A) * op(B) + beta * C, where op(A) is m x k and op(B) is
// k x n, and op(X) = X, or X transposed when its trans flag is SGEMM_TRANS
// (then X is stored k x m / n x k). beta == 0 overwrites C without reading
// it, so C may start uninitialized.
void sgemm(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
           float alpha, const &float a, unsigned long lda,
           const &float b, unsigned long ldb,
           float beta, &float c, unsigned long ldc);

// sgemm, but always on the calling thread (no pool, no wake-ups): for code
// that is already parallel, or that must not block on other threads.
void sgemm_serial(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
                  float alpha, const &float a, unsigned long lda,
                  const &float b, unsigned long ldb,
                  float beta, &float c, unsigned long ldc);

} // namespace std
