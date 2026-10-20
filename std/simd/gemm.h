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
//
// Vendor BLAS: on hosted targets (not Windows yet), a program that links a
// CBLAS gets its kernels for sgemm's larger products: AOCL-BLAS on AMD
// CPUs, oneMKL on Intel, Accelerate on Apple. Except with Accelerate, std
// runs the library single-threaded and splits the product across its own
// thread pool (the libraries' OpenMP workers busy-wait between calls and
// would compete with the pool for cores); that setting applies to the
// program's own calls into the library too.
// safeguard's "blas" feature links the right one for the build machine;
// otherwise link one by hand (e.g. -lblis-mt, -lmkl_rt, -framework
// Accelerate). The choice is made at link time; with no BLAS linked the
// built-in kernel runs.
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

// sgemm, but always on the calling thread (no pool, no wake-ups; a vendor
// library, if linked, runs single-threaded): for code that is already
// parallel, or that must not block on other threads.
void sgemm_serial(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
                  float alpha, const &float a, unsigned long lda,
                  const &float b, unsigned long ldb,
                  float beta, &float c, unsigned long ldc);

// 1 if sgemm is using a linked vendor BLAS, and which one: "AOCL-BLAS
// (BLIS)", "oneMKL", "Accelerate", "OpenBLAS", "CBLAS" (another library) or
// "built-in".
int sgemm_vendor();
const char* sgemm_backend();

} // namespace std
