#pragma once
// SafeC Standard Library — BLAS-accelerated Tensor matmul.
//
// tensor.sc's tensor_matmul is a portable, dependency-free triple loop --
// deliberately so (see tensor.h's own header comment: no framework
// dependency baked into the portable core). Real BLAS libraries
// (Accelerate on macOS, OpenBLAS/MKL elsewhere) are cache-blocked,
// hand-vectorized, and multithreaded in ways a compiler-auto-vectorized
// triple loop doesn't match — see the Benchmarks page. This file closes
// that gap by calling the same underlying library PyTorch does (cblas_sgemm,
// float32 like Tensor itself), instead of re-implementing a hand-tuned
// GEMM from scratch.
//
// The products go through std/simd/gemm.h's sgemm: any standard CBLAS the
// program links (Accelerate on macOS, AOCL-BLAS, oneMKL or OpenBLAS
// elsewhere; '-framework Accelerate' / '-lblis' / '-lopenblas' ...) takes
// the large ones, and without one sgemm's own vectorized, threaded kernel
// runs -- so linking a BLAS is optional, not required. The one Apple-only
// piece, vDSP_vadd in the backward pass, has a portable fallback elsewhere
// (see tensor_blas.sc). There is no runtime "unavailable" case the way GPU
// backends have one.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_matmul_blas(const &Tensor a, const &Tensor b);

} // namespace std
