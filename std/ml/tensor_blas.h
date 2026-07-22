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
// Any standard CBLAS (Accelerate on macOS, OpenBLAS/MKL elsewhere) —
// cblas_sgemm's C ABI is identical across all of them, verified against
// real OpenBLAS builds on Linux and Windows in addition to Accelerate on
// macOS (link with '-framework Accelerate' there, '-lopenblas'
// elsewhere). The one genuinely Apple-only piece, vDSP_vadd in the
// backward pass, has a portable fallback on non-Apple targets — see
// tensor_blas.sc. Every function here falls back to tensor.sc's
// tensor_matmul internally only in the sense that it computes the
// identical result via a different, faster implementation; there's no
// runtime "unavailable" case the way GPU backends have one (whichever
// CBLAS the program links against is statically/dynamically linked, not
// optional hardware — if the program links it at all, it's there).
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_matmul_blas(const &Tensor a, const &Tensor b);

} // namespace std
