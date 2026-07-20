#pragma once
// SafeC Standard Library — BLAS-accelerated Tensor matmul (Apple Accelerate).
//
// tensor.sc's tensor_matmul is a portable, dependency-free triple loop --
// deliberately so (see tensor.h's own header comment: no framework
// dependency baked into the portable core). Real BLAS libraries
// (Accelerate on macOS, OpenBLAS/MKL elsewhere) are cache-blocked,
// hand-vectorized, and multithreaded in ways a compiler-auto-vectorized
// triple loop doesn't match: measured on this machine, PyTorch's CPU
// matmul (itself Accelerate-backed on macOS) trains this codebase's MLP
// benchmark shape (128->256(relu)->64, batch 64) about 6x faster and
// runs inference about 36x faster than tensor.sc's tensor_matmul — see
// the Benchmarks page. This file closes that gap by calling the same
// underlying library PyTorch does, instead of re-implementing a
// hand-tuned GEMM from scratch.
//
// macOS/Accelerate only for now (matches gpu_mps.h's own precedent: one
// real, hand-verified backend rather than an unverified multi-platform
// stub set) — link with '-framework Accelerate'. Every function here
// falls back to tensor.sc's tensor_matmul internally only in the sense
// that it computes the identical result via a different, faster
// implementation; there's no runtime "unavailable" case the way GPU
// backends have one (Accelerate is a statically-linked framework, not
// optional hardware — if the program links it at all, it's there).
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_matmul_blas(const &Tensor a, const &Tensor b);

} // namespace std
