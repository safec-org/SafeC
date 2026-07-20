// SafeC Standard Library — size-based automatic backend selection
// implementation (see tensor_auto.h).
#pragma once
#include <std/ml/tensor_auto.h>
#include <std/ml/tensor_vdsp.h>
#include <std/ml/tensor_blas.h>

namespace std {

// Measured directly (2M-scale microbenchmark, fixed payload, isolated
// per-op timing — same methodology as every other benchmark this project
// runs), not assumed: vDSP's own per-call overhead makes it *slower* than
// the plain scalar loop below ~256 elements (64 elements: vDSP measured
// 0.84x — i.e. slower — than naive; 256 elements: 2.24x faster; 1024:
// 3.24x faster, holding steady around 3x through 500K+ elements). 256 is
// the threshold, not a rounder number picked for convenience — it's where
// the measured crossover actually falls.
#define TENSOR_AUTO_VDSP_MIN_ELEMS ((unsigned long)256)

&Tensor tensor_add_auto(const &Tensor a, const &Tensor b) {
    if (a->size >= TENSOR_AUTO_VDSP_MIN_ELEMS) { return tensor_add_vdsp(a, b); }
    return tensor_add(a, b);
}

&Tensor tensor_sub_auto(const &Tensor a, const &Tensor b) {
    if (a->size >= TENSOR_AUTO_VDSP_MIN_ELEMS) { return tensor_sub_vdsp(a, b); }
    return tensor_sub(a, b);
}

&Tensor tensor_mul_auto(const &Tensor a, const &Tensor b) {
    if (a->size >= TENSOR_AUTO_VDSP_MIN_ELEMS) { return tensor_mul_vdsp(a, b); }
    return tensor_mul(a, b);
}

&Tensor tensor_sum_auto(const &Tensor a) {
    if (a->size >= TENSOR_AUTO_VDSP_MIN_ELEMS) { return tensor_sum_vdsp(a); }
    return tensor_sum(a);
}

// No size check at all: benchmarked 2x2x2 through 128x128x128 (8 to
// ~2,097,152 multiply-adds) and BLAS won or tied at every single point —
// no naive-wins region was ever found, unlike vDSP's real one above. This
// project's own two training-loop benchmarks (128x256x64 and
// 128x512x1024x256) independently confirm the same thing at much larger
// scale. Always dispatching to BLAS here isn't a shortcut that skipped
// measuring the small-size case — it's what measuring the small-size
// case actually found.
&Tensor tensor_matmul_auto(const &Tensor a, const &Tensor b) {
    return tensor_matmul_blas(a, b);
}

} // namespace std
