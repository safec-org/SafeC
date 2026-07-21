// SafeC Standard Library — vDSP-accelerated elementwise Tensor ops
// implementation (see tensor_vdsp.h).
#pragma once
#include <std/ml/tensor_vdsp.h>
#include <std/ml/tensor.h>

namespace std {

// Same Accelerate framework as tensor_blas.sc's cblas_sgemm, a different
// corner of it: vectorized elementwise float32 ops instead of matrix
// multiply. Verified directly (not assumed) before using any of these:
// vDSP_vsub computes C = B - A (second argument minus first — the
// reverse of what the argument order suggests), the other two are the
// ordinary commutative operations their names imply.
extern void vDSP_vadd(const float* A, long IA, const float* B, long IB,
                       float* C, long IC, unsigned long N);
extern void vDSP_vsub(const float* A, long IA, const float* B, long IB,
                       float* C, long IC, unsigned long N);
extern void vDSP_vmul(const float* A, long IA, const float* B, long IB,
                       float* C, long IC, unsigned long N);
extern void vDSP_sve(const float* A, long IA, float* result, unsigned long N);

&Tensor tensor_add_vdsp(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        vDSP_vadd((const float*)a->data, 1L, (const float*)b->data, 1L,
                  (float*)out->data, 1L, out->size);
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub_vdsp(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        // out = a - b: vDSP_vsub(X, ..., Y, ..., Z, ...) computes Z = Y - X,
        // so X=b, Y=a gives Z=a-b.
        vDSP_vsub((const float*)b->data, 1L, (const float*)a->data, 1L,
                  (float*)out->data, 1L, out->size);
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

&Tensor tensor_mul_vdsp(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        vDSP_vmul((const float*)a->data, 1L, (const float*)b->data, 1L,
                  (float*)out->data, 1L, out->size);
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

&Tensor tensor_sum_vdsp(const &Tensor a) {
    unsigned long one[1];
    unsafe { one[0] = 1UL; }
    struct Tensor* out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        float acc = (float)0.0;
        vDSP_sve((const float*)a->data, 1L, &acc, a->size);
        out->data[0] = acc;
    }
    __tensor_link1(out, a, (void*)__sum_backward);
    return out;
}

} // namespace std
