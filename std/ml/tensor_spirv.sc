// SafeC Standard Library — GPU-backed Tensor ops implementation (Vulkan/SPIR-V (cross-vendor), see
// tensor_spirv.h). UNVERIFIED — see gpu_spirv.h/.sc's warnings; every op falls back to CPU today.
#pragma once
#include <std/ml/tensor_spirv.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/gpu_spirv.h>

namespace std {

&Tensor tensor_add_spirv(const &Tensor a, const &Tensor b) {
    if (!spirv_available()) { return tensor_add(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = spirv_add_f32((const float*)a->data, (const float*)b->data, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub_spirv(const &Tensor a, const &Tensor b) {
    if (!spirv_available()) { return tensor_sub(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = spirv_sub_f32((const float*)a->data, (const float*)b->data, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

&Tensor tensor_mul_spirv(const &Tensor a, const &Tensor b) {
    if (!spirv_available()) { return tensor_mul(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = spirv_mul_f32((const float*)a->data, (const float*)b->data, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

&Tensor tensor_scale_spirv(const &Tensor a, float k) {
    if (!spirv_available()) { return tensor_scale(a, k); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = spirv_scale_f32((const float*)a->data, k, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * k; i = i + 1UL; }
        }
    }
    unsafe { out->extraScalar = k; }
    __tensor_link1(out, a, (void*)__scale_backward);
    return out;
}

&Tensor tensor_relu_spirv(const &Tensor a) {
    if (!spirv_available()) { return tensor_relu(a); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = spirv_relu_f32((const float*)a->data, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) {
                float v = a->data[i];
                out->data[i] = (v > (float)0.0) ? v : (float)0.0;
                i = i + 1UL;
            }
        }
    }
    __tensor_link1(out, a, (void*)__relu_backward);
    return out;
}

&Tensor tensor_sum_spirv(const &Tensor a) {
    if (!spirv_available()) { return tensor_sum(a); }
    unsigned long one[1];
    unsafe { one[0] = 1UL; }
    struct Tensor* out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        float fout[1];
        int ok = spirv_sum_f32((const float*)a->data, (float*)fout, a->size);
        if (ok) {
            out->data[0] = fout[0];
        } else {
            float acc = (float)0.0;
            unsigned long i = 0UL;
            while (i < a->size) { acc = acc + a->data[i]; i = i + 1UL; }
            out->data[0] = acc;
        }
    }
    __tensor_link1(out, a, (void*)__sum_backward);
    return out;
}

&Tensor tensor_matmul_spirv(const &Tensor a, const &Tensor b) {
    if (!spirv_available()) { return tensor_matmul(a, b); }

    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    struct Tensor* out = tensor_new_2d(m, n, 0);
    unsafe {
        int ok = spirv_matmul_f32((const float*)a->data, (const float*)b->data, (float*)out->data, m, k, n);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < m) {
                unsigned long j0 = 0UL;
                while (j0 < n) { out->data[i * n + j0] = (float)0.0; j0 = j0 + 1UL; }
                i = i + 1UL;
            }
            i = 0UL;
            while (i < m) {
                unsigned long p = 0UL;
                while (p < k) {
                    float aVal = a->data[i * k + p];
                    unsigned long j = 0UL;
                    while (j < n) {
                        out->data[i * n + j] = out->data[i * n + j] + aVal * b->data[p * n + j];
                        j = j + 1UL;
                    }
                    p = p + 1UL;
                }
                i = i + 1UL;
            }
        }
    }
    __tensor_link2(out, a, b, (void*)__matmul_backward);
    return out;
}

} // namespace std
