// SafeC Standard Library — GPU-backed Tensor ops implementation (WebGPU (wgpu-native/Dawn), see
// tensor_webgpu.h). UNVERIFIED — see gpu_webgpu.h/.sc's warnings.
#pragma once
#include <std/ml/tensor_webgpu.h>
#include <std/ml/tensor.sc>
#include <std/ml/gpu_webgpu.h>
#include <std/ml/gpu_webgpu.sc>

namespace std {

static float* __tensor_webgpu_to_f32(const double* src, unsigned long n) {
    unsafe {
        float* out = (float*)alloc(checked_mul_size(sizeof(float), n));
        unsigned long i = 0UL;
        while (i < n) { out[i] = (float)src[i]; i = i + 1UL; }
        return out;
    }
}

static void __tensor_webgpu_from_f32(double* dst, const float* src, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { dst[i] = (double)src[i]; i = i + 1UL; }
    }
}

&Tensor tensor_add_webgpu(const &Tensor a, const &Tensor b) {
    if (!webgpu_available()) { return tensor_add(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_webgpu_to_f32(a->data, out->size);
        float* fb = __tensor_webgpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = webgpu_add_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_webgpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub_webgpu(const &Tensor a, const &Tensor b) {
    if (!webgpu_available()) { return tensor_sub(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_webgpu_to_f32(a->data, out->size);
        float* fb = __tensor_webgpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = webgpu_sub_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_webgpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

&Tensor tensor_mul_webgpu(const &Tensor a, const &Tensor b) {
    if (!webgpu_available()) { return tensor_mul(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_webgpu_to_f32(a->data, out->size);
        float* fb = __tensor_webgpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = webgpu_mul_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_webgpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

&Tensor tensor_scale_webgpu(const &Tensor a, double k) {
    if (!webgpu_available()) { return tensor_scale(a, k); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_webgpu_to_f32(a->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = webgpu_scale_f32((const float*)fa, (float)k, fout, out->size);
        if (ok) {
            __tensor_webgpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * k; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fout);
    }
    unsafe { out->extraScalar = k; }
    __tensor_link1(out, a, (void*)__scale_backward);
    return out;
}

&Tensor tensor_relu_webgpu(const &Tensor a) {
    if (!webgpu_available()) { return tensor_relu(a); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_webgpu_to_f32(a->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = webgpu_relu_f32((const float*)fa, fout, out->size);
        if (ok) {
            __tensor_webgpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) {
                double v = a->data[i];
                out->data[i] = (v > 0.0) ? v : 0.0;
                i = i + 1UL;
            }
        }
        dealloc((void*)fa); dealloc((void*)fout);
    }
    __tensor_link1(out, a, (void*)__relu_backward);
    return out;
}

&Tensor tensor_sum_webgpu(const &Tensor a) {
    if (!webgpu_available()) { return tensor_sum(a); }
    unsigned long one[1];
    unsafe { one[0] = 1UL; }
    struct Tensor* out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        float* fa = __tensor_webgpu_to_f32(a->data, a->size);
        float fout[1];
        int ok = webgpu_sum_f32((const float*)fa, (float*)fout, a->size);
        if (ok) {
            out->data[0] = (double)fout[0];
        } else {
            double acc = 0.0;
            unsigned long i = 0UL;
            while (i < a->size) { acc = acc + a->data[i]; i = i + 1UL; }
            out->data[0] = acc;
        }
        dealloc((void*)fa);
    }
    __tensor_link1(out, a, (void*)__sum_backward);
    return out;
}

&Tensor tensor_matmul_webgpu(const &Tensor a, const &Tensor b) {
    if (!webgpu_available()) { return tensor_matmul(a, b); }

    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    struct Tensor* out = tensor_new_2d(m, n, 0);
    unsafe {
        float* fa   = (float*)alloc(checked_mul_size(sizeof(float), m * k));
        float* fb   = (float*)alloc(checked_mul_size(sizeof(float), k * n));
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), m * n));

        unsigned long z = 0UL;
        while (z < m * k) { fa[z] = (float)a->data[z]; z = z + 1UL; }
        z = 0UL;
        while (z < k * n) { fb[z] = (float)b->data[z]; z = z + 1UL; }

        int ok = webgpu_matmul_f32((const float*)fa, (const float*)fb, fout, m, k, n);
        if (ok) {
            z = 0UL;
            while (z < m * n) { out->data[z] = (double)fout[z]; z = z + 1UL; }
        } else {
            unsigned long i = 0UL;
            while (i < m) {
                unsigned long j0 = 0UL;
                while (j0 < n) { out->data[i * n + j0] = 0.0; j0 = j0 + 1UL; }
                i = i + 1UL;
            }
            i = 0UL;
            while (i < m) {
                unsigned long p = 0UL;
                while (p < k) {
                    double aVal = a->data[i * k + p];
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

        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__matmul_backward);
    return out;
}

} // namespace std
