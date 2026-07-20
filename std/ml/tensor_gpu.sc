// SafeC Standard Library — GPU-backed Tensor ops implementation (see
// tensor_gpu.h).
#pragma once
#include <std/ml/tensor_gpu.h>
#include <std/ml/tensor.sc>
#include <std/ml/gpu_mps.h>
#include <std/ml/gpu_mps.sc>

namespace std {

// double -> float32 and back, shared by every op below. Metal has no
// 'double' type at all (see gpu_mps.sc), so this conversion is unavoidable
// anywhere a Tensor's 'double' data crosses into an mps_* call.
static float* __tensor_gpu_to_f32(const double* src, unsigned long n) {
    unsafe {
        float* out = (float*)alloc(checked_mul_size(sizeof(float), n));
        unsigned long i = 0UL;
        while (i < n) { out[i] = (float)src[i]; i = i + 1UL; }
        return out;
    }
}

static void __tensor_gpu_from_f32(double* dst, const float* src, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { dst[i] = (double)src[i]; i = i + 1UL; }
    }
}

&Tensor tensor_add_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_add(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fb = __tensor_gpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_add_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_sub(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fb = __tensor_gpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_sub_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

&Tensor tensor_mul_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_mul(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fb = __tensor_gpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_mul_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

&Tensor tensor_scale_gpu(const &Tensor a, double k) {
    if (!mps_available()) { return tensor_scale(a, k); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_scale_f32((const float*)fa, (float)k, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
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

&Tensor tensor_relu_gpu(const &Tensor a) {
    if (!mps_available()) { return tensor_relu(a); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_relu_f32((const float*)fa, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
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

&Tensor tensor_sum_gpu(const &Tensor a) {
    if (!mps_available()) { return tensor_sum(a); }
    unsigned long one[1];
    unsafe { one[0] = 1UL; }
    struct Tensor* out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, a->size);
        float fout[1];
        int ok = mps_sum_f32((const float*)fa, (float*)fout, a->size);
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

&Tensor tensor_matmul_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_matmul(a, b); }

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

        int ok = mps_matmul_f32((const float*)fa, (const float*)fb, fout, m, k, n);
        if (ok) {
            z = 0UL;
            while (z < m * n) { out->data[z] = (double)fout[z]; z = z + 1UL; }
        } else {
            // GPU dispatch failed despite a device being present (pipeline
            // compile error, etc, rather than "no GPU" -- that's already
            // handled by the mps_available() check above). Compute
            // directly into the already-allocated 'out' instead of leaking
            // it and calling tensor_matmul for a second Tensor allocation.
            // Same i,p,j loop as tensor_matmul's CPU path (see its comment
            // for why that order).
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
