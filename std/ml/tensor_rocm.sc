// SafeC Standard Library — GPU-backed Tensor ops implementation (ROCm/HIP (AMD), see
// tensor_rocm.h). Each op falls back to the CPU loop if the GPU call fails.
#pragma once
#include <std/ml/tensor_rocm.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/gpu_rocm.h>

namespace std {

&Tensor tensor_add_rocm(const &Tensor a, const &Tensor b) {
    if (!rocm_available()) { return tensor_add(a, b); }
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = rocm_add_f32((const &float)a->data, (const &float)b->data, (&float)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub_rocm(const &Tensor a, const &Tensor b) {
    if (!rocm_available()) { return tensor_sub(a, b); }
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = rocm_sub_f32((const &float)a->data, (const &float)b->data, (&float)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

&Tensor tensor_mul_rocm(const &Tensor a, const &Tensor b) {
    if (!rocm_available()) { return tensor_mul(a, b); }
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = rocm_mul_f32((const &float)a->data, (const &float)b->data, (&float)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

&Tensor Tensor::scale_rocm(float k) const {
    if (!rocm_available()) { return self.scale(k); }
    &Tensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        int ok = rocm_scale_f32((const &float)self->data, k, (&float)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = self->data[i] * k; i = i + 1UL; }
        }
    }
    unsafe { out->extraScalar = k; }
    __tensor_link1(out, self, (void*)__scale_backward);
    return out;
}

&Tensor Tensor::relu_rocm() const {
    if (!rocm_available()) { return self.relu(); }
    &Tensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        int ok = rocm_relu_f32((const &float)self->data, (&float)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) {
                float v = self->data[i];
                out->data[i] = (v > (float)0.0) ? v : (float)0.0;
                i = i + 1UL;
            }
        }
    }
    __tensor_link1(out, self, (void*)__relu_backward);
    return out;
}

&Tensor Tensor::sum_rocm() const {
    if (!rocm_available()) { return self.sum(); }
    unsigned long one[1];
    unsafe { one[0] = 1UL; }
    &Tensor out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        float fout[1];
        int ok = rocm_sum_f32((const &float)self->data, (&float)fout, self->size);
        if (ok) {
            out->data[0] = fout[0];
        } else {
            float acc = (float)0.0;
            unsigned long i = 0UL;
            while (i < self->size) { acc = acc + self->data[i]; i = i + 1UL; }
            out->data[0] = acc;
        }
    }
    __tensor_link1(out, self, (void*)__sum_backward);
    return out;
}

&Tensor tensor_matmul_rocm(const &Tensor a, const &Tensor b) {
    if (!rocm_available()) { return tensor_matmul(a, b); }

    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    &Tensor out = Tensor::new_2d(m, n, 0);
    unsafe {
        int ok = rocm_matmul_f32((const &float)a->data, (const &float)b->data, (&float)out->data, m, k, n);
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

&Tensor tensor_matmul_rocm_blas(const &Tensor a, const &Tensor b) {
    if (!rocm_available()) { return tensor_matmul(a, b); }

    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    &Tensor out = Tensor::new_2d(m, n, 0);
    unsafe {
        int ok = rocm_matmul_f32_blas((const &float)a->data, (const &float)b->data, (&float)out->data, m, k, n);
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
