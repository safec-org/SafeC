// SafeC Standard Library — Activation functions implementation (see
// activations.h).
#pragma once
#include <std/ml/activations.h>
#include <std/ml/tensor.h>
#include <std/math.h>
#include <std/simd/vmath.h>

namespace std {

// Uninitialized outputs (every element is written), the shared elementwise
// kernels and vmath.h's vectorized exp/tanh/sigmoid.

&FloatTensor FloatTensor::relu_fwd() const {
    unsafe { tensor_host_raw_((void*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe { __tensor_ew(EW_RELU, (const float*)self->data, (const float*)0, (float)0.0, (float*)out->data, out->size); }
    return out;
}

&FloatTensor FloatTensor::sigmoid_fwd() const {
    unsafe { tensor_host_raw_((void*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe { vm_sigmoid_f32_n_((const float*)self->data, (float*)out->data, out->size); }
    return out;
}

&FloatTensor FloatTensor::tanh_fwd() const {
    unsafe { tensor_host_raw_((void*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe { vm_tanh_f32_n_((const float*)self->data, (float*)out->data, out->size); }
    return out;
}

&FloatTensor FloatTensor::silu_fwd() const {
    unsafe { tensor_host_raw_((void*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        vm_sigmoid_f32_n_((const float*)self->data, (float*)out->data, out->size);
        __tensor_ew(EW_MUL, (const float*)self->data, (const float*)out->data, (float)0.0, (float*)out->data, out->size);
    }
    return out;
}

&FloatTensor FloatTensor::gelu_fwd() const {
    unsafe { tensor_host_raw_((void*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    float c = (float)0.7978845608028654; // sqrt(2/pi)
    unsafe {
        unsigned long n = out->size;
        float* y = (float*)out->data;
        unsigned long i = 0UL;
        while (i < n) { float x = self->data[i]; y[i] = c * (x + (float)0.044715 * x * x * x); i = i + 1UL; }
        vm_tanh_f32_n_((const float*)y, y, n);
        i = 0UL;
        while (i < n) { float x = self->data[i]; y[i] = (float)0.5 * x * ((float)1.0 + y[i]); i = i + 1UL; }
    }
    return out;
}

&FloatTensor FloatTensor::layernorm_rows(float eps) const {
    unsafe { tensor_host_raw_((void*)self); }
    unsigned long rows; unsigned long cols;
    unsafe { rows = self->shape[0]; cols = self->shape[1]; }
    &FloatTensor out = FloatTensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            float mean = (float)0.0;
            unsigned long c = 0UL;
            while (c < cols) { mean = mean + self->data[r * cols + c]; c = c + 1UL; }
            mean = mean / (float)cols;

            float varAcc = (float)0.0;
            c = 0UL;
            while (c < cols) {
                float d = self->data[r * cols + c] - mean;
                varAcc = varAcc + d * d;
                c = c + 1UL;
            }
            float variance = varAcc / (float)cols;
            float invStd = (float)1.0 / __builtin_sqrt(variance + eps);

            c = 0UL;
            while (c < cols) {
                out->data[r * cols + c] = (self->data[r * cols + c] - mean) * invStd;
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
    return out;
}

&FloatTensor tensor_residual_add(const &FloatTensor x, const &FloatTensor sublayerOut) {
    return tensor_add(x, sublayerOut);
}

} // namespace std
