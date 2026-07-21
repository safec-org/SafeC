// SafeC Standard Library — Activation functions implementation (see
// activations.h).
#pragma once
#include <std/ml/activations.h>
#include <std/ml/tensor.h>
#include <std/math.h>

namespace std {

&Tensor tensor_relu_fwd(const &Tensor a) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            float v = a->data[i];
            out->data[i] = (v > (float)0.0) ? v : (float)0.0;
            i = i + 1UL;
        }
    }
    return out;
}

&Tensor tensor_sigmoid_fwd(const &Tensor a) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            out->data[i] = (float)1.0 / ((float)1.0 + exp_f(-(a->data[i])));
            i = i + 1UL;
        }
    }
    return out;
}

&Tensor tensor_tanh_fwd(const &Tensor a) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            out->data[i] = tanh_f(a->data[i]);
            i = i + 1UL;
        }
    }
    return out;
}

&Tensor tensor_silu_fwd(const &Tensor a) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            float x = a->data[i];
            float sig = (float)1.0 / ((float)1.0 + exp_f(-x));
            out->data[i] = x * sig;
            i = i + 1UL;
        }
    }
    return out;
}

&Tensor tensor_gelu_fwd(const &Tensor a) {
    struct Tensor* out = tensor_zeros_like(a);
    float c = (float)0.7978845608028654; // sqrt(2/pi)
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            float x = a->data[i];
            float inner = c * (x + (float)0.044715 * x * x * x);
            out->data[i] = (float)0.5 * x * ((float)1.0 + tanh_f(inner));
            i = i + 1UL;
        }
    }
    return out;
}

&Tensor tensor_layernorm_rows(const &Tensor x, float eps) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = x->shape[0]; cols = x->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            float mean = (float)0.0;
            unsigned long c = 0UL;
            while (c < cols) { mean = mean + x->data[r * cols + c]; c = c + 1UL; }
            mean = mean / (float)cols;

            float varAcc = (float)0.0;
            c = 0UL;
            while (c < cols) {
                float d = x->data[r * cols + c] - mean;
                varAcc = varAcc + d * d;
                c = c + 1UL;
            }
            float variance = varAcc / (float)cols;
            float invStd = (float)1.0 / sqrt_f(variance + eps);

            c = 0UL;
            while (c < cols) {
                out->data[r * cols + c] = (x->data[r * cols + c] - mean) * invStd;
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
    return out;
}

&Tensor tensor_residual_add(const &Tensor x, const &Tensor sublayerOut) {
    return tensor_add(x, sublayerOut);
}

} // namespace std
