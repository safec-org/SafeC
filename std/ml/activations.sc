// SafeC Standard Library — Activation functions implementation (see
// activations.h).
#pragma once
#include <std/ml/activations.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor.sc>
#include <std/math.h>
#include <std/math.sc>

namespace std {

struct Tensor* tensor_sigmoid(struct Tensor* a) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            out->data[i] = 1.0 / (1.0 + exp_d(-(a->data[i])));
            i = i + 1UL;
        }
    }
    return out;
}

struct Tensor* tensor_tanh(struct Tensor* a) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            out->data[i] = tanh_d(a->data[i]);
            i = i + 1UL;
        }
    }
    return out;
}

struct Tensor* tensor_silu(struct Tensor* a) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            double x = a->data[i];
            double sig = 1.0 / (1.0 + exp_d(-x));
            out->data[i] = x * sig;
            i = i + 1UL;
        }
    }
    return out;
}

struct Tensor* tensor_gelu(struct Tensor* a) {
    struct Tensor* out = tensor_zeros_like(a);
    double c = 0.7978845608028654; // sqrt(2/pi)
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            double x = a->data[i];
            double inner = c * (x + 0.044715 * x * x * x);
            out->data[i] = 0.5 * x * (1.0 + tanh_d(inner));
            i = i + 1UL;
        }
    }
    return out;
}

struct Tensor* tensor_layernorm_rows(struct Tensor* x, double eps) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = x->shape[0]; cols = x->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            double mean = 0.0;
            unsigned long c = 0UL;
            while (c < cols) { mean = mean + x->data[r * cols + c]; c = c + 1UL; }
            mean = mean / (double)cols;

            double varAcc = 0.0;
            c = 0UL;
            while (c < cols) {
                double d = x->data[r * cols + c] - mean;
                varAcc = varAcc + d * d;
                c = c + 1UL;
            }
            double variance = varAcc / (double)cols;
            double invStd = 1.0 / sqrt_d(variance + eps);

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

struct Tensor* tensor_residual_add(struct Tensor* x, struct Tensor* sublayerOut) {
    return tensor_add(x, sublayerOut);
}

} // namespace std
