// SafeC Standard Library — neural-net building blocks implementation (see
// tensor_nn.h). Internal 'static' helpers (every __X_backward function)
// are defined in tensor_nn.h itself, not here -- see tensor.h's own
// comment for why. This file only implements the public API declared in
// tensor_nn.h, and #includes only headers (never another .sc file).
#pragma once
#include <std/ml/tensor_nn.h>

namespace std {

// ══ Activations ═══════════════════════════════════════════════════════════════
// Backward math for each of these lives in tensor_nn.h, right next to the
// helpers it shares with the GPU backends -- see __relu_backward etc. there.

// ── ReLU ─────────────────────────────────────────────────────────────────────
// Moved here from tensor.sc -- it's a neural-net activation function like
// every other one in this section, not core tensor arithmetic.
&Tensor tensor_relu(const &Tensor a) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            double v = a->data[i];
            out->data[i] = (v > 0.0) ? v : 0.0;
            i = i + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__relu_backward);
    return out;
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────
&Tensor tensor_sigmoid(const &Tensor a) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            out->data[i] = 1.0 / (1.0 + exp_d(-a->data[i]));
            i = i + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__sigmoid_backward);
    return out;
}

// ── Tanh ─────────────────────────────────────────────────────────────────────
&Tensor tensor_tanh_t(const &Tensor a) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            out->data[i] = tanh_d(a->data[i]);
            i = i + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__tanh_t_backward);
    return out;
}

// ── Softmax (last axis) ───────────────────────────────────────────────────────
&Tensor tensor_softmax(const &Tensor a) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long lastDim = a->shape[a->ndim - 1UL];
        unsigned long rows = a->size / lastDim;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long base = r * lastDim;
            double mx = a->data[base];
            unsigned long j = 1UL;
            while (j < lastDim) {
                if (a->data[base + j] > mx) { mx = a->data[base + j]; }
                j = j + 1UL;
            }
            double sum = 0.0;
            j = 0UL;
            while (j < lastDim) {
                double e = exp_d(a->data[base + j] - mx);
                out->data[base + j] = e;
                sum = sum + e;
                j = j + 1UL;
            }
            j = 0UL;
            while (j < lastDim) {
                out->data[base + j] = out->data[base + j] / sum;
                j = j + 1UL;
            }
            r = r + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__softmax_backward);
    return out;
}

// ── ELU ──────────────────────────────────────────────────────────────────────
&Tensor tensor_elu(const &Tensor a, double alpha) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            double x = a->data[i];
            out->data[i] = (x > 0.0) ? x : (alpha * (exp_d(x) - 1.0));
            i = i + 1UL;
        }
    }
    unsafe { out->extraScalar = alpha; }
    __tensor_link1(out, a, (void*)__elu_backward);
    return out;
}

// ── GELU (tanh approximation) ─────────────────────────────────────────────────
&Tensor tensor_gelu(const &Tensor a) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        double c = 0.7978845608028654;
        unsigned long i = 0UL;
        while (i < out->size) {
            double x = a->data[i];
            double u = c * (x + 0.044715 * x * x * x);
            out->data[i] = 0.5 * x * (1.0 + tanh_d(u));
            i = i + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__gelu_backward);
    return out;
}

// ── SiLU / Swish ───────────────────────────────────────────────────────────────
&Tensor tensor_silu(const &Tensor a) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            double x = a->data[i];
            double s = 1.0 / (1.0 + exp_d(-x));
            out->data[i] = x * s;
            i = i + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__silu_backward);
    return out;
}

// ── GLU: split last axis in half, out = a1 * sigmoid(a2) ─────────────────────
&Tensor tensor_glu(const &Tensor a) {
    unsigned long ndim;
    unsafe { ndim = a->ndim; }
    unsigned long* outShape = (unsigned long*)alloc(checked_mul_size(sizeof(unsigned long), ndim));
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outShape[ndim - 1UL] / 2UL;
    }
    struct Tensor* out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc((void*)outShape); }
    unsafe {
        unsigned long outLast = out->shape[out->ndim - 1UL];
        unsigned long inLast = outLast * 2UL;
        unsigned long rows = out->size / outLast;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long outBase = r * outLast;
            unsigned long inBase = r * inLast;
            unsigned long j = 0UL;
            while (j < outLast) {
                double a1 = a->data[inBase + j];
                double a2 = a->data[inBase + outLast + j];
                double s = 1.0 / (1.0 + exp_d(-a2));
                out->data[outBase + j] = a1 * s;
                j = j + 1UL;
            }
            r = r + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__glu_backward);
    return out;
}

// ── SwiGLU: split last axis in half, out = silu(a1) * a2 ──────────────────────
&Tensor tensor_swiglu(const &Tensor a) {
    unsigned long ndim;
    unsafe { ndim = a->ndim; }
    unsigned long* outShape = (unsigned long*)alloc(checked_mul_size(sizeof(unsigned long), ndim));
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outShape[ndim - 1UL] / 2UL;
    }
    struct Tensor* out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc((void*)outShape); }
    unsafe {
        unsigned long outLast = out->shape[out->ndim - 1UL];
        unsigned long inLast = outLast * 2UL;
        unsigned long rows = out->size / outLast;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long outBase = r * outLast;
            unsigned long inBase = r * inLast;
            unsigned long j = 0UL;
            while (j < outLast) {
                double a1 = a->data[inBase + j];
                double a2 = a->data[inBase + outLast + j];
                double s1 = 1.0 / (1.0 + exp_d(-a1));
                out->data[outBase + j] = (a1 * s1) * a2;
                j = j + 1UL;
            }
            r = r + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)__swiglu_backward);
    return out;
}

// ══ Optimizers ════════════════════════════════════════════════════════════════

// ── Adam ─────────────────────────────────────────────────────────────────────
struct AdamState adam_new(unsigned long size, double lr, double beta1, double beta2, double eps) {
    struct AdamState s;
    unsafe {
        double* mBuf = (double*)malloc(sizeof(double) * size);
        double* vBuf = (double*)malloc(sizeof(double) * size);
        unsigned long i = 0UL;
        while (i < size) { mBuf[i] = 0.0; vBuf[i] = 0.0; i = i + 1UL; }
        s.m = (&heap double)mBuf;
        s.v = (&heap double)vBuf;
    }
    s.size = size;
    s.lr = lr; s.beta1 = beta1; s.beta2 = beta2; s.eps = eps;
    s.t = 0UL;
    return s;
}

struct AdamState adam_new_default(unsigned long size, double lr) {
    return adam_new(size, lr, 0.9, 0.999, 0.00000001);
}

void adam_step(&AdamState state, &Tensor param) {
    unsafe {
        state.t = state.t + 1UL;
        double b1 = state.beta1;
        double b2 = state.beta2;
        double biasCorr1 = 1.0 - pow_d(b1, (double)state.t);
        double biasCorr2 = 1.0 - pow_d(b2, (double)state.t);
        unsigned long i = 0UL;
        while (i < state.size) {
            double g = param->grad[i];
            state.m[i] = b1 * state.m[i] + (1.0 - b1) * g;
            state.v[i] = b2 * state.v[i] + (1.0 - b2) * g * g;
            double mHat = state.m[i] / biasCorr1;
            double vHat = state.v[i] / biasCorr2;
            param->data[i] = param->data[i] - state.lr * mHat / (sqrt_d(vHat) + state.eps);
            i = i + 1UL;
        }
    }
}

void adam_free(&AdamState state) {
    unsafe {
        if (state.m != (double*)0) { free((void*)state.m); }
        if (state.v != (double*)0) { free((void*)state.v); }
        state.m = (&heap double)0;
        state.v = (&heap double)0;
    }
}

// ══ Pooling ═══════════════════════════════════════════════════════════════════

// ── 1D max pooling ───────────────────────────────────────────────────────────
&Tensor tensor_maxpool1d(const &Tensor a, unsigned long kernel, unsigned long stride) {
    unsigned long ndim; unsigned long lastDim;
    unsafe { ndim = a->ndim; lastDim = a->shape[ndim - 1UL]; }
    unsigned long outLen = (lastDim - kernel) / stride + 1UL;
    unsigned long* outShape = (unsigned long*)alloc(checked_mul_size(sizeof(unsigned long), ndim));
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outLen;
    }
    struct Tensor* out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    struct Tensor* idxCache = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc((void*)outShape); }
    unsafe {
        unsigned long rows = out->size / outLen;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long inBase = r * lastDim;
            unsigned long outBase = r * outLen;
            unsigned long o = 0UL;
            while (o < outLen) {
                unsigned long winStart = o * stride;
                double best = a->data[inBase + winStart];
                unsigned long bestIdx = inBase + winStart;
                unsigned long k = 1UL;
                while (k < kernel) {
                    double v = a->data[inBase + winStart + k];
                    if (v > best) { best = v; bestIdx = inBase + winStart + k; }
                    k = k + 1UL;
                }
                out->data[outBase + o] = best;
                idxCache->data[outBase + o] = (double)bestIdx;
                o = o + 1UL;
            }
            r = r + 1UL;
        }
        out->requiresGrad = a->requiresGrad;
        if (out->requiresGrad) {
            out->parents.push((const void*)&a);
            out->parents.push((const void*)&idxCache);
            out->gradFn = (void*)__maxpool1d_backward;
        }
    }
    return out;
}

// ── 1D average pooling ───────────────────────────────────────────────────────
&Tensor tensor_avgpool1d(const &Tensor a, unsigned long kernel, unsigned long stride) {
    unsigned long ndim; unsigned long lastDim;
    unsafe { ndim = a->ndim; lastDim = a->shape[ndim - 1UL]; }
    unsigned long outLen = (lastDim - kernel) / stride + 1UL;
    unsigned long* outShape = (unsigned long*)alloc(checked_mul_size(sizeof(unsigned long), ndim));
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outLen;
    }
    struct Tensor* out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc((void*)outShape); }
    unsafe {
        double invK = 1.0 / (double)kernel;
        unsigned long rows = out->size / outLen;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long inBase = r * lastDim;
            unsigned long outBase = r * outLen;
            unsigned long o = 0UL;
            while (o < outLen) {
                unsigned long winStart = o * stride;
                double sum = 0.0;
                unsigned long k = 0UL;
                while (k < kernel) { sum = sum + a->data[inBase + winStart + k]; k = k + 1UL; }
                out->data[outBase + o] = sum * invK;
                o = o + 1UL;
            }
            r = r + 1UL;
        }
        out->extraScalar = (double)(kernel * 1000000UL + stride);
    }
    __tensor_link1(out, a, (void*)__avgpool1d_backward);
    return out;
}

} // namespace std
