// SafeC Standard Library — neural-net building blocks implementation (see
// tensor_nn.h). Internal 'static' helpers (every __X_backward function)
// are defined in tensor_nn.h itself, not here -- see tensor.h's own
// comment for why. This file only implements the public API declared in
// tensor_nn.h, and #includes only headers (never another .sc file).
#pragma once
#include <std/ml/tensor_nn.h>
#include <std/simd/vmath.h>
#include <std/sync/parallel.h>

namespace std {

// ══ Activations ═══════════════════════════════════════════════════════════════
// Backward math for each of these lives in tensor_nn.h, right next to the
// helpers it shares with the GPU backends -- see __relu_backward etc. there.

// ── ReLU ─────────────────────────────────────────────────────────────────────
// Moved here from tensor.sc -- it's a neural-net activation function like
// every other one in this section, not core tensor arithmetic.
&Tensor tensor_relu(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe { __tensor_ew(EW_RELU, (const float*)a->data, (const float*)0, (float)0.0, (float*)out->data, out->size); }
    __tensor_link1(out, a, (void*)__relu_backward);
    return out;
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────
&Tensor tensor_sigmoid(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe { vm_sigmoid_f32((const float*)a->data, (float*)out->data, out->size); }
    __tensor_link1(out, a, (void*)__sigmoid_backward);
    return out;
}

// ── Tanh ─────────────────────────────────────────────────────────────────────
&Tensor tensor_tanh_t(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe { vm_tanh_f32((const float*)a->data, (float*)out->data, out->size); }
    __tensor_link1(out, a, (void*)__tanh_t_backward);
    return out;
}

// ── Softmax (last axis) ───────────────────────────────────────────────────────
// Rows are independent; large tensors split them across the thread pool.
struct SoftmaxJob { const float* in; float* out; unsigned long cols; };

static void softmax_rows_(&SoftmaxJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long n = j.cols;
        unsigned long r = lo;
        while (r < hi) {
            const float* x = &j.in[r * n];
            float* y = &j.out[r * n];
            // max with 8 running lanes, so it vectorizes
            float m[8];
            unsigned long q = 0UL;
            while (q < 8UL) { m[q] = x[0]; q = q + 1UL; }
            unsigned long i = 0UL;
            while (i + 8UL <= n) {
                q = 0UL;
                while (q < 8UL) { float v = x[i + q]; m[q] = v > m[q] ? v : m[q]; q = q + 1UL; }
                i = i + 8UL;
            }
            float mx = m[0];
            q = 1UL;
            while (q < 8UL) { mx = m[q] > mx ? m[q] : mx; q = q + 1UL; }
            while (i < n) { mx = x[i] > mx ? x[i] : mx; i = i + 1UL; }
            i = 0UL;
            while (i < n) { y[i] = x[i] - mx; i = i + 1UL; }
            vm_exp_f32((const float*)y, y, n);
            float sum = (float)0.0;
            i = 0UL;
            while (i < n) { sum = sum + y[i]; i = i + 1UL; }
            float inv = (float)1.0 / sum;
            i = 0UL;
            while (i < n) { y[i] = y[i] * inv; i = i + 1UL; }
            r = r + 1UL;
        }
    }
}

&Tensor tensor_softmax(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        struct SoftmaxJob j;
        j.in = (const float*)a->data;
        j.out = (float*)out->data;
        j.cols = a->shape[a->ndim - 1UL];
        unsigned long rows = a->size / j.cols;
        if (a->size >= TENSOR_EW_PAR && rows > 1UL) parallel_for(rows, 16384UL / j.cols + 1UL, softmax_rows_, &j);
        else softmax_rows_(&j, 0UL, rows);
    }
    __tensor_link1(out, a, (void*)__softmax_backward);
    return out;
}

// ── ELU ──────────────────────────────────────────────────────────────────────
&Tensor tensor_elu(const &Tensor a, float alpha) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long n = out->size;
        float* y = (float*)out->data;
        unsigned long i = 0UL;
        while (i < n) { float x = a->data[i]; y[i] = x < (float)0.0 ? x : (float)0.0; i = i + 1UL; }
        vm_exp_f32((const float*)y, y, n);
        i = 0UL;
        while (i < n) {
            float x = a->data[i];
            y[i] = (x > (float)0.0) ? x : (alpha * (y[i] - (float)1.0));
            i = i + 1UL;
        }
    }
    unsafe { out->extraScalar = alpha; }
    __tensor_link1(out, a, (void*)__elu_backward);
    return out;
}

// ── GELU (tanh approximation) ─────────────────────────────────────────────────
&Tensor tensor_gelu(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        float c = (float)0.7978845608028654;
        unsigned long n = out->size;
        float* y = (float*)out->data;
        unsigned long i = 0UL;
        while (i < n) { float x = a->data[i]; y[i] = c * (x + (float)0.044715 * x * x * x); i = i + 1UL; }
        vm_tanh_f32((const float*)y, y, n);
        i = 0UL;
        while (i < n) { float x = a->data[i]; y[i] = (float)0.5 * x * ((float)1.0 + y[i]); i = i + 1UL; }
    }
    __tensor_link1(out, a, (void*)__gelu_backward);
    return out;
}

// ── SiLU / Swish ───────────────────────────────────────────────────────────────
&Tensor tensor_silu(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long n = out->size;
        float* y = (float*)out->data;
        vm_sigmoid_f32((const float*)a->data, y, n);
        unsigned long i = 0UL;
        while (i < n) { y[i] = a->data[i] * y[i]; i = i + 1UL; }
    }
    __tensor_link1(out, a, (void*)__silu_backward);
    return out;
}

// ── GLU: split last axis in half, out = a1 * sigmoid(a2) ─────────────────────
&Tensor tensor_glu(const &Tensor a) {
    unsigned long ndim;
    unsafe { ndim = a->ndim; }
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outShape[ndim - 1UL] / 2UL;
    }
    &Tensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc_raw_((void*)outShape); }
    unsafe {
        unsigned long outLast = out->shape[out->ndim - 1UL];
        unsigned long inLast = outLast * 2UL;
        unsigned long rows = out->size / outLast;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long outBase = r * outLast;
            unsigned long inBase = r * inLast;
            float* y = &out->data[outBase];
            vm_sigmoid_f32((const float*)&a->data[inBase + outLast], y, outLast);
            unsigned long j = 0UL;
            while (j < outLast) { y[j] = a->data[inBase + j] * y[j]; j = j + 1UL; }
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
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outShape[ndim - 1UL] / 2UL;
    }
    &Tensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc_raw_((void*)outShape); }
    unsafe {
        unsigned long outLast = out->shape[out->ndim - 1UL];
        unsigned long inLast = outLast * 2UL;
        unsigned long rows = out->size / outLast;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long outBase = r * outLast;
            unsigned long inBase = r * inLast;
            float* y = &out->data[outBase];
            vm_sigmoid_f32((const float*)&a->data[inBase], y, outLast);
            unsigned long j = 0UL;
            while (j < outLast) {
                float a1 = a->data[inBase + j];
                y[j] = (a1 * y[j]) * a->data[inBase + outLast + j];
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
struct AdamState adam_new(unsigned long size, float lr, float beta1, float beta2, float eps) {
    struct AdamState s;
    unsafe {
        float* mBuf = (float*)malloc(sizeof(float) * size);
        float* vBuf = (float*)malloc(sizeof(float) * size);
        unsigned long i = 0UL;
        while (i < size) { mBuf[i] = (float)0.0; vBuf[i] = (float)0.0; i = i + 1UL; }
        s.m = (&heap float)mBuf;
        s.v = (&heap float)vBuf;
    }
    s.size = size;
    s.lr = lr; s.beta1 = beta1; s.beta2 = beta2; s.eps = eps;
    s.t = 0UL;
    return s;
}

struct AdamState adam_new_default(unsigned long size, float lr) {
    return adam_new(size, lr, (float)0.9, (float)0.999, (float)0.00000001);
}

void adam_step(&AdamState state, &Tensor param) {
    unsafe {
        state.t = state.t + 1UL;
        float b1 = state.beta1;
        float b2 = state.beta2;
        float biasCorr1 = (float)1.0 - pow_f(b1, (float)state.t);
        float biasCorr2 = (float)1.0 - pow_f(b2, (float)state.t);
        unsigned long i = 0UL;
        while (i < state.size) {
            float g = param->grad[i];
            state.m[i] = b1 * state.m[i] + ((float)1.0 - b1) * g;
            state.v[i] = b2 * state.v[i] + ((float)1.0 - b2) * g * g;
            float mHat = state.m[i] / biasCorr1;
            float vHat = state.v[i] / biasCorr2;
            param->data[i] = param->data[i] - state.lr * mHat / (__builtin_sqrt(vHat) + state.eps);
            i = i + 1UL;
        }
    }
}

void adam_free(&AdamState state) {
    unsafe {
        if (state.m != (float*)0) { free((void*)state.m); }
        if (state.v != (float*)0) { free((void*)state.v); }
        state.m = (&heap float)0;
        state.v = (&heap float)0;
    }
}

// ══ Pooling ═══════════════════════════════════════════════════════════════════

// ── 1D max pooling ───────────────────────────────────────────────────────────
&Tensor tensor_maxpool1d(const &Tensor a, unsigned long kernel, unsigned long stride) {
    unsigned long ndim; unsigned long lastDim;
    unsafe { ndim = a->ndim; lastDim = a->shape[ndim - 1UL]; }
    unsigned long outLen = (lastDim - kernel) / stride + 1UL;
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outLen;
    }
    &Tensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    &Tensor idxCache = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc_raw_((void*)outShape); }
    unsafe {
        unsigned long rows = out->size / outLen;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long inBase = r * lastDim;
            unsigned long outBase = r * outLen;
            unsigned long o = 0UL;
            while (o < outLen) {
                unsigned long winStart = o * stride;
                float best = a->data[inBase + winStart];
                unsigned long bestIdx = inBase + winStart;
                unsigned long k = 1UL;
                while (k < kernel) {
                    float v = a->data[inBase + winStart + k];
                    if (v > best) { best = v; bestIdx = inBase + winStart + k; }
                    k = k + 1UL;
                }
                out->data[outBase + o] = best;
                idxCache->data[outBase + o] = (float)bestIdx;
                o = o + 1UL;
            }
            r = r + 1UL;
        }
        out->requiresGrad = a->requiresGrad;
        if (out->requiresGrad) {
            out->parents.push(a);
            out->parents.push(idxCache);
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
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = a->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outLen;
    }
    &Tensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc_raw_((void*)outShape); }
    unsafe {
        float invK = (float)1.0 / (float)kernel;
        unsigned long rows = out->size / outLen;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long inBase = r * lastDim;
            unsigned long outBase = r * outLen;
            unsigned long o = 0UL;
            while (o < outLen) {
                unsigned long winStart = o * stride;
                float sum = (float)0.0;
                unsigned long k = 0UL;
                while (k < kernel) { sum = sum + a->data[inBase + winStart + k]; k = k + 1UL; }
                out->data[outBase + o] = sum * invK;
                o = o + 1UL;
            }
            r = r + 1UL;
        }
        // See tensor_nn.h's __avgpool1d_backward comment for the base-10000 choice.
        out->extraScalar = (float)(kernel * 10000UL + stride);
    }
    __tensor_link1(out, a, (void*)__avgpool1d_backward);
    return out;
}

} // namespace std
