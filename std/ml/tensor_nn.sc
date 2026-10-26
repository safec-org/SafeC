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
&FloatTensor FloatTensor::relu() const {
    match (tdev_unary_(self, EW_RELU, (float)0.0, (float)0.0, EW_ACC_RELU, 1, (float)0.0, 0)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe { __tensor_ew(EW_RELU, (const float*)self->data, (const float*)0, (float)0.0, (float*)out->data, out->size); }
    __tensor_link1(out, self, __relu_backward);
    return out;
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────
&FloatTensor FloatTensor::sigmoid() const {
    match (tdev_unary_(self, EW_SIGMOID, (float)0.0, (float)0.0, EW_ACC_SIGB, 2, (float)0.0, 0)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe { vm_sigmoid_f32_n_((const float*)self->data, (float*)out->data, out->size); }
    __tensor_link1(out, self, __sigmoid_backward);
    return out;
}

// ── Tanh ─────────────────────────────────────────────────────────────────────
&FloatTensor FloatTensor::tanh() const {
    match (tdev_unary_(self, EW_TANH, (float)0.0, (float)0.0, EW_ACC_TANHB, 2, (float)0.0, 0)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe { vm_tanh_f32_n_((const float*)self->data, (float*)out->data, out->size); }
    __tensor_link1(out, self, __tanh_t_backward);
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
            vm_exp_f32_n_((const float*)y, y, n);
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

&FloatTensor FloatTensor::softmax() const {
    match (tdev_softmax_(self, 0)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        struct SoftmaxJob j;
        j.in = (const float*)self->data;
        j.out = (float*)out->data;
        j.cols = self->shape[self->ndim - 1UL];
        unsigned long rows = self->size / j.cols;
        if (self->size >= TENSOR_EW_PAR && rows > 1UL) parallel_for(rows, 16384UL / j.cols + 1UL, softmax_rows_, &j);
        else softmax_rows_(&j, 0UL, rows);
    }
    __tensor_link1(out, self, __softmax_backward);
    return out;
}

// ── ELU ──────────────────────────────────────────────────────────────────────
&FloatTensor FloatTensor::elu(float alpha) const {
    match (tdev_unary_(self, EW_ELU, alpha, (float)0.0, EW_ACC_ELUB, 1, alpha, 2)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        unsigned long n = out->size;
        float* y = (float*)out->data;
        unsigned long i = 0UL;
        while (i < n) { float x = self->data[i]; y[i] = x < (float)0.0 ? x : (float)0.0; i = i + 1UL; }
        vm_exp_f32_n_((const float*)y, y, n);
        i = 0UL;
        while (i < n) {
            float x = self->data[i];
            y[i] = (x > (float)0.0) ? x : (alpha * (y[i] - (float)1.0));
            i = i + 1UL;
        }
    }
    unsafe { out->extraScalar = alpha; }
    __tensor_link1(out, self, __elu_backward);
    return out;
}

// ── GELU (tanh approximation) ─────────────────────────────────────────────────
&FloatTensor FloatTensor::gelu() const {
    match (tdev_unary_(self, EW_GELU, (float)0.0, (float)0.0, EW_ACC_GELUB, 1, (float)0.0, 0)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        float c = (float)0.7978845608028654;
        unsigned long n = out->size;
        float* y = (float*)out->data;
        unsigned long i = 0UL;
        while (i < n) { float x = self->data[i]; y[i] = c * (x + (float)0.044715 * x * x * x); i = i + 1UL; }
        vm_tanh_f32_n_((const float*)y, y, n);
        i = 0UL;
        while (i < n) { float x = self->data[i]; y[i] = (float)0.5 * x * ((float)1.0 + y[i]); i = i + 1UL; }
    }
    __tensor_link1(out, self, __gelu_backward);
    return out;
}

// ── SiLU / Swish ───────────────────────────────────────────────────────────────
&FloatTensor FloatTensor::silu() const {
    match (tdev_unary_(self, EW_SILU, (float)0.0, (float)0.0, EW_ACC_SILUB, 1, (float)0.0, 0)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        unsigned long n = out->size;
        float* y = (float*)out->data;
        vm_sigmoid_f32_n_((const float*)self->data, y, n);
        unsigned long i = 0UL;
        while (i < n) { y[i] = self->data[i] * y[i]; i = i + 1UL; }
    }
    __tensor_link1(out, self, __silu_backward);
    return out;
}

// ── GLU: split last axis in half, out = a1 * sigmoid(a2) ─────────────────────
&FloatTensor FloatTensor::glu() const {
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    unsigned long ndim;
    unsafe { ndim = self->ndim; }
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = self->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outShape[ndim - 1UL] / 2UL;
    }
    &FloatTensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc(outShape); }
    unsafe {
        unsigned long outLast = out->shape[out->ndim - 1UL];
        unsigned long inLast = outLast * 2UL;
        unsigned long rows = out->size / outLast;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long outBase = r * outLast;
            unsigned long inBase = r * inLast;
            float* y = &out->data[outBase];
            vm_sigmoid_f32_n_((const float*)&self->data[inBase + outLast], y, outLast);
            unsigned long j = 0UL;
            while (j < outLast) { y[j] = self->data[inBase + j] * y[j]; j = j + 1UL; }
            r = r + 1UL;
        }
    }
    __tensor_link1(out, self, __glu_backward);
    return out;
}

// ── SwiGLU: split last axis in half, out = silu(a1) * a2 ──────────────────────
&FloatTensor FloatTensor::swiglu() const {
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    unsigned long ndim;
    unsafe { ndim = self->ndim; }
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = self->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outShape[ndim - 1UL] / 2UL;
    }
    &FloatTensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc(outShape); }
    unsafe {
        unsigned long outLast = out->shape[out->ndim - 1UL];
        unsigned long inLast = outLast * 2UL;
        unsigned long rows = out->size / outLast;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long outBase = r * outLast;
            unsigned long inBase = r * inLast;
            float* y = &out->data[outBase];
            vm_sigmoid_f32_n_((const float*)&self->data[inBase], y, outLast);
            unsigned long j = 0UL;
            while (j < outLast) {
                float a1 = self->data[inBase + j];
                y[j] = (a1 * y[j]) * self->data[inBase + outLast + j];
                j = j + 1UL;
            }
            r = r + 1UL;
        }
    }
    __tensor_link1(out, self, __swiglu_backward);
    return out;
}

// ══ Optimizers ════════════════════════════════════════════════════════════════

// ── Adam ─────────────────────────────────────────────────────────────────────
struct AdamState AdamState::new(unsigned long size, float lr, float beta1, float beta2, float eps) {
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

struct AdamState AdamState::new_default(unsigned long size, float lr) {
    return AdamState::new(size, lr, (float)0.9, (float)0.999, (float)0.00000001);
}

void AdamState::step(&FloatTensor param) {
    unsafe { tensor_host_write_raw_((FloatTensor*)param); }
    unsafe {
        self.t = self.t + 1UL;
        float b1 = self.beta1;
        float b2 = self.beta2;
        float biasCorr1 = (float)1.0 - pow_f(b1, (float)self.t);
        float biasCorr2 = (float)1.0 - pow_f(b2, (float)self.t);
        unsigned long i = 0UL;
        while (i < self.size) {
            float g = param->grad[i];
            self.m[i] = b1 * self.m[i] + ((float)1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + ((float)1.0 - b2) * g * g;
            float mHat = self.m[i] / biasCorr1;
            float vHat = self.v[i] / biasCorr2;
            param->data[i] = param->data[i] - self.lr * mHat / (__builtin_sqrt(vHat) + self.eps);
            i = i + 1UL;
        }
    }
}

void AdamState::free() { self.drop(); }

void AdamState::drop() {
    unsafe {
        if (self.m != (float*)0) { free((unsigned char*)self.m); }
        if (self.v != (float*)0) { free((unsigned char*)self.v); }
        self.m = (&heap float)0;
        self.v = (&heap float)0;
    }
}

// ══ Pooling ═══════════════════════════════════════════════════════════════════

// ── 1D max pooling ───────────────────────────────────────────────────────────
&FloatTensor FloatTensor::maxpool1d(unsigned long kernel, unsigned long stride) const {
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    unsigned long ndim; unsigned long lastDim;
    unsafe { ndim = self->ndim; lastDim = self->shape[ndim - 1UL]; }
    unsigned long outLen = (lastDim - kernel) / stride + 1UL;
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = self->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outLen;
    }
    &FloatTensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    &FloatTensor idxCache = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc(outShape); }
    unsafe {
        unsigned long rows = out->size / outLen;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long inBase = r * lastDim;
            unsigned long outBase = r * outLen;
            unsigned long o = 0UL;
            while (o < outLen) {
                unsigned long winStart = o * stride;
                float best = self->data[inBase + winStart];
                unsigned long bestIdx = inBase + winStart;
                unsigned long k = 1UL;
                while (k < kernel) {
                    float v = self->data[inBase + winStart + k];
                    if (v > best) { best = v; bestIdx = inBase + winStart + k; }
                    k = k + 1UL;
                }
                out->data[outBase + o] = best;
                idxCache->data[outBase + o] = (float)bestIdx;
                o = o + 1UL;
            }
            r = r + 1UL;
        }
        out->requiresGrad = self->requiresGrad;
        if (out->requiresGrad) {
            out->parents.push(self);
            out->parents.push(idxCache);
            out->gradFn = __maxpool1d_backward;
        }
    }
    return out;
}

// ── 1D average pooling ───────────────────────────────────────────────────────
&FloatTensor FloatTensor::avgpool1d(unsigned long kernel, unsigned long stride) const {
    unsafe { tensor_host_raw_((FloatTensor*)self); }
    unsigned long ndim; unsigned long lastDim;
    unsafe { ndim = self->ndim; lastDim = self->shape[ndim - 1UL]; }
    unsigned long outLen = (lastDim - kernel) / stride + 1UL;
    unsigned long* outShape;
    unsafe { outShape = (unsigned long*)alloc_raw_(checked_mul_size(sizeof(unsigned long), ndim)); }
    unsafe {
        unsigned long i = 0UL;
        while (i < ndim) { outShape[i] = self->shape[i]; i = i + 1UL; }
        outShape[ndim - 1UL] = outLen;
    }
    &FloatTensor out = __tensor_alloc((const unsigned long*)outShape, ndim, 0);
    unsafe { dealloc(outShape); }
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
                while (k < kernel) { sum = sum + self->data[inBase + winStart + k]; k = k + 1UL; }
                out->data[outBase + o] = sum * invK;
                o = o + 1UL;
            }
            r = r + 1UL;
        }
        // See tensor_nn.h's __avgpool1d_backward comment for the base-10000 choice.
        out->extraScalar = (float)(kernel * 10000UL + stride);
    }
    __tensor_link1(out, self, __avgpool1d_backward);
    return out;
}

} // namespace std
