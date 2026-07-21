#pragma once
// SafeC Standard Library — neural-net building blocks on top of std::ml's
// Tensor/autograd core (tensor.h/tensor.sc): activation functions, the
// Adam optimizer, and 1D pooling. Separate file from tensor.sc for the
// same reason tensor_gpu.h is separate — an independently useful slice of
// functionality, not a change to the core type or its existing ops.
//
// Every activation below is elementwise, allocates via
// __tensor_alloc_uninit_like (like tensor_relu), and links into the same
// autograd graph via its own backward function — same shape as
// tensor.sc's existing tensor_relu/tensor_scale, just more of them.
//
// Internal 'static' helpers (every __X_backward function) are defined
// right here, not in tensor_nn.sc — same reasoning as tensor.h's own
// header comment: each including .sc file gets its own private copy,
// which is what lets tensor_nn.sc and the tensor_<gpu-backend>.sc family
// compile as separate translation units and link together at build time
// instead of textually pasting tensor_nn.sc's whole body into each backend.
#include <std/ml/tensor.h>
#include <std/math.h>
#include <std/mem.h>

namespace std {

// Moved here from tensor.h -- a neural-net activation function, grouped
// with the rest of them rather than with tensor.h's core arithmetic ops.
&Tensor tensor_relu(const &Tensor a);
&Tensor tensor_sigmoid(const &Tensor a);
&Tensor tensor_tanh_t(const &Tensor a);      // 'tensor_tanh' would collide with math.h's tanh_f-style naming convention if this were 'tanh'; matches this file's other tensor_-prefixed names instead
&Tensor tensor_softmax(const &Tensor a);     // last-axis softmax (row-wise for a rank-2 tensor)
&Tensor tensor_elu(const &Tensor a, float alpha);
&Tensor tensor_gelu(const &Tensor a);        // tanh approximation (the common fast form, not the exact erf one)
&Tensor tensor_silu(const &Tensor a);        // aka Swish: x * sigmoid(x)

// GLU/SwiGLU split the input's last axis in half (must be even) into
// [a1, a2] and gate one half with (a function of) the other:
//   GLU(x)    = a1 * sigmoid(a2)
//   SwiGLU(x) = silu(a1) * a2
// Output shape is the input's shape with the last axis halved.
&Tensor tensor_glu(const &Tensor a);
&Tensor tensor_swiglu(const &Tensor a);

// ── Adam optimizer ───────────────────────────────────────────────────────────
// Per-parameter state (first/second moment buffers, same size as the
// parameter) plus the shared hyperparameters and step count. One
// AdamState per parameter tensor (W1, W2, ... each get their own) --
// unlike sgd_update (a pure function of data/grad/lr with no persistent
// state), Adam needs m/v carried across steps.
struct AdamState {
    &heap float m;
    &heap float v;
    unsigned long size;
    float lr;
    float beta1;
    float beta2;
    float eps;
    unsigned long t; // step count, starts at 0, incremented by adam_step
};

struct AdamState adam_new(unsigned long size, float lr, float beta1, float beta2, float eps);
// Convenience default (lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8 — the
// values from the original Adam paper and every framework's default).
struct AdamState adam_new_default(unsigned long size, float lr);
void adam_step(&AdamState state, &Tensor param);
void adam_free(&AdamState state);

// ── 1D pooling ────────────────────────────────────────────────────────────────
// Pools along the tensor's LAST axis with the given kernel size and
// stride (kernel == stride is non-overlapping "classic" pooling; stride <
// kernel gives overlapping windows). Every axis before the last is
// treated as an independent batch/channel dimension pooled the same way
// (e.g. a (batch, channels, length) rank-3 tensor pools each
// (batch, channel) row's length axis independently) -- this is genuine
// N-D-aware pooling via Tensor's existing arbitrary-ndim shape, not
// limited to rank-2.
&Tensor tensor_maxpool1d(const &Tensor a, unsigned long kernel, unsigned long stride);
&Tensor tensor_avgpool1d(const &Tensor a, unsigned long kernel, unsigned long stride);

// ══ Internal helpers (static — each including file gets its own copy) ═══════

static void __relu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        unsigned long i = 0UL;
        while (i < selfT->size) {
            if (a->data[i] > (float)0.0) { a->grad[i] = a->grad[i] + selfT->grad[i]; }
            i = i + 1UL;
        }
    }
}

static void __sigmoid_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long i = 0UL;
            while (i < selfT->size) {
                float y = selfT->data[i];
                a->grad[i] = a->grad[i] + selfT->grad[i] * y * ((float)1.0 - y);
                i = i + 1UL;
            }
        }
    }
}

static void __tanh_t_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long i = 0UL;
            while (i < selfT->size) {
                float y = selfT->data[i];
                a->grad[i] = a->grad[i] + selfT->grad[i] * ((float)1.0 - y * y);
                i = i + 1UL;
            }
        }
    }
}

static void __softmax_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long lastDim = selfT->shape[selfT->ndim - 1UL];
            unsigned long rows = selfT->size / lastDim;
            unsigned long r = 0UL;
            while (r < rows) {
                unsigned long base = r * lastDim;
                float dot = (float)0.0;
                unsigned long j = 0UL;
                while (j < lastDim) {
                    dot = dot + selfT->grad[base + j] * selfT->data[base + j];
                    j = j + 1UL;
                }
                j = 0UL;
                while (j < lastDim) {
                    float y = selfT->data[base + j];
                    a->grad[base + j] = a->grad[base + j] + y * (selfT->grad[base + j] - dot);
                    j = j + 1UL;
                }
                r = r + 1UL;
            }
        }
    }
}

static void __elu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        float alpha = selfT->extraScalar;
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long i = 0UL;
            while (i < selfT->size) {
                float x = a->data[i];
                float y = selfT->data[i];
                float d = (x > (float)0.0) ? (float)1.0 : (y + alpha);
                a->grad[i] = a->grad[i] + selfT->grad[i] * d;
                i = i + 1UL;
            }
        }
    }
}

static void __gelu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            float c = (float)0.7978845608028654; // sqrt(2/pi)
            unsigned long i = 0UL;
            while (i < selfT->size) {
                float x = a->data[i];
                float u = c * (x + (float)0.044715 * x * x * x);
                float t = tanh_f(u);
                float dudx = c * ((float)1.0 + (float)0.134145 * x * x);
                float d = (float)0.5 * ((float)1.0 + t) + (float)0.5 * x * ((float)1.0 - t * t) * dudx;
                a->grad[i] = a->grad[i] + selfT->grad[i] * d;
                i = i + 1UL;
            }
        }
    }
}

static void __silu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long i = 0UL;
            while (i < selfT->size) {
                float x = a->data[i];
                float s = (float)1.0 / ((float)1.0 + exp_f(-x));
                float d = s + x * s * ((float)1.0 - s);
                a->grad[i] = a->grad[i] + selfT->grad[i] * d;
                i = i + 1UL;
            }
        }
    }
}

static void __glu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long outLast = selfT->shape[selfT->ndim - 1UL];
            unsigned long inLast = outLast * 2UL;
            unsigned long rows = selfT->size / outLast;
            unsigned long r = 0UL;
            while (r < rows) {
                unsigned long outBase = r * outLast;
                unsigned long inBase = r * inLast;
                unsigned long j = 0UL;
                while (j < outLast) {
                    float a1 = a->data[inBase + j];
                    float a2 = a->data[inBase + outLast + j];
                    float s = (float)1.0 / ((float)1.0 + exp_f(-a2));
                    float dy = selfT->grad[outBase + j];
                    a->grad[inBase + j] = a->grad[inBase + j] + dy * s;
                    a->grad[inBase + outLast + j] = a->grad[inBase + outLast + j] + dy * a1 * s * ((float)1.0 - s);
                    j = j + 1UL;
                }
                r = r + 1UL;
            }
        }
    }
}

static void __swiglu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long outLast = selfT->shape[selfT->ndim - 1UL];
            unsigned long inLast = outLast * 2UL;
            unsigned long rows = selfT->size / outLast;
            unsigned long r = 0UL;
            while (r < rows) {
                unsigned long outBase = r * outLast;
                unsigned long inBase = r * inLast;
                unsigned long j = 0UL;
                while (j < outLast) {
                    float a1 = a->data[inBase + j];
                    float a2 = a->data[inBase + outLast + j];
                    float s1 = (float)1.0 / ((float)1.0 + exp_f(-a1));
                    float silu1 = a1 * s1;
                    float dsilu1 = s1 + a1 * s1 * ((float)1.0 - s1);
                    float dy = selfT->grad[outBase + j];
                    a->grad[inBase + j] = a->grad[inBase + j] + dy * a2 * dsilu1;
                    a->grad[inBase + outLast + j] = a->grad[inBase + outLast + j] + dy * silu1;
                    j = j + 1UL;
                }
                r = r + 1UL;
            }
        }
    }
}

static void __maxpool1d_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* idxCache = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long i = 0UL;
            while (i < selfT->size) {
                unsigned long srcIdx = (unsigned long)idxCache->data[i];
                a->grad[srcIdx] = a->grad[srcIdx] + selfT->grad[i];
                i = i + 1UL;
            }
        }
    }
}

static void __avgpool1d_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            // Base 10000, not the double-precision predecessor's
            // 1000000 -- float32 only represents integers exactly up to
            // 2^24 (~16.7M), so kernel*BASE+stride must stay well under
            // that. 10000 covers any realistic kernel/stride (both
            // usually single- or double-digit) with a wide margin.
            unsigned long encoded = (unsigned long)selfT->extraScalar;
            unsigned long kernel = encoded / 10000UL;
            unsigned long stride = encoded % 10000UL;
            unsigned long lastDim = a->shape[a->ndim - 1UL];
            unsigned long outLen = selfT->shape[selfT->ndim - 1UL];
            unsigned long rows = selfT->size / outLen;
            float invK = (float)1.0 / (float)kernel;
            unsigned long r = 0UL;
            while (r < rows) {
                unsigned long inBase = r * lastDim;
                unsigned long outBase = r * outLen;
                unsigned long o = 0UL;
                while (o < outLen) {
                    unsigned long winStart = o * stride;
                    float dy = selfT->grad[outBase + o] * invK;
                    unsigned long k = 0UL;
                    while (k < kernel) {
                        a->grad[inBase + winStart + k] = a->grad[inBase + winStart + k] + dy;
                        k = k + 1UL;
                    }
                    o = o + 1UL;
                }
                r = r + 1UL;
            }
        }
    }
}

} // namespace std
