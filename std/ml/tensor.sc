// SafeC Standard Library — Tensor implementation (see tensor.h). Internal
// 'static' helpers (allocation, autograd-graph plumbing, every
// __X_backward function) are defined in tensor.h itself, not here -- see
// that header's own comment for why. This file only implements the
// public API declared there, and #includes only headers (never another
// .sc file): tensor.sc, tensor_nn.sc, and each tensor_<gpu-backend>.sc
// are meant to be compiled as separate translation units and linked
// together at build time, not textually pasted into one another.
#pragma once
#include <std/ml/tensor.h>

namespace std {

// Multithreading tensor_matmul's row loop (spawn N worker threads, one per
// row range, join) was tried and measured *slower* on this benchmark, not
// faster -- reverted. A synthetic isolated measurement (spawn+join a single
// no-op thread, ~20us; a single 64x128@128x256 matmul, ~550us sequentially)
// suggested real headroom. The actual training loop calls tensor_matmul
// ~2,200 times (100 steps x 2 forward matmuls, plus 1000 inference passes x
// 2), each spawning 3 new threads under that scheme -- ~6,600 real
// thread_create/join pairs total, not one. At that call frequency, real
// per-call OS thread-creation cost (stack allocation, kernel scheduling
// churn) turned out far higher than the isolated single-spawn measurement
// implied: train_ms went from ~165ms to ~212ms and inference_ms/1000 from
// ~517ms to ~946ms, both regressions, not the improvement the isolated
// numbers predicted. A persistent thread pool (spawn once, reuse across
// every matmul call) would avoid this repeated-spawn cost and might still
// be a net win -- not implemented here since std/ has no such primitive
// yet and it's real, standalone infrastructure work, not a one-line
// change like this was.

// ══ Tensor methods ═══════════════════════════════════════════════════════════

inline double Tensor::at1(unsigned long i) const {
    unsafe { return self.data[i]; }
}

inline double Tensor::at2(unsigned long r, unsigned long c) const {
    unsafe { return self.data[r * self.shape[1] + c]; }
}

inline void Tensor::set1(unsigned long i, double v) {
    unsafe { self.data[i] = v; }
}

inline void Tensor::set2(unsigned long r, unsigned long c, double v) {
    unsafe { self.data[r * self.shape[1] + c] = v; }
}

inline void Tensor::free() {
    unsafe {
        if (self.data != (double*)0) { std::free((void*)self.data); self.data = (double*)0; }
        if (self.shape != (unsigned long*)0) { std::free((void*)self.shape); self.shape = (unsigned long*)0; }
        if (self.grad != (double*)0) { std::free((void*)self.grad); self.grad = (double*)0; }
        self.parents.free();
    }
}

// ══ Allocation & lifecycle ═══════════════════════════════════════════════════

&Tensor tensor_new_1d(unsigned long n, int requiresGrad) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    struct Tensor* t = __tensor_alloc(shape, 1UL, requiresGrad);
    tensor_fill(t, 0.0);
    return t;
}

&Tensor tensor_new_2d(unsigned long rows, unsigned long cols, int requiresGrad) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    struct Tensor* t = __tensor_alloc(shape, 2UL, requiresGrad);
    tensor_fill(t, 0.0);
    return t;
}

&Tensor tensor_from_1d(const double* values, unsigned long n, int requiresGrad) {
    struct Tensor* t = tensor_new_1d(n, requiresGrad);
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { t->data[i] = values[i]; i = i + 1UL; }
    }
    return t;
}

&Tensor tensor_from_2d(const double* values, unsigned long rows, unsigned long cols, int requiresGrad) {
    struct Tensor* t = tensor_new_2d(rows, cols, requiresGrad);
    unsigned long total = rows * cols;
    unsafe {
        unsigned long i = 0UL;
        while (i < total) { t->data[i] = values[i]; i = i + 1UL; }
    }
    return t;
}

&Tensor tensor_zeros_like(const &Tensor t) {
    struct Tensor* out;
    unsafe { out = __tensor_alloc((const unsigned long*)t->shape, t->ndim, 0); }
    tensor_fill(out, 0.0);
    return out;
}

&Tensor tensor_fill(&Tensor t, double v) {
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->data[i] = v; i = i + 1UL; }
    }
    return t;
}

// ══ Elementwise & reduction ops ══════════════════════════════════════════════
// Backward math for each of these lives in tensor.h, right next to the
// helpers it shares with tensor_nn.sc and the GPU backends -- see
// __add_backward etc. there.

// ── Add ──────────────────────────────────────────────────────────────────────
&Tensor tensor_add(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

// ── Sub ──────────────────────────────────────────────────────────────────────
&Tensor tensor_sub(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

// ── Mul (elementwise) ──────────────────────────────────────────────────────────
&Tensor tensor_mul(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

// ── Scale (multiply by a scalar constant) ─────────────────────────────────────
&Tensor tensor_scale(const &Tensor a, double k) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] * k; i = i + 1UL; }
    }
    unsafe { out->extraScalar = k; }
    __tensor_link1(out, a, (void*)__scale_backward);
    return out;
}

// tensor_relu/__relu_backward moved to tensor_nn.h/.sc (see those files) --
// it's a neural-net activation function, not core tensor arithmetic, and
// tensor_nn.sc is where every other activation (sigmoid, tanh, elu, gelu,
// silu, glu, swiglu) already lives. Include tensor_nn.h wherever relu is
// needed (and link tensor_nn.sc at build time).

// ── Sum (full reduction to a 1-element tensor) ────────────────────────────────
&Tensor tensor_sum(const &Tensor a) {
    unsigned long one[1];
    unsafe { one[0] = 1UL; }
    struct Tensor* out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        double acc = 0.0;
        unsigned long i = 0UL;
        while (i < a->size) { acc = acc + a->data[i]; i = i + 1UL; }
        out->data[0] = acc;
    }
    __tensor_link1(out, a, (void*)__sum_backward);
    return out;
}

// ══ Matrix multiply ══════════════════════════════════════════════════════════

&Tensor tensor_matmul(const &Tensor a, const &Tensor b) {
    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    struct Tensor* out = tensor_new_2d(m, n, 0);
    // i,p,j loop order, not the textbook i,j,p: with i,j,p, the innermost
    // loop's 'b->data[p*n+j]' access strides by n bytes per step (p is
    // the loop variable, but n is the stride) -- a cache miss on nearly
    // every access for any matrix past L1 size, and not something the
    // backend's auto-vectorizer can turn into packed SIMD loads either,
    // since consecutive iterations touch non-adjacent memory. Reordering
    // to i,p,j instead makes the innermost loop a running
    // 'out[i,j] += a[i,p] * b[p,j]' accumulation over j -- both b's row
    // and out's row are read/written sequentially, which is both
    // cache-friendly and exactly the shape LLVM's loop vectorizer
    // recognizes and turns into packed multiply-add instructions at -O2.
    // Verified: same output values as the previous i,j,p version on every
    // existing tensor.sc correctness check, meaningfully faster on the
    // MLP training benchmark on safec-docs's Benchmarks page.
    unsafe {
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
    __tensor_link2(out, a, b, (void*)__matmul_backward);
    return out;
}

// ══ Graph traversal & tensor_backward() ══════════════════════════════════════

static void __tensor_toposort(struct Tensor* t, struct Vec* order) {
    unsafe { if (t->visited) return; }
    unsafe { t->visited = 1; }
    unsigned long n;
    unsafe { n = t->parents.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        __tensor_toposort(__tensor_parent(t, i), order);
        i = i + 1UL;
    }
    unsafe { order->push((const void*)&t); }
}

void tensor_backward(&Tensor t) {
    __tensor_ensure_grad(t);
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->grad[i] = 1.0; i = i + 1UL; }
    }

    struct Vec order = vec_new(sizeof(struct Tensor*));
    __tensor_toposort(t, &order);

    unsafe {
        unsigned long n = order.length();
        long long idx = (long long)n - 1LL;
        while (idx >= 0LL) {
            struct Tensor** pp = (struct Tensor**)order.get_raw((unsigned long)idx);
            struct Tensor* node = *pp;
            if (node->gradFn != (void*)0) {
                TensorBackwardFn backwardFn = (TensorBackwardFn)node->gradFn;
                backwardFn((void*)node);
            }
            idx = idx - 1LL;
        }

        unsigned long j = 0UL;
        while (j < n) {
            struct Tensor** pp2 = (struct Tensor**)order.get_raw(j);
            struct Tensor* node2 = *pp2;
            node2->visited = 0;
            j = j + 1UL;
        }
        order.free();
    }
}

void tensor_zero_grad(&Tensor t) {
    unsafe {
        if (t->grad != (double*)0) {
            unsigned long i = 0UL;
            while (i < t->size) { t->grad[i] = 0.0; i = i + 1UL; }
        }
    }
}

// ── Gradient mode (no_grad) ──────────────────────────────────────────────────
// Backing storage stays a private 'static' in this one TU -- every other
// file only ever touches it through the two externally-linked accessor
// functions below, so there's exactly one copy of the flag no matter how
// many files get linked together (see tensor.h's comment on why that
// matters here, unlike the header's own per-TU-private static helpers).
static int __gradEnabled = 1;

int tensor_is_grad_enabled() { return __gradEnabled; }
void tensor_set_grad_enabled(int enabled) { __gradEnabled = enabled; }

} // namespace std
