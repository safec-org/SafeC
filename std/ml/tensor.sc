// SafeC Standard Library — Tensor implementation (see tensor.h).
#pragma once
#include <std/ml/tensor.h>
#include <std/collections/vec.h>
#include <std/collections/vec.sc>
#include <std/mem.sc>

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

static struct Tensor* __tensor_alloc(const unsigned long* shape, unsigned long ndim, int requiresGrad) {
    unsafe {
        struct Tensor* t = (struct Tensor*)malloc(sizeof(struct Tensor));

        unsigned long size = 1UL;
        unsigned long* shapeCopy = (unsigned long*)malloc(sizeof(unsigned long) * ndim);
        unsigned long i = 0UL;
        while (i < ndim) {
            shapeCopy[i] = shape[i];
            size = size * shape[i];
            i = i + 1UL;
        }

        double* buf = (double*)malloc(sizeof(double) * size);

        t->data = (&heap double)buf;
        t->shape = (&heap unsigned long)shapeCopy;
        t->ndim = ndim;
        t->size = size;
        t->requiresGrad = requiresGrad;
        t->grad = (&heap double)0;
        t->gradFn = (void*)0;
        t->parents = vec_new(sizeof(struct Tensor*));
        t->visited = 0;
        t->extraScalar = 0.0;
        return t;
    }
}

static void __tensor_ensure_grad(struct Tensor* t) {
    unsafe {
        if (t->grad == (double*)0) {
            double* g = (double*)malloc(sizeof(double) * t->size);
            unsigned long i = 0UL;
            while (i < t->size) { g[i] = 0.0; i = i + 1UL; }
            t->grad = (&heap double)g;
        }
    }
}

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

// Same shape/alloc as tensor_zeros_like, but skips the zero-fill --
// profiled (via 'sample' on a real training run): tensor_add/sub/mul/
// scale/relu were all allocating their output through tensor_zeros_like
// and then immediately overwriting every single element in the very next
// loop, making the zero-fill (a real, measured cost -- __bzero showed up
// as its own line in the profile) pure wasted memory-bandwidth work. Any
// caller using this MUST write every element before returning; it is not
// a safe drop-in replacement for tensor_zeros_like everywhere.
static struct Tensor* __tensor_alloc_uninit_like(struct Tensor* t) {
    unsafe { return __tensor_alloc((const unsigned long*)t->shape, t->ndim, 0); }
}

&Tensor tensor_fill(&Tensor t, double v) {
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->data[i] = v; i = i + 1UL; }
    }
    return t;
}

// ── Autograd plumbing ────────────────────────────────────────────────────────

static struct Tensor* __tensor_parent(struct Tensor* t, unsigned long idx) {
    unsafe {
        struct Tensor** pp = (struct Tensor**)t->parents.get_raw(idx);
        return *pp;
    }
}

static void __tensor_accumulate(struct Tensor* dst, const double* delta) {
    __tensor_ensure_grad(dst);
    unsafe {
        unsigned long i = 0UL;
        while (i < dst->size) { dst->grad[i] = dst->grad[i] + delta[i]; i = i + 1UL; }
    }
}

static void __add_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const double*)selfT->grad);
        if (b->requiresGrad) __tensor_accumulate(b, (const double*)selfT->grad);
    }
}

// __sub_backward through __sum_backward below used to each build a
// throwaway 'delta' buffer (malloc, fill it, __tensor_accumulate it into
// the destination's grad, free it) instead of accumulating straight into
// the destination -- a full extra malloc/free and a full extra pass over
// the data for every single backward call. Profiled: matmul_backward
// alone (same shape of waste, fixed separately below) was ~43% of a real
// training run's samples; these elementwise ones are individually
// smaller but the pattern is identical and just as pointless to keep,
// now that __tensor_ensure_grad can be called up front and the loop can
// write straight into 'grad[i] +=' instead of 'delta[i] ='.

static void __sub_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const double*)selfT->grad);
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            unsigned long i = 0UL;
            while (i < selfT->size) { b->grad[i] = b->grad[i] - selfT->grad[i]; i = i + 1UL; }
        }
    }
}

static void __mul_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            unsigned long i = 0UL;
            while (i < selfT->size) { a->grad[i] = a->grad[i] + selfT->grad[i] * b->data[i]; i = i + 1UL; }
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            unsigned long i = 0UL;
            while (i < selfT->size) { b->grad[i] = b->grad[i] + selfT->grad[i] * a->data[i]; i = i + 1UL; }
        }
    }
}

static void __scale_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        unsigned long i = 0UL;
        while (i < selfT->size) { a->grad[i] = a->grad[i] + selfT->grad[i] * selfT->extraScalar; i = i + 1UL; }
    }
}

static void __relu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        unsigned long i = 0UL;
        while (i < selfT->size) {
            if (a->data[i] > 0.0) { a->grad[i] = a->grad[i] + selfT->grad[i]; }
            i = i + 1UL;
        }
    }
}

static void __sum_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        double seed = selfT->grad[0];
        unsigned long i = 0UL;
        while (i < a->size) { a->grad[i] = a->grad[i] + seed; i = i + 1UL; }
    }
}

static void __matmul_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        unsigned long m = a->shape[0];
        unsigned long k = a->shape[1];
        unsigned long n = b->shape[1];

        // dA[m,k] = dC[m,n] . B^T[n,k]      dA[i,j] = sum_p dC[i,p]*B[j,p]
        // dB[k,n] = A^T[k,m] . dC[m,n]      dB[i,j] = sum_p A[p,i]*dC[p,j]
        //
        // These DO still build a throwaway dA/dB buffer and copy it in
        // via __tensor_accumulate, unlike every other backward function
        // in this file -- tried fusing the accumulation straight into
        // a->grad/b->grad here too (the same change that was a clear win
        // for add/sub/mul/scale/relu/sum below), and measured it with
        // 'sample' on a real training run instead of assuming it was
        // also a win: it wasn't. __matmul_backward's sample count nearly
        // *doubled* (1751 -> 3021 out of ~4000 total). a->grad/b->grad
        // are struct-field loads the compiler can't prove don't alias
        // selfT->grad/a->data/b->data inside this tight, O(m*k*n)
        // innermost loop, so fusing the write into them defeats
        // auto-vectorization here specifically -- a cost that swamps the
        // one extra malloc+pass+free this keeps, because this loop runs
        // vastly more times than the elementwise ones do. A fresh local
        // buffer has no such aliasing ambiguity, so it vectorizes fine.
        if (a->requiresGrad) {
            double* dA = (double*)malloc(sizeof(double) * m * k);
            unsigned long i = 0UL;
            while (i < m) {
                unsigned long j = 0UL;
                while (j < k) {
                    double acc = 0.0;
                    unsigned long p = 0UL;
                    while (p < n) {
                        acc = acc + selfT->grad[i * n + p] * b->data[j * n + p];
                        p = p + 1UL;
                    }
                    dA[i * k + j] = acc;
                    j = j + 1UL;
                }
                i = i + 1UL;
            }
            __tensor_accumulate(a, (const double*)dA);
            free((void*)dA);
        }
        if (b->requiresGrad) {
            // Same i,j,p -> p,i,j reordering as tensor_matmul's forward
            // pass, for the same reason: 'a->data[p*k+i]' with p as the
            // innermost loop variable strides by k per step, defeating
            // both cache locality and auto-vectorization. Looping p
            // outermost instead makes the innermost loop over j a
            // sequential 'dB[i,j] += aVal * selfT->grad[p,j]'
            // accumulation, with aVal ('A[p,i]') loaded once per (p,i)
            // pair rather than re-derived in the hot loop.
            double* dB = (double*)malloc(sizeof(double) * k * n);
            unsigned long z = 0UL;
            while (z < k * n) { dB[z] = 0.0; z = z + 1UL; }
            unsigned long p = 0UL;
            while (p < m) {
                unsigned long i = 0UL;
                while (i < k) {
                    double aVal = a->data[p * k + i];
                    unsigned long j = 0UL;
                    while (j < n) {
                        dB[i * n + j] = dB[i * n + j] + aVal * selfT->grad[p * n + j];
                        j = j + 1UL;
                    }
                    i = i + 1UL;
                }
                p = p + 1UL;
            }
            __tensor_accumulate(b, (const double*)dB);
            free((void*)dB);
        }
    }
}

// ── Forward ops ──────────────────────────────────────────────────────────────

static void __tensor_link2(struct Tensor* out, struct Tensor* a, struct Tensor* b, void* backwardFn) {
    unsafe {
        out->requiresGrad = a->requiresGrad || b->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push((const void*)&a);
        out->parents.push((const void*)&b);
        out->gradFn = backwardFn;
    }
}

static void __tensor_link1(struct Tensor* out, struct Tensor* a, void* backwardFn) {
    unsafe {
        out->requiresGrad = a->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push((const void*)&a);
        out->gradFn = backwardFn;
    }
}

&Tensor tensor_add(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

&Tensor tensor_mul(const &Tensor a, const &Tensor b) {
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

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

// ── tensor_backward() ─────────────────────────────────────────────────────────

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

} // namespace std
