// SafeC Standard Library — Tensor implementation (see tensor.h).
#pragma once
#include <std/ml/tensor.h>
#include <std/collections/vec.h>
#include <std/collections/vec.sc>
#include <std/mem.sc>

namespace std {

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

struct Tensor* tensor_new_1d(unsigned long n, int requiresGrad) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    struct Tensor* t = __tensor_alloc(shape, 1UL, requiresGrad);
    tensor_fill(t, 0.0);
    return t;
}

struct Tensor* tensor_new_2d(unsigned long rows, unsigned long cols, int requiresGrad) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    struct Tensor* t = __tensor_alloc(shape, 2UL, requiresGrad);
    tensor_fill(t, 0.0);
    return t;
}

struct Tensor* tensor_from_1d(const double* values, unsigned long n, int requiresGrad) {
    struct Tensor* t = tensor_new_1d(n, requiresGrad);
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { t->data[i] = values[i]; i = i + 1UL; }
    }
    return t;
}

struct Tensor* tensor_from_2d(const double* values, unsigned long rows, unsigned long cols, int requiresGrad) {
    struct Tensor* t = tensor_new_2d(rows, cols, requiresGrad);
    unsigned long total = rows * cols;
    unsafe {
        unsigned long i = 0UL;
        while (i < total) { t->data[i] = values[i]; i = i + 1UL; }
    }
    return t;
}

struct Tensor* tensor_zeros_like(const struct Tensor* t) {
    struct Tensor* out;
    unsafe { out = __tensor_alloc((const unsigned long*)t->shape, t->ndim, 0); }
    tensor_fill(out, 0.0);
    return out;
}

struct Tensor* tensor_fill(struct Tensor* t, double v) {
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

static void __sub_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const double*)selfT->grad);
        if (b->requiresGrad) {
            double* neg = (double*)malloc(sizeof(double) * selfT->size);
            unsigned long i = 0UL;
            while (i < selfT->size) { neg[i] = -(selfT->grad[i]); i = i + 1UL; }
            __tensor_accumulate(b, (const double*)neg);
            free((void*)neg);
        }
    }
}

static void __mul_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) {
            double* d = (double*)malloc(sizeof(double) * selfT->size);
            unsigned long i = 0UL;
            while (i < selfT->size) { d[i] = selfT->grad[i] * b->data[i]; i = i + 1UL; }
            __tensor_accumulate(a, (const double*)d);
            free((void*)d);
        }
        if (b->requiresGrad) {
            double* d = (double*)malloc(sizeof(double) * selfT->size);
            unsigned long i = 0UL;
            while (i < selfT->size) { d[i] = selfT->grad[i] * a->data[i]; i = i + 1UL; }
            __tensor_accumulate(b, (const double*)d);
            free((void*)d);
        }
    }
}

static void __scale_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        double* d = (double*)malloc(sizeof(double) * selfT->size);
        unsigned long i = 0UL;
        while (i < selfT->size) { d[i] = selfT->grad[i] * selfT->extraScalar; i = i + 1UL; }
        __tensor_accumulate(a, (const double*)d);
        free((void*)d);
    }
}

static void __relu_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        double* d = (double*)malloc(sizeof(double) * selfT->size);
        unsigned long i = 0UL;
        while (i < selfT->size) {
            d[i] = (a->data[i] > 0.0) ? selfT->grad[i] : 0.0;
            i = i + 1UL;
        }
        __tensor_accumulate(a, (const double*)d);
        free((void*)d);
    }
}

static void __sum_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        double seed = selfT->grad[0];
        double* d = (double*)malloc(sizeof(double) * a->size);
        unsigned long i = 0UL;
        while (i < a->size) { d[i] = seed; i = i + 1UL; }
        __tensor_accumulate(a, (const double*)d);
        free((void*)d);
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
            double* dB = (double*)malloc(sizeof(double) * k * n);
            unsigned long i = 0UL;
            while (i < k) {
                unsigned long j = 0UL;
                while (j < n) {
                    double acc = 0.0;
                    unsigned long p = 0UL;
                    while (p < m) {
                        acc = acc + a->data[p * k + i] * selfT->grad[p * n + j];
                        p = p + 1UL;
                    }
                    dB[i * n + j] = acc;
                    j = j + 1UL;
                }
                i = i + 1UL;
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

struct Tensor* tensor_add(struct Tensor* a, struct Tensor* b) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

struct Tensor* tensor_sub(struct Tensor* a, struct Tensor* b) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

struct Tensor* tensor_mul(struct Tensor* a, struct Tensor* b) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

struct Tensor* tensor_scale(struct Tensor* a, double k) {
    struct Tensor* out = tensor_zeros_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = a->data[i] * k; i = i + 1UL; }
    }
    unsafe { out->extraScalar = k; }
    __tensor_link1(out, a, (void*)__scale_backward);
    return out;
}

struct Tensor* tensor_relu(struct Tensor* a) {
    struct Tensor* out = tensor_zeros_like(a);
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

struct Tensor* tensor_sum(struct Tensor* a) {
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

struct Tensor* tensor_matmul(struct Tensor* a, struct Tensor* b) {
    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    struct Tensor* out = tensor_new_2d(m, n, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long j = 0UL;
            while (j < n) {
                double acc = 0.0;
                unsigned long p = 0UL;
                while (p < k) {
                    acc = acc + a->data[i * k + p] * b->data[p * n + j];
                    p = p + 1UL;
                }
                out->data[i * n + j] = acc;
                j = j + 1UL;
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

void tensor_backward(struct Tensor* t) {
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

void tensor_zero_grad(struct Tensor* t) {
    unsafe {
        if (t->grad != (double*)0) {
            unsigned long i = 0UL;
            while (i < t->size) { t->grad[i] = 0.0; i = i + 1UL; }
        }
    }
}

} // namespace std
