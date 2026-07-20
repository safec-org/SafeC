#pragma once
// SafeC Standard Library — Tensor: CPU tensors with reverse-mode autograd.
//
// PyTorch-style ergonomics (build a graph implicitly by calling ops;
// tensor_backward() walks it) over a plain flat f64 buffer — no unified-
// memory story here (that's MPS's job; see gpu_mps.h), just row-major
// double-precision arrays with a recorded computation graph. Scope: 1D
// and 2D tensors (vectors and matrices — everything a small MLP/attention
// block needs), add/sub/mul/relu/sum/matmul, and reverse-mode autodiff
// over that op set. Not a general n-dimensional array library — no
// broadcasting beyond 2D, no int/complex dtypes, no in-place ops.
//
// Internal helpers (__tensor_alloc, __tensor_link1/2, every __X_backward
// function, etc.) are defined right here in the header as 'static'
// functions, not in tensor.sc — this is what lets tensor.sc, tensor_nn.sc,
// and the tensor_<gpu-backend>.sc family compile as genuinely separate
// translation units (each #includes this header, gets its own private
// copy of these helpers, and is linked against the others' *public*
// symbols — tensor_add, tensor_matmul, etc. — at build time) instead of
// every file needing to textually #include tensor.sc's entire body just
// to reach a handful of internal-linkage helpers it happens to reuse.
#include <std/collections/vec.h>

namespace std {

// The static helpers below call libc malloc/free directly (not
// std::alloc/dealloc, so no dependency on mem.sc's allocator machinery
// for this hot path) -- declared here rather than pulled in via
// std/mem.sc, matching every other .sc file in std/ that touches libc
// memory functions directly (see mem.sc, heap.sc, result.sc, etc., each
// of which declares its own copy of these same externs rather than
// including one another's .sc file for them).
extern void* malloc(unsigned long size);
extern void  free(void* ptr);

typedef fn void(void* selfTensor) TensorBackwardFn;

struct Tensor {
    &heap double  data;       // flat row-major buffer, length == size
    &heap unsigned long shape; // shape[0..ndim)
    unsigned long ndim;
    unsigned long size;        // product of shape
    int           requiresGrad;
    &heap double  grad;        // same length as data; only allocated if requiresGrad
    // Autograd graph edges. 'gradFn' is void* (not TensorBackwardFn)
    // purely to dodge the typedef-ordering constraint every other
    // callback-carrying struct in std/ hits (see std/gui/gui_widget.h's
    // header comment for the exact same workaround) — cast at the handful
    // of call sites that read/write it.
    void*         gradFn;
    struct Vec    parents;     // Vec<struct Tensor*> — inputs this tensor was computed from
    int           visited;     // topological-sort scratch flag, used and cleared by tensor_backward()
    double        extraScalar; // op-specific constant (e.g. tensor_scale's k) the gradFn needs

    // Element access (row-major; 'idx1D' for a rank-1 tensor,
    // 'row,col' for a rank-2 one).
    double  at1(unsigned long i) const;
    double  at2(unsigned long r, unsigned long c) const;
    void    set1(unsigned long i, double v);
    void    set2(unsigned long r, unsigned long c, double v);

    void    free();
};

// Every function below takes/returns 'Tensor' by a region-less reference
// ('&Tensor'/'const &Tensor' — see README's "Outliving references"
// section): a Tensor is never null anywhere in this library, and nothing
// here cares whether the caller's tensor is heap/stack/static/arena-
// backed, so pinning one specific region would only narrow who can call
// these for no safety benefit. Internally, tensors are still allocated
// with a plain 'malloc' (see '__tensor_alloc' below) and freed with
// 'free()' via 'Tensor::free()' below — 'free()' takes no argument beyond
// 'self' precisely because there's no separate heap-region bookkeeping to
// release; call it, then let the reference go out of scope.

// ── Construction ─────────────────────────────────────────────────────────────
&Tensor tensor_new_1d(unsigned long n, int requiresGrad);
&Tensor tensor_new_2d(unsigned long rows, unsigned long cols, int requiresGrad);
// Copies 'values' (length must equal the tensor's size) into a freshly
// allocated tensor of the given shape.
&Tensor tensor_from_1d(const double* values, unsigned long n, int requiresGrad);
&Tensor tensor_from_2d(const double* values, unsigned long rows, unsigned long cols, int requiresGrad);
&Tensor tensor_zeros_like(const &Tensor t);
&Tensor tensor_fill(&Tensor t, double v);

// ── Forward ops (each records a backward edge when either operand
// requires grad) ────────────────────────────────────────────────────────────
&Tensor tensor_add(const &Tensor a, const &Tensor b);
&Tensor tensor_sub(const &Tensor a, const &Tensor b);
&Tensor tensor_mul(const &Tensor a, const &Tensor b);      // elementwise
&Tensor tensor_scale(const &Tensor a, double k);            // a * scalar k
&Tensor tensor_matmul(const &Tensor a, const &Tensor b);   // 2D x 2D
&Tensor tensor_sum(const &Tensor a);                        // -> scalar (1-element) tensor
// tensor_relu lives in tensor_nn.h alongside the rest of the activation
// functions (sigmoid, tanh, elu, gelu, silu, glu, swiglu) -- include that
// header too wherever relu is needed.

// ── Autograd ─────────────────────────────────────────────────────────────────
// 't' must be a scalar (size == 1, e.g. a loss). Seeds t->grad = 1, walks
// the graph in reverse topological order, and accumulates gradients into
// every requiresGrad ancestor's 'grad' buffer (allocating it on first
// write). Safe to call multiple times — gradients accumulate across
// calls, matching PyTorch's default (call tensor_zero_grad() between
// optimizer steps if that's not what you want).
void tensor_backward(&Tensor t);
void tensor_zero_grad(&Tensor t);

// ── Gradient mode (no_grad) ──────────────────────────────────────────────────
// Global, not per-TU: every tensor_<op> across every separately-compiled
// file linked into the same program shares one flag (that's why these
// are ordinary externally-linked functions, defined once in tensor.sc,
// not 'static' — unlike the internal helpers below, this state must be
// visible and consistent everywhere, not private per including file).
//
// Mirrors PyTorch's torch.no_grad(): while disabled, every tensor
// produced by tensor_add/matmul/etc. gets requiresGrad=0 regardless of
// its inputs' — __tensor_link1/__tensor_link2 (below) skip pushing to
// 'parents' and setting 'gradFn' entirely, which is the real cost this
// exists to avoid (Vec::push's first call allocates; skipping it removes
// a real std::alloc from every op call during inference). Doesn't affect
// LEAF tensors' own requiresGrad (tensor_new_2d's explicit argument is
// unaffected) — only what happens when ops COMPOSE tensors.
int  tensor_is_grad_enabled();
void tensor_set_grad_enabled(int enabled);

// ══ Internal helpers (static — each including file gets its own copy) ═══════
// Not part of the public API; used by tensor.sc's own op implementations
// and reused directly (as raw function-pointer values, e.g.
// '(void*)__add_backward') by tensor_nn.sc and the tensor_<backend>.sc
// family so a GPU-computed forward result can link into the exact same
// backward math as the CPU path, without duplicating that math.

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

static void __tensor_link2(struct Tensor* out, struct Tensor* a, struct Tensor* b, void* backwardFn) {
    unsafe {
        if (!tensor_is_grad_enabled()) { out->requiresGrad = 0; return; }
        out->requiresGrad = a->requiresGrad || b->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push((const void*)&a);
        out->parents.push((const void*)&b);
        out->gradFn = backwardFn;
    }
}

static void __tensor_link1(struct Tensor* out, struct Tensor* a, void* backwardFn) {
    unsafe {
        if (!tensor_is_grad_enabled()) { out->requiresGrad = 0; return; }
        out->requiresGrad = a->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push((const void*)&a);
        out->gradFn = backwardFn;
    }
}

// ── Backward functions (one per forward op in tensor.sc) ─────────────────────
// __sub_backward through __sum_backward accumulate straight into the
// destination's grad (x->grad[i] +=) instead of building a throwaway
// 'delta' buffer and calling __tensor_accumulate -- profiled (via 'sample'
// on a real training run): the old throwaway-buffer version cost a full
// extra malloc/free and a full extra pass over the data on every single
// backward call, for no benefit at this size. __matmul_backward is the one
// exception that still uses a throwaway buffer, for the opposite reason --
// see its own comment.

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
        // here -- tried fusing the accumulation straight into a->grad/
        // b->grad here too (the same change that was a clear win for
        // add/sub/mul/scale/relu/sum), and measured it with 'sample' on a
        // real training run instead of assuming it was also a win: it
        // wasn't. __matmul_backward's sample count nearly *doubled* (1751
        // -> 3021 out of ~4000 total). a->grad/b->grad are struct-field
        // loads the compiler can't prove don't alias selfT->grad/a->data/
        // b->data inside this tight, O(m*k*n) innermost loop, so fusing
        // the write into them defeats auto-vectorization here
        // specifically -- a cost that swamps the one extra malloc+pass+
        // free this keeps, because this loop runs vastly more times than
        // the elementwise ones do. A fresh local buffer has no such
        // aliasing ambiguity, so it vectorizes fine.
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

} // namespace std
