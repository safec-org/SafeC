#pragma once
#include <std/stddef.h>
// SafeC Standard Library — Tensor: CPU tensors with reverse-mode autograd.
//
// PyTorch-style ergonomics (build a graph implicitly by calling ops;
// tensor_backward() walks it) over a plain flat f32 buffer — no unified-
// memory story here (that's MPS's job; see gpu_mps.h), just row-major
// single-precision arrays with a recorded computation graph. Scope: 1D
// and 2D tensors (vectors and matrices — everything a small MLP/attention
// block needs), add/sub/mul/relu/sum/matmul, and reverse-mode autodiff
// over that op set. Not a general n-dimensional array library — no
// broadcasting beyond 2D, no int/complex dtypes, no in-place ops.
//
// float32, not float64: PyTorch defaults to float32, and this codebase's
// training-loop shapes measured ~30% faster on Accelerate's cblas_sgemm
// than cblas_dgemm for identical work, on top of halving memory
// bandwidth for every elementwise pass -- see tensor_blas.h. Also the
// same precision Metal itself supports (Metal has no double at all), so
// this removes the double<->float conversion every GPU-backed op used to
// pay at its CPU/GPU boundary (see gpu_mps.h's history) instead of just
// moving where the precision loss happens.
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
//
// Every op here is its own small function with one loop -- not just
// style. A hand-rolled training loop with everything (BLAS calls,
// elementwise passes) inlined into one large function was measured to
// make this compiler's -O2 optimizer hang (40+ minutes, not just slow)
// on exactly this kind of training-loop shape; splitting into small
// single-purpose functions avoided it entirely and let -O2's
// auto-vectorization actually run. Keeping every op as its own function
// isn't just for consistency -- it's load-bearing.
#include <std/collections/vec.h>
#include <std/simd/gemm.h>
#include <std/sync/parallel.h>
#include <std/simd/cpu.h>

namespace std {

// The static helpers below call libc malloc/free directly (not
// std::alloc/dealloc, so no dependency on mem.sc's allocator machinery
// for this hot path) -- declared here rather than pulled in via
// std/mem.sc, matching every other .sc file in std/ that touches libc
// memory functions directly (see mem.sc, heap.sc, result.sc, etc., each
// of which declares its own copy of these same externs rather than
// including one another's .sc file for them).
extern void* malloc(size_t size);
extern void  free(void* ptr);

typedef fn void(void* selfTensor) TensorBackwardFn;

// ── Tensor arenas ─────────────────────────────────────────────────────────────
// Every op allocates its output. In a training loop that is a malloc (and,
// for fresh memory, page faults) per op per step: on a small MLP that cost
// more than the arithmetic. A TensorArena is a chunked bump allocator for a
// step's temporaries: between tensor_arena_begin(&a) and tensor_arena_end()
// every tensor created on this thread (data, shape, gradient) comes from
// it, and tensor_arena_reset(&a) drops them all at once while keeping the
// memory -- already faulted in -- for the next step.
//
//   struct TensorArena a = tensor_arena_new(0UL);
//   while (step < steps) {
//       tensor_arena_begin(&a);
//       ... forward, tensor_backward, optimizer update ...
//       tensor_arena_end();
//       tensor_arena_reset(&a);     // this step's tensors are gone
//   }
//   tensor_arena_free(&a);
//
// Tensors created outside the arena (parameters) stay on the heap, and so do
// their gradients, so they survive resets. Tensor::free() on an arena tensor
// does nothing. A tensor must not be used after its arena is reset.
struct TensorArena {
    unsigned long first;       // chunk list (internal)
    unsigned long current;
    unsigned long chunkBytes;
    unsigned long tracked;     // tensors to release on reset (internal)
    unsigned long trackedCount;
    unsigned long trackedCap;
};

// chunkBytes 0 = 16 MiB chunks (larger requests get their own chunk).
struct TensorArena tensor_arena_new(unsigned long chunkBytes);
void tensor_arena_begin(&TensorArena a);   // this thread's new tensors go to a
void tensor_arena_end();                   // back to the heap
void tensor_arena_reset(&TensorArena a);   // drop every tensor in it, keep the memory
void tensor_arena_free(&TensorArena a);
// Bytes handed out since the last reset.
unsigned long tensor_arena_used(const &TensorArena a);

// Internal (tensor.h's allocation helpers): the arena new tensors use on this
// thread, or null; 64-byte-aligned memory from it; and registering a tensor
// so a reset releases what it holds outside the arena.
?&TensorArena tensor_arena_current_();
void* tensor_arena_alloc_raw_(&TensorArena a, unsigned long bytes);
void tensor_arena_track_(&TensorArena a, void* t);

struct Tensor {
    &heap float   data;       // flat row-major buffer, length == size
    &heap unsigned long shape; // shape[0..ndim)
    unsigned long ndim;
    unsigned long size;        // product of shape
    int           requiresGrad;
    &heap float   grad;        // same length as data; only allocated if requiresGrad
    // Autograd graph edges. 'gradFn' is void* (not TensorBackwardFn)
    // purely to dodge the typedef-ordering constraint every other
    // callback-carrying struct in std/ hits (see std/gui/gui_widget.h's
    // header comment for the exact same workaround) — cast at the handful
    // of call sites that read/write it.
    void*         gradFn;
    struct Vec<&Tensor> parents; // inputs this tensor was computed from
    int           visited;     // topological-sort scratch flag, used and cleared by tensor_backward()
    float         extraScalar; // op-specific constant (e.g. tensor_scale's k) the gradFn needs
    ?&TensorArena owner;       // the TensorArena that owns this tensor's memory, or null (heap)

    // Element access (row-major; 'idx1D' for a rank-1 tensor,
    // 'row,col' for a rank-2 one).
    float   at1(unsigned long i) const;
    float   at2(unsigned long r, unsigned long c) const;
    void    set1(unsigned long i, float v);
    void    set2(unsigned long r, unsigned long c, float v);

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
&Tensor tensor_from_1d(const &float values, unsigned long n, int requiresGrad);
&Tensor tensor_from_2d(const &float values, unsigned long rows, unsigned long cols, int requiresGrad);
&Tensor tensor_zeros_like(const &Tensor t);
&Tensor tensor_fill(&Tensor t, float v);

// ── Forward ops (each records a backward edge when either operand
// requires grad) ────────────────────────────────────────────────────────────
&Tensor tensor_add(const &Tensor a, const &Tensor b);
&Tensor tensor_sub(const &Tensor a, const &Tensor b);
&Tensor tensor_mul(const &Tensor a, const &Tensor b);      // elementwise
&Tensor tensor_scale(const &Tensor a, float k);             // a * scalar k
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

// ── Elementwise kernels ───────────────────────────────────────────────────────
// One loop per op over raw arrays, which the compiler vectorizes; arrays of
// TENSOR_EW_PAR or more elements are split across std/sync/parallel.h's
// thread pool. Used by the forward ops in tensor.sc / tensor_nn.sc and by
// the backward functions below.
#define TENSOR_EW_PAR 65536UL

#define EW_ADD        0   // out = a + b
#define EW_SUB        1   // out = a - b
#define EW_MUL        2   // out = a * b
#define EW_SCALE      3   // out = a * k
#define EW_RELU       4   // out = max(a, 0)
#define EW_FILL       5   // out = k
#define EW_COPY       6   // out = a
#define EW_ACC        7   // out += a
#define EW_DEC        8   // out -= a
#define EW_ACC_MUL    9   // out += a * b
#define EW_ACC_SCALE  10  // out += a * k
#define EW_ACC_RELU   11  // out += b > 0 ? a : 0
#define EW_ACC_K      12  // out += k
#define EW_ACC_SIGB   13  // out += a * b * (1 - b)   (sigmoid backward, b = output)
#define EW_ACC_TANHB  14  // out += a * (1 - b * b)   (tanh backward, b = output)

struct __TensorEwJob {
    const float* a;
    const float* b;
    float*       out;
    float        k;
    int          op;
};

static void __tensor_ew_body(const &__TensorEwJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsafe {
        const float* a = j.a;
        const float* b = j.b;
        float* o = j.out;
        float k = j.k;
        unsigned long i = lo;
        int op = j.op;
        if (op == EW_ADD)            { while (i < hi) { o[i] = a[i] + b[i]; i = i + 1UL; } }
        else if (op == EW_SUB)       { while (i < hi) { o[i] = a[i] - b[i]; i = i + 1UL; } }
        else if (op == EW_MUL)       { while (i < hi) { o[i] = a[i] * b[i]; i = i + 1UL; } }
        else if (op == EW_SCALE)     { while (i < hi) { o[i] = a[i] * k; i = i + 1UL; } }
        else if (op == EW_RELU)      { while (i < hi) { float v = a[i]; o[i] = v > (float)0.0 ? v : (float)0.0; i = i + 1UL; } }
        else if (op == EW_FILL)      { while (i < hi) { o[i] = k; i = i + 1UL; } }
        else if (op == EW_COPY)      { while (i < hi) { o[i] = a[i]; i = i + 1UL; } }
        else if (op == EW_ACC)       { while (i < hi) { o[i] = o[i] + a[i]; i = i + 1UL; } }
        else if (op == EW_DEC)       { while (i < hi) { o[i] = o[i] - a[i]; i = i + 1UL; } }
        else if (op == EW_ACC_MUL)   { while (i < hi) { o[i] = __builtin_fmuladd(a[i], b[i], o[i]); i = i + 1UL; } }
        else if (op == EW_ACC_SCALE) { while (i < hi) { o[i] = __builtin_fmuladd(a[i], k, o[i]); i = i + 1UL; } }
        else if (op == EW_ACC_RELU)  { while (i < hi) { o[i] = o[i] + (b[i] > (float)0.0 ? a[i] : (float)0.0); i = i + 1UL; } }
        else if (op == EW_ACC_K)     { while (i < hi) { o[i] = o[i] + k; i = i + 1UL; } }
        else if (op == EW_ACC_SIGB)  { while (i < hi) { float y = b[i]; o[i] = __builtin_fmuladd(a[i] * y, (float)1.0 - y, o[i]); i = i + 1UL; } }
        else if (op == EW_ACC_TANHB) { while (i < hi) { float y = b[i]; o[i] = __builtin_fmuladd(a[i], __builtin_fmuladd(-y, y, (float)1.0), o[i]); i = i + 1UL; } }
    }
}

// The loops per ISA level (std/simd/cpu.h): wider vectors and FMA.
static void __tensor_ew_base(const &__TensorEwJob j, unsigned long lo, unsigned long hi) { __tensor_ew_body(j, lo, hi); }
#if defined(__x86_64__)
static void __tensor_ew_avx2(const &__TensorEwJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { __tensor_ew_body(j, lo, hi); }
static void __tensor_ew_avx512(const &__TensorEwJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma"))) { __tensor_ew_body(j, lo, hi); }
#endif

static void __tensor_ew_range(&__TensorEwJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) { __tensor_ew_avx512(j, lo, hi); return; }
    if (l >= SIMD_AVX2) { __tensor_ew_avx2(j, lo, hi); return; }
#endif
    __tensor_ew_base(j, lo, hi);
}

static void __tensor_ew(int op, const float* a, const float* b, float k, float* out, unsigned long n) {
    struct __TensorEwJob j;
    j.a = a;
    j.b = b;
    j.out = out;
    j.k = k;
    j.op = op;
    if (n >= TENSOR_EW_PAR) parallel_for(n, 16384UL, __tensor_ew_range, &j);
    else __tensor_ew_range(&j, 0UL, n);
}

static &Tensor __tensor_alloc(const unsigned long* shape, unsigned long ndim, int requiresGrad) {
    unsafe {
        ?&TensorArena ar = tensor_arena_current_();
        unsigned long size = 1UL;
        unsigned long i = 0UL;
        while (i < ndim) { size = size * shape[i]; i = i + 1UL; }
        struct Tensor* t;
        unsigned long* shapeCopy;
        float* buf;
        match (ar) {
            case some(a): {
                t = (struct Tensor*)tensor_arena_alloc_raw_(a, sizeof(struct Tensor));
                shapeCopy = (unsigned long*)tensor_arena_alloc_raw_(a, sizeof(unsigned long) * ndim);
                buf = (float*)tensor_arena_alloc_raw_(a, sizeof(float) * size);
            }
            case null: {
                t = (struct Tensor*)malloc(sizeof(struct Tensor));
                shapeCopy = (unsigned long*)malloc(sizeof(unsigned long) * ndim);
                buf = (float*)malloc(sizeof(float) * size);
            }
        }
        i = 0UL;
        while (i < ndim) { shapeCopy[i] = shape[i]; i = i + 1UL; }

        t->data = (&heap float)buf;
        t->shape = (&heap unsigned long)shapeCopy;
        t->ndim = ndim;
        t->size = size;
        t->requiresGrad = requiresGrad;
        t->grad = (&heap float)0;
        t->gradFn = (void*)0;
        t->parents = vec_new();
        t->visited = 0;
        t->extraScalar = (float)0.0;
        t->owner = ar;
        match (ar) {
            case some(a): { tensor_arena_track_(a, (void*)t); }
            case null: { }
        }
        return (&Tensor)t;
    }
}

static void __tensor_ensure_grad(struct Tensor* t) {
    unsafe {
        if (t->grad == (float*)0) {
            float* g;
            match (t->owner) {
                case some(a): { g = (float*)tensor_arena_alloc_raw_(a, sizeof(float) * t->size); }
                case null: { g = (float*)malloc(sizeof(float) * t->size); }
            }
            __tensor_ew(EW_FILL, (const float*)0, (const float*)0, (float)0.0, g, t->size);
            t->grad = (&heap float)g;
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
static &Tensor __tensor_alloc_uninit_like(struct Tensor* t) {
    unsafe { return __tensor_alloc((const unsigned long*)t->shape, t->ndim, 0); }
}

static &Tensor __tensor_parent(struct Tensor* t, unsigned long idx) {
    unsafe {
        return (&Tensor)*t->parents.at(idx);
    }
}

static void __tensor_accumulate(struct Tensor* dst, const float* delta) {
    __tensor_ensure_grad(dst);
    unsafe {
        __tensor_ew(EW_ACC, delta, (const float*)0, (float)0.0, (float*)dst->grad, dst->size);
    }
}

static void __tensor_link2(struct Tensor* out, struct Tensor* a, struct Tensor* b, void* backwardFn) {
    unsafe {
        if (!tensor_is_grad_enabled()) { out->requiresGrad = 0; return; }
        out->requiresGrad = a->requiresGrad || b->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push(a);
        out->parents.push(b);
        out->gradFn = backwardFn;
    }
}

static void __tensor_link1(struct Tensor* out, struct Tensor* a, void* backwardFn) {
    unsafe {
        if (!tensor_is_grad_enabled()) { out->requiresGrad = 0; return; }
        out->requiresGrad = a->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push(a);
        out->gradFn = backwardFn;
    }
}

// ── Backward functions (one per forward op in tensor.sc) ─────────────────────
// __sub_backward through __sum_backward accumulate straight into the
// destination's grad (x->grad[i] +=) instead of building a throwaway
// 'delta' buffer and calling __tensor_accumulate -- profiled (via 'sample'
// on a real training run): the old throwaway-buffer version cost a full
// extra malloc/free and a full extra pass over the data on every single
// backward call, for no benefit at this size. __matmul_backward does the
// same through sgemm's beta = 1.

static void __add_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const float*)selfT->grad);
        if (b->requiresGrad) __tensor_accumulate(b, (const float*)selfT->grad);
    }
}

static void __sub_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const float*)selfT->grad);
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            __tensor_ew(EW_DEC, (const float*)selfT->grad, (const float*)0, (float)0.0, (float*)b->grad, selfT->size);
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
            __tensor_ew(EW_ACC_MUL, (const float*)selfT->grad, (const float*)b->data, (float)0.0, (float*)a->grad, selfT->size);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            __tensor_ew(EW_ACC_MUL, (const float*)selfT->grad, (const float*)a->data, (float)0.0, (float*)b->grad, selfT->size);
        }
    }
}

static void __scale_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        __tensor_ew(EW_ACC_SCALE, (const float*)selfT->grad, (const float*)0, selfT->extraScalar, (float*)a->grad, selfT->size);
    }
}

static void __sum_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        float seed = selfT->grad[0];
        __tensor_ew(EW_ACC_K, (const float*)0, (const float*)0, seed, (float*)a->grad, a->size);
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

        // dA[m,k] += dC[m,n] . B^T[n,k]
        // dB[k,n] += A^T[k,m] . dC[m,n]
        // Both accumulate straight into the existing gradient (beta = 1).
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            sgemm(SGEMM_NO_TRANS, SGEMM_TRANS, m, k, n, (float)1.0,
                  (const &float)selfT->grad, n, (const &float)b->data, n, (float)1.0, (&float)a->grad, k);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            sgemm(SGEMM_TRANS, SGEMM_NO_TRANS, k, n, m, (float)1.0,
                  (const &float)a->data, k, (const &float)selfT->grad, n, (float)1.0, (&float)b->grad, n);
        }
    }
}

} // namespace std
