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
#include <std/panic.h>
#include <std/mem.h>

namespace std {

// ══ Tensor arenas ════════════════════════════════════════════════════════════
// Chunk layout: [next][size][used] header (3 words), then the data. Chunks
// are kept across resets; 'current' is where allocation continues.

#define TENSOR_ARENA_DEFAULT_CHUNK 16777216UL
#define TENSOR_ARENA_HEADER        64UL      // keeps chunk data 64-byte aligned

#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__)
static ?&TensorArena g_tensor_arena = null;
#else
static thread_local ?&TensorArena g_tensor_arena = null;
#endif

struct TensorArena tensor_arena_new(unsigned long chunkBytes) {
    struct TensorArena a;
    a.first = 0UL;
    a.current = 0UL;
    a.chunkBytes = chunkBytes == 0UL ? TENSOR_ARENA_DEFAULT_CHUNK : chunkBytes;
    a.tracked = 0UL;
    a.trackedCount = 0UL;
    a.trackedCap = 0UL;
    return a;
}

void tensor_arena_begin(&TensorArena a) { g_tensor_arena = (?&TensorArena)a; }
void tensor_arena_end() { g_tensor_arena = null; }
?&TensorArena tensor_arena_current_() { return g_tensor_arena; }

static unsigned long* tensor_arena_chunk_new_(unsigned long dataBytes) {
    unsafe {
        unsigned long* c = (unsigned long*)alloc_raw_(TENSOR_ARENA_HEADER + dataBytes);
        if (c == (unsigned long*)0) panic_at("tensor arena: out of memory", "tensor.sc", __LINE__);
        c[0] = 0UL;          // next
        c[1] = dataBytes;    // size
        c[2] = 0UL;          // used
        return c;
    }
}

void* tensor_arena_alloc_raw_(&TensorArena a, unsigned long bytes) {
    unsigned long need = (bytes + 63UL) / 64UL * 64UL;
    if (need == 0UL) need = 64UL;
    unsafe {
        if (a.first == 0UL) {
            unsigned long* c = tensor_arena_chunk_new_(need > a.chunkBytes ? need : a.chunkBytes);
            a.first = (unsigned long)c;
            a.current = a.first;
        }
        unsigned long* cur = (unsigned long*)a.current;
        while (cur[2] + need > cur[1]) {
            // next chunk: reuse a kept one if it is big enough, else insert a new one
            unsigned long* next = (unsigned long*)cur[0];
            if (next != (unsigned long*)0 && next[1] >= need) {
                next[2] = 0UL;
                cur = next;
            } else {
                unsigned long* c = tensor_arena_chunk_new_(need > a.chunkBytes ? need : a.chunkBytes);
                c[0] = (unsigned long)next;
                cur[0] = (unsigned long)c;
                cur = c;
            }
            a.current = (unsigned long)cur;
        }
        char* base = (char*)cur;
        void* p = (void*)&base[TENSOR_ARENA_HEADER + cur[2]];
        cur[2] = cur[2] + need;
        return p;
    }
}

void tensor_arena_track_(&TensorArena a, void* t) {
    unsafe {
        if (a.trackedCount == a.trackedCap) {
            unsigned long cap = a.trackedCap == 0UL ? 256UL : a.trackedCap * 2UL;
            void** list = (void**)alloc_raw_(sizeof(void*) * cap);
            if (list == (void**)0) panic_at("tensor arena: out of memory", "tensor.sc", __LINE__);
            void** old = (void**)a.tracked;
            unsigned long i = 0UL;
            while (i < a.trackedCount) { list[i] = old[i]; i = i + 1UL; }
            if (old != (void**)0) dealloc_raw_((void*)old);
            a.tracked = (unsigned long)list;
            a.trackedCap = cap;
        }
        void** list = (void**)a.tracked;
        list[a.trackedCount] = t;
        a.trackedCount = a.trackedCount + 1UL;
    }
}

void tensor_arena_reset(&TensorArena a) {
    unsafe {
        // the parents vectors live on the heap
        void** list = (void**)a.tracked;
        unsigned long i = 0UL;
        while (i < a.trackedCount) {
            struct Tensor* t = (struct Tensor*)list[i];
            t->parents.free();
            i = i + 1UL;
        }
        a.trackedCount = 0UL;
        unsigned long* c = (unsigned long*)a.first;
        while (c != (unsigned long*)0) { c[2] = 0UL; c = (unsigned long*)c[0]; }
        a.current = a.first;
    }
}

void tensor_arena_free(&TensorArena a) {
    tensor_arena_reset(a);
    unsafe {
        unsigned long* c = (unsigned long*)a.first;
        while (c != (unsigned long*)0) {
            unsigned long* next = (unsigned long*)c[0];
            dealloc_raw_((void*)c);
            c = next;
        }
        if (a.tracked != 0UL) dealloc_raw_((void*)a.tracked);
    }
    a.first = 0UL;
    a.current = 0UL;
    a.tracked = 0UL;
    a.trackedCap = 0UL;
}

unsigned long tensor_arena_used(const &TensorArena a) {
    unsigned long used = 0UL;
    unsafe {
        unsigned long* c = (unsigned long*)a.first;
        while (c != (unsigned long*)0) {
            used = used + c[2];
            if ((unsigned long)c == a.current) break;
            c = (unsigned long*)c[0];
        }
    }
    return used;
}

// ══ Tensor methods ═══════════════════════════════════════════════════════════

inline float Tensor::at1(unsigned long i) const {
    unsafe { return self.data[i]; }
}

inline float Tensor::at2(unsigned long r, unsigned long c) const {
    unsafe { return self.data[r * self.shape[1] + c]; }
}

inline void Tensor::set1(unsigned long i, float v) {
    unsafe { self.data[i] = v; }
}

inline void Tensor::set2(unsigned long r, unsigned long c, float v) {
    unsafe { self.data[r * self.shape[1] + c] = v; }
}

inline void Tensor::free() {
    match (self.owner) { case some(o): { return; } case null: { } }   // its TensorArena owns it
    unsafe {
        if (self.data != (float*)0) { std::free((void*)self.data); self.data = (float*)0; }
        if (self.shape != (unsigned long*)0) { std::free((void*)self.shape); self.shape = (unsigned long*)0; }
        if (self.grad != (float*)0) { std::free((void*)self.grad); self.grad = (float*)0; }
        self.parents.free();
    }
}

// ══ Allocation & lifecycle ═══════════════════════════════════════════════════

&Tensor tensor_new_1d(unsigned long n, int requiresGrad) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    &Tensor t = __tensor_alloc(shape, 1UL, requiresGrad);
    tensor_fill(t, (float)0.0);
    return t;
}

&Tensor tensor_new_2d(unsigned long rows, unsigned long cols, int requiresGrad) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    &Tensor t = __tensor_alloc(shape, 2UL, requiresGrad);
    tensor_fill(t, (float)0.0);
    return t;
}

&Tensor tensor_from_1d(const &float values, unsigned long n, int requiresGrad) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    &Tensor t = __tensor_alloc(shape, 1UL, requiresGrad);
    unsafe { __tensor_ew(EW_COPY, (const float*)values, (const float*)0, (float)0.0, (float*)t->data, n); }
    return t;
}

&Tensor tensor_from_2d(const &float values, unsigned long rows, unsigned long cols, int requiresGrad) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    &Tensor t = __tensor_alloc(shape, 2UL, requiresGrad);
    unsafe { __tensor_ew(EW_COPY, (const float*)values, (const float*)0, (float)0.0, (float*)t->data, rows * cols); }
    return t;
}

&Tensor tensor_zeros_like(const &Tensor t) {
    &Tensor out;
    unsafe { out = __tensor_alloc((const unsigned long*)t->shape, t->ndim, 0); }
    tensor_fill(out, (float)0.0);
    return out;
}

&Tensor tensor_fill(&Tensor t, float v) {
    unsafe { __tensor_ew(EW_FILL, (const float*)0, (const float*)0, v, (float*)t->data, t->size); }
    return t;
}

// ══ Elementwise & reduction ops ══════════════════════════════════════════════
// Backward math for each of these lives in tensor.h, right next to the
// helpers it shares with tensor_nn.sc and the GPU backends -- see
// __add_backward etc. there.

// ── Add ──────────────────────────────────────────────────────────────────────
&Tensor tensor_add(const &Tensor a, const &Tensor b) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        __tensor_ew(EW_ADD, (const float*)a->data, (const float*)b->data, (float)0.0, (float*)out->data, out->size);
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

// ── Sub ──────────────────────────────────────────────────────────────────────
&Tensor tensor_sub(const &Tensor a, const &Tensor b) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        __tensor_ew(EW_SUB, (const float*)a->data, (const float*)b->data, (float)0.0, (float*)out->data, out->size);
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

// ── Mul (elementwise) ──────────────────────────────────────────────────────────
&Tensor tensor_mul(const &Tensor a, const &Tensor b) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        __tensor_ew(EW_MUL, (const float*)a->data, (const float*)b->data, (float)0.0, (float*)out->data, out->size);
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

// ── Scale (multiply by a scalar constant) ─────────────────────────────────────
&Tensor tensor_scale(const &Tensor a, float k) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        __tensor_ew(EW_SCALE, (const float*)a->data, (const float*)0, k, (float*)out->data, out->size);
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
    &Tensor out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        // Eight running sums (one f32x8), so the loop runs at vector width
        // and loses less precision than one long serial sum.
        const float* p = (const float*)a->data;
        unsigned long n = a->size;
        f32x8 acc8;
        unsigned long i = 0UL;
        while (i + 8UL <= n) { acc8 = acc8 + *(const f32x8*)&p[i]; i = i + 8UL; }
        float acc = ((acc8[0] + acc8[1]) + (acc8[2] + acc8[3])) + ((acc8[4] + acc8[5]) + (acc8[6] + acc8[7]));
        while (i < n) { acc = acc + p[i]; i = i + 1UL; }
        out->data[0] = acc;
    }
    __tensor_link1(out, a, (void*)__sum_backward);
    return out;
}

// ══ Matrix multiply ══════════════════════════════════════════════════════════

&Tensor tensor_matmul(const &Tensor a, const &Tensor b) {
    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    unsigned long shape[2];
    unsafe { shape[0] = m; shape[1] = n; }
    &Tensor out = __tensor_alloc(shape, 2UL, 0);   // sgemm's beta = 0 writes every element
    // Packed SIMD GEMM (std/simd/gemm.h); large products run on the
    // std/sync/parallel.h thread pool.
    unsafe {
        sgemm(SGEMM_NO_TRANS, SGEMM_NO_TRANS, m, n, k, (float)1.0,
              (const &float)a->data, k, (const &float)b->data, n, (float)0.0, (&float)out->data, n);
    }
    __tensor_link2(out, a, b, (void*)__matmul_backward);
    return out;
}

// ══ Graph traversal & tensor_backward() ══════════════════════════════════════

static void __tensor_toposort(&Tensor t, &Vec<&Tensor> order) {
    unsafe { if (t->visited) return; }
    unsafe { t->visited = 1; }
    unsigned long n;
    unsafe { n = t->parents.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        __tensor_toposort(__tensor_parent(t, i), order);
        i = i + 1UL;
    }
    order.push(t);
}

void tensor_backward(&Tensor t) {
    __tensor_ensure_grad(t);
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->grad[i] = (float)1.0; i = i + 1UL; }
    }

    struct Vec<&Tensor> order = vec_new();
    __tensor_toposort(t, &order);

    unsafe {
        unsigned long n = order.length();
        long long idx = (long long)n - 1LL;
        while (idx >= 0LL) {
            &Tensor node = *order.at((unsigned long)idx);
            if (node->gradFn != (void*)0) {
                TensorBackwardFn backwardFn = (TensorBackwardFn)node->gradFn;
                backwardFn((void*)node);
            }
            idx = idx - 1LL;
        }

        unsigned long j = 0UL;
        while (j < n) {
            &Tensor node2 = *order.at(j);
            node2->visited = 0;
            j = j + 1UL;
        }
        order.free();
    }
}

void tensor_zero_grad(&Tensor t) {
    unsafe {
        if (t->grad != (float*)0) {
            unsigned long i = 0UL;
            while (i < t->size) { t->grad[i] = (float)0.0; i = i + 1UL; }
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
