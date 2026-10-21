// SafeC Standard Library — LayerNorm and AdaLN (see norm.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/norm.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/math.h>

namespace std {

// ── Row LayerNorm ────────────────────────────────────────────────────────────
// The output's rows are xhat; the backward pass needs 1/sigma per row,
// kept in an aux parent ([rows]).
static void norm_ln_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* x = __tensor_parent(s, 0UL);
        struct Tensor* inv = __tensor_parent(s, 1UL);
        if (!x->requiresGrad) return;
        __tensor_ensure_grad(x);
        unsigned long rows = inv->size;
        unsigned long n = s->size / rows;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long base = r * n;
            float mg = (float)0.0;
            float mgx = (float)0.0;
            unsigned long c = 0UL;
            while (c < n) {
                float g = s->grad[base + c];
                mg = mg + g;
                mgx = mgx + g * s->data[base + c];
                c = c + 1UL;
            }
            mg = mg / (float)n;
            mgx = mgx / (float)n;
            float is = inv->data[r];
            c = 0UL;
            while (c < n) {
                x->grad[base + c] = x->grad[base + c] + is * (s->grad[base + c] - mg - s->data[base + c] * mgx);
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
}

&Tensor tensor_layer_norm(const &Tensor x, float eps) {
    unsigned long n; unsigned long rows;
    unsafe { n = x->shape[x->ndim - 1UL]; rows = x->size / n; }
    &Tensor out = __tensor_alloc_uninit_like(x);
    &Tensor inv = tensor_new_1d(rows, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long base = r * n;
            double s = 0.0;
            unsigned long c = 0UL;
            while (c < n) { s = s + (double)x->data[base + c]; c = c + 1UL; }
            double mean = s / (double)n;
            double v = 0.0;
            c = 0UL;
            while (c < n) { double d = (double)x->data[base + c] - mean; v = v + d * d; c = c + 1UL; }
            float is = (float)(1.0 / sqrt_d(v / (double)n + (double)eps));
            inv->data[r] = is;
            c = 0UL;
            while (c < n) { out->data[base + c] = (float)((double)x->data[base + c] - mean) * is; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    __tensor_link2(out, x, inv, (void*)norm_ln_backward_);
    unsafe {
        // not in a graph: release the aux now (an arena's goes with its reset)
        if (!out->requiresGrad) {
            match (inv->owner) {
                case some(ar): { }
                case null: { inv->free(); free((void*)inv); }
            }
        }
    }
    return out;
}

// ── Row-broadcast multiply ───────────────────────────────────────────────────
static void norm_mul_row_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* x = __tensor_parent(s, 0UL);
        struct Tensor* g = __tensor_parent(s, 1UL);
        unsigned long n = g->size;
        unsigned long rows = s->size / n;
        if (x->requiresGrad) __tensor_ensure_grad(x);
        if (g->requiresGrad) __tensor_ensure_grad(g);
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < n) {
                unsigned long k = r * n + c;
                if (x->requiresGrad) x->grad[k] = x->grad[k] + s->grad[k] * g->data[c];
                if (g->requiresGrad) g->grad[c] = g->grad[c] + s->grad[k] * x->data[k];
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
}

&Tensor tensor_mul_row(const &Tensor x, const &Tensor g) {
    &Tensor out = __tensor_alloc_uninit_like(x);
    unsafe {
        unsigned long n = g->size;
        unsigned long i = 0UL;
        while (i < x->size) { out->data[i] = x->data[i] * g->data[i % n]; i = i + 1UL; }
    }
    __tensor_link2(out, x, g, (void*)norm_mul_row_backward_);
    return out;
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

static ?&TensorArena norm_heap_begin_() {
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    return cur;
}

static void norm_heap_end_(?&TensorArena cur) {
    match (cur) {
        case some(a): { tensor_arena_begin(a); }
        case null: { }
    }
}

static void norm_drop_(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

struct LayerNorm layer_norm_new(unsigned long dim) {
    struct LayerNorm ln;
    ?&TensorArena cur = norm_heap_begin_();
    ln.gamma = tensor_new_1d(dim, 1);
    ln.beta = tensor_new_1d(dim, 1);
    norm_heap_end_(cur);
    tensor_fill(ln.gamma, (float)1.0);
    ln.dim = dim;
    ln.eps = (float)0.00001;
    return ln;
}

&Tensor layer_norm_forward(const &LayerNorm ln, const &Tensor x) {
    return tensor_add_bias(tensor_mul_row(tensor_layer_norm(x, ln.eps), ln.gamma), ln.beta);
}

void layer_norm_add_params(&Optimizer opt, const &LayerNorm ln) {
    optimizer_add(opt, ln.gamma);
    optimizer_add(opt, ln.beta);
}

void layer_norm_free(&LayerNorm ln) {
    norm_drop_(ln.gamma);
    norm_drop_(ln.beta);
}

// ── AdaLN ────────────────────────────────────────────────────────────────────

struct AdaLN adaln_new(unsigned long dim, unsigned long condDim, int gated) {
    struct AdaLN a;
    unsigned long chunks = gated ? 3UL : 2UL;
    ?&TensorArena cur = norm_heap_begin_();
    a.W = tensor_new_2d(condDim, chunks * dim, 1);
    a.b = tensor_new_1d(chunks * dim, 1);
    norm_heap_end_(cur);
    a.dim = dim;
    a.condDim = condDim;
    a.gated = gated;
    a.eps = (float)0.000001;
    return a;
}

struct AdaLNOut adaln_forward(const &AdaLN a, const &Tensor x, const &Tensor cond) {
    unsigned long n = a.dim;
    &Tensor mod = tensor_add_bias(tensor_matmul(cond, a.W), a.b);
    &Tensor shift = tensor_slice_cols(mod, 0UL, n);
    &Tensor scale = tensor_slice_cols(mod, n, n);
    &Tensor xh = tensor_layer_norm(x, a.eps);
    struct AdaLNOut o;
    o.out = tensor_add(tensor_mul(xh, tensor_add_scalar(scale, (float)1.0)), shift);
    if (a.gated) {
        o.gate = tensor_slice_cols(mod, 2UL * n, n);
    } else {
        unsigned long m;
        unsafe { m = x->shape[0]; }
        o.gate = tensor_fill(tensor_new_2d(m, n, 0), (float)1.0);
    }
    return o;
}

void adaln_add_params(&Optimizer opt, const &AdaLN a) {
    optimizer_add(opt, a.W);
    optimizer_add(opt, a.b);
}

void adaln_free(&AdaLN a) {
    norm_drop_(a.W);
    norm_drop_(a.b);
}

} // namespace std
