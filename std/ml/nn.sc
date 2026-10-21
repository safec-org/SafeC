// SafeC Standard Library — trainable networks (see nn.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/nn.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/math.h>
#include <std/io_file.h>

namespace std {

// ══ Random numbers ═══════════════════════════════════════════════════════════

struct Rng rng_new(unsigned long seed) {
    struct Rng r;
    r.state = seed * 6364136223846793005UL + 1442695040888963407UL;
    r.hasSpare = 0;
    r.spare = (float)0.0;
    return r;
}

unsigned long rng_u64(&Rng r) {
    r.state = r.state + 0x9E3779B97F4A7C15UL;
    unsigned long z = r.state;
    z = (z ^ (z >> 30UL)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27UL)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31UL);
}

float rng_uniform(&Rng r) {
    return (float)(rng_u64(r) >> 40UL) / (float)16777216.0;
}

float rng_normal(&Rng r) {
    if (r.hasSpare) { r.hasSpare = 0; return r.spare; }
    float u1 = (float)1.0 - rng_uniform(r);    // (0, 1]
    float u2 = rng_uniform(r);
    float mag = sqrt_f((float)-2.0 * log_f(u1));
    float ang = (float)6.283185307179586 * u2;
    r.spare = mag * sin_f(ang);
    r.hasSpare = 1;
    return mag * cos_f(ang);
}

unsigned long rng_below(&Rng r, unsigned long n) {
    if (n == 0UL) return 0UL;
    return rng_u64(r) % n;
}

void rng_fill_uniform(&Rng r, &Tensor t, float lo, float hi) {
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->data[i] = lo + (hi - lo) * rng_uniform(r); i = i + 1UL; }
    }
}

void rng_fill_normal(&Rng r, &Tensor t, float mean, float std) {
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->data[i] = mean + std * rng_normal(r); i = i + 1UL; }
    }
}

&Tensor rng_normal_tensor(&Rng r, unsigned long rows, unsigned long cols) {
    &Tensor t = tensor_new_2d(rows, cols, 0);
    rng_fill_normal(r, t, (float)0.0, (float)1.0);
    return t;
}

// ══ Ops ══════════════════════════════════════════════════════════════════════

static &Tensor nn_alloc2_(unsigned long rows, unsigned long cols) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    return __tensor_alloc(shape, 2UL, 0);
}

static &Tensor nn_alloc1_(unsigned long n) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    return __tensor_alloc(shape, 1UL, 0);
}

// Frees a tensor that is not part of any graph (heap tensors only; an
// arena tensor goes with its arena's reset).
static void nn_drop_(&Tensor t) {
    unsafe {
        match (t->owner) {
            case some(a): { }
            case null: { t->free(); free((void*)t); }
        }
    }
}

static unsigned long nn_rows_(const &Tensor t) {
    unsafe { return t->ndim >= 2UL ? t->shape[0] : 1UL; }
}

static unsigned long nn_cols_(const &Tensor t) {
    unsafe { return t->shape[t->ndim - 1UL]; }
}

// ── Bias add ─────────────────────────────────────────────────────────────────
static void nn_add_bias_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* x = __tensor_parent(s, 0UL);
        struct Tensor* b = __tensor_parent(s, 1UL);
        if (x->requiresGrad) __tensor_accumulate(x, (const float*)s->grad);
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            unsigned long n = b->size;
            unsigned long rows = s->size / n;
            unsigned long r = 0UL;
            while (r < rows) {
                unsigned long c = 0UL;
                while (c < n) { b->grad[c] = b->grad[c] + s->grad[r * n + c]; c = c + 1UL; }
                r = r + 1UL;
            }
        }
    }
}

&Tensor tensor_add_bias(const &Tensor x, const &Tensor b) {
    &Tensor out = __tensor_alloc_uninit_like(x);
    unsafe {
        unsigned long n = b->size;
        unsigned long rows = x->size / n;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < n) { out->data[r * n + c] = x->data[r * n + c] + b->data[c]; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    __tensor_link2(out, x, b, (void*)nn_add_bias_backward_);
    return out;
}

// ── Elementwise unary ops ────────────────────────────────────────────────────
#define NN_U_ADDK   0
#define NN_U_EXP    1
#define NN_U_LOG    2
#define NN_U_ABS    3
#define NN_U_LEAKY  4
#define NN_U_CLAMP  5

// The op tag is kept in 'extraScalar' of a 2-element aux parent (op, k)
// plus a second constant for clamp; the aux tensor never needs a gradient.
static void nn_unary_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(s, 0UL);
        struct Tensor* aux = __tensor_parent(s, 1UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        int op = (int)aux->data[0];
        float k = aux->data[1];
        unsigned long i = 0UL;
        unsigned long n = s->size;
        if (op == NN_U_ADDK) {
            while (i < n) { a->grad[i] = a->grad[i] + s->grad[i]; i = i + 1UL; }
        } else if (op == NN_U_EXP) {
            while (i < n) { a->grad[i] = a->grad[i] + s->grad[i] * s->data[i]; i = i + 1UL; }
        } else if (op == NN_U_LOG) {
            while (i < n) { a->grad[i] = a->grad[i] + s->grad[i] / a->data[i]; i = i + 1UL; }
        } else if (op == NN_U_ABS) {
            while (i < n) {
                float x = a->data[i];
                float sg = x > (float)0.0 ? (float)1.0 : (x < (float)0.0 ? (float)-1.0 : (float)0.0);
                a->grad[i] = a->grad[i] + s->grad[i] * sg;
                i = i + 1UL;
            }
        } else if (op == NN_U_LEAKY) {
            while (i < n) { a->grad[i] = a->grad[i] + s->grad[i] * (a->data[i] > (float)0.0 ? (float)1.0 : k); i = i + 1UL; }
        } else {
            // clamp: the gradient passes where the value was not clamped
            while (i < n) { if (s->data[i] == a->data[i]) a->grad[i] = a->grad[i] + s->grad[i]; i = i + 1UL; }
        }
    }
}

static &Tensor nn_unary_(const &Tensor a, int op, float k, float k2) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    &Tensor aux = nn_alloc1_(2UL);
    unsafe {
        aux->data[0] = (float)op;
        aux->data[1] = k;
        unsigned long n = a->size;
        unsigned long i = 0UL;
        if (op == NN_U_ADDK) {
            while (i < n) { out->data[i] = a->data[i] + k; i = i + 1UL; }
        } else if (op == NN_U_EXP) {
            while (i < n) { out->data[i] = exp_f(a->data[i]); i = i + 1UL; }
        } else if (op == NN_U_LOG) {
            while (i < n) { out->data[i] = log_f(a->data[i]); i = i + 1UL; }
        } else if (op == NN_U_ABS) {
            while (i < n) { float x = a->data[i]; out->data[i] = x < (float)0.0 ? -x : x; i = i + 1UL; }
        } else if (op == NN_U_LEAKY) {
            while (i < n) { float x = a->data[i]; out->data[i] = x > (float)0.0 ? x : k * x; i = i + 1UL; }
        } else {
            while (i < n) {
                float x = a->data[i];
                out->data[i] = x < k ? k : (x > k2 ? k2 : x);
                i = i + 1UL;
            }
        }
    }
    __tensor_link2(out, a, aux, (void*)nn_unary_backward_);
    unsafe { if (!out->requiresGrad) nn_drop_(aux); }
    return out;
}

&Tensor tensor_add_scalar(const &Tensor a, float k) { return nn_unary_(a, NN_U_ADDK, k, (float)0.0); }
&Tensor tensor_exp(const &Tensor a) { return nn_unary_(a, NN_U_EXP, (float)0.0, (float)0.0); }
&Tensor tensor_log(const &Tensor a) { return nn_unary_(a, NN_U_LOG, (float)0.0, (float)0.0); }
&Tensor tensor_abs(const &Tensor a) { return nn_unary_(a, NN_U_ABS, (float)0.0, (float)0.0); }
&Tensor tensor_leaky_relu(const &Tensor a, float slope) { return nn_unary_(a, NN_U_LEAKY, slope, (float)0.0); }
&Tensor tensor_clamp(const &Tensor a, float lo, float hi) { return nn_unary_(a, NN_U_CLAMP, lo, hi); }

// ── Minimum ──────────────────────────────────────────────────────────────────
static void nn_minimum_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(s, 0UL);
        struct Tensor* b = __tensor_parent(s, 1UL);
        if (a->requiresGrad) __tensor_ensure_grad(a);
        if (b->requiresGrad) __tensor_ensure_grad(b);
        unsigned long i = 0UL;
        while (i < s->size) {
            if (a->data[i] <= b->data[i]) { if (a->requiresGrad) a->grad[i] = a->grad[i] + s->grad[i]; }
            else { if (b->requiresGrad) b->grad[i] = b->grad[i] + s->grad[i]; }
            i = i + 1UL;
        }
    }
}

&Tensor tensor_minimum(const &Tensor a, const &Tensor b) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < a->size) {
            float x = a->data[i]; float y = b->data[i];
            out->data[i] = x <= y ? x : y;
            i = i + 1UL;
        }
    }
    __tensor_link2(out, a, b, (void*)nn_minimum_backward_);
    return out;
}

&Tensor tensor_mean(const &Tensor a) {
    float n;
    unsafe { n = (float)a->size; }
    return tensor_scale(tensor_sum(a), (float)1.0 / n);
}

// ── Row sums ─────────────────────────────────────────────────────────────────
static void nn_sum_rows_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(s, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        unsigned long rows = s->size;
        unsigned long cols = a->size / rows;
        unsigned long r = 0UL;
        while (r < rows) {
            float g = s->grad[r];
            unsigned long c = 0UL;
            while (c < cols) { a->grad[r * cols + c] = a->grad[r * cols + c] + g; c = c + 1UL; }
            r = r + 1UL;
        }
    }
}

&Tensor tensor_sum_rows(const &Tensor a) {
    unsigned long rows = nn_rows_(a);
    unsigned long cols = nn_cols_(a);
    &Tensor out = nn_alloc2_(rows, 1UL);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            float acc = (float)0.0;
            unsigned long c = 0UL;
            while (c < cols) { acc = acc + a->data[r * cols + c]; c = c + 1UL; }
            out->data[r] = acc;
            r = r + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)nn_sum_rows_backward_);
    return out;
}

// ── Column concatenation ─────────────────────────────────────────────────────
static void nn_concat_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(s, 0UL);
        struct Tensor* b = __tensor_parent(s, 1UL);
        unsigned long rows = s->shape[0];
        unsigned long p = a->size / rows;
        unsigned long q = b->size / rows;
        if (a->requiresGrad) __tensor_ensure_grad(a);
        if (b->requiresGrad) __tensor_ensure_grad(b);
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            if (a->requiresGrad) { while (c < p) { a->grad[r * p + c] = a->grad[r * p + c] + s->grad[r * (p + q) + c]; c = c + 1UL; } }
            c = 0UL;
            if (b->requiresGrad) { while (c < q) { b->grad[r * q + c] = b->grad[r * q + c] + s->grad[r * (p + q) + p + c]; c = c + 1UL; } }
            r = r + 1UL;
        }
    }
}

&Tensor tensor_concat_cols(const &Tensor a, const &Tensor b) {
    unsigned long rows = nn_rows_(a);
    unsigned long p = nn_cols_(a);
    unsigned long q = nn_cols_(b);
    &Tensor out = nn_alloc2_(rows, p + q);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < p) { out->data[r * (p + q) + c] = a->data[r * p + c]; c = c + 1UL; }
            c = 0UL;
            while (c < q) { out->data[r * (p + q) + p + c] = b->data[r * q + c]; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    __tensor_link2(out, a, b, (void*)nn_concat_backward_);
    return out;
}

// ── Row gather ───────────────────────────────────────────────────────────────
static void nn_gather_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* table = __tensor_parent(s, 0UL);
        struct Tensor* idx = __tensor_parent(s, 1UL);
        if (!table->requiresGrad) return;
        __tensor_ensure_grad(table);
        unsigned long d = table->shape[1];
        unsigned long i = 0UL;
        while (i < idx->size) {
            unsigned long row = (unsigned long)idx->data[i];
            unsigned long c = 0UL;
            while (c < d) { table->grad[row * d + c] = table->grad[row * d + c] + s->grad[i * d + c]; c = c + 1UL; }
            i = i + 1UL;
        }
    }
}

&Tensor tensor_gather_rows(const &Tensor table, const &Tensor idx) {
    unsigned long d = nn_cols_(table);
    unsigned long m;
    unsafe { m = idx->size; }
    &Tensor out = nn_alloc2_(m, d);
    unsafe {
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long row = (unsigned long)idx->data[i];
            unsigned long c = 0UL;
            while (c < d) { out->data[i * d + c] = table->data[row * d + c]; c = c + 1UL; }
            i = i + 1UL;
        }
    }
    __tensor_link2(out, table, idx, (void*)nn_gather_backward_);
    return out;
}

// ── Log-softmax ──────────────────────────────────────────────────────────────
static void nn_log_softmax_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(s, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        unsigned long cols = s->shape[s->ndim - 1UL];
        unsigned long rows = s->size / cols;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long base = r * cols;
            float gs = (float)0.0;
            unsigned long c = 0UL;
            while (c < cols) { gs = gs + s->grad[base + c]; c = c + 1UL; }
            c = 0UL;
            while (c < cols) {
                a->grad[base + c] = a->grad[base + c] + s->grad[base + c] - exp_f(s->data[base + c]) * gs;
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
}

static float nn_row_logsumexp_(const float* x, unsigned long n) {
    unsafe {
        float mx = x[0];
        unsigned long c = 1UL;
        while (c < n) { if (x[c] > mx) mx = x[c]; c = c + 1UL; }
        float acc = (float)0.0;
        c = 0UL;
        while (c < n) { acc = acc + exp_f(x[c] - mx); c = c + 1UL; }
        return mx + log_f(acc);
    }
}

&Tensor tensor_log_softmax(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long cols = a->shape[a->ndim - 1UL];
        unsigned long rows = a->size / cols;
        unsigned long r = 0UL;
        while (r < rows) {
            const float* x = (const float*)&a->data[r * cols];
            float lse = nn_row_logsumexp_(x, cols);
            unsigned long c = 0UL;
            while (c < cols) { out->data[r * cols + c] = x[c] - lse; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    __tensor_link1(out, a, (void*)nn_log_softmax_backward_);
    return out;
}

&Tensor tensor_detach(const &Tensor a) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe { __tensor_ew(EW_COPY, (const float*)a->data, (const float*)0, (float)0.0, (float*)out->data, a->size); }
    return out;
}

&Tensor tensor_one_hot(const &Tensor labels, unsigned long classes) {
    unsigned long m;
    unsafe { m = labels->size; }
    &Tensor out = tensor_new_2d(m, classes, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long c = (unsigned long)labels->data[i];
            if (c < classes) out->data[i * classes + c] = (float)1.0;
            i = i + 1UL;
        }
    }
    return out;
}

&Tensor tensor_argmax_rows(const &Tensor a) {
    unsigned long rows = nn_rows_(a);
    unsigned long cols = nn_cols_(a);
    &Tensor out = nn_alloc1_(rows);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long best = 0UL;
            unsigned long c = 1UL;
            while (c < cols) { if (a->data[r * cols + c] > a->data[r * cols + best]) best = c; c = c + 1UL; }
            out->data[r] = (float)best;
            r = r + 1UL;
        }
    }
    return out;
}

// ══ Losses ═══════════════════════════════════════════════════════════════════

&Tensor tensor_mse_loss(const &Tensor pred, const &Tensor target) {
    &Tensor d = tensor_sub(pred, target);
    return tensor_mean(tensor_mul(d, d));
}

#define NN_L_HUBER  0
#define NN_L_BCE    1
#define NN_L_CE     2

// Fused losses: the scalar output keeps the loss kind and its constant in
// an aux parent (parent 2); the target / labels are parent 1.
static void nn_loss_backward_(void* selfPtr) {
    unsafe {
        struct Tensor* s = (struct Tensor*)selfPtr;
        struct Tensor* x = __tensor_parent(s, 0UL);
        struct Tensor* t = __tensor_parent(s, 1UL);
        struct Tensor* aux = __tensor_parent(s, 2UL);
        if (!x->requiresGrad) return;
        __tensor_ensure_grad(x);
        int kind = (int)aux->data[0];
        float k = aux->data[1];
        float g = s->grad[0];
        unsigned long n = x->size;
        unsigned long i = 0UL;
        if (kind == NN_L_HUBER) {
            float sc = g / (float)n;
            while (i < n) {
                float d = x->data[i] - t->data[i];
                if (d > k) d = k;
                if (d < -k) d = -k;
                x->grad[i] = x->grad[i] + sc * d;
                i = i + 1UL;
            }
        } else if (kind == NN_L_BCE) {
            float sc = g / (float)n;
            while (i < n) {
                float sg = (float)1.0 / ((float)1.0 + exp_f(-x->data[i]));
                x->grad[i] = x->grad[i] + sc * (sg - t->data[i]);
                i = i + 1UL;
            }
        } else {
            unsigned long cols = x->shape[x->ndim - 1UL];
            unsigned long rows = n / cols;
            float sc = g / (float)rows;
            unsigned long r = 0UL;
            while (r < rows) {
                const float* row = (const float*)&x->data[r * cols];
                float lse = nn_row_logsumexp_(row, cols);
                unsigned long lab = (unsigned long)t->data[r];
                unsigned long c = 0UL;
                while (c < cols) {
                    float p = exp_f(row[c] - lse);
                    if (c == lab) p = p - (float)1.0;
                    x->grad[r * cols + c] = x->grad[r * cols + c] + sc * p;
                    c = c + 1UL;
                }
                r = r + 1UL;
            }
        }
    }
}

static &Tensor nn_loss_(const &Tensor x, const &Tensor t, int kind, float k) {
    &Tensor out = nn_alloc1_(1UL);
    &Tensor aux = nn_alloc1_(2UL);
    unsafe {
        aux->data[0] = (float)kind;
        aux->data[1] = k;
        unsigned long n = x->size;
        double acc = 0.0;
        unsigned long i = 0UL;
        if (kind == NN_L_HUBER) {
            while (i < n) {
                float d = x->data[i] - t->data[i];
                float ad = d < (float)0.0 ? -d : d;
                acc = acc + (double)(ad <= k ? (float)0.5 * d * d : k * (ad - (float)0.5 * k));
                i = i + 1UL;
            }
            acc = acc / (double)n;
        } else if (kind == NN_L_BCE) {
            while (i < n) {
                float v = x->data[i];
                float av = v < (float)0.0 ? -v : v;
                float relu = v > (float)0.0 ? v : (float)0.0;
                acc = acc + (double)(relu - v * t->data[i] + log_f((float)1.0 + exp_f(-av)));
                i = i + 1UL;
            }
            acc = acc / (double)n;
        } else {
            unsigned long cols = x->shape[x->ndim - 1UL];
            unsigned long rows = n / cols;
            unsigned long r = 0UL;
            while (r < rows) {
                const float* row = (const float*)&x->data[r * cols];
                unsigned long lab = (unsigned long)t->data[r];
                acc = acc + (double)(nn_row_logsumexp_(row, cols) - row[lab]);
                r = r + 1UL;
            }
            acc = acc / (double)rows;
        }
        out->data[0] = (float)acc;
        if (tensor_is_grad_enabled() && x->requiresGrad) {
            out->requiresGrad = 1;
            out->parents.push(x);
            out->parents.push(t);
            out->parents.push(aux);
            out->gradFn = (void*)nn_loss_backward_;
        } else {
            nn_drop_(aux);
        }
    }
    return out;
}

&Tensor tensor_huber_loss(const &Tensor pred, const &Tensor target, float delta) {
    return nn_loss_(pred, target, NN_L_HUBER, delta);
}

&Tensor tensor_bce_logits_loss(const &Tensor logits, const &Tensor target) {
    return nn_loss_(logits, target, NN_L_BCE, (float)0.0);
}

&Tensor tensor_cross_entropy_loss(const &Tensor logits, const &Tensor labels) {
    return nn_loss_(logits, labels, NN_L_CE, (float)0.0);
}

float tensor_accuracy(const &Tensor logits, const &Tensor labels) {
    unsigned long rows = nn_rows_(logits);
    unsigned long cols = nn_cols_(logits);
    unsigned long hit = 0UL;
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long best = 0UL;
            unsigned long c = 1UL;
            while (c < cols) { if (logits->data[r * cols + c] > logits->data[r * cols + best]) best = c; c = c + 1UL; }
            if (best == (unsigned long)labels->data[r]) hit = hit + 1UL;
            r = r + 1UL;
        }
    }
    return rows == 0UL ? (float)0.0 : (float)hit / (float)rows;
}

// ══ Activations by tag ═══════════════════════════════════════════════════════

&Tensor tensor_activate(const &Tensor x, int act) {
    if (act == ACT_RELU) return tensor_relu(x);
    if (act == ACT_LEAKY) return tensor_leaky_relu(x, (float)0.2);
    if (act == ACT_TANH) return tensor_tanh_t(x);
    if (act == ACT_SIGMOID) return tensor_sigmoid(x);
    if (act == ACT_GELU) return tensor_gelu(x);
    if (act == ACT_SILU) return tensor_silu(x);
    return x;
}

// ══ MLP ══════════════════════════════════════════════════════════════════════

// Parameters always live on the heap, also when an arena is active.
static ?&TensorArena nn_heap_begin_() {
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    return cur;
}

static void nn_heap_end_(?&TensorArena cur) {
    match (cur) {
        case some(a): { tensor_arena_begin(a); }
        case null: { }
    }
}

struct Mlp mlp_new(unsigned long inDim, unsigned long hidden, unsigned long hiddenLayers,
                   unsigned long outDim, int act, int outAct, &Rng rng) {
    ?&TensorArena cur = nn_heap_begin_();
    struct Mlp m;
    m.params = vec_new();
    m.nLayers = hiddenLayers + 1UL;
    m.inDim = inDim;
    m.outDim = outDim;
    m.act = act;
    m.outAct = outAct;
    unsigned long l = 0UL;
    while (l < m.nLayers) {
        unsigned long fin = l == 0UL ? inDim : hidden;
        unsigned long fout = l + 1UL == m.nLayers ? outDim : hidden;
        &Tensor W = tensor_new_2d(fin, fout, 1);
        &Tensor b = tensor_new_1d(fout, 1);
        float bound = (float)1.0 / sqrt_f((float)fin);
        rng_fill_uniform(rng, W, -bound, bound);
        rng_fill_uniform(rng, b, -bound, bound);
        m.params.push(W);
        m.params.push(b);
        l = l + 1UL;
    }
    nn_heap_end_(cur);
    return m;
}

static &Tensor mlp_layer_(const &Mlp m, unsigned long l, const &Tensor x) {
    &Tensor W = *m.params.at(2UL * l);
    &Tensor b = *m.params.at(2UL * l + 1UL);
    &Tensor y = tensor_add_bias(tensor_matmul(x, W), b);
    return tensor_activate(y, l + 1UL == m.nLayers ? m.outAct : m.act);
}

&Tensor mlp_forward(const &Mlp m, const &Tensor x) {
    &Tensor h = mlp_layer_(m, 0UL, x);
    unsigned long l = 1UL;
    while (l < m.nLayers) {
        h = mlp_layer_(m, l, h);
        l = l + 1UL;
    }
    return h;
}

unsigned long mlp_param_count(const &Mlp m) {
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < m.params.length()) {
        &Tensor p = *m.params.at(i);
        unsafe { n = n + p->size; }
        i = i + 1UL;
    }
    return n;
}

void mlp_soft_update(&Mlp dst, const &Mlp src, float tau) {
    unsigned long i = 0UL;
    while (i < dst.params.length()) {
        &Tensor d = *dst.params.at(i);
        &Tensor s = *src.params.at(i);
        unsafe {
            unsigned long j = 0UL;
            while (j < d->size) { d->data[j] = tau * s->data[j] + ((float)1.0 - tau) * d->data[j]; j = j + 1UL; }
        }
        i = i + 1UL;
    }
}

void mlp_copy(&Mlp dst, const &Mlp src) { mlp_soft_update(dst, src, (float)1.0); }

void mlp_free(&Mlp m) {
    unsigned long i = 0UL;
    while (i < m.params.length()) {
        &Tensor p = *m.params.at(i);
        p->free();
        unsafe { free((void*)p); }
        i = i + 1UL;
    }
    m.params.free();
}

// ══ Optimizers ═══════════════════════════════════════════════════════════════

struct Optimizer optimizer_new(int kind, float lr) {
    struct Optimizer o;
    o.params = vec_new();
    o.m = vec_new();
    o.v = vec_new();
    o.kind = kind;
    o.lr = lr;
    o.momentum = (float)0.9;
    o.beta1 = (float)0.9;
    o.beta2 = kind == OPT_RMSPROP || kind == OPT_LION ? (float)0.99 : (float)0.999;
    o.eps = kind == OPT_ADAGRAD ? (float)0.0000000001 : (float)0.00000001;
    o.weightDecay = kind == OPT_ADAMW ? (float)0.01 : (float)0.0;
    o.t = 0UL;
    return o;
}

void optimizer_add(&Optimizer o, &Tensor param) {
    ?&TensorArena cur = nn_heap_begin_();
    unsigned long n;
    unsafe { n = param->size; }
    o.params.push(param);
    o.m.push(tensor_new_1d(n, 0));
    o.v.push(tensor_new_1d(n, 0));
    nn_heap_end_(cur);
}

void optimizer_add_mlp(&Optimizer o, const &Mlp m) {
    unsigned long i = 0UL;
    while (i < m.params.length()) { optimizer_add(o, *m.params.at(i)); i = i + 1UL; }
}

void optimizer_zero_grad(&Optimizer o) {
    unsigned long i = 0UL;
    while (i < o.params.length()) { tensor_zero_grad(*o.params.at(i)); i = i + 1UL; }
}

float optimizer_clip_grad_norm(&Optimizer o, float maxNorm) {
    double sq = 0.0;
    unsigned long i = 0UL;
    while (i < o.params.length()) {
        &Tensor p = *o.params.at(i);
        unsafe {
            if (p->grad != (float*)0) {
                unsigned long j = 0UL;
                while (j < p->size) { sq = sq + (double)p->grad[j] * (double)p->grad[j]; j = j + 1UL; }
            }
        }
        i = i + 1UL;
    }
    float norm = (float)sqrt_d(sq);
    if (norm > maxNorm && norm > (float)0.0) {
        float sc = maxNorm / norm;
        i = 0UL;
        while (i < o.params.length()) {
            &Tensor p = *o.params.at(i);
            unsafe {
                if (p->grad != (float*)0) {
                    unsigned long j = 0UL;
                    while (j < p->size) { p->grad[j] = p->grad[j] * sc; j = j + 1UL; }
                }
            }
            i = i + 1UL;
        }
    }
    return norm;
}

// One parameter's update over [lo, hi): the loops compile once per ISA
// level (the target() copies below) and split across the thread pool for
// large tensors, like tensor.h's elementwise kernels.
struct OptJob {
    float*       w;
    const float* g;
    float*       m;
    float*       v;
    int   kind;
    float lr;
    float wd;
    float mu;
    float b1;
    float b2;
    float eps;
    float ibc1;     // Adam: 1 / (1 - beta1^t)
    float isbc2;    // Adam: 1 / sqrt(1 - beta2^t)
};

static void opt_body_(const &OptJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsafe {
        float* w = j.w;
        const float* gr = j.g;
        float* m = j.m;
        float* v = j.v;
        float lr = j.lr;
        float wd = j.wd;
        unsigned long i = lo;
        int kind = j.kind;
        if (kind == OPT_SGD) {
            while (i < hi) { w[i] = w[i] - lr * (gr[i] + wd * w[i]); i = i + 1UL; }
        } else if (kind == OPT_MOMENTUM) {
            float mu = j.mu;
            while (i < hi) {
                float g = gr[i] + wd * w[i];
                float mi = mu * m[i] + g;
                m[i] = mi;
                w[i] = w[i] - lr * mi;
                i = i + 1UL;
            }
        } else if (kind == OPT_NESTEROV) {
            float mu = j.mu;
            while (i < hi) {
                float g = gr[i] + wd * w[i];
                float mi = mu * m[i] + g;
                m[i] = mi;
                w[i] = w[i] - lr * (g + mu * mi);
                i = i + 1UL;
            }
        } else if (kind == OPT_ADAGRAD) {
            float eps = j.eps;
            while (i < hi) {
                float g = gr[i] + wd * w[i];
                float vi = v[i] + g * g;
                v[i] = vi;
                w[i] = w[i] - lr * g / (__builtin_sqrt(vi) + eps);
                i = i + 1UL;
            }
        } else if (kind == OPT_LION) {
            // update = sign(b1 m + (1 - b1) g); m = b2 m + (1 - b2) g
            float b1 = j.b1;
            float b2 = j.b2;
            while (i < hi) {
                float g = gr[i];
                float c = b1 * m[i] + ((float)1.0 - b1) * g;
                float sg = c > (float)0.0 ? (float)1.0 : (c < (float)0.0 ? (float)-1.0 : (float)0.0);
                w[i] = w[i] - lr * (sg + wd * w[i]);
                m[i] = b2 * m[i] + ((float)1.0 - b2) * g;
                i = i + 1UL;
            }
        } else if (kind == OPT_RMSPROP) {
            float a = j.b2;
            float eps = j.eps;
            while (i < hi) {
                float g = gr[i] + wd * w[i];
                float vi = a * v[i] + ((float)1.0 - a) * g * g;
                v[i] = vi;
                w[i] = w[i] - lr * g / (__builtin_sqrt(vi) + eps);
                i = i + 1UL;
            }
        } else {
            // Adam / AdamW: m_hat / (sqrt(v_hat) + eps), with
            // sqrt(v / bc2) = sqrt(v) / sqrt(bc2)
            float b1 = j.b1;
            float b2 = j.b2;
            float eps = j.eps;
            float ibc1 = j.ibc1;
            float isbc2 = j.isbc2;
            float l2 = kind == OPT_ADAMW ? (float)0.0 : wd;
            float dec = kind == OPT_ADAMW ? wd : (float)0.0;
            while (i < hi) {
                float g = gr[i] + l2 * w[i];
                float mi = b1 * m[i] + ((float)1.0 - b1) * g;
                float vi = b2 * v[i] + ((float)1.0 - b2) * g * g;
                m[i] = mi;
                v[i] = vi;
                float upd = (mi * ibc1) / (__builtin_sqrt(vi) * isbc2 + eps) + dec * w[i];
                w[i] = w[i] - lr * upd;
                i = i + 1UL;
            }
        }
    }
}

static void opt_base_(const &OptJob j, unsigned long lo, unsigned long hi) { opt_body_(j, lo, hi); }
#if defined(__x86_64__)
static void opt_avx2_(const &OptJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { opt_body_(j, lo, hi); }
static void opt_avx512_(const &OptJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma"))) { opt_body_(j, lo, hi); }
#endif

static void opt_range_(&OptJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) { opt_avx512_(j, lo, hi); return; }
    if (l >= SIMD_AVX2) { opt_avx2_(j, lo, hi); return; }
#endif
    opt_base_(j, lo, hi);
}

void optimizer_step(&Optimizer o) {
    o.t = o.t + 1UL;
    float bc1 = (float)1.0 - pow_f(o.beta1, (float)o.t);
    float bc2 = (float)1.0 - pow_f(o.beta2, (float)o.t);
    struct OptJob j;
    j.kind = o.kind;
    j.lr = o.lr;
    j.wd = o.weightDecay;
    j.mu = o.momentum;
    j.b1 = o.beta1;
    j.b2 = o.beta2;
    j.eps = o.eps;
    j.ibc1 = (float)1.0 / bc1;
    j.isbc2 = (float)1.0 / sqrt_f(bc2);
    unsigned long i = 0UL;
    while (i < o.params.length()) {
        &Tensor p = *o.params.at(i);
        &Tensor mt = *o.m.at(i);
        &Tensor vt = *o.v.at(i);
        unsafe {
            if (p->grad != (float*)0) {
                j.w = (float*)p->data;
                j.g = (const float*)p->grad;
                j.m = (float*)mt->data;
                j.v = (float*)vt->data;
                unsigned long n = p->size;
                if (n >= TENSOR_EW_PAR) parallel_for(n, 16384UL, opt_range_, &j);
                else opt_range_(&j, 0UL, n);
            }
        }
        i = i + 1UL;
    }
}

void optimizer_free(&Optimizer o) {
    unsigned long i = 0UL;
    while (i < o.m.length()) {
        &Tensor a = *o.m.at(i);
        &Tensor b = *o.v.at(i);
        a->free(); b->free();
        unsafe { free((void*)a); free((void*)b); }
        i = i + 1UL;
    }
    o.params.free(); o.m.free(); o.v.free();
}

// ══ Saving and loading tensors ═══════════════════════════════════════════════

static int nn_write_(&File f, const void* p, unsigned long bytes) {
    unsafe { return file_write(f, (const &unsigned char)(const unsigned char*)p, bytes) == bytes; }
}

static int nn_read_(&File f, void* p, unsigned long bytes) {
    unsafe { return file_read(f, (&unsigned char)(unsigned char*)p, bytes) == bytes; }
}

int tensors_save(const &Vec<&Tensor> ts, const &char path) {
    ?&File fo = file_open(path, "wb");
    match (fo) {
        case null: { return 0; }
        case some(f): {
            int ok = 1;
            unsigned char magic[4];
            unsafe {
                magic[0] = (unsigned char)83; magic[1] = (unsigned char)67;   // "SCTN"
                magic[2] = (unsigned char)84; magic[3] = (unsigned char)78;
                ok = nn_write_(f, (const void*)&magic[0], 4UL);
                unsigned long count = ts.length();
                ok = ok && nn_write_(f, (const void*)&count, sizeof(unsigned long));
                unsigned long i = 0UL;
                while (ok && i < count) {
                    &Tensor t = *ts.at(i);
                    unsigned long nd = t->ndim;
                    ok = nn_write_(f, (const void*)&nd, sizeof(unsigned long));
                    ok = ok && nn_write_(f, (const void*)t->shape, sizeof(unsigned long) * nd);
                    ok = ok && nn_write_(f, (const void*)t->data, sizeof(float) * t->size);
                    i = i + 1UL;
                }
            }
            if (file_close(f) != 0) ok = 0;
            return ok;
        }
    }
    return 0;
}

int tensors_load(&Vec<&Tensor> ts, const &char path) {
    ?&File fo = file_open(path, "rb");
    match (fo) {
        case null: { return 0; }
        case some(f): {
            int ok = 1;
            unsigned char magic[4];
            unsafe {
                ok = nn_read_(f, (void*)&magic[0], 4UL)
                     && magic[0] == (unsigned char)83 && magic[1] == (unsigned char)67
                     && magic[2] == (unsigned char)84 && magic[3] == (unsigned char)78;
                unsigned long count = 0UL;
                ok = ok && nn_read_(f, (void*)&count, sizeof(unsigned long)) && count == ts.length();
                // check every shape before overwriting anything
                long long start = file_tell(f);
                unsigned long i = 0UL;
                while (ok && i < count) {
                    &Tensor t = *ts.at(i);
                    unsigned long nd = 0UL;
                    ok = nn_read_(f, (void*)&nd, sizeof(unsigned long)) && nd == t->ndim;
                    unsigned long d = 0UL;
                    while (ok && d < nd) {
                        unsigned long dim = 0UL;
                        ok = nn_read_(f, (void*)&dim, sizeof(unsigned long)) && dim == t->shape[d];
                        d = d + 1UL;
                    }
                    ok = ok && file_seek(f, (long long)(sizeof(float) * t->size), 1) == 0;
                    i = i + 1UL;
                }
                ok = ok && file_seek(f, start, 0) == 0;
                i = 0UL;
                while (ok && i < count) {
                    &Tensor t = *ts.at(i);
                    unsigned long skip = sizeof(unsigned long) * (1UL + t->ndim);
                    ok = file_seek(f, (long long)skip, 1) == 0;
                    ok = ok && nn_read_(f, (void*)t->data, sizeof(float) * t->size);
                    i = i + 1UL;
                }
            }
            file_close(f);
            return ok;
        }
    }
    return 0;
}

int mlp_save_file(const &Mlp m, const &char path) { return tensors_save(&m.params, path); }
int mlp_load_file(&Mlp m, const &char path) { return tensors_load(&m.params, path); }

} // namespace std
