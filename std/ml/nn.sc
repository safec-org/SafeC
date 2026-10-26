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

struct Rng Rng::new(unsigned long seed) {
    struct Rng r;
    r.state = seed * 6364136223846793005UL + 1442695040888963407UL;
    r.hasSpare = 0;
    r.spare = (float)0.0;
    return r;
}

unsigned long Rng::u64() {
    self.state = self.state + 0x9E3779B97F4A7C15UL;
    unsigned long z = self.state;
    z = (z ^ (z >> 30UL)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27UL)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31UL);
}

float Rng::uniform() {
    return (float)(self.u64() >> 40UL) / (float)16777216.0;
}

float Rng::normal() {
    if (self.hasSpare) { self.hasSpare = 0; return self.spare; }
    float u1 = (float)1.0 - self.uniform();    // (0, 1]
    float u2 = self.uniform();
    float mag = sqrt_f((float)-2.0 * log_f(u1));
    float ang = (float)6.283185307179586 * u2;
    self.spare = mag * sin_f(ang);
    self.hasSpare = 1;
    return mag * cos_f(ang);
}

unsigned long Rng::below(unsigned long n) {
    if (n == 0UL) return 0UL;
    return self.u64() % n;
}

void Rng::fill_uniform(&FloatTensor t, float lo, float hi) {
    unsafe { tensor_host_write_raw_((void*)t); }
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->data[i] = lo + (hi - lo) * self.uniform(); i = i + 1UL; }
    }
}

void Rng::fill_normal(&FloatTensor t, float mean, float std) {
    unsafe { tensor_host_write_raw_((void*)t); }
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { t->data[i] = mean + std * self.normal(); i = i + 1UL; }
    }
}

&FloatTensor Rng::normal_tensor(unsigned long rows, unsigned long cols) {
    &FloatTensor t = FloatTensor::new_2d(rows, cols, 0);
    self.fill_normal(t, (float)0.0, (float)1.0);
    return t;
}

// ══ Ops ══════════════════════════════════════════════════════════════════════

static &FloatTensor nn_alloc2_(unsigned long rows, unsigned long cols) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    return __tensor_alloc(shape, 2UL, 0);
}

static &FloatTensor nn_alloc1_(unsigned long n) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    return __tensor_alloc(shape, 1UL, 0);
}

// Frees a tensor that is not part of any graph (heap tensors only; an
// arena tensor goes with its arena's reset).
static void nn_drop_(&FloatTensor t) {
    unsafe {
        match (t->owner) {
            case some(a): { }
            case null: { t->free(); free((void*)t); }
        }
    }
}

static unsigned long nn_rows_(const &FloatTensor t) {
    unsafe { tensor_host_raw_((void*)t); }
    unsafe { return t->ndim >= 2UL ? t->shape[0] : 1UL; }
}

static unsigned long nn_cols_(const &FloatTensor t) {
    unsafe { tensor_host_raw_((void*)t); }
    unsafe { return t->shape[t->ndim - 1UL]; }
}

// ── Bias add ─────────────────────────────────────────────────────────────────
static void nn_add_bias_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* b = __tensor_parent(s, 1UL);
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

&FloatTensor tensor_add_bias(const &FloatTensor x, const &FloatTensor b) {
    match (tdev_add_bias_(x, b)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)x); tensor_host_raw_((void*)b); }
    &FloatTensor out = __tensor_alloc_uninit_like(x);
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
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        FloatTensor* aux = __tensor_parent(s, 1UL);
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

static &FloatTensor nn_unary_(const &FloatTensor a, int op, float k, float k2) {
    // on a device: the same op as one elementwise kernel each way
    if (op == NN_U_ADDK) { match (tdev_unary_(a, EW_ADDK, k, (float)0.0, EW_ACC, 0, (float)0.0, 0)) { case some(o): { return o; } case null: { } } }
    else if (op == NN_U_EXP) { match (tdev_unary_(a, EW_EXP, (float)0.0, (float)0.0, EW_ACC_EXPB, 2, (float)0.0, 0)) { case some(o): { return o; } case null: { } } }
    else if (op == NN_U_LOG) { match (tdev_unary_(a, EW_LOG, (float)0.0, (float)0.0, EW_ACC_LOGB, 1, (float)0.0, 0)) { case some(o): { return o; } case null: { } } }
    else if (op == NN_U_ABS) { match (tdev_unary_(a, EW_ABS, (float)0.0, (float)0.0, EW_ACC_ABSB, 1, (float)0.0, 0)) { case some(o): { return o; } case null: { } } }
    else if (op == NN_U_LEAKY) { match (tdev_unary_(a, EW_LEAKY, k, (float)0.0, EW_ACC_LEAKYB, 1, k, 0)) { case some(o): { return o; } case null: { } } }
    else if (op == NN_U_CLAMP) { match (tdev_unary_(a, EW_CLAMP, k, k2, EW_ACC_CLAMPB, 1, (float)0.0, 2)) { case some(o): { return o; } case null: { } } }
    unsafe { tensor_host_raw_((void*)a); }
    &FloatTensor out = __tensor_alloc_uninit_like(a);
    &FloatTensor aux = nn_alloc1_(2UL);
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

&FloatTensor FloatTensor::add_scalar(float k) const { return nn_unary_(self, NN_U_ADDK, k, (float)0.0); }
&FloatTensor FloatTensor::exp() const { return nn_unary_(self, NN_U_EXP, (float)0.0, (float)0.0); }
&FloatTensor FloatTensor::log() const { return nn_unary_(self, NN_U_LOG, (float)0.0, (float)0.0); }
&FloatTensor FloatTensor::abs() const { return nn_unary_(self, NN_U_ABS, (float)0.0, (float)0.0); }
&FloatTensor FloatTensor::leaky_relu(float slope) const { return nn_unary_(self, NN_U_LEAKY, slope, (float)0.0); }
&FloatTensor FloatTensor::clamp(float lo, float hi) const { return nn_unary_(self, NN_U_CLAMP, lo, hi); }

// ── Minimum ──────────────────────────────────────────────────────────────────
static void nn_minimum_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        FloatTensor* b = __tensor_parent(s, 1UL);
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

&FloatTensor tensor_minimum(const &FloatTensor a, const &FloatTensor b) {
    match (tdev_binary_(a, b, 3)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)a); tensor_host_raw_((void*)b); }
    &FloatTensor out = __tensor_alloc_uninit_like(a);
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

&FloatTensor FloatTensor::mean() const {
    unsigned long count;
    unsafe { count = self->size; }
    match (tdev_sum_(self, (float)1.0 / (float)count)) { case some(o): { return o; } case null: { } }
    float n;
    unsafe { n = (float)self->size; }
    return self.sum().scale((float)1.0 / n);
}

// ── Row sums ─────────────────────────────────────────────────────────────────
static void nn_sum_rows_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
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

&FloatTensor FloatTensor::sum_rows() const {
    match (tdev_sum_rows_(self)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)self); }
    unsigned long rows = nn_rows_(self);
    unsigned long cols = nn_cols_(self);
    &FloatTensor out = nn_alloc2_(rows, 1UL);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            float acc = (float)0.0;
            unsigned long c = 0UL;
            while (c < cols) { acc = acc + self->data[r * cols + c]; c = c + 1UL; }
            out->data[r] = acc;
            r = r + 1UL;
        }
    }
    __tensor_link1(out, self, (void*)nn_sum_rows_backward_);
    return out;
}

// ── Column concatenation ─────────────────────────────────────────────────────
static void nn_concat_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        FloatTensor* b = __tensor_parent(s, 1UL);
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

&FloatTensor tensor_concat_cols(const &FloatTensor a, const &FloatTensor b) {
    match (tdev_concat_cols_(a, b)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)a); tensor_host_raw_((void*)b); }
    unsigned long rows = nn_rows_(a);
    unsigned long p = nn_cols_(a);
    unsigned long q = nn_cols_(b);
    &FloatTensor out = nn_alloc2_(rows, p + q);
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
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* table = __tensor_parent(s, 0UL);
        FloatTensor* idx = __tensor_parent(s, 1UL);
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

&FloatTensor tensor_gather_rows(const &FloatTensor table, const &FloatTensor idx) {
    match (tdev_gather_rows_(table, idx)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)table); tensor_host_raw_((void*)idx); }
    unsigned long d = nn_cols_(table);
    unsigned long m;
    unsafe { m = idx->size; }
    &FloatTensor out = nn_alloc2_(m, d);
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
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
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

&FloatTensor FloatTensor::log_softmax() const {
    match (tdev_softmax_(self, 1)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        unsigned long cols = self->shape[self->ndim - 1UL];
        unsigned long rows = self->size / cols;
        unsigned long r = 0UL;
        while (r < rows) {
            const float* x = (const float*)&self->data[r * cols];
            float lse = nn_row_logsumexp_(x, cols);
            unsigned long c = 0UL;
            while (c < cols) { out->data[r * cols + c] = x[c] - lse; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    __tensor_link1(out, self, (void*)nn_log_softmax_backward_);
    return out;
}

&FloatTensor FloatTensor::detach() const {
    match (tdev_detach_(self)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)self); }
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsafe { __tensor_ew(EW_COPY, (const float*)self->data, (const float*)0, (float)0.0, (float*)out->data, self->size); }
    return out;
}

&FloatTensor FloatTensor::one_hot(unsigned long classes) const {
    unsafe { tensor_host_raw_((void*)self); }
    unsigned long m;
    unsafe { m = self->size; }
    &FloatTensor out = FloatTensor::new_2d(m, classes, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long c = (unsigned long)self->data[i];
            if (c < classes) out->data[i * classes + c] = (float)1.0;
            i = i + 1UL;
        }
    }
    return out;
}

&FloatTensor FloatTensor::argmax_rows() const {
    unsafe { tensor_host_raw_((void*)self); }
    unsigned long rows = nn_rows_(self);
    unsigned long cols = nn_cols_(self);
    &FloatTensor out = nn_alloc1_(rows);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long best = 0UL;
            unsigned long c = 1UL;
            while (c < cols) { if (self->data[r * cols + c] > self->data[r * cols + best]) best = c; c = c + 1UL; }
            out->data[r] = (float)best;
            r = r + 1UL;
        }
    }
    return out;
}

// ══ Losses ═══════════════════════════════════════════════════════════════════

&FloatTensor tensor_mse_loss(const &FloatTensor pred, const &FloatTensor target) {
    &FloatTensor d = tensor_sub(pred, target);
    return tensor_mul(d, d).mean();
}

#define NN_L_HUBER  0
#define NN_L_BCE    1
#define NN_L_CE     2

// Fused losses: the scalar output keeps the loss kind and its constant in
// an aux parent (parent 2); the target / labels are parent 1.
static void nn_loss_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* t = __tensor_parent(s, 1UL);
        FloatTensor* aux = __tensor_parent(s, 2UL);
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

static &FloatTensor nn_loss_(const &FloatTensor x, const &FloatTensor t, int kind, float k) {
    unsafe { tensor_host_raw_((void*)x); tensor_host_raw_((void*)t); }
    &FloatTensor out = nn_alloc1_(1UL);
    &FloatTensor aux = nn_alloc1_(2UL);
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
        if (Autograd::get().enabled() && x->requiresGrad) {
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

&FloatTensor tensor_huber_loss(const &FloatTensor pred, const &FloatTensor target, float delta) {
    match (tdev_loss_(pred, target, 2, delta)) { case some(o): { return o; } case null: { } }
    return nn_loss_(pred, target, NN_L_HUBER, delta);
}

&FloatTensor tensor_bce_logits_loss(const &FloatTensor logits, const &FloatTensor target) {
    match (tdev_loss_(logits, target, 1, (float)0.0)) { case some(o): { return o; } case null: { } }
    return nn_loss_(logits, target, NN_L_BCE, (float)0.0);
}

&FloatTensor tensor_cross_entropy_loss(const &FloatTensor logits, const &FloatTensor labels) {
    match (tdev_loss_(logits, labels, 0, (float)0.0)) { case some(o): { return o; } case null: { } }
    return nn_loss_(logits, labels, NN_L_CE, (float)0.0);
}

float tensor_accuracy(const &FloatTensor logits, const &FloatTensor labels) {
    unsafe { tensor_host_raw_((void*)logits); tensor_host_raw_((void*)labels); }
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

&FloatTensor FloatTensor::activate(int act) const {
    if (act == ACT_RELU) return self.relu();
    if (act == ACT_LEAKY) return self.leaky_relu((float)0.2);
    if (act == ACT_TANH) return self.tanh();
    if (act == ACT_SIGMOID) return self.sigmoid();
    if (act == ACT_GELU) return self.gelu();
    if (act == ACT_SILU) return self.silu();
    // ACT_NONE: the receiver itself (a caller's region-less reference, so it
    // may leave the method)
    unsafe { return (&FloatTensor)(const FloatTensor*)self; }
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
        case some(a): { a.begin(); }
        case null: { }
    }
}

struct Mlp Mlp::new(unsigned long inDim, unsigned long hidden, unsigned long hiddenLayers, unsigned long outDim,
                    int act, int outAct, &Rng rng) {
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
        &FloatTensor W = FloatTensor::new_2d(fin, fout, 1);
        &FloatTensor b = FloatTensor::new_1d(fout, 1);
        float bound = (float)1.0 / sqrt_f((float)fin);
        rng.fill_uniform(W, -bound, bound);
        rng.fill_uniform(b, -bound, bound);
        m.params.push(W);
        m.params.push(b);
        l = l + 1UL;
    }
    nn_heap_end_(cur);
    return m;
}

static &FloatTensor mlp_layer_(const &Mlp m, unsigned long l, const &FloatTensor x) {
    &FloatTensor W = *m.params.at(2UL * l);
    &FloatTensor b = *m.params.at(2UL * l + 1UL);
    &FloatTensor y = tensor_add_bias(tensor_matmul(x, W), b);
    return y.activate(l + 1UL == m.nLayers ? m.outAct : m.act);
}

&FloatTensor Mlp::forward(const &FloatTensor x) const {
    &FloatTensor h = mlp_layer_(self, 0UL, x);
    unsigned long l = 1UL;
    while (l < self.nLayers) {
        h = mlp_layer_(self, l, h);
        l = l + 1UL;
    }
    return h;
}

void Mlp::collect_params(&Vec<&FloatTensor> out) const {
    unsigned long i = 0UL;
    while (i < self.params.length()) { out.push(*self.params.at(i)); i = i + 1UL; }
}

void Mlp::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

void Mlp::free() { self.drop(); }

void Mlp::drop() {
    unsigned long i = 0UL;
    while (i < self.params.length()) {
        &FloatTensor p = *self.params.at(i);
        p->free();
        unsafe { free((void*)p); }
        i = i + 1UL;
    }
    self.params.free();
}

// ══ Optimizers ═══════════════════════════════════════════════════════════════

struct Optimizer Optimizer::new(int kind, float lr) {
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

struct Optimizer Optimizer::from(const &Optimizer o) {
    struct Optimizer d = Optimizer::new(o.kind, o.lr);
    d.momentum = o.momentum;
    d.beta1 = o.beta1;
    d.beta2 = o.beta2;
    d.eps = o.eps;
    d.weightDecay = o.weightDecay;
    return d;
}

void Optimizer::add(&FloatTensor param) {
    ?&TensorArena cur = nn_heap_begin_();
    unsigned long n;
    unsafe { n = param->size; }
    self.params.push(param);
    self.m.push(FloatTensor::new_1d(n, 0));
    self.v.push(FloatTensor::new_1d(n, 0));
    nn_heap_end_(cur);
}

void Optimizer::zero_grad() {
    unsigned long i = 0UL;
    while (i < self.params.length()) { (*self.params.at(i)).zero_grad(); i = i + 1UL; }
}

// 1 if p has a gradient anywhere (a parameter never reached by a loss has none).
static int nn_has_grad_(FloatTensor* p) {
    unsafe { return p->grad != (float*)0 || p->dgrad != 0UL; }
}

// A one-float device scratch buffer per device, for gradient-norm sums.
static unsigned long g_nn_dev_scratch[DEVICE_COUNT];

static unsigned long nn_dev_scratch_(int d) {
    unsafe {
        if (g_nn_dev_scratch[d] == 0UL) g_nn_dev_scratch[d] = device_mem_alloc_(d, 4UL);
        return g_nn_dev_scratch[d];
    }
}

float Optimizer::clip_grad_norm(float maxNorm) {
    double sq = 0.0;
    unsigned long i = 0UL;
    // device gradients: sum of squares on each device, read back once per device
    int d = 0;
    while (d < DEVICE_COUNT) {
        if (d != DEVICE_CPU) {
            unsigned long acc = 0UL;
            int any = 0;
            i = 0UL;
            while (i < self.params.length()) {
                &FloatTensor p = *self.params.at(i);
                unsafe {
                    if (p->dev == d && nn_has_grad_((FloatTensor*)p) && tensor_grad_dev_raw_((void*)p, d)) {
                        if (!any) {
                            acc = nn_dev_scratch_(d);
                            float z = (float)0.0;
                            device_upload_(d, acc, (const &float)&z, 1UL);
                            any = 1;
                        }
                        struct DevArgs a = dev_args_();
                        a.p[0] = p->dgrad; a.p[1] = acc; a.u[0] = 1U;
                        device_launch_(d, DK_SUM, p->size, &a);
                    }
                }
                i = i + 1UL;
            }
            if (any) {
                float v = (float)0.0;
                unsafe { device_download_(d, (&float)&v, acc, 1UL); }
                sq = sq + (double)v;
            }
        }
        d = d + 1;
    }
    i = 0UL;
    while (i < self.params.length()) {
        &FloatTensor p = *self.params.at(i);
        unsafe {
            if (p->dev == DEVICE_CPU && p->grad != (float*)0) {
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
        while (i < self.params.length()) {
            &FloatTensor p = *self.params.at(i);
            unsafe {
                if (p->dev != DEVICE_CPU) {
                    if (p->dgrad != 0UL) {
                        struct DevArgs a = dev_args_();
                        a.p[0] = p->dgrad; a.p[2] = p->dgrad; a.u[0] = (unsigned int)EW_SCALE; a.f[0] = sc;
                        device_launch_(p->dev, DK_EW, p->size, &a);
                        tensor_grad_dev_wrote_raw_((void*)p);
                    }
                } else if (p->grad != (float*)0) {
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

// The device's parameter table for dk_opt_multi, kept while the same
// buffers come back step after step (one cached table per device).
#define NN_OPT_TABLE_MAX 1024UL
static unsigned long g_nn_opt_dev[DEVICE_COUNT];        // device copy
static unsigned long g_nn_opt_cap[DEVICE_COUNT];        // its entries
static unsigned long g_nn_opt_count[DEVICE_COUNT];
static unsigned long g_nn_opt_host[DEVICE_COUNT * NN_OPT_TABLE_MAX * 6UL];

// Every device-d parameter with a gradient, stepped in one launch; 0 when
// the backend can't (the caller steps them one by one).
static int nn_opt_multi_(&Optimizer o, int d, const &DevArgs base) {
    unsafe {
        unsigned long tab[NN_OPT_TABLE_MAX * 6UL];
        unsigned long cnt = 0UL;
        unsigned long total = 0UL;
        unsigned long i = 0UL;
        while (i < o.params.length()) {
            &FloatTensor p = *o.params.at(i);
            &FloatTensor mt = *o.m.at(i);
            &FloatTensor vt = *o.v.at(i);
            if (p->dev == d && nn_has_grad_((FloatTensor*)p)) {
                if (cnt >= NN_OPT_TABLE_MAX || total + p->size >= 4294967295UL) return 0;
                if (!tensor_grad_dev_raw_((void*)p, d) || !tensor_place_raw_((void*)mt, d) || !tensor_place_raw_((void*)vt, d)) return 0;
                tab[cnt * 6UL] = p->dbuf; tab[cnt * 6UL + 1UL] = p->dgrad;
                tab[cnt * 6UL + 2UL] = mt->dbuf; tab[cnt * 6UL + 3UL] = vt->dbuf;
                tab[cnt * 6UL + 4UL] = p->size; tab[cnt * 6UL + 5UL] = total;
                total = total + p->size;
                cnt = cnt + 1UL;
            }
            i = i + 1UL;
        }
        if (cnt == 0UL) return 1;
        // the cached table, uploaded again only when it changed
        unsigned long* host = &g_nn_opt_host[(unsigned long)d * NN_OPT_TABLE_MAX * 6UL];
        int same = g_nn_opt_count[d] == cnt && g_nn_opt_dev[d] != 0UL;
        unsigned long k = 0UL;
        while (same && k < cnt * 6UL) { if (host[k] != tab[k]) same = 0; k = k + 1UL; }
        if (!same) {
            if (g_nn_opt_cap[d] < cnt) {
                if (g_nn_opt_dev[d] != 0UL) device_mem_free_(d, g_nn_opt_dev[d], 48UL * g_nn_opt_cap[d]);
                g_nn_opt_dev[d] = device_mem_alloc_(d, 48UL * cnt);
                g_nn_opt_cap[d] = g_nn_opt_dev[d] != 0UL ? cnt : 0UL;
                if (g_nn_opt_dev[d] == 0UL) { g_nn_opt_count[d] = 0UL; return 0; }
            }
            // 12 floats' worth per entry
            if (!device_upload_(d, g_nn_opt_dev[d], (const &float)(const float*)&tab[0], 12UL * cnt)) { g_nn_opt_count[d] = 0UL; return 0; }
            k = 0UL;
            while (k < cnt * 6UL) { host[k] = tab[k]; k = k + 1UL; }
            g_nn_opt_count[d] = cnt;
        }
        struct DevArgs a = *base;
        a.p[0] = g_nn_opt_dev[d];
        a.p[5] = device_amp_state_(d);   // fp16: an overflowed step is skipped
        a.u[1] = (unsigned int)cnt;
        if (!device_launch_(d, DK_OPT_MULTI, total, &a)) return 0;
        i = 0UL;
        while (i < o.params.length()) {
            &FloatTensor p = *o.params.at(i);
            if (p->dev == d && nn_has_grad_((FloatTensor*)p)) {
                tensor_dev_wrote_raw_((void*)p);
                tensor_dev_wrote_raw_((void*)*o.m.at(i));
                tensor_dev_wrote_raw_((void*)*o.v.at(i));
            }
            i = i + 1UL;
        }
        return 1;
    }
}

void Optimizer::step() {
    self.t = self.t + 1UL;
    float bc1 = (float)1.0 - pow_f(self.beta1, (float)self.t);
    float bc2 = (float)1.0 - pow_f(self.beta2, (float)self.t);
    struct OptJob j;
    j.kind = self.kind;
    j.lr = self.lr;
    j.wd = self.weightDecay;
    j.mu = self.momentum;
    j.b1 = self.beta1;
    j.b2 = self.beta2;
    j.eps = self.eps;
    j.ibc1 = (float)1.0 / bc1;
    j.isbc2 = (float)1.0 / sqrt_f(bc2);
    struct DevArgs base = dev_args_();
    base.u[0] = (unsigned int)j.kind;
    base.f[0] = j.lr; base.f[1] = j.wd; base.f[2] = j.mu; base.f[3] = j.b1; base.f[4] = j.b2; base.f[5] = j.eps;
    base.f[6] = j.ibc1; base.f[7] = j.isbc2;
    int doneDev[DEVICE_COUNT];
    int d0 = 0;
    while (d0 < DEVICE_COUNT) {
        doneDev[d0] = d0 != DEVICE_CPU && device_available(d0) && nn_opt_multi_(self, d0, &base);
        d0 = d0 + 1;
    }
    unsigned long i = 0UL;
    while (i < self.params.length()) {
        &FloatTensor p = *self.params.at(i);
        &FloatTensor mt = *self.m.at(i);
        &FloatTensor vt = *self.v.at(i);
        unsafe {
            if (p->dev != DEVICE_CPU) {
                // on its device: the same update as one kernel, the moments living there too
                int d = p->dev;
                if (!doneDev[d] && nn_has_grad_((FloatTensor*)p) && tensor_grad_dev_raw_((void*)p, d) &&
                    tensor_place_raw_((void*)mt, d) && tensor_place_raw_((void*)vt, d)) {
                    struct DevArgs a = dev_args_();
                    a.p[0] = p->dbuf; a.p[1] = p->dgrad; a.p[2] = mt->dbuf; a.p[3] = vt->dbuf; a.p[4] = 0UL;
                    a.u[0] = (unsigned int)j.kind;
                    a.f[0] = j.lr; a.f[1] = j.wd; a.f[2] = j.mu; a.f[3] = j.b1; a.f[4] = j.b2; a.f[5] = j.eps;
                    a.f[6] = j.ibc1; a.f[7] = j.isbc2;
                    a.p[5] = device_amp_state_(d);
                    if (device_launch_(d, DK_OPT, p->size, &a)) {
                        tensor_dev_wrote_raw_((void*)p);
                        tensor_dev_wrote_raw_((void*)mt);
                        tensor_dev_wrote_raw_((void*)vt);
                    }
                }
            } else if (p->grad != (float*)0) {
                tensor_host_write_raw_((void*)mt);
                tensor_host_write_raw_((void*)vt);
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
    // fp16 devices: the loss scale moves on (halved after an overflow,
    // doubled after 2000 clean steps) and the overflow flag clears
    int used[DEVICE_COUNT];
    d0 = 0;
    while (d0 < DEVICE_COUNT) { used[d0] = 0; d0 = d0 + 1; }
    i = 0UL;
    while (i < self.params.length()) { used[(*self.params.at(i)).device()] = 1; i = i + 1UL; }
    d0 = 0;
    while (d0 < DEVICE_COUNT) {
        if (d0 != DEVICE_CPU && used[d0] && device_precision(d0) == DEVICE_FP16) {
            unsigned long amp = device_amp_state_(d0);
            if (amp != 0UL) {
                struct DevArgs u = dev_args_();
                u.p[0] = amp; u.u[0] = 2000U;
                device_launch_(d0, DK_AMP_UPDATE, 1UL, &u);
            }
        }
        d0 = d0 + 1;
    }
}

void Optimizer::free() { self.drop(); }

void Optimizer::drop() {
    unsigned long i = 0UL;
    while (i < self.m.length()) {
        &FloatTensor a = *self.m.at(i);
        &FloatTensor b = *self.v.at(i);
        a->free(); b->free();
        unsafe { free((void*)a); free((void*)b); }
        i = i + 1UL;
    }
    self.params.free(); self.m.free(); self.v.free();
}

// ══ Saving and loading tensors ═══════════════════════════════════════════════

static int nn_write_(&File f, const void* p, unsigned long bytes) {
    unsafe { return file_write(f, ((const unsigned char*)p)[0..bytes]) == bytes; }
}

static int nn_read_(&File f, void* p, unsigned long bytes) {
    unsafe { return file_read(f, ((unsigned char*)p)[0..bytes]) == bytes; }
}

int tensors_save(const &Vec<&FloatTensor> ts, const &char path) {
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
                    &FloatTensor t = *ts.at(i);
        unsafe { tensor_host_raw_((void*)t); }
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

int tensors_load(&Vec<&FloatTensor> ts, const &char path) {
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
                    &FloatTensor t = *ts.at(i);
        unsafe { tensor_host_write_raw_((void*)t); }
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
                    &FloatTensor t = *ts.at(i);
        unsafe { tensor_host_write_raw_((void*)t); }
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


} // namespace std
