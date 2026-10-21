// SafeC Standard Library — LoRA and QLoRA (see lora.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/lora.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/math.h>

namespace std {

static ?&TensorArena lora_heap_begin_() {
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    return cur;
}

static void lora_heap_end_(?&TensorArena cur) {
    match (cur) {
        case some(a): { tensor_arena_begin(a); }
        case null: { }
    }
}

static void lora_drop_(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

static unsigned long lora_rows_(const &Tensor t) {
    unsafe { return t->shape[0]; }
}

static unsigned long lora_cols_(const &Tensor t) {
    unsafe { return t->shape[t->ndim - 1UL]; }
}

struct LoraConfig lora_config_default(unsigned long rank) {
    struct LoraConfig c;
    c.rank = rank;
    c.alpha = (float)rank;
    c.dropout = (float)0.0;
    c.seed = 1UL;
    return c;
}

// A [in, r] Kaiming-uniform (as PEFT initializes it), B [r, out] zero.
static void lora_make_pair_(unsigned long in, unsigned long out, unsigned long r, &Rng rng,
                            &Vec<&Tensor> As, &Vec<&Tensor> Bs) {
    ?&TensorArena cur = lora_heap_begin_();
    &Tensor A = tensor_new_2d(in, r, 1);
    &Tensor B = tensor_new_2d(r, out, 1);
    lora_heap_end_(cur);
    float bound = (float)1.0 / sqrt_f((float)in);
    rng_fill_uniform(rng, A, -bound, bound);
    As.push(A);
    Bs.push(B);
}

// One adapted layer: act(x W + b + scale * (drop(x) A) B).
static &Tensor lora_layer_(const &Tensor x, const &Tensor W, const &Tensor b, const &Tensor A,
                           const &Tensor B, float scale, float dropout, &Rng rng, int act) {
    &Tensor base = tensor_add_bias(tensor_matmul(x, W), b);
    &Tensor xin = tensor_add_scalar(x, (float)0.0);
    if (dropout > (float)0.0 && tensor_is_grad_enabled()) {
        &Tensor mask = tensor_new_2d(lora_rows_(x), lora_cols_(x), 0);
        float keep = (float)1.0 / ((float)1.0 - dropout);
        unsafe {
            unsigned long i = 0UL;
            while (i < mask->size) { mask->data[i] = rng_uniform(rng) < dropout ? (float)0.0 : keep; i = i + 1UL; }
        }
        xin = tensor_mul(x, mask);
    }
    &Tensor upd = tensor_scale(tensor_matmul(tensor_matmul(xin, A), B), scale);
    return tensor_activate(tensor_add(base, upd), act);
}

// ══ LoRA ═════════════════════════════════════════════════════════════════════

struct LoraMlp lora_attach(&Mlp base, const &LoraConfig cfg) {
    struct LoraMlp l;
    l.base = base;
    l.A = vec_new();
    l.B = vec_new();
    l.cfg = *cfg;
    l.scale = cfg.alpha / (float)cfg.rank;
    l.rng = rng_new(cfg.seed);
    unsigned long i = 0UL;
    while (i < base.params.length()) {
        &Tensor p = *base.params.at(i);
        unsafe { p->requiresGrad = 0; }
        i = i + 1UL;
    }
    unsigned long k = 0UL;
    while (k < base.nLayers) {
        &Tensor W = *base.params.at(2UL * k);
        lora_make_pair_(lora_rows_(W), lora_cols_(W), cfg.rank, &l.rng, &l.A, &l.B);
        k = k + 1UL;
    }
    return l;
}

&Tensor lora_forward(&LoraMlp l, const &Tensor x) {
    &Mlp m = l.base;
    &Tensor h = lora_layer_(x, *m.params.at(0UL), *m.params.at(1UL), *l.A.at(0UL), *l.B.at(0UL),
                            l.scale, l.cfg.dropout, &l.rng, m.nLayers == 1UL ? m.outAct : m.act);
    unsigned long k = 1UL;
    while (k < m.nLayers) {
        h = lora_layer_(h, *m.params.at(2UL * k), *m.params.at(2UL * k + 1UL), *l.A.at(k), *l.B.at(k),
                        l.scale, l.cfg.dropout, &l.rng, k + 1UL == m.nLayers ? m.outAct : m.act);
        k = k + 1UL;
    }
    return h;
}

void lora_add_params(&Optimizer opt, const &LoraMlp l) {
    unsigned long k = 0UL;
    while (k < l.A.length()) { optimizer_add(opt, *l.A.at(k)); optimizer_add(opt, *l.B.at(k)); k = k + 1UL; }
}

static unsigned long lora_count_(const &Vec<&Tensor> As, const &Vec<&Tensor> Bs) {
    unsigned long n = 0UL;
    unsigned long k = 0UL;
    while (k < As.length()) {
        &Tensor a = *As.at(k);
        &Tensor b = *Bs.at(k);
        unsafe { n = n + a->size + b->size; }
        k = k + 1UL;
    }
    return n;
}

unsigned long lora_param_count(const &LoraMlp l) { return lora_count_(&l.A, &l.B); }

void lora_merge(&LoraMlp l) {
    unsigned long k = 0UL;
    while (k < l.base.nLayers) {
        &Tensor W = *l.base.params.at(2UL * k);
        &Tensor A = *l.A.at(k);
        &Tensor B = *l.B.at(k);
        unsigned long in = lora_rows_(W);
        unsigned long out = lora_cols_(W);
        unsigned long r = l.cfg.rank;
        unsafe {
            unsigned long i = 0UL;
            while (i < in) {
                unsigned long j = 0UL;
                while (j < out) {
                    float acc = (float)0.0;
                    unsigned long t = 0UL;
                    while (t < r) { acc = acc + A->data[i * r + t] * B->data[t * out + j]; t = t + 1UL; }
                    W->data[i * out + j] = W->data[i * out + j] + l.scale * acc;
                    j = j + 1UL;
                }
                i = i + 1UL;
            }
        }
        tensor_fill(B, (float)0.0);
        k = k + 1UL;
    }
}

void lora_detach(&LoraMlp l) {
    unsigned long i = 0UL;
    while (i < l.base.params.length()) {
        &Tensor p = *l.base.params.at(i);
        unsafe { p->requiresGrad = 1; }
        i = i + 1UL;
    }
    unsigned long k = 0UL;
    while (k < l.A.length()) { lora_drop_(*l.A.at(k)); lora_drop_(*l.B.at(k)); k = k + 1UL; }
    l.A.free();
    l.B.free();
}

// ══ NF4 ══════════════════════════════════════════════════════════════════════

// The 16 NormalFloat-4 levels (QLoRA, Appendix E / bitsandbytes).
static float nf4_level_(unsigned long i) {
    if (i == 0UL) return (float)-1.0;
    if (i == 1UL) return (float)-0.6961928009986877;
    if (i == 2UL) return (float)-0.5250730514526367;
    if (i == 3UL) return (float)-0.39491748809814453;
    if (i == 4UL) return (float)-0.28444138169288635;
    if (i == 5UL) return (float)-0.18477343022823334;
    if (i == 6UL) return (float)-0.09105003625154495;
    if (i == 7UL) return (float)0.0;
    if (i == 8UL) return (float)0.07958029955625534;
    if (i == 9UL) return (float)0.16093020141124725;
    if (i == 10UL) return (float)0.24611230194568634;
    if (i == 11UL) return (float)0.33791524171829224;
    if (i == 12UL) return (float)0.44070982933044434;
    if (i == 13UL) return (float)0.5626170039176941;
    if (i == 14UL) return (float)0.7229568362236023;
    return (float)1.0;
}

static unsigned long nf4_nearest_(float v) {
    unsigned long best = 0UL;
    float bestD = (float)3.0;
    unsigned long i = 0UL;
    while (i < 16UL) {
        float d = v - nf4_level_(i);
        if (d < (float)0.0) d = -d;
        if (d < bestD) { bestD = d; best = i; }
        i = i + 1UL;
    }
    return best;
}

struct Nf4Tensor nf4_quantize(const &Tensor w, unsigned long blockSize, int doubleQuant) {
    struct Nf4Tensor q;
    unsigned long n;
    unsafe { n = w->size; }
    q.rows = lora_rows_(w);
    q.cols = lora_cols_(w);
    q.blockSize = blockSize == 0UL ? NF4_BLOCK : blockSize;
    q.blocks = (n + q.blockSize - 1UL) / q.blockSize;
    q.doubleQuant = doubleQuant;
    q.absmaxMean = (float)0.0;
    unsafe {
        unsigned char* codes = (unsigned char*)malloc((n + 1UL) / 2UL + 1UL);
        float* absmax = (float*)malloc(sizeof(float) * q.blocks);
        unsigned long i = 0UL;
        while (i < (n + 1UL) / 2UL + 1UL) { codes[i] = (unsigned char)0; i = i + 1UL; }
        unsigned long bl = 0UL;
        while (bl < q.blocks) {
            unsigned long lo = bl * q.blockSize;
            unsigned long hi = lo + q.blockSize < n ? lo + q.blockSize : n;
            float m = (float)0.0;
            i = lo;
            while (i < hi) { float a = w->data[i]; if (a < (float)0.0) a = -a; if (a > m) m = a; i = i + 1UL; }
            absmax[bl] = m;
            float inv = m > (float)0.0 ? (float)1.0 / m : (float)0.0;
            i = lo;
            while (i < hi) {
                unsigned long c = nf4_nearest_(w->data[i] * inv);
                if (i % 2UL == 0UL) codes[i / 2UL] = (unsigned char)c;
                else codes[i / 2UL] = (unsigned char)((unsigned long)codes[i / 2UL] | (c << 4UL));
                i = i + 1UL;
            }
            bl = bl + 1UL;
        }
        q.codes = (&heap unsigned char)codes;
        if (doubleQuant) {
            // scales - mean, in 8 bits per NF4_SCALE_BLOCK group
            double s = 0.0;
            bl = 0UL;
            while (bl < q.blocks) { s = s + (double)absmax[bl]; bl = bl + 1UL; }
            float mean = (float)(s / (double)q.blocks);
            unsigned long groups = (q.blocks + NF4_SCALE_BLOCK - 1UL) / NF4_SCALE_BLOCK;
            unsigned char* aq = (unsigned char*)malloc(q.blocks);
            float* gs = (float*)malloc(sizeof(float) * groups);
            unsigned long g = 0UL;
            while (g < groups) {
                unsigned long lo = g * NF4_SCALE_BLOCK;
                unsigned long hi = lo + NF4_SCALE_BLOCK < q.blocks ? lo + NF4_SCALE_BLOCK : q.blocks;
                float m = (float)0.0;
                bl = lo;
                while (bl < hi) { float d = absmax[bl] - mean; if (d < (float)0.0) d = -d; if (d > m) m = d; bl = bl + 1UL; }
                float sc = m > (float)0.0 ? m / (float)127.0 : (float)1.0;
                gs[g] = sc;
                bl = lo;
                while (bl < hi) {
                    float v = (absmax[bl] - mean) / sc;
                    long r = (long)(v < (float)0.0 ? v - (float)0.5 : v + (float)0.5);
                    if (r > 127L) r = 127L;
                    if (r < -127L) r = -127L;
                    aq[bl] = (unsigned char)(r + 128L);
                    bl = bl + 1UL;
                }
                g = g + 1UL;
            }
            free((void*)absmax);
            q.absmaxMean = mean;
            q.absmaxQ = (&heap unsigned char)aq;
            q.absmaxScale = (&heap float)gs;
            q.absmax = (&heap float)0;
        } else {
            q.absmax = (&heap float)absmax;
            q.absmaxQ = (&heap unsigned char)0;
            q.absmaxScale = (&heap float)0;
        }
    }
    return q;
}

static float nf4_block_scale_(const &Nf4Tensor q, unsigned long bl) {
    unsafe {
        if (!q.doubleQuant) return q.absmax[bl];
        float d = (float)((long)q.absmaxQ[bl] - 128L) * q.absmaxScale[bl / NF4_SCALE_BLOCK];
        return d + q.absmaxMean;
    }
}

&Tensor nf4_dequantize(const &Nf4Tensor q) {
    &Tensor w = tensor_new_2d(q.rows, q.cols, 0);
    unsafe {
        unsigned long n = q.rows * q.cols;
        unsigned long bl = 0UL;
        while (bl < q.blocks) {
            float m = nf4_block_scale_(q, bl);
            unsigned long lo = bl * q.blockSize;
            unsigned long hi = lo + q.blockSize < n ? lo + q.blockSize : n;
            unsigned long i = lo;
            while (i < hi) {
                unsigned long byte = (unsigned long)q.codes[i / 2UL];
                unsigned long c = i % 2UL == 0UL ? byte & 15UL : byte >> 4UL;
                w->data[i] = nf4_level_(c) * m;
                i = i + 1UL;
            }
            bl = bl + 1UL;
        }
    }
    return w;
}

unsigned long nf4_bytes(const &Nf4Tensor q) {
    unsigned long n = q.rows * q.cols;
    unsigned long b = (n + 1UL) / 2UL;
    if (q.doubleQuant) {
        unsigned long groups = (q.blocks + NF4_SCALE_BLOCK - 1UL) / NF4_SCALE_BLOCK;
        return b + q.blocks + 4UL * groups + 4UL;
    }
    return b + 4UL * q.blocks;
}

void nf4_free(&Nf4Tensor q) {
    unsafe {
        free((void*)q.codes);
        if (q.absmax != (float*)0) free((void*)q.absmax);
        if (q.absmaxQ != (unsigned char*)0) free((void*)q.absmaxQ);
        if (q.absmaxScale != (float*)0) free((void*)q.absmaxScale);
    }
}

// ══ QLoRA ════════════════════════════════════════════════════════════════════

struct QLoraMlp qlora_attach(const &Mlp base, const &LoraConfig cfg, unsigned long blockSize, int doubleQuant) {
    struct QLoraMlp q;
    q.W = vec_new();
    q.bias = vec_new();
    q.A = vec_new();
    q.B = vec_new();
    q.nLayers = base.nLayers;
    q.act = base.act;
    q.outAct = base.outAct;
    q.cfg = *cfg;
    q.scale = cfg.alpha / (float)cfg.rank;
    q.rng = rng_new(cfg.seed);
    unsigned long k = 0UL;
    while (k < base.nLayers) {
        &Tensor W = *base.params.at(2UL * k);
        &Tensor b = *base.params.at(2UL * k + 1UL);
        q.W.push(nf4_quantize(W, blockSize, doubleQuant));
        ?&TensorArena cur = lora_heap_begin_();
        &Tensor bc = tensor_detach(b);
        lora_heap_end_(cur);
        q.bias.push(bc);
        lora_make_pair_(lora_rows_(W), lora_cols_(W), cfg.rank, &q.rng, &q.A, &q.B);
        k = k + 1UL;
    }
    return q;
}

&Tensor qlora_forward(&QLoraMlp q, const &Tensor x) {
    // the dequantized weights are temporaries (in the caller's arena, if any)
    &Tensor h = lora_layer_(x, nf4_dequantize(q.W.at(0UL)), *q.bias.at(0UL), *q.A.at(0UL), *q.B.at(0UL),
                            q.scale, q.cfg.dropout, &q.rng, q.nLayers == 1UL ? q.outAct : q.act);
    unsigned long k = 1UL;
    while (k < q.nLayers) {
        h = lora_layer_(h, nf4_dequantize(q.W.at(k)), *q.bias.at(k), *q.A.at(k), *q.B.at(k),
                        q.scale, q.cfg.dropout, &q.rng, k + 1UL == q.nLayers ? q.outAct : q.act);
        k = k + 1UL;
    }
    return h;
}

void qlora_add_params(&Optimizer opt, const &QLoraMlp q) {
    unsigned long k = 0UL;
    while (k < q.A.length()) { optimizer_add(opt, *q.A.at(k)); optimizer_add(opt, *q.B.at(k)); k = k + 1UL; }
}

unsigned long qlora_param_count(const &QLoraMlp q) { return lora_count_(&q.A, &q.B); }

unsigned long qlora_base_bytes(const &QLoraMlp q) {
    unsigned long n = 0UL;
    unsigned long k = 0UL;
    while (k < q.nLayers) {
        n = n + nf4_bytes(q.W.at(k));
        &Tensor b = *q.bias.at(k);
        unsafe { n = n + 4UL * b->size; }
        k = k + 1UL;
    }
    return n;
}

void qlora_free(&QLoraMlp q) {
    unsigned long k = 0UL;
    while (k < q.nLayers) {
        nf4_free(q.W.at(k));
        lora_drop_(*q.bias.at(k));
        lora_drop_(*q.A.at(k));
        lora_drop_(*q.B.at(k));
        k = k + 1UL;
    }
    q.W.free(); q.bias.free(); q.A.free(); q.B.free();
}

} // namespace std
