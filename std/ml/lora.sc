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
        case some(a): { a.begin(); }
        case null: { }
    }
}

static void lora_drop_(&FloatTensor t) {
    t->free();
    unsafe { free((void*)t); }
}

static unsigned long lora_rows_(const &FloatTensor t) {
    unsafe { return t->shape[0]; }
}

static unsigned long lora_cols_(const &FloatTensor t) {
    unsafe { return t->shape[t->ndim - 1UL]; }
}

struct LoraConfig LoraConfig::default(unsigned long rank) {
    struct LoraConfig c;
    c.rank = rank;
    c.alpha = (float)rank;
    c.dropout = (float)0.0;
    c.seed = 1UL;
    return c;
}

// A [in, r] Kaiming-uniform (as PEFT initializes it), B [r, out] zero.
static void lora_make_pair_(unsigned long in, unsigned long out, unsigned long r, &Rng rng,
                            &Vec<&FloatTensor> As, &Vec<&FloatTensor> Bs) {
    ?&TensorArena cur = lora_heap_begin_();
    &FloatTensor A = FloatTensor::new_2d(in, r, 1);
    &FloatTensor B = FloatTensor::new_2d(r, out, 1);
    lora_heap_end_(cur);
    float bound = (float)1.0 / sqrt_f((float)in);
    rng.fill_uniform(A, -bound, bound);
    As.push(A);
    Bs.push(B);
}

// One adapted layer: act(x W + b + scale * (drop(x) A) B).
static &FloatTensor lora_layer_(const &FloatTensor x, const &FloatTensor W, const &FloatTensor b, const &FloatTensor A,
                           const &FloatTensor B, float scale, float dropout, &Rng rng, int act) {
    &FloatTensor base = tensor_add_bias(tensor_matmul(x, W), b);
    &FloatTensor xin = x.add_scalar((float)0.0);
    if (dropout > (float)0.0 && Autograd::get().enabled()) {
        &FloatTensor mask = FloatTensor::new_2d(lora_rows_(x), lora_cols_(x), 0);
        float keep = (float)1.0 / ((float)1.0 - dropout);
        unsafe {
            unsigned long i = 0UL;
            while (i < mask->size) { mask->data[i] = rng.uniform() < dropout ? (float)0.0 : keep; i = i + 1UL; }
        }
        xin = tensor_mul(x, mask);
    }
    &FloatTensor upd = tensor_matmul(tensor_matmul(xin, A), B).scale(scale);
    return tensor_add(base, upd).activate(act);
}

// ══ LoRA ═════════════════════════════════════════════════════════════════════

struct LoraMlp LoraMlp::attach(&Mlp base, const &LoraConfig cfg) {
    struct LoraMlp l;
    l.base = base;
    l.A = vec_new();
    l.B = vec_new();
    l.cfg = *cfg;
    l.scale = cfg.alpha / (float)cfg.rank;
    l.rng = Rng::new(cfg.seed);
    unsigned long i = 0UL;
    while (i < base.params.length()) {
        &FloatTensor p = *base.params.at(i);
        unsafe { p->requiresGrad = 0; }
        i = i + 1UL;
    }
    unsigned long k = 0UL;
    while (k < base.nLayers) {
        &FloatTensor W = *base.params.at(2UL * k);
        lora_make_pair_(lora_rows_(W), lora_cols_(W), cfg.rank, &l.rng, &l.A, &l.B);
        k = k + 1UL;
    }
    return l;
}

&FloatTensor LoraMlp::forward(const &FloatTensor x) {
    &Mlp m = self.base;
    &FloatTensor h = lora_layer_(x, *m.params.at(0UL), *m.params.at(1UL), *self.A.at(0UL), *self.B.at(0UL),
                            self.scale, self.cfg.dropout, &self.rng, m.nLayers == 1UL ? m.outAct : m.act);
    unsigned long k = 1UL;
    while (k < m.nLayers) {
        h = lora_layer_(h, *m.params.at(2UL * k), *m.params.at(2UL * k + 1UL), *self.A.at(k), *self.B.at(k),
                        self.scale, self.cfg.dropout, &self.rng, k + 1UL == m.nLayers ? m.outAct : m.act);
        k = k + 1UL;
    }
    return h;
}

static void lora_collect_(const &Vec<&FloatTensor> As, const &Vec<&FloatTensor> Bs, &Vec<&FloatTensor> out) {
    unsigned long k = 0UL;
    while (k < As.length()) { out.push(*As.at(k)); out.push(*Bs.at(k)); k = k + 1UL; }
}

void LoraMlp::collect_params(&Vec<&FloatTensor> out) const { lora_collect_(&self.A, &self.B, out); }
void LoraMlp::collect_state(&Vec<&FloatTensor> out) const { lora_collect_(&self.A, &self.B, out); }

void LoraMlp::merge() {
    unsigned long k = 0UL;
    while (k < self.base.nLayers) {
        &FloatTensor W = *self.base.params.at(2UL * k);
        &FloatTensor A = *self.A.at(k);
        &FloatTensor B = *self.B.at(k);
        unsigned long in = lora_rows_(W);
        unsigned long out = lora_cols_(W);
        unsigned long r = self.cfg.rank;
        unsafe {
            unsigned long i = 0UL;
            while (i < in) {
                unsigned long j = 0UL;
                while (j < out) {
                    float acc = (float)0.0;
                    unsigned long t = 0UL;
                    while (t < r) { acc = acc + A->data[i * r + t] * B->data[t * out + j]; t = t + 1UL; }
                    W->data[i * out + j] = W->data[i * out + j] + self.scale * acc;
                    j = j + 1UL;
                }
                i = i + 1UL;
            }
        }
        B.fill((float)0.0);
        k = k + 1UL;
    }
}

void LoraMlp::detach() {
    unsigned long i = 0UL;
    while (i < self.base.params.length()) {
        &FloatTensor p = *self.base.params.at(i);
        unsafe { p->requiresGrad = 1; }
        i = i + 1UL;
    }
    unsigned long k = 0UL;
    while (k < self.A.length()) { lora_drop_(*self.A.at(k)); lora_drop_(*self.B.at(k)); k = k + 1UL; }
    self.A.free();
    self.B.free();
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

struct Nf4Tensor Nf4Tensor::quantize(const &FloatTensor w, unsigned long blockSize, int doubleQuant) {
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

&FloatTensor Nf4Tensor::dequantize() const {
    &FloatTensor w = FloatTensor::new_2d(self.rows, self.cols, 0);
    unsafe {
        unsigned long n = self.rows * self.cols;
        unsigned long bl = 0UL;
        while (bl < self.blocks) {
            float m = nf4_block_scale_(self, bl);
            unsigned long lo = bl * self.blockSize;
            unsigned long hi = lo + self.blockSize < n ? lo + self.blockSize : n;
            unsigned long i = lo;
            while (i < hi) {
                unsigned long byte = (unsigned long)self.codes[i / 2UL];
                unsigned long c = i % 2UL == 0UL ? byte & 15UL : byte >> 4UL;
                w->data[i] = nf4_level_(c) * m;
                i = i + 1UL;
            }
            bl = bl + 1UL;
        }
    }
    return w;
}

unsigned long Nf4Tensor::bytes() const {
    unsigned long n = self.rows * self.cols;
    unsigned long b = (n + 1UL) / 2UL;
    if (self.doubleQuant) {
        unsigned long groups = (self.blocks + NF4_SCALE_BLOCK - 1UL) / NF4_SCALE_BLOCK;
        return b + self.blocks + 4UL * groups + 4UL;
    }
    return b + 4UL * self.blocks;
}

void Nf4Tensor::free() {
    unsafe {
        free((void*)self.codes);
        if (self.absmax != (float*)0) free((void*)self.absmax);
        if (self.absmaxQ != (unsigned char*)0) free((void*)self.absmaxQ);
        if (self.absmaxScale != (float*)0) free((void*)self.absmaxScale);
    }
}

// ══ QLoRA ════════════════════════════════════════════════════════════════════

struct QLoraMlp QLoraMlp::attach(const &Mlp base, const &LoraConfig cfg, unsigned long blockSize, int doubleQuant) {
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
    q.rng = Rng::new(cfg.seed);
    unsigned long k = 0UL;
    while (k < base.nLayers) {
        &FloatTensor W = *base.params.at(2UL * k);
        &FloatTensor b = *base.params.at(2UL * k + 1UL);
        q.W.push(Nf4Tensor::quantize(W, blockSize, doubleQuant));
        ?&TensorArena cur = lora_heap_begin_();
        &FloatTensor bc = b.detach();
        lora_heap_end_(cur);
        q.bias.push(bc);
        lora_make_pair_(lora_rows_(W), lora_cols_(W), cfg.rank, &q.rng, &q.A, &q.B);
        k = k + 1UL;
    }
    return q;
}

&FloatTensor QLoraMlp::forward(const &FloatTensor x) {
    // the dequantized weights are temporaries (in the caller's arena, if any)
    &FloatTensor h = lora_layer_(x, self.W.at(0UL).dequantize(), *self.bias.at(0UL), *self.A.at(0UL), *self.B.at(0UL),
                            self.scale, self.cfg.dropout, &self.rng, self.nLayers == 1UL ? self.outAct : self.act);
    unsigned long k = 1UL;
    while (k < self.nLayers) {
        h = lora_layer_(h, self.W.at(k).dequantize(), *self.bias.at(k), *self.A.at(k), *self.B.at(k),
                        self.scale, self.cfg.dropout, &self.rng, k + 1UL == self.nLayers ? self.outAct : self.act);
        k = k + 1UL;
    }
    return h;
}

void QLoraMlp::collect_params(&Vec<&FloatTensor> out) const { lora_collect_(&self.A, &self.B, out); }
void QLoraMlp::collect_state(&Vec<&FloatTensor> out) const { lora_collect_(&self.A, &self.B, out); }

unsigned long QLoraMlp::base_bytes() const {
    unsigned long n = 0UL;
    unsigned long k = 0UL;
    while (k < self.nLayers) {
        n = n + self.W.at(k).bytes();
        &FloatTensor b = *self.bias.at(k);
        unsafe { n = n + 4UL * b->size; }
        k = k + 1UL;
    }
    return n;
}

void QLoraMlp::free() {
    unsigned long k = 0UL;
    while (k < self.nLayers) {
        self.W.at(k).free();
        lora_drop_(*self.bias.at(k));
        lora_drop_(*self.A.at(k));
        lora_drop_(*self.B.at(k));
        k = k + 1UL;
    }
    self.W.free(); self.bias.free(); self.A.free(); self.B.free();
}

} // namespace std
