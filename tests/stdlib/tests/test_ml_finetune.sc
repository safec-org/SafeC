// std::ml fine-tuning and normalization: Adagrad / Lion, LayerNorm and
// AdaLN (norm.h), LoRA, NF4 and QLoRA (lora.h).
#include <std/test/test.h>
#include <std/ml/nn.h>
#include <std/ml/module.h>
#include <std/ml/norm.h>
#include <std/ml/lora.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/math.h>

extern unsigned char* malloc(unsigned long size);
extern void free(unsigned char* ptr);

static void drop_(&FloatTensor t) {
    unsafe { t->free(); }   // a heap tensor, freed below
    unsafe { free((unsigned char*)t); }
}

// ── Optimizers ───────────────────────────────────────────────────────────────
void t_adagrad_lion() {
    int kinds[2];
    kinds[0] = OPT_ADAGRAD; kinds[1] = OPT_LION;
    int k = 0;
    while (k < 2) {
        &FloatTensor w = FloatTensor::new_2d(1UL, 3UL, 1);
        &FloatTensor target = FloatTensor::new_2d(1UL, 3UL, 0);
        unsafe { target->data[0] = (float)1.0; target->data[1] = (float)-2.0; target->data[2] = (float)0.5; }
        struct Optimizer opt = Optimizer::new(kinds[k], kinds[k] == OPT_ADAGRAD ? (float)0.5 : (float)0.01);
        opt.weightDecay = (float)0.0;
        opt.add(w);
        struct TensorArena ar = TensorArena::new(0UL);
        unsigned long s = 0UL;
        while (s < 600UL) {
            ar.begin();
            trainer_step_(&opt, tensor_mse_loss(w, target), (float)0.0);
            tensor_arena_end();
            ar.reset();
            s = s + 1UL;
        }
        unsafe {
            float e = (float)0.0;
            unsigned long i = 0UL;
            while (i < 3UL) { float d = w->data[i] - target->data[i]; e = e + d * d; i = i + 1UL; }
            ASSERT_TRUE(e < (float)1e-3);
        }
        ar.free();
        opt.free();
        drop_(w); drop_(target);
        k = k + 1;
    }
}

// ── Gradient checks ──────────────────────────────────────────────────────────
static &FloatTensor gc_loss_(int which, &FloatTensor x, const &FloatTensor aux, const &AdaLN ad, const &FloatTensor cond) {
    if (which == 0) return (tensor_mul(x.layer_norm((float)0.00001), aux)).sum();
    if (which == 1) return (tensor_mul(tensor_mul_row(x, aux.slice_cols(0UL, 1UL)), aux)).sum();
    if (which == 2) return tensor_mul(x.slice_cols(1UL, 2UL), aux.slice_cols(0UL, 2UL)).sum();
    struct AdaLNOut o = ad.forward(x, cond);
    return tensor_mul(tensor_mul(o.out, o.gate), aux).sum();
}

static double grad_check_(int which) {
    struct Rng rng = Rng::new(3UL + (unsigned long)which);
    &FloatTensor x = FloatTensor::new_2d(4UL, 4UL, 1);
    &FloatTensor aux = FloatTensor::new_2d(4UL, 4UL, 0);
    &FloatTensor cond = FloatTensor::new_2d(4UL, 3UL, 0);
    rng.fill_uniform(x, (float)-1.0, (float)1.0);
    rng.fill_uniform(aux, (float)-1.0, (float)1.0);
    rng.fill_uniform(cond, (float)-1.0, (float)1.0);
    // which 1 multiplies by the row [aux[0..4)] -- slice_cols(aux, 0, 1) is [4, 1]; use a [4] row instead
    struct AdaLN ad = AdaLN::new(4UL, 3UL, 1);
    rng.fill_uniform(ad.W, (float)-0.5, (float)0.5);   // away from the zero init
    rng.fill_uniform(ad.b, (float)-0.5, (float)0.5);
    struct TensorArena ar = TensorArena::new(0UL);
    ar.begin();
    gc_loss_(which, x, aux, &ad, cond).backward();
    ar.reset();
    double worst = 0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < 16UL) {
            float keep = x->data[i];
            float h = (float)0.002;
            x->data[i] = keep + h;
            float lp = gc_loss_(which, x, aux, &ad, cond)->data[0];
            x->data[i] = keep - h;
            float lm = gc_loss_(which, x, aux, &ad, cond)->data[0];
            x->data[i] = keep;
            ar.reset();
            double num = ((double)lp - (double)lm) / (2.0 * (double)h);
            double e = num - (double)x->grad[i];
            if (e < 0.0) e = -e;
            double sc = num < 0.0 ? -num : num;
            e = e / (sc > 1.0 ? sc : 1.0);
            if (e > worst) worst = e;
            i = i + 1UL;
        }
        // AdaLN: also the modulation weights
        if (which == 3) {
            i = 0UL;
            while (i < ad.W->size) {
                float keep = ad.W->data[i];
                float h = (float)0.002;
                ad.W->data[i] = keep + h;
                float lp = gc_loss_(which, x, aux, &ad, cond)->data[0];
                ad.W->data[i] = keep - h;
                float lm = gc_loss_(which, x, aux, &ad, cond)->data[0];
                ad.W->data[i] = keep;
                ar.reset();
                double num = ((double)lp - (double)lm) / (2.0 * (double)h);
                double e = num - (double)ad.W->grad[i];
                if (e < 0.0) e = -e;
                double sc = num < 0.0 ? -num : num;
                e = e / (sc > 1.0 ? sc : 1.0);
                if (e > worst) worst = e;
                i = i + 1UL;
            }
        }
    }
    tensor_arena_end();
    ar.free();
    ad.free();
    drop_(x); drop_(aux); drop_(cond);
    return worst;
}

void t_norm_grad_checks() {
    int w = 0;
    while (w < 4) {
        ASSERT_TRUE(grad_check_(w) < 2e-3);
        w = w + 1;
    }
}

void t_mul_row_grad() {
    // tensor_mul_row's gradient for the row vector itself
    &FloatTensor x = FloatTensor::new_2d(3UL, 2UL, 0);
    &FloatTensor g = FloatTensor::new_1d(2UL, 1);
    unsafe {
        unsigned long i = 0UL;
        while (i < 6UL) { x->data[i] = (float)(i + 1UL); i = i + 1UL; }
        g->data[0] = (float)2.0; g->data[1] = (float)-1.0;
    }
    &FloatTensor y = tensor_mul_row(x, g);
    unsafe { ASSERT_TRUE(y->data[0] == (float)2.0 && y->data[1] == (float)-2.0 && y->data[5] == (float)-6.0); }
    y.sum().backward();
    unsafe { ASSERT_TRUE(g->grad[0] == (float)9.0 && g->grad[1] == (float)12.0); }   // column sums
}

// ── LayerNorm / AdaLN ────────────────────────────────────────────────────────
void t_layer_norm() {
    struct Rng rng = Rng::new(9UL);
    &FloatTensor x = FloatTensor::new_2d(5UL, 16UL, 0);
    rng.fill_normal(x, (float)3.0, (float)2.0);
    struct LayerNorm ln = LayerNorm::new(16UL);
    &FloatTensor y = ln.forward(x);
    unsafe {
        unsigned long r = 0UL;
        while (r < 5UL) {
            double s = 0.0; double sq = 0.0;
            unsigned long c = 0UL;
            while (c < 16UL) { double v = (double)y->data[r * 16UL + c]; s = s + v; sq = sq + v * v; c = c + 1UL; }
            ASSERT_TRUE(s / 16.0 < 1e-5 && s / 16.0 > -1e-5);
            ASSERT_TRUE(sq / 16.0 > 0.999 && sq / 16.0 < 1.001);
            r = r + 1UL;
        }
    }
    // a fresh AdaLN is a plain LayerNorm with a zero gate
    struct AdaLN ad = AdaLN::new(16UL, 4UL, 1);
    &FloatTensor cond = FloatTensor::new_2d(5UL, 4UL, 0);
    rng.fill_normal(cond, (float)0.0, (float)1.0);
    struct AdaLNOut o = ad.forward(x, cond);
    unsafe {
        unsigned long i = 0UL;
        while (i < 80UL) {
            float d = o.out->data[i] - y->data[i];
            ASSERT_TRUE(d < (float)1e-4 && d > (float)-1e-4);
            ASSERT_TRUE(o.gate->data[i] == (float)0.0);
            i = i + 1UL;
        }
    }
    ad.free();
    ln.free();
}

// AdaLN learns a class-dependent affine map that a plain LayerNorm can't:
// target = LN(x) * (1 + 2c) - c for class c in {0, 1, 2}.
struct CondModel {
    struct AdaLN ad;
    unsigned long classes;
};

static &FloatTensor cond_loss_(&CondModel m, const &FloatTensor xy, const &FloatTensor labels) {
    unsigned long n = 8UL;
    &FloatTensor x = xy.slice_cols(0UL, n);
    &FloatTensor target = xy.slice_cols(n, n);
    struct AdaLNOut o = m.ad.forward(x, labels.one_hot(m.classes));
    return tensor_mse_loss(o.out, target);
}

void t_adaln_conditioning() {
    struct Rng rng = Rng::new(2UL);
    unsigned long n = 8UL;
    unsigned long N = 600UL;
    struct Dataset d = Dataset::blobs(N, 3UL, 2UL * n, (float)1.0, &rng);   // reuse: x then target
    unsafe {
        unsigned long i = 0UL;
        while (i < N) {
            float c = d.y->data[i];
            float* row = (float*)&d.x->data[i * 2UL * n];
            unsigned long k = 0UL;
            while (k < n) { row[k] = rng.normal(); k = k + 1UL; }
            double s = 0.0; double sq = 0.0;
            k = 0UL;
            while (k < n) { s = s + (double)row[k]; k = k + 1UL; }
            double mean = s / (double)n;
            k = 0UL;
            while (k < n) { double dv = (double)row[k] - mean; sq = sq + dv * dv; k = k + 1UL; }
            float is = (float)(1.0 / sqrt_d(sq / (double)n + 1e-6));
            k = 0UL;
            while (k < n) {
                row[n + k] = ((row[k] - (float)mean) * is) * ((float)1.0 + (float)2.0 * c) - c;
                k = k + 1UL;
            }
            i = i + 1UL;
        }
    }
    struct CondModel m;
    m.ad = AdaLN::new(n, 3UL, 0);
    m.classes = 3UL;
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.05);
    model_add_params(&m.ad, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 40UL;
    struct TrainReport rep = trainer_fit(&m, cond_loss_, &opt, &d, &tc);
    ASSERT_TRUE(rep.firstEpochLoss > (float)0.5);
    ASSERT_TRUE(rep.lastEpochLoss < (float)0.001);
    opt.free();
    m.ad.free();
    d.free();
}

// ── LoRA ─────────────────────────────────────────────────────────────────────
static &FloatTensor base_loss_(&Mlp m, const &FloatTensor x, const &FloatTensor y) {
    return tensor_cross_entropy_loss(m.forward(x), y);
}
static &FloatTensor base_logits_(&Mlp m, const &FloatTensor x) { return m.forward(x); }
static &FloatTensor lora_loss_(&LoraMlp l, const &FloatTensor x, const &FloatTensor y) {
    return tensor_cross_entropy_loss(l.forward(x), y);
}
static &FloatTensor lora_logits_(&LoraMlp l, const &FloatTensor x) { return l.forward(x); }
static &FloatTensor qlora_loss_(&QLoraMlp q, const &FloatTensor x, const &FloatTensor y) {
    return tensor_cross_entropy_loss(q.forward(x), y);
}
static &FloatTensor qlora_logits_(&QLoraMlp q, const &FloatTensor x) { return q.forward(x); }

// Task A: moons. Task B: the same points, rotated 90 degrees, with the
// labels swapped -- a new task the frozen base gets wrong.
static struct Dataset task_b_(const &Dataset a) {
    struct Rng rng = Rng::new(77UL);
    struct Dataset b = Dataset::moons(a.n, (float)0.0, &rng);
    unsafe {
        unsigned long i = 0UL;
        while (i < a.n) {
            float x0 = a.x->data[i * 2UL];
            float x1 = a.x->data[i * 2UL + 1UL];
            b.x->data[i * 2UL] = -x1;
            b.x->data[i * 2UL + 1UL] = x0;
            b.y->data[i] = (float)1.0 - a.y->data[i];
            i = i + 1UL;
        }
    }
    return b;
}

static float checksum_(const &Mlp m) {
    double s = 0.0;
    unsigned long i = 0UL;
    while (i < m.params.length()) {
        &FloatTensor p = *m.params.at(i);
        unsafe { unsigned long j = 0UL; while (j < p->size) { s = s + (double)p->data[j] * (double)(j % 7UL + 1UL); j = j + 1UL; } }
        i = i + 1UL;
    }
    return (float)s;
}

static struct Mlp pretrain_(&Dataset a) {
    struct Rng rng = Rng::new(6UL);
    struct Mlp base = Mlp::new(2UL, 64UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &rng);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.01);
    model_add_params(&base, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 60UL;
    trainer_fit(&base, base_loss_, &opt, a, &tc);
    opt.free();
    return base;
}

void t_lora() {
    struct Rng rng = Rng::new(4UL);
    struct Dataset a = Dataset::moons(512UL, (float)0.08, &rng);
    struct Dataset b = task_b_(&a);
    struct Mlp base = pretrain_(&a);
    ASSERT_TRUE(trainer_accuracy(&base, base_logits_, &a, 128UL) > (float)0.97);
    ASSERT_TRUE(trainer_accuracy(&base, base_logits_, &b, 128UL) < (float)0.7);
    float before = checksum_(&base);

    struct LoraConfig cfg = LoraConfig::default(4UL);
    cfg.alpha = (float)8.0;
    struct LoraMlp ft = LoraMlp::attach(&base, &cfg); defer drop(ft);
    // B starts at zero: the adapted network equals the base
    Autograd::get().set_enabled(0);
    &FloatTensor y0 = base.forward(b.x);
    &FloatTensor y1 = ft.forward(b.x);
    unsafe { unsigned long i = 0UL; while (i < y0->size) { ASSERT_TRUE(y0->data[i] == y1->data[i]); i = i + 1UL; } }
    Autograd::get().set_enabled(1);
    // trainable: r * (in + out) per layer (1040 here, 23% of this small base;
    // the fraction r (in + out) / (in out) shrinks with layer width)
    ASSERT_EQ(model_param_count(&ft), 4UL * (2UL + 64UL) + 4UL * (64UL + 64UL) + 4UL * (64UL + 2UL));
    ASSERT_TRUE(model_param_count(&ft) * 4UL < model_param_count(&base));

    struct Optimizer opt = Optimizer::new(OPT_ADAMW, (float)0.01);
    model_add_params(&ft, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 60UL;
    trainer_fit(&ft, lora_loss_, &opt, &b, &tc);
    ASSERT_TRUE(trainer_accuracy(&ft, lora_logits_, &b, 128UL) > (float)0.95);
    ASSERT_TRUE(checksum_(&base) == before);          // the base never moved

    // merging folds the update into the base weights
    Autograd::get().set_enabled(0);
    &FloatTensor ya = ft.forward(b.x);
    ft.merge();
    &FloatTensor yb = base.forward(b.x);
    unsafe {
        unsigned long i = 0UL;
        while (i < ya->size) { float d = ya->data[i] - yb->data[i]; ASSERT_TRUE(d < (float)1e-3 && d > (float)-1e-3); i = i + 1UL; }
    }
    Autograd::get().set_enabled(1);
    ASSERT_TRUE(trainer_accuracy(&base, base_logits_, &b, 128UL) > (float)0.95);
    ft.detach();
    opt.free();
    base.free();
    a.free(); b.free();
}

// ── NF4 / QLoRA ──────────────────────────────────────────────────────────────
void t_nf4() {
    // values on the NF4 grid come back exactly
    &FloatTensor g = FloatTensor::new_2d(2UL, 8UL, 0);
    unsafe {
        float lv[16];
        lv[0] = (float)-1.0; lv[1] = (float)-0.6961928009986877; lv[2] = (float)-0.5250730514526367;
        lv[3] = (float)-0.39491748809814453; lv[4] = (float)-0.28444138169288635; lv[5] = (float)-0.18477343022823334;
        lv[6] = (float)-0.09105003625154495; lv[7] = (float)0.0; lv[8] = (float)0.07958029955625534;
        lv[9] = (float)0.16093020141124725; lv[10] = (float)0.24611230194568634; lv[11] = (float)0.33791524171829224;
        lv[12] = (float)0.44070982933044434; lv[13] = (float)0.5626170039176941; lv[14] = (float)0.7229568362236023;
        lv[15] = (float)1.0;
        unsigned long i = 0UL;
        while (i < 16UL) { g->data[i] = lv[i] * (float)0.5; i = i + 1UL; }
    }
    struct Nf4Tensor qg = Nf4Tensor::quantize(g, 16UL, 0);
    &FloatTensor gd = qg.dequantize();
    unsafe { unsigned long i = 0UL; while (i < 16UL) { ASSERT_TRUE(gd->data[i] == g->data[i]); i = i + 1UL; } }
    qg.free();

    // Gaussian weights: small relative error at ~4.5 bits (4.1 with double quantization)
    struct Rng rng = Rng::new(1UL);
    unsigned long rows = 256UL; unsigned long cols = 256UL;
    &FloatTensor w = FloatTensor::new_2d(rows, cols, 0);
    rng.fill_normal(w, (float)0.0, (float)0.02);
    int dq = 0;
    double errs[2];
    while (dq < 2) {
        struct Nf4Tensor q = Nf4Tensor::quantize(w, NF4_BLOCK, dq);
        &FloatTensor wd = q.dequantize();
        double se = 0.0; double s2 = 0.0;
        unsafe {
            unsigned long i = 0UL;
            while (i < rows * cols) {
                double d = (double)wd->data[i] - (double)w->data[i];
                se = se + d * d;
                s2 = s2 + (double)w->data[i] * (double)w->data[i];
                i = i + 1UL;
            }
        }
        errs[dq] = sqrt_d(se / s2);
        double bits = 8.0 * (double)q.bytes() / (double)(rows * cols);
        if (dq) ASSERT_TRUE(bits < 4.2);
        else ASSERT_TRUE(bits > 4.45 && bits < 4.55);
        ASSERT_TRUE(errs[dq] < 0.12);
        q.free();
        drop_(wd);
        dq = dq + 1;
    }
    ASSERT_TRUE(errs[1] < errs[0] * 1.05 + 0.001);   // double quantization costs almost nothing
    drop_(w);
}

void t_qlora() {
    struct Rng rng = Rng::new(4UL);
    struct Dataset a = Dataset::moons(512UL, (float)0.08, &rng);
    struct Dataset b = task_b_(&a);
    struct Mlp base = pretrain_(&a);
    unsigned long fp32Bytes = 4UL * model_param_count(&base);
    struct LoraConfig cfg = LoraConfig::default(4UL);
    cfg.alpha = (float)8.0;
    struct QLoraMlp q = QLoraMlp::attach(&base, &cfg, NF4_BLOCK, 1);
    base.free();                                   // the quantized copy is all that's needed
    ASSERT_TRUE(q.base_bytes() * 6UL < fp32Bytes);
    // the 4-bit base still solves task A
    ASSERT_TRUE(trainer_accuracy(&q, qlora_logits_, &a, 128UL) > (float)0.95);
    struct Optimizer opt = Optimizer::new(OPT_ADAMW, (float)0.01);
    model_add_params(&q, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 60UL;
    trainer_fit(&q, qlora_loss_, &opt, &b, &tc);
    ASSERT_TRUE(trainer_accuracy(&q, qlora_logits_, &b, 128UL) > (float)0.95);
    ASSERT_EQ(model_param_count(&q), 4UL * (2UL + 64UL) + 4UL * (64UL + 64UL) + 4UL * (64UL + 2UL));
    opt.free();
    q.free();
    a.free(); b.free();
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("Adagrad and Lion converge", t_adagrad_lion);
        t.add("gradient checks (layer norm, slices, AdaLN)", t_norm_grad_checks);
        t.add("row-broadcast multiply gradient", t_mul_row_grad);
        t.add("LayerNorm / fresh AdaLN", t_layer_norm);
        t.add("AdaLN learns class conditioning", t_adaln_conditioning);
        t.add("LoRA fine-tune, frozen base, merge", t_lora);
        t.add("NF4 quantization", t_nf4);
        t.add("QLoRA fine-tune on a 4-bit base", t_qlora);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
