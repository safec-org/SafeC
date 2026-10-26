// SafeC Standard Library — generative adversarial networks (see gan.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/gan.h>
#include <std/ml/module.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/ml/convnet.h>
#include <std/panic.h>
#include <std/ml/datasets.h>

namespace std {

struct GanConfig GanConfig::default(unsigned long latent) {
    struct GanConfig c;
    c.latent = latent;
    c.hidden = 128UL;
    c.hiddenLayers = 2UL;
    c.loss = GAN_STANDARD;
    c.genAct = ACT_RELU;
    c.outAct = ACT_NONE;
    c.lrG = (float)0.0002;
    c.lrD = (float)0.0002;
    c.nCritic = 1UL;
    c.clip = (float)0.01;
    c.gpWeight = (float)10.0;
    c.classes = 0UL;
    c.emaDecay = (float)0.999;
    c.seed = 1UL;
    return c;
}

static struct Optimizer gan_opt_(int loss, float lr) {
    if (loss == GAN_WGAN) return Optimizer::new(OPT_RMSPROP, lr);
    struct Optimizer o = Optimizer::new(OPT_ADAM, lr);
    o.beta1 = (float)0.5;
    if (loss == GAN_WGAN_GP) { o.beta1 = (float)0.0; o.beta2 = (float)0.9; }
    return o;
}

struct Gan Gan::new(unsigned long dataDim, const &GanConfig cfg) {
    struct Gan g;
    g.cfg = *cfg;
    g.rng = Rng::new(cfg.seed);
    unsigned long c = cfg.classes;
    g.gen = Mlp::new(cfg.latent + c, cfg.hidden, cfg.hiddenLayers, dataDim, cfg.genAct, cfg.outAct, &g.rng);
    g.genEma = Mlp::new(cfg.latent + c, cfg.hidden, cfg.hiddenLayers, dataDim, cfg.genAct, cfg.outAct, &g.rng);
    model_copy(&g.genEma, &g.gen);
    g.disc = Mlp::new(dataDim + c, cfg.hidden, cfg.hiddenLayers, 1UL, ACT_LEAKY, ACT_NONE, &g.rng);
    g.optG = gan_opt_(cfg.loss, cfg.lrG);
    g.optD = gan_opt_(cfg.loss, cfg.lrD);
    model_add_params(&g.gen, &g.optG);
    model_add_params(&g.disc, &g.optD);
    g.stepArena = TensorArena::new(0UL);
    g.steps = 0UL;
    return g;
}

static &FloatTensor gan_cond_(const &Gan g, const &FloatTensor x, ?&FloatTensor labels) {
    unsafe { tensor_host_raw_((void*)x); }
    if (g.cfg.classes > 0UL) {
        match (labels) {
            case some(l): { return tensor_concat_cols(x, l.one_hot(g.cfg.classes)); }
            case null: { }
        }
        unsigned long m;
        unsafe { m = x->shape[0]; }
        return tensor_concat_cols(x, FloatTensor::new_1d(m, 0).one_hot(g.cfg.classes));
    }
    return x.add_scalar((float)0.0);
}

static &FloatTensor gan_fake_(&Gan g, unsigned long m, ?&FloatTensor labels, int averaged) {
    &FloatTensor z = g.rng.normal_tensor(m, g.cfg.latent);
    if (averaged) return g.genEma.forward(gan_cond_(g, z, labels));
    return g.gen.forward(gan_cond_(g, z, labels));
}

static &FloatTensor gan_full_(unsigned long m, float v) {
    &FloatTensor t = FloatTensor::new_2d(m, 1UL, 0);
    t.fill(v);
    return t;
}

static &FloatTensor gan_d_loss_(int k, &FloatTensor dReal, &FloatTensor dFake, unsigned long m) {
    if (k == GAN_LSGAN) {
        &FloatTensor a = tensor_mse_loss(dReal, gan_full_(m, (float)1.0));
        &FloatTensor b = tensor_mse_loss(dFake, gan_full_(m, (float)0.0));
        return tensor_add(a, b).scale((float)0.5);
    }
    if (k == GAN_WGAN || k == GAN_WGAN_GP) return tensor_sub(dFake.mean(), dReal.mean());
    if (k == GAN_HINGE) {
        &FloatTensor a = dReal.scale((float)-1.0).add_scalar((float)1.0).relu().mean();
        &FloatTensor b = dFake.add_scalar((float)1.0).relu().mean();
        return tensor_add(a, b);
    }
    &FloatTensor a = tensor_bce_logits_loss(dReal, gan_full_(m, (float)1.0));
    &FloatTensor b = tensor_bce_logits_loss(dFake, gan_full_(m, (float)0.0));
    return tensor_add(a, b);
}

static &FloatTensor gan_g_loss_(int k, &FloatTensor dFake, unsigned long m) {
    if (k == GAN_LSGAN) return (tensor_mse_loss(dFake, gan_full_(m, (float)1.0))).scale((float)0.5);
    if (k == GAN_WGAN || k == GAN_WGAN_GP || k == GAN_HINGE) return dFake.mean().scale((float)-1.0);
    return tensor_bce_logits_loss(dFake, gan_full_(m, (float)1.0));
}

static void gan_clip_(const &Vec<&FloatTensor> params, float c) {
    unsigned long i = 0UL;
    while (i < params.length()) {
        &FloatTensor p = *params.at(i);
        unsafe { tensor_host_write_raw_((void*)p); }
        unsafe {
            unsigned long j = 0UL;
            while (j < p->size) {
                float v = p->data[j];
                p->data[j] = v > c ? c : (v < -c ? -c : v);
                j = j + 1UL;
            }
        }
        i = i + 1UL;
    }
}

static float gan_value_(const &FloatTensor t) {
    unsafe { tensor_host_raw_((void*)t); }
    unsafe { return t->data[0]; }
}

// warm-up: the average follows closely over the first steps
static float gan_ema_decay_(const &GanConfig cfg, unsigned long steps) {
    float t = (float)steps;
    float warm = ((float)1.0 + t) / ((float)10.0 + t);
    return cfg.emaDecay < warm ? cfg.emaDecay : warm;
}

static void gan_arena_restore_(?&TensorArena outer) {
    match (outer) {
        case some(a): { a.begin(); }
        case null: { }
    }
}

// ── Gradient penalty (GAN_WGAN_GP) ───────────────────────────────────────────
// With g_i = grad_x D(x^_i) and P = w / m * sum_i (|g_i| - 1)^2,
//   dP/dtheta = w / m * sum_i 2 (|g_i| - 1) * d|g_i|/dtheta
//             = w / m * sum_i 2 (|g_i| - 1) * d/dtheta [u_i . grad_x D(x^_i)],  u_i = g_i / |g_i| held fixed
// and u . grad_x D(x) = (D(x + eps u) - D(x - eps u)) / (2 eps) + O(eps^2).
// So the term returned is sum_i c_i (D(x^_i + eps u_i) - D(x^_i - eps u_i)),
// c_i = w / m * 2 (|g_i| - 1) / (2 eps), whose first-order gradient is the
// penalty's.

#define GAN_GP_EPS 0.01

generic<T>
static &FloatTensor gan_gp_core_(&T g, fn &FloatTensor(&T, const &FloatTensor, ?&FloatTensor) disc, const &FloatTensor xh,
                            ?&FloatTensor labels, float weight, &float value) {
    unsafe { tensor_host_raw_((void*)xh); }
    unsigned long m;
    unsigned long d;
    unsafe { m = xh->shape[0]; d = xh->size / m; }
    &FloatTensor xg = xh.detach();
    unsafe { xg->requiresGrad = 1; }
    disc(g, xg, labels).sum().backward();
    &FloatTensor xp = xh.detach();
    unsafe { tensor_host_write_raw_((void*)xp); }
    &FloatTensor xm = xh.detach();
    unsafe { tensor_host_write_raw_((void*)xm); }
    &FloatTensor coef = FloatTensor::new_2d(m, 1UL, 0);
    double pen = 0.0;
    float eps = (float)GAN_GP_EPS;
    unsafe {
        unsigned long r = 0UL;
        while (r < m) {
            const float* gr = (const float*)&xg->grad[r * d];
            double sq = 0.0;
            unsigned long c = 0UL;
            while (c < d) { sq = sq + (double)gr[c] * (double)gr[c]; c = c + 1UL; }
            double norm = sqrt_d(sq);
            pen = pen + (norm - 1.0) * (norm - 1.0);
            float inv = norm > 1e-12 ? (float)(1.0 / norm) : (float)0.0;
            c = 0UL;
            while (c < d) {
                float step = eps * gr[c] * inv;
                xp->data[r * d + c] = xp->data[r * d + c] + step;
                xm->data[r * d + c] = xm->data[r * d + c] - step;
                c = c + 1UL;
            }
            coef->data[r] = (float)((double)weight / (double)m * 2.0 * (norm - 1.0) / (2.0 * (double)eps));
            r = r + 1UL;
        }
    }
    *value = (float)((double)weight * pen / (double)m);
    &FloatTensor diff = tensor_sub(disc(g, xp, labels), disc(g, xm, labels));
    return tensor_mul(diff, coef).sum();
}

// Rows t*real + (1 - t)*fake, t ~ U(0, 1) per row.
static &FloatTensor gan_interpolates_(&Rng rng, const &FloatTensor real, const &FloatTensor fake) {
    unsafe { tensor_host_raw_((void*)real); tensor_host_raw_((void*)fake); }
    &FloatTensor xh = real.detach();
    unsafe { tensor_host_write_raw_((void*)xh); }
    unsafe {
        unsigned long m = real->shape[0];
        unsigned long d = real->size / m;
        unsigned long r = 0UL;
        while (r < m) {
            float t = rng.uniform();
            unsigned long c = 0UL;
            while (c < d) {
                unsigned long at = r * d + c;
                xh->data[at] = t * real->data[at] + ((float)1.0 - t) * fake->data[at];
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
    return xh;
}

static &FloatTensor gan_disc_mlp_(&Gan g, const &FloatTensor x, ?&FloatTensor labels) {
    return g.disc.forward(gan_cond_(g, x, labels));
}

&FloatTensor gan_gradient_penalty_(&Gan g, const &FloatTensor xh, ?&FloatTensor labels, &float value) {
    return gan_gp_core_(g, gan_disc_mlp_, xh, labels, g.cfg.gpWeight, value);
}

struct GanLosses Gan::train_step(const &FloatTensor real, ?&FloatTensor labels) {
    struct GanLosses out;
    out.d = (float)0.0;
    out.g = (float)0.0;
    unsigned long m;
    unsafe { m = real->shape[0]; }
    ?&TensorArena outer = tensor_arena_current_();
    self.stepArena.begin();
    unsigned long c = 0UL;
    unsigned long nc = self.cfg.nCritic == 0UL ? 1UL : self.cfg.nCritic;
    while (c < nc) {
        &FloatTensor fake = gan_fake_(self, m, labels, 0).detach();
        &FloatTensor dReal = self.disc.forward(gan_cond_(self, real, labels));
        &FloatTensor dFake = self.disc.forward(gan_cond_(self, fake, labels));
        &FloatTensor dl = gan_d_loss_(self.cfg.loss, dReal, dFake, m);
        out.d = gan_value_(dl);
        if (self.cfg.loss == GAN_WGAN_GP) {
            float gp = (float)0.0;
            &FloatTensor term = gan_gp_core_(self, gan_disc_mlp_, gan_interpolates_(&self.rng, real, fake), labels,
                                        self.cfg.gpWeight, &gp);
            dl = tensor_add(dl, term);
            out.d = out.d + gp;
        }
        self.optD.zero_grad();
        dl.backward();
        self.optD.step();
        if (self.cfg.loss == GAN_WGAN) gan_clip_(&self.disc.params, self.cfg.clip);
        self.stepArena.reset();
        c = c + 1UL;
    }
    &FloatTensor fake2 = gan_fake_(self, m, labels, 0);
    &FloatTensor dFake2 = self.disc.forward(gan_cond_(self, fake2, labels));
    &FloatTensor gl = gan_g_loss_(self.cfg.loss, dFake2, m);
    self.optG.zero_grad();
    gl.backward();
    self.optG.step();
    out.g = gan_value_(gl);
    if (self.cfg.emaDecay > (float)0.0) {
        model_soft_update(&self.genEma, &self.gen, (float)1.0 - gan_ema_decay_(&self.cfg, self.steps));
    } else {
        model_copy(&self.genEma, &self.gen);
    }
    self.stepArena.reset();
    tensor_arena_end();
    gan_arena_restore_(outer);
    self.steps = self.steps + 1UL;
    return out;
}

struct GanLosses Gan::train(const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    struct GanLosses last;
    last.d = (float)0.0;
    last.g = (float)0.0;
    struct DataLoader dl = DataLoader::new(data, batchSize, 1, self.cfg.seed + 101UL);
    dl.to(model_device(self));
    struct TensorArena batchArena = TensorArena::new(0UL);
    struct Batch b;
    unsigned long e = 0UL;
    while (e < epochs) {
        batchArena.begin();
        while (dl.next(&b)) {
            tensor_arena_end();
            if (self.cfg.classes > 0UL) last = self.train_step(b.x, (?&FloatTensor)b.y);
            else last = self.train_step(b.x, null);
            batchArena.reset();
            batchArena.begin();
        }
        tensor_arena_end();
        batchArena.reset();
        e = e + 1UL;
    }
    batchArena.free();
    dl.free();
    return last;
}

&FloatTensor Gan::sample(unsigned long n, ?&FloatTensor labels) {
    ?&TensorArena outer = tensor_arena_current_();
    struct GradGuard noGrad = Autograd::get().no_grad();
    self.stepArena.begin();
    &FloatTensor fake = gan_fake_(self, n, labels, 1);
    tensor_arena_end();
    &FloatTensor out = fake.detach();      // no arena active: on the heap
    self.stepArena.reset();
    gan_arena_restore_(outer);
    noGrad.restore();
    return out;
}

&FloatTensor Gan::score(const &FloatTensor x, ?&FloatTensor labels) {
    return self.disc.forward(gan_cond_(self, x, labels));
}

void Gan::free() { self.drop(); }

void Gan::drop() {
    self.optG.free();
    self.optD.free();
    self.gen.free();
    self.genEma.free();
    self.disc.free();
    self.stepArena.free();
}

// ══ DCGAN ════════════════════════════════════════════════════════════════════

struct Dcgan Dcgan::new(struct ConvNet gen, struct ConvNet disc, const &GanConfig cfg) {
    struct Dcgan g;
    g.cfg = *cfg;
    g.rng = Rng::new(cfg.seed);
    g.gen = gen;
    g.disc = disc;
    unsigned long c = cfg.classes;
    if (g.disc.inC <= c) panic_at("Dcgan::new: D needs the image's channels plus one per class", "gan.sc", __LINE__);
    unsigned long dataDim = (g.disc.inC - c) * g.disc.inH * g.disc.inW;
    if (g.gen.in_dim() != cfg.latent + c || g.gen.out_dim() != dataDim
        || g.disc.out_dim() != 1UL)
        panic_at("Dcgan::new: G must map latent (+ classes) to the image's size, D the image to one score", "gan.sc", __LINE__);
    g.genEma = g.gen.clone();
    g.optG = gan_opt_(cfg.loss, cfg.lrG);
    g.optD = gan_opt_(cfg.loss, cfg.lrD);
    model_add_params(&g.gen, &g.optG);
    model_add_params(&g.disc, &g.optD);
    g.stepArena = TensorArena::new(0UL);
    g.steps = 0UL;
    return g;
}

// Labels for a conditional DCGAN: the given ones, or class 0.
static &FloatTensor dcgan_labels_(unsigned long m, ?&FloatTensor labels) {
    match (labels) {
        case some(l): { return l; }
        case null: { }
    }
    return FloatTensor::new_1d(m, 0);
}

static &FloatTensor dcgan_disc_(&Dcgan g, const &FloatTensor x, ?&FloatTensor labels) {
    unsafe { tensor_host_raw_((void*)x); }
    if (g.cfg.classes == 0UL) return g.disc.forward(x);
    unsigned long m;
    unsafe { m = x->ndim >= 2UL ? x->shape[0] : 1UL; }
    &FloatTensor in = tensor_append_label_channels(x, dcgan_labels_(m, labels), g.cfg.classes, g.disc.inH, g.disc.inW);
    return g.disc.forward(in);
}

static &FloatTensor dcgan_fake_(&Dcgan g, unsigned long m, ?&FloatTensor labels, int averaged) {
    &FloatTensor z = g.rng.normal_tensor(m, g.cfg.latent);
    if (g.cfg.classes > 0UL) z = tensor_concat_cols(z, dcgan_labels_(m, labels).one_hot(g.cfg.classes));
    if (averaged) return g.genEma.forward(z);
    return g.gen.forward(z);
}

struct GanLosses Dcgan::train_step(const &FloatTensor real, ?&FloatTensor labels) {
    struct GanLosses out;
    out.d = (float)0.0;
    out.g = (float)0.0;
    unsigned long m;
    unsafe { m = real->ndim >= 2UL ? real->shape[0] : 1UL; }
    ?&TensorArena outer = tensor_arena_current_();
    self.stepArena.begin();
    unsigned long c = 0UL;
    unsigned long nc = self.cfg.nCritic == 0UL ? 1UL : self.cfg.nCritic;
    while (c < nc) {
        &FloatTensor fake = dcgan_fake_(self, m, labels, 0).detach();
        &FloatTensor dReal = dcgan_disc_(self, real, labels);
        &FloatTensor dFake = dcgan_disc_(self, fake, labels);
        &FloatTensor dl = gan_d_loss_(self.cfg.loss, dReal, dFake, m);
        out.d = gan_value_(dl);
        if (self.cfg.loss == GAN_WGAN_GP) {
            float gp = (float)0.0;
            &FloatTensor term = gan_gp_core_(self, dcgan_disc_, gan_interpolates_(&self.rng, real, fake), labels,
                                        self.cfg.gpWeight, &gp);
            dl = tensor_add(dl, term);
            out.d = out.d + gp;
        }
        self.optD.zero_grad();
        dl.backward();
        self.optD.step();
        if (self.cfg.loss == GAN_WGAN) gan_clip_(&self.disc.params, self.cfg.clip);
        self.stepArena.reset();
        c = c + 1UL;
    }
    &FloatTensor gl = gan_g_loss_(self.cfg.loss, dcgan_disc_(self, dcgan_fake_(self, m, labels, 0), labels), m);
    self.optG.zero_grad();
    gl.backward();
    self.optG.step();
    out.g = gan_value_(gl);
    if (self.cfg.emaDecay > (float)0.0) model_soft_update(&self.genEma, &self.gen, (float)1.0 - gan_ema_decay_(&self.cfg, self.steps));
    else model_copy(&self.genEma, &self.gen);
    self.stepArena.reset();
    tensor_arena_end();
    gan_arena_restore_(outer);
    self.steps = self.steps + 1UL;
    return out;
}

struct GanLosses Dcgan::train(const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    struct GanLosses last;
    last.d = (float)0.0;
    last.g = (float)0.0;
    struct DataLoader dl = DataLoader::new(data, batchSize, 1, self.cfg.seed + 101UL);
    dl.to(model_device(self));
    struct TensorArena batchArena = TensorArena::new(0UL);
    struct Batch b;
    unsigned long e = 0UL;
    while (e < epochs) {
        batchArena.begin();
        while (dl.next(&b)) {
            tensor_arena_end();
            if (self.cfg.classes > 0UL) last = self.train_step(b.x, (?&FloatTensor)b.y);
            else last = self.train_step(b.x, null);
            batchArena.reset();
            batchArena.begin();
        }
        tensor_arena_end();
        batchArena.reset();
        e = e + 1UL;
    }
    batchArena.free();
    dl.free();
    return last;
}

&FloatTensor Dcgan::sample(unsigned long n, ?&FloatTensor labels) {
    ?&TensorArena outer = tensor_arena_current_();
    struct GradGuard noGrad = Autograd::get().no_grad();
    int mode = self.genEma.set_training(0);
    self.stepArena.begin();
    &FloatTensor fake = dcgan_fake_(self, n, labels, 1);
    tensor_arena_end();
    &FloatTensor out = fake.detach();      // no arena active: on the heap
    self.stepArena.reset();
    self.genEma.set_training(mode);
    gan_arena_restore_(outer);
    noGrad.restore();
    return out;
}

&FloatTensor Dcgan::score(const &FloatTensor x, ?&FloatTensor labels) {
    int mode = self.disc.set_training(0);
    &FloatTensor s = dcgan_disc_(self, x, labels);
    self.disc.set_training(mode);
    return s;
}

&FloatTensor dcgan_gradient_penalty_(&Dcgan g, const &FloatTensor xh, ?&FloatTensor labels, &float value) {
    return gan_gp_core_(g, dcgan_disc_, xh, labels, g.cfg.gpWeight, value);
}

void Dcgan::free() { self.drop(); }

void Dcgan::drop() {
    self.optG.free();
    self.optD.free();
    self.gen.free();
    self.genEma.free();
    self.disc.free();
    self.stepArena.free();
}

void Gan::collect_params(&Vec<&FloatTensor> out) const {
    self.gen.collect_params(out);
    self.disc.collect_params(out);
}

void Gan::collect_state(&Vec<&FloatTensor> out) const {
    self.gen.collect_state(out);
    self.genEma.collect_state(out);
    self.disc.collect_state(out);
}

void Dcgan::collect_params(&Vec<&FloatTensor> out) const {
    self.gen.collect_params(out);
    self.disc.collect_params(out);
}

void Dcgan::collect_state(&Vec<&FloatTensor> out) const {
    self.gen.collect_state(out);
    self.genEma.collect_state(out);
    self.disc.collect_state(out);
}

} // namespace std
