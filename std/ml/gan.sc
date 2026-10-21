// SafeC Standard Library — generative adversarial networks (see gan.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/gan.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/ml/convnet.h>
#include <std/panic.h>
#include <std/ml/datasets.h>

namespace std {

struct GanConfig gan_config_default(unsigned long latent) {
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
    if (loss == GAN_WGAN) return optimizer_new(OPT_RMSPROP, lr);
    struct Optimizer o = optimizer_new(OPT_ADAM, lr);
    o.beta1 = (float)0.5;
    if (loss == GAN_WGAN_GP) { o.beta1 = (float)0.0; o.beta2 = (float)0.9; }
    return o;
}

struct Gan gan_new(unsigned long dataDim, const &GanConfig cfg) {
    struct Gan g;
    g.cfg = *cfg;
    g.rng = rng_new(cfg.seed);
    unsigned long c = cfg.classes;
    g.gen = mlp_new(cfg.latent + c, cfg.hidden, cfg.hiddenLayers, dataDim, cfg.genAct, cfg.outAct, &g.rng);
    g.genEma = mlp_new(cfg.latent + c, cfg.hidden, cfg.hiddenLayers, dataDim, cfg.genAct, cfg.outAct, &g.rng);
    mlp_copy(&g.genEma, &g.gen);
    g.disc = mlp_new(dataDim + c, cfg.hidden, cfg.hiddenLayers, 1UL, ACT_LEAKY, ACT_NONE, &g.rng);
    g.optG = gan_opt_(cfg.loss, cfg.lrG);
    g.optD = gan_opt_(cfg.loss, cfg.lrD);
    optimizer_add_mlp(&g.optG, &g.gen);
    optimizer_add_mlp(&g.optD, &g.disc);
    g.stepArena = tensor_arena_new(0UL);
    g.steps = 0UL;
    return g;
}

static &Tensor gan_cond_(const &Gan g, const &Tensor x, ?&Tensor labels) {
    if (g.cfg.classes > 0UL) {
        match (labels) {
            case some(l): { return tensor_concat_cols(x, tensor_one_hot(l, g.cfg.classes)); }
            case null: { }
        }
        unsigned long m;
        unsafe { m = x->shape[0]; }
        return tensor_concat_cols(x, tensor_one_hot(tensor_new_1d(m, 0), g.cfg.classes));
    }
    return tensor_add_scalar(x, (float)0.0);
}

static &Tensor gan_fake_(&Gan g, unsigned long m, ?&Tensor labels, int averaged) {
    &Tensor z = rng_normal_tensor(&g.rng, m, g.cfg.latent);
    if (averaged) return mlp_forward(&g.genEma, gan_cond_(g, z, labels));
    return mlp_forward(&g.gen, gan_cond_(g, z, labels));
}

static &Tensor gan_full_(unsigned long m, float v) {
    &Tensor t = tensor_new_2d(m, 1UL, 0);
    return tensor_fill(t, v);
}

static &Tensor gan_d_loss_(int k, &Tensor dReal, &Tensor dFake, unsigned long m) {
    if (k == GAN_LSGAN) {
        &Tensor a = tensor_mse_loss(dReal, gan_full_(m, (float)1.0));
        &Tensor b = tensor_mse_loss(dFake, gan_full_(m, (float)0.0));
        return tensor_scale(tensor_add(a, b), (float)0.5);
    }
    if (k == GAN_WGAN || k == GAN_WGAN_GP) return tensor_sub(tensor_mean(dFake), tensor_mean(dReal));
    if (k == GAN_HINGE) {
        &Tensor a = tensor_mean(tensor_relu(tensor_add_scalar(tensor_scale(dReal, (float)-1.0), (float)1.0)));
        &Tensor b = tensor_mean(tensor_relu(tensor_add_scalar(dFake, (float)1.0)));
        return tensor_add(a, b);
    }
    &Tensor a = tensor_bce_logits_loss(dReal, gan_full_(m, (float)1.0));
    &Tensor b = tensor_bce_logits_loss(dFake, gan_full_(m, (float)0.0));
    return tensor_add(a, b);
}

static &Tensor gan_g_loss_(int k, &Tensor dFake, unsigned long m) {
    if (k == GAN_LSGAN) return tensor_scale(tensor_mse_loss(dFake, gan_full_(m, (float)1.0)), (float)0.5);
    if (k == GAN_WGAN || k == GAN_WGAN_GP || k == GAN_HINGE) return tensor_scale(tensor_mean(dFake), (float)-1.0);
    return tensor_bce_logits_loss(dFake, gan_full_(m, (float)1.0));
}

static void gan_clip_(const &Vec<&Tensor> params, float c) {
    unsigned long i = 0UL;
    while (i < params.length()) {
        &Tensor p = *params.at(i);
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

static float gan_value_(const &Tensor t) {
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
        case some(a): { tensor_arena_begin(a); }
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
static &Tensor gan_gp_core_(&T g, fn &Tensor(&T, const &Tensor, ?&Tensor) disc, const &Tensor xh,
                            ?&Tensor labels, float weight, &float value) {
    unsigned long m;
    unsigned long d;
    unsafe { m = xh->shape[0]; d = xh->size / m; }
    &Tensor xg = tensor_detach(xh);
    unsafe { xg->requiresGrad = 1; }
    tensor_backward(tensor_sum(disc(g, xg, labels)));
    &Tensor xp = tensor_detach(xh);
    &Tensor xm = tensor_detach(xh);
    &Tensor coef = tensor_new_2d(m, 1UL, 0);
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
    &Tensor diff = tensor_sub(disc(g, xp, labels), disc(g, xm, labels));
    return tensor_sum(tensor_mul(diff, coef));
}

// Rows t*real + (1 - t)*fake, t ~ U(0, 1) per row.
static &Tensor gan_interpolates_(&Rng rng, const &Tensor real, const &Tensor fake) {
    &Tensor xh = tensor_detach(real);
    unsafe {
        unsigned long m = real->shape[0];
        unsigned long d = real->size / m;
        unsigned long r = 0UL;
        while (r < m) {
            float t = rng_uniform(rng);
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

static &Tensor gan_disc_mlp_(&Gan g, const &Tensor x, ?&Tensor labels) {
    return mlp_forward(&g.disc, gan_cond_(g, x, labels));
}

&Tensor gan_gradient_penalty_(&Gan g, const &Tensor xh, ?&Tensor labels, &float value) {
    return gan_gp_core_(g, gan_disc_mlp_, xh, labels, g.cfg.gpWeight, value);
}

struct GanLosses gan_train_step(&Gan g, const &Tensor real, ?&Tensor labels) {
    struct GanLosses out;
    out.d = (float)0.0;
    out.g = (float)0.0;
    unsigned long m;
    unsafe { m = real->shape[0]; }
    ?&TensorArena outer = tensor_arena_current_();
    tensor_arena_begin(&g.stepArena);
    unsigned long c = 0UL;
    unsigned long nc = g.cfg.nCritic == 0UL ? 1UL : g.cfg.nCritic;
    while (c < nc) {
        &Tensor fake = tensor_detach(gan_fake_(g, m, labels, 0));
        &Tensor dReal = mlp_forward(&g.disc, gan_cond_(g, real, labels));
        &Tensor dFake = mlp_forward(&g.disc, gan_cond_(g, fake, labels));
        &Tensor dl = gan_d_loss_(g.cfg.loss, dReal, dFake, m);
        out.d = gan_value_(dl);
        if (g.cfg.loss == GAN_WGAN_GP) {
            float gp = (float)0.0;
            &Tensor term = gan_gp_core_(g, gan_disc_mlp_, gan_interpolates_(&g.rng, real, fake), labels,
                                        g.cfg.gpWeight, &gp);
            dl = tensor_add(dl, term);
            out.d = out.d + gp;
        }
        optimizer_zero_grad(&g.optD);
        tensor_backward(dl);
        optimizer_step(&g.optD);
        if (g.cfg.loss == GAN_WGAN) gan_clip_(&g.disc.params, g.cfg.clip);
        tensor_arena_reset(&g.stepArena);
        c = c + 1UL;
    }
    &Tensor fake2 = gan_fake_(g, m, labels, 0);
    &Tensor dFake2 = mlp_forward(&g.disc, gan_cond_(g, fake2, labels));
    &Tensor gl = gan_g_loss_(g.cfg.loss, dFake2, m);
    optimizer_zero_grad(&g.optG);
    tensor_backward(gl);
    optimizer_step(&g.optG);
    out.g = gan_value_(gl);
    if (g.cfg.emaDecay > (float)0.0) {
        mlp_soft_update(&g.genEma, &g.gen, (float)1.0 - gan_ema_decay_(&g.cfg, g.steps));
    } else {
        mlp_copy(&g.genEma, &g.gen);
    }
    tensor_arena_reset(&g.stepArena);
    tensor_arena_end();
    gan_arena_restore_(outer);
    g.steps = g.steps + 1UL;
    return out;
}

struct GanLosses gan_train(&Gan g, const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    struct GanLosses last;
    last.d = (float)0.0;
    last.g = (float)0.0;
    struct DataLoader dl = dataloader_new(data, batchSize, 1, g.cfg.seed + 101UL);
    struct TensorArena batchArena = tensor_arena_new(0UL);
    struct Batch b;
    unsigned long e = 0UL;
    while (e < epochs) {
        tensor_arena_begin(&batchArena);
        while (dataloader_next(&dl, &b)) {
            tensor_arena_end();
            if (g.cfg.classes > 0UL) last = gan_train_step(g, b.x, (?&Tensor)b.y);
            else last = gan_train_step(g, b.x, null);
            tensor_arena_reset(&batchArena);
            tensor_arena_begin(&batchArena);
        }
        tensor_arena_end();
        tensor_arena_reset(&batchArena);
        e = e + 1UL;
    }
    tensor_arena_free(&batchArena);
    dataloader_free(&dl);
    return last;
}

&Tensor gan_generate(&Gan g, unsigned long n, ?&Tensor labels) {
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    tensor_arena_begin(&g.stepArena);
    &Tensor fake = gan_fake_(g, n, labels, 1);
    tensor_arena_end();
    &Tensor out = tensor_detach(fake);      // no arena active: on the heap
    tensor_arena_reset(&g.stepArena);
    gan_arena_restore_(outer);
    tensor_set_grad_enabled(was);
    return out;
}

&Tensor gan_score(&Gan g, const &Tensor x, ?&Tensor labels) {
    return mlp_forward(&g.disc, gan_cond_(g, x, labels));
}

void gan_free(&Gan g) {
    optimizer_free(&g.optG);
    optimizer_free(&g.optD);
    mlp_free(&g.gen);
    mlp_free(&g.genEma);
    mlp_free(&g.disc);
    tensor_arena_free(&g.stepArena);
}

// ══ DCGAN ════════════════════════════════════════════════════════════════════

struct Dcgan dcgan_new(struct ConvNet gen, struct ConvNet disc, const &GanConfig cfg) {
    struct Dcgan g;
    g.cfg = *cfg;
    g.rng = rng_new(cfg.seed);
    g.gen = gen;
    g.disc = disc;
    unsigned long c = cfg.classes;
    if (g.disc.inC <= c) panic_at("dcgan_new: D needs the image's channels plus one per class", "gan.sc", __LINE__);
    unsigned long dataDim = (g.disc.inC - c) * g.disc.inH * g.disc.inW;
    if (convnet_in_dim(&g.gen) != cfg.latent + c || convnet_out_dim(&g.gen) != dataDim
        || convnet_out_dim(&g.disc) != 1UL)
        panic_at("dcgan_new: G must map latent (+ classes) to the image's size, D the image to one score", "gan.sc", __LINE__);
    g.genEma = convnet_clone(&g.gen);
    g.optG = gan_opt_(cfg.loss, cfg.lrG);
    g.optD = gan_opt_(cfg.loss, cfg.lrD);
    optimizer_add_convnet(&g.optG, &g.gen);
    optimizer_add_convnet(&g.optD, &g.disc);
    g.stepArena = tensor_arena_new(0UL);
    g.steps = 0UL;
    return g;
}

// Labels for a conditional DCGAN: the given ones, or class 0.
static &Tensor dcgan_labels_(unsigned long m, ?&Tensor labels) {
    match (labels) {
        case some(l): { return l; }
        case null: { }
    }
    return tensor_new_1d(m, 0);
}

static &Tensor dcgan_disc_(&Dcgan g, const &Tensor x, ?&Tensor labels) {
    if (g.cfg.classes == 0UL) return convnet_forward(&g.disc, x);
    unsigned long m;
    unsafe { m = x->ndim >= 2UL ? x->shape[0] : 1UL; }
    &Tensor in = tensor_append_label_channels(x, dcgan_labels_(m, labels), g.cfg.classes, g.disc.inH, g.disc.inW);
    return convnet_forward(&g.disc, in);
}

static &Tensor dcgan_fake_(&Dcgan g, unsigned long m, ?&Tensor labels, int averaged) {
    &Tensor z = rng_normal_tensor(&g.rng, m, g.cfg.latent);
    if (g.cfg.classes > 0UL) z = tensor_concat_cols(z, tensor_one_hot(dcgan_labels_(m, labels), g.cfg.classes));
    if (averaged) return convnet_forward(&g.genEma, z);
    return convnet_forward(&g.gen, z);
}

struct GanLosses dcgan_train_step(&Dcgan g, const &Tensor real, ?&Tensor labels) {
    struct GanLosses out;
    out.d = (float)0.0;
    out.g = (float)0.0;
    unsigned long m;
    unsafe { m = real->ndim >= 2UL ? real->shape[0] : 1UL; }
    ?&TensorArena outer = tensor_arena_current_();
    tensor_arena_begin(&g.stepArena);
    unsigned long c = 0UL;
    unsigned long nc = g.cfg.nCritic == 0UL ? 1UL : g.cfg.nCritic;
    while (c < nc) {
        &Tensor fake = tensor_detach(dcgan_fake_(g, m, labels, 0));
        &Tensor dReal = dcgan_disc_(g, real, labels);
        &Tensor dFake = dcgan_disc_(g, fake, labels);
        &Tensor dl = gan_d_loss_(g.cfg.loss, dReal, dFake, m);
        out.d = gan_value_(dl);
        if (g.cfg.loss == GAN_WGAN_GP) {
            float gp = (float)0.0;
            &Tensor term = gan_gp_core_(g, dcgan_disc_, gan_interpolates_(&g.rng, real, fake), labels,
                                        g.cfg.gpWeight, &gp);
            dl = tensor_add(dl, term);
            out.d = out.d + gp;
        }
        optimizer_zero_grad(&g.optD);
        tensor_backward(dl);
        optimizer_step(&g.optD);
        if (g.cfg.loss == GAN_WGAN) gan_clip_(&g.disc.params, g.cfg.clip);
        tensor_arena_reset(&g.stepArena);
        c = c + 1UL;
    }
    &Tensor gl = gan_g_loss_(g.cfg.loss, dcgan_disc_(g, dcgan_fake_(g, m, labels, 0), labels), m);
    optimizer_zero_grad(&g.optG);
    tensor_backward(gl);
    optimizer_step(&g.optG);
    out.g = gan_value_(gl);
    if (g.cfg.emaDecay > (float)0.0) convnet_soft_update(&g.genEma, &g.gen, (float)1.0 - gan_ema_decay_(&g.cfg, g.steps));
    else convnet_copy(&g.genEma, &g.gen);
    tensor_arena_reset(&g.stepArena);
    tensor_arena_end();
    gan_arena_restore_(outer);
    g.steps = g.steps + 1UL;
    return out;
}

struct GanLosses dcgan_train(&Dcgan g, const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    struct GanLosses last;
    last.d = (float)0.0;
    last.g = (float)0.0;
    struct DataLoader dl = dataloader_new(data, batchSize, 1, g.cfg.seed + 101UL);
    struct TensorArena batchArena = tensor_arena_new(0UL);
    struct Batch b;
    unsigned long e = 0UL;
    while (e < epochs) {
        tensor_arena_begin(&batchArena);
        while (dataloader_next(&dl, &b)) {
            tensor_arena_end();
            if (g.cfg.classes > 0UL) last = dcgan_train_step(g, b.x, (?&Tensor)b.y);
            else last = dcgan_train_step(g, b.x, null);
            tensor_arena_reset(&batchArena);
            tensor_arena_begin(&batchArena);
        }
        tensor_arena_end();
        tensor_arena_reset(&batchArena);
        e = e + 1UL;
    }
    tensor_arena_free(&batchArena);
    dataloader_free(&dl);
    return last;
}

&Tensor dcgan_generate(&Dcgan g, unsigned long n, ?&Tensor labels) {
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    int mode = convnet_set_training(&g.genEma, 0);
    tensor_arena_begin(&g.stepArena);
    &Tensor fake = dcgan_fake_(g, n, labels, 1);
    tensor_arena_end();
    &Tensor out = tensor_detach(fake);      // no arena active: on the heap
    tensor_arena_reset(&g.stepArena);
    convnet_set_training(&g.genEma, mode);
    gan_arena_restore_(outer);
    tensor_set_grad_enabled(was);
    return out;
}

&Tensor dcgan_score(&Dcgan g, const &Tensor x, ?&Tensor labels) {
    int mode = convnet_set_training(&g.disc, 0);
    &Tensor s = dcgan_disc_(g, x, labels);
    convnet_set_training(&g.disc, mode);
    return s;
}

&Tensor dcgan_gradient_penalty_(&Dcgan g, const &Tensor xh, ?&Tensor labels, &float value) {
    return gan_gp_core_(g, dcgan_disc_, xh, labels, g.cfg.gpWeight, value);
}

void dcgan_free(&Dcgan g) {
    optimizer_free(&g.optG);
    optimizer_free(&g.optD);
    convnet_free(&g.gen);
    convnet_free(&g.genEma);
    convnet_free(&g.disc);
    tensor_arena_free(&g.stepArena);
}

} // namespace std
