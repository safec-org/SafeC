// SafeC Standard Library — generative adversarial networks (see gan.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/gan.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
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
    c.classes = 0UL;
    c.emaDecay = (float)0.999;
    c.seed = 1UL;
    return c;
}

static struct Optimizer gan_opt_(int loss, float lr) {
    if (loss == GAN_WGAN) return optimizer_new(OPT_RMSPROP, lr);
    struct Optimizer o = optimizer_new(OPT_ADAM, lr);
    o.beta1 = (float)0.5;
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

static &Tensor gan_d_loss_(const &Gan g, &Tensor dReal, &Tensor dFake, unsigned long m) {
    int k = g.cfg.loss;
    if (k == GAN_LSGAN) {
        &Tensor a = tensor_mse_loss(dReal, gan_full_(m, (float)1.0));
        &Tensor b = tensor_mse_loss(dFake, gan_full_(m, (float)0.0));
        return tensor_scale(tensor_add(a, b), (float)0.5);
    }
    if (k == GAN_WGAN) return tensor_sub(tensor_mean(dFake), tensor_mean(dReal));
    if (k == GAN_HINGE) {
        &Tensor a = tensor_mean(tensor_relu(tensor_add_scalar(tensor_scale(dReal, (float)-1.0), (float)1.0)));
        &Tensor b = tensor_mean(tensor_relu(tensor_add_scalar(dFake, (float)1.0)));
        return tensor_add(a, b);
    }
    &Tensor a = tensor_bce_logits_loss(dReal, gan_full_(m, (float)1.0));
    &Tensor b = tensor_bce_logits_loss(dFake, gan_full_(m, (float)0.0));
    return tensor_add(a, b);
}

static &Tensor gan_g_loss_(const &Gan g, &Tensor dFake, unsigned long m) {
    int k = g.cfg.loss;
    if (k == GAN_LSGAN) return tensor_scale(tensor_mse_loss(dFake, gan_full_(m, (float)1.0)), (float)0.5);
    if (k == GAN_WGAN || k == GAN_HINGE) return tensor_scale(tensor_mean(dFake), (float)-1.0);
    return tensor_bce_logits_loss(dFake, gan_full_(m, (float)1.0));
}

static void gan_clip_(&Gan g) {
    float c = g.cfg.clip;
    unsigned long i = 0UL;
    while (i < g.disc.params.length()) {
        &Tensor p = *g.disc.params.at(i);
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
        &Tensor dl = gan_d_loss_(g, dReal, dFake, m);
        optimizer_zero_grad(&g.optD);
        tensor_backward(dl);
        optimizer_step(&g.optD);
        if (g.cfg.loss == GAN_WGAN) gan_clip_(g);
        out.d = gan_value_(dl);
        tensor_arena_reset(&g.stepArena);
        c = c + 1UL;
    }
    &Tensor fake2 = gan_fake_(g, m, labels, 0);
    &Tensor dFake2 = mlp_forward(&g.disc, gan_cond_(g, fake2, labels));
    &Tensor gl = gan_g_loss_(g, dFake2, m);
    optimizer_zero_grad(&g.optG);
    tensor_backward(gl);
    optimizer_step(&g.optG);
    out.g = gan_value_(gl);
    if (g.cfg.emaDecay > (float)0.0) {
        // warm-up: the average follows closely over the first steps
        float t = (float)g.steps;
        float warm = ((float)1.0 + t) / ((float)10.0 + t);
        float decay = g.cfg.emaDecay < warm ? g.cfg.emaDecay : warm;
        mlp_soft_update(&g.genEma, &g.gen, (float)1.0 - decay);
    } else {
        mlp_copy(&g.genEma, &g.gen);
    }
    tensor_arena_reset(&g.stepArena);
    tensor_arena_end();
    match (outer) {
        case some(a): { tensor_arena_begin(a); }
        case null: { }
    }
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
    match (outer) {
        case some(a): { tensor_arena_begin(a); }
        case null: { }
    }
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

} // namespace std
