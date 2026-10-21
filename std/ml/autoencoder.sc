// SafeC Standard Library — autoencoders (see autoencoder.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/autoencoder.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/math.h>

namespace std {

static float ae_cols_(const &Tensor t) {
    unsafe { return (float)t->shape[t->ndim - 1UL]; }
}

static unsigned long ae_rows_(const &Tensor t) {
    unsafe { return t->ndim >= 2UL ? t->shape[0] : 1UL; }
}

// Reconstruction loss averaged over elements.
static &Tensor ae_recon_mean_(const &Tensor out, const &Tensor x, int recon) {
    if (recon == RECON_BCE) return tensor_bce_logits_loss(out, x);
    return tensor_mse_loss(out, x);
}

// The decoder's output as data: sigmoid of the logits for BCE.
static &Tensor ae_output_(&Tensor out, int recon) {
    if (recon == RECON_BCE) return tensor_sigmoid(out);
    return out;
}

static float ae_value_(const &Tensor t) {
    unsafe { return t->data[0]; }
}

// ══ Autoencoder ══════════════════════════════════════════════════════════════

struct AutoencoderConfig autoencoder_config_default(unsigned long latent) {
    struct AutoencoderConfig c;
    c.hidden = 128UL;
    c.hiddenLayers = 1UL;
    c.latent = latent;
    c.act = ACT_RELU;
    c.latentAct = ACT_NONE;
    c.recon = RECON_MSE;
    c.noiseStd = (float)0.0;
    c.dropProb = (float)0.0;
    c.sparsity = (float)0.0;
    c.seed = 1UL;
    return c;
}

struct Autoencoder autoencoder_new(unsigned long inDim, const &AutoencoderConfig cfg) {
    struct Autoencoder ae;
    ae.cfg = *cfg;
    ae.rng = rng_new(cfg.seed);
    ae.enc = mlp_new(inDim, cfg.hidden, cfg.hiddenLayers, cfg.latent, cfg.act, cfg.latentAct, &ae.rng);
    ae.dec = mlp_new(cfg.latent, cfg.hidden, cfg.hiddenLayers, inDim, cfg.act, ACT_NONE, &ae.rng);
    ae.lastRecon = (float)0.0;
    ae.lastPenalty = (float)0.0;
    return ae;
}

&Tensor autoencoder_encode(&Autoencoder ae, const &Tensor x) { return mlp_forward(&ae.enc, x); }

&Tensor autoencoder_decode(&Autoencoder ae, const &Tensor z) {
    return ae_output_(mlp_forward(&ae.dec, z), ae.cfg.recon);
}

&Tensor autoencoder_reconstruct(&Autoencoder ae, const &Tensor x) {
    return autoencoder_decode(ae, autoencoder_encode(ae, x));
}

&Tensor autoencoder_loss(&Autoencoder ae, const &Tensor x, const &Tensor y) {
    // the encoder's input: a copy of x, corrupted for a denoising AE
    &Tensor noisy = tensor_detach(x);
    int corrupt = ae.cfg.noiseStd > (float)0.0 || ae.cfg.dropProb > (float)0.0;
    if (corrupt) {
        unsafe {
            unsigned long i = 0UL;
            while (i < noisy->size) {
                float v = noisy->data[i] + ae.cfg.noiseStd * rng_normal(&ae.rng);
                if (ae.cfg.dropProb > (float)0.0 && rng_uniform(&ae.rng) < ae.cfg.dropProb) v = (float)0.0;
                noisy->data[i] = v;
                i = i + 1UL;
            }
        }
    }
    &Tensor z = mlp_forward(&ae.enc, noisy);
    &Tensor out = mlp_forward(&ae.dec, z);
    &Tensor loss = ae_recon_mean_(out, x, ae.cfg.recon);
    ae.lastRecon = ae_value_(loss);
    ae.lastPenalty = (float)0.0;
    if (ae.cfg.sparsity > (float)0.0) {
        &Tensor pen = tensor_scale(tensor_mean(tensor_abs(z)), ae.cfg.sparsity);
        ae.lastPenalty = ae_value_(pen);
        loss = tensor_add(loss, pen);
    }
    return loss;
}

void autoencoder_add_params(&Optimizer opt, const &Autoencoder ae) {
    optimizer_add_mlp(opt, &ae.enc);
    optimizer_add_mlp(opt, &ae.dec);
}

void autoencoder_free(&Autoencoder ae) {
    mlp_free(&ae.enc);
    mlp_free(&ae.dec);
}

// ══ VAE ══════════════════════════════════════════════════════════════════════

struct VaeConfig vae_config_default(unsigned long latent) {
    struct VaeConfig c;
    c.hidden = 256UL;
    c.hiddenLayers = 1UL;
    c.latent = latent;
    c.act = ACT_RELU;
    c.recon = RECON_BCE;
    c.beta = (float)1.0;
    c.classes = 0UL;
    c.seed = 1UL;
    return c;
}

struct Vae vae_new(unsigned long inDim, const &VaeConfig cfg) {
    struct Vae v;
    v.cfg = *cfg;
    if (v.cfg.hiddenLayers == 0UL) v.cfg.hiddenLayers = 1UL;
    v.rng = rng_new(cfg.seed);
    unsigned long c = cfg.classes;
    v.enc = mlp_new(inDim + c, cfg.hidden, v.cfg.hiddenLayers - 1UL, cfg.hidden, cfg.act, cfg.act, &v.rng);
    v.muHead = mlp_new(cfg.hidden, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.logvarHead = mlp_new(cfg.hidden, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.dec = mlp_new(cfg.latent + c, cfg.hidden, v.cfg.hiddenLayers, inDim, cfg.act, ACT_NONE, &v.rng);
    v.lastRecon = (float)0.0;
    v.lastKl = (float)0.0;
    return v;
}

// Appends the one-hot label for a CVAE.
static &Tensor vae_cond_(const &Vae v, const &Tensor x, const &Tensor y) {
    if (v.cfg.classes == 0UL) return tensor_add_scalar(x, (float)0.0);
    return tensor_concat_cols(x, tensor_one_hot(y, v.cfg.classes));
}

&Tensor vae_loss(&Vae v, const &Tensor x, const &Tensor y) {
    &Tensor h = mlp_forward(&v.enc, vae_cond_(v, x, y));
    &Tensor mu = mlp_forward(&v.muHead, h);
    &Tensor logvar = mlp_forward(&v.logvarHead, h);
    unsigned long m = ae_rows_(x);
    float rows = (float)m;
    // z = mu + exp(logvar / 2) * eps
    &Tensor eps = rng_normal_tensor(&v.rng, m, v.cfg.latent);
    &Tensor z = tensor_add(mu, tensor_mul(tensor_exp(tensor_scale(logvar, (float)0.5)), eps));
    &Tensor out = mlp_forward(&v.dec, vae_cond_(v, z, y));
    // reconstruction summed over features, averaged over the batch
    &Tensor recon = tensor_scale(ae_recon_mean_(out, x, v.cfg.recon), ae_cols_(x));
    // KL(N(mu, sigma^2) || N(0, 1)) = -1/2 sum(1 + logvar - mu^2 - exp(logvar))
    &Tensor inner = tensor_sub(tensor_add_scalar(logvar, (float)1.0),
                               tensor_add(tensor_mul(mu, mu), tensor_exp(logvar)));
    &Tensor kl = tensor_scale(tensor_sum(inner), (float)-0.5 / rows);
    v.lastRecon = ae_value_(recon);
    v.lastKl = ae_value_(kl);
    return tensor_add(recon, tensor_scale(kl, v.cfg.beta));
}

&Tensor vae_encode(&Vae v, const &Tensor x, const &Tensor y) {
    return mlp_forward(&v.muHead, mlp_forward(&v.enc, vae_cond_(v, x, y)));
}

&Tensor vae_decode(&Vae v, const &Tensor z, const &Tensor y) {
    return ae_output_(mlp_forward(&v.dec, vae_cond_(v, z, y)), v.cfg.recon);
}

&Tensor vae_reconstruct(&Vae v, const &Tensor x, const &Tensor y) {
    return vae_decode(v, vae_encode(v, x, y), y);
}

&Tensor vae_sample(&Vae v, unsigned long n, ?&Tensor labels) {
    &Tensor z = rng_normal_tensor(&v.rng, n, v.cfg.latent);
    match (labels) {
        case some(l): { return vae_decode(v, z, l); }
        case null: { }
    }
    &Tensor none = tensor_new_1d(n, 0);   // class 0 if a CVAE is sampled without labels
    return vae_decode(v, z, none);
}

void vae_add_params(&Optimizer opt, const &Vae v) {
    optimizer_add_mlp(opt, &v.enc);
    optimizer_add_mlp(opt, &v.muHead);
    optimizer_add_mlp(opt, &v.logvarHead);
    optimizer_add_mlp(opt, &v.dec);
}

void vae_free(&Vae v) {
    mlp_free(&v.enc); mlp_free(&v.muHead); mlp_free(&v.logvarHead); mlp_free(&v.dec);
}

// ══ VQ-VAE ═══════════════════════════════════════════════════════════════════

struct VqVaeConfig vqvae_config_default(unsigned long latent) {
    struct VqVaeConfig c;
    c.hidden = 128UL;
    c.hiddenLayers = 1UL;
    c.latent = latent;
    c.codes = 32UL;
    c.commitment = (float)0.25;
    c.resetDeadAfter = 20UL;
    c.act = ACT_RELU;
    c.recon = RECON_MSE;
    c.seed = 1UL;
    return c;
}

struct VqVae vqvae_new(unsigned long inDim, const &VqVaeConfig cfg) {
    struct VqVae v;
    v.cfg = *cfg;
    v.rng = rng_new(cfg.seed);
    v.enc = mlp_new(inDim, cfg.hidden, cfg.hiddenLayers, cfg.latent, cfg.act, ACT_NONE, &v.rng);
    v.dec = mlp_new(cfg.latent, cfg.hidden, cfg.hiddenLayers, inDim, cfg.act, ACT_NONE, &v.rng);
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    v.codebook = tensor_new_2d(cfg.codes, cfg.latent, 1);
    v.idle = tensor_new_1d(cfg.codes, 0);
    match (cur) {
        case some(a): { tensor_arena_begin(a); }
        case null: { }
    }
    float k = (float)1.0 / (float)cfg.codes;
    rng_fill_uniform(&v.rng, v.codebook, -k, k);
    v.initialized = 0;
    v.lastRecon = (float)0.0;
    v.lastVq = (float)0.0;
    v.lastPerplexity = (float)0.0;
    return v;
}

// Index of the nearest code for every row of ze.
static &Tensor vq_nearest_(const &VqVae v, const &Tensor ze) {
    unsigned long m = ae_rows_(ze);
    unsigned long d = v.cfg.latent;
    unsigned long K = v.cfg.codes;
    &Tensor idx = tensor_new_1d(m, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < m) {
            unsigned long best = 0UL;
            float bestD = (float)3.0e38;
            unsigned long k = 0UL;
            while (k < K) {
                float acc = (float)0.0;
                unsigned long c = 0UL;
                while (c < d) {
                    float diff = ze->data[r * d + c] - v.codebook->data[k * d + c];
                    acc = acc + diff * diff;
                    c = c + 1UL;
                }
                if (acc < bestD) { bestD = acc; best = k; }
                k = k + 1UL;
            }
            idx->data[r] = (float)best;
            r = r + 1UL;
        }
    }
    return idx;
}

static float vq_perplexity_(const &Tensor idx, unsigned long K) {
    float H = (float)0.0;
    unsafe {
        unsigned long m = idx->size;
        float* counts = (float*)malloc(sizeof(float) * K);
        unsigned long k = 0UL;
        while (k < K) { counts[k] = (float)0.0; k = k + 1UL; }
        unsigned long i = 0UL;
        while (i < m) { unsigned long c = (unsigned long)idx->data[i]; counts[c] = counts[c] + (float)1.0; i = i + 1UL; }
        k = 0UL;
        while (k < K) {
            if (counts[k] > (float)0.0) { float p = counts[k] / (float)m; H = H - p * log_f(p); }
            k = k + 1UL;
        }
        free((void*)counts);
    }
    return exp_f(H);
}

// Copies encoder output row 'row' into code 'k'.
static void vq_set_code_(&VqVae v, unsigned long k, const &Tensor ze, unsigned long row) {
    unsigned long d = v.cfg.latent;
    unsafe {
        unsigned long c = 0UL;
        while (c < d) { v.codebook->data[k * d + c] = ze->data[row * d + c]; c = c + 1UL; }
        v.idle->data[k] = (float)0.0;
    }
}

// Data-dependent init on the first call; afterwards restart codes that
// have been idle for resetDeadAfter batches.
static void vq_maintain_(&VqVae v, const &Tensor ze) {
    unsigned long m = ae_rows_(ze);
    unsigned long K = v.cfg.codes;
    if (m == 0UL) return;
    unsigned long k = 0UL;
    if (!v.initialized) {
        while (k < K) { vq_set_code_(v, k, ze, rng_below(&v.rng, m)); k = k + 1UL; }
        v.initialized = 1;
        return;
    }
    if (v.cfg.resetDeadAfter == 0UL) return;
    unsafe {
        while (k < K) {
            if (v.idle->data[k] >= (float)v.cfg.resetDeadAfter) vq_set_code_(v, k, ze, rng_below(&v.rng, m));
            k = k + 1UL;
        }
    }
}

static void vq_count_use_(&VqVae v, const &Tensor idx) {
    unsafe {
        unsigned long k = 0UL;
        while (k < v.cfg.codes) { v.idle->data[k] = v.idle->data[k] + (float)1.0; k = k + 1UL; }
        unsigned long i = 0UL;
        while (i < idx->size) { v.idle->data[(unsigned long)idx->data[i]] = (float)0.0; i = i + 1UL; }
    }
}

&Tensor vqvae_loss(&VqVae v, const &Tensor x, const &Tensor y) {
    &Tensor ze = mlp_forward(&v.enc, x);
    vq_maintain_(v, ze);
    &Tensor idx = vq_nearest_(v, ze);
    vq_count_use_(v, idx);
    &Tensor zq = tensor_gather_rows(v.codebook, idx);
    // straight-through: forward zq, backward d/dze = identity
    &Tensor st = tensor_add(ze, tensor_detach(tensor_sub(zq, ze)));
    &Tensor out = mlp_forward(&v.dec, st);
    &Tensor recon = ae_recon_mean_(out, x, v.cfg.recon);
    &Tensor book = tensor_mse_loss(zq, tensor_detach(ze));
    &Tensor commit = tensor_mse_loss(ze, tensor_detach(zq));
    &Tensor vq = tensor_add(book, tensor_scale(commit, v.cfg.commitment));
    v.lastRecon = ae_value_(recon);
    v.lastVq = ae_value_(vq);
    v.lastPerplexity = vq_perplexity_(idx, v.cfg.codes);
    return tensor_add(recon, vq);
}

&Tensor vqvae_encode(&VqVae v, const &Tensor x) { return vq_nearest_(v, mlp_forward(&v.enc, x)); }

&Tensor vqvae_decode(&VqVae v, const &Tensor codes) {
    return ae_output_(mlp_forward(&v.dec, tensor_gather_rows(v.codebook, codes)), v.cfg.recon);
}

&Tensor vqvae_reconstruct(&VqVae v, const &Tensor x) { return vqvae_decode(v, vqvae_encode(v, x)); }

void vqvae_add_params(&Optimizer opt, const &VqVae v) {
    optimizer_add_mlp(opt, &v.enc);
    optimizer_add_mlp(opt, &v.dec);
    optimizer_add(opt, v.codebook);
}

void vqvae_free(&VqVae v) {
    mlp_free(&v.enc);
    mlp_free(&v.dec);
    v.codebook->free();
    v.idle->free();
    unsafe { free((void*)v.codebook); free((void*)v.idle); }
}

} // namespace std
