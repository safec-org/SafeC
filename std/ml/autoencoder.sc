// SafeC Standard Library — autoencoders (see autoencoder.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/autoencoder.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/ml/convnet.h>
#include <std/math.h>
#include <std/panic.h>

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

&Tensor Autoencoder::encode(const &Tensor x) { return self.enc.forward(x); }

&Tensor Autoencoder::decode(const &Tensor z) {
    return ae_output_(self.dec.forward(z), self.cfg.recon);
}

&Tensor Autoencoder::reconstruct(const &Tensor x) {
    return self.decode(self.encode(x));
}

&Tensor Autoencoder::loss(const &Tensor x, const &Tensor y) {
    // the encoder's input: a copy of x, corrupted for a denoising AE
    &Tensor noisy = tensor_detach(x);
    int corrupt = self.cfg.noiseStd > (float)0.0 || self.cfg.dropProb > (float)0.0;
    if (corrupt) {
        unsafe {
            unsigned long i = 0UL;
            while (i < noisy->size) {
                float v = noisy->data[i] + self.cfg.noiseStd * self.rng.normal();
                if (self.cfg.dropProb > (float)0.0 && self.rng.uniform() < self.cfg.dropProb) v = (float)0.0;
                noisy->data[i] = v;
                i = i + 1UL;
            }
        }
    }
    &Tensor z = self.enc.forward(noisy);
    &Tensor out = self.dec.forward(z);
    &Tensor loss = ae_recon_mean_(out, x, self.cfg.recon);
    self.lastRecon = ae_value_(loss);
    self.lastPenalty = (float)0.0;
    if (self.cfg.sparsity > (float)0.0) {
        &Tensor pen = tensor_scale(tensor_mean(tensor_abs(z)), self.cfg.sparsity);
        self.lastPenalty = ae_value_(pen);
        loss = tensor_add(loss, pen);
    }
    return loss;
}

void Autoencoder::free() {
    self.enc.free();
    self.dec.free();
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

// z = mu + exp(logvar / 2) * eps
static &Tensor vae_reparam_(&Rng rng, const &Tensor mu, const &Tensor logvar, unsigned long m, unsigned long latent) {
    &Tensor eps = rng.normal_tensor(m, latent);
    return tensor_add(mu, tensor_mul(tensor_exp(tensor_scale(logvar, (float)0.5)), eps));
}

// KL(N(mu, sigma^2) || N(0, 1)) = -1/2 sum(1 + logvar - mu^2 - exp(logvar)),
// averaged over the batch.
static &Tensor vae_kl_(const &Tensor mu, const &Tensor logvar, float rows) {
    &Tensor inner = tensor_sub(tensor_add_scalar(logvar, (float)1.0),
                               tensor_add(tensor_mul(mu, mu), tensor_exp(logvar)));
    return tensor_scale(tensor_sum(inner), (float)-0.5 / rows);
}

&Tensor Vae::loss(const &Tensor x, const &Tensor y) {
    &Tensor h = self.enc.forward(vae_cond_(self, x, y));
    &Tensor mu = self.muHead.forward(h);
    &Tensor logvar = self.logvarHead.forward(h);
    unsigned long m = ae_rows_(x);
    float rows = (float)m;
    &Tensor z = vae_reparam_(&self.rng, mu, logvar, m, self.cfg.latent);
    &Tensor out = self.dec.forward(vae_cond_(self, z, y));
    // reconstruction summed over features, averaged over the batch
    &Tensor recon = tensor_scale(ae_recon_mean_(out, x, self.cfg.recon), ae_cols_(x));
    &Tensor kl = vae_kl_(mu, logvar, rows);
    self.lastRecon = ae_value_(recon);
    self.lastKl = ae_value_(kl);
    return tensor_add(recon, tensor_scale(kl, self.cfg.beta));
}

&Tensor Vae::encode(const &Tensor x, const &Tensor y) {
    return self.muHead.forward(self.enc.forward(vae_cond_(self, x, y)));
}

&Tensor Vae::decode(const &Tensor z, const &Tensor y) {
    return ae_output_(self.dec.forward(vae_cond_(self, z, y)), self.cfg.recon);
}

&Tensor Vae::reconstruct(const &Tensor x, const &Tensor y) {
    return self.decode(self.encode(x, y), y);
}

&Tensor Vae::sample(unsigned long n, ?&Tensor labels) {
    &Tensor z = self.rng.normal_tensor(n, self.cfg.latent);
    match (labels) {
        case some(l): { return self.decode(z, l); }
        case null: { }
    }
    &Tensor none = tensor_new_1d(n, 0);   // class 0 if a CVAE is sampled without labels
    return self.decode(z, none);
}

void Vae::free() {
    self.enc.free(); self.muHead.free(); self.logvarHead.free(); self.dec.free();
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
        case some(a): { a.begin(); }
        case null: { }
    }
    float k = (float)1.0 / (float)cfg.codes;
    v.rng.fill_uniform(v.codebook, -k, k);
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
        while (k < K) { vq_set_code_(v, k, ze, v.rng.below(m)); k = k + 1UL; }
        v.initialized = 1;
        return;
    }
    if (v.cfg.resetDeadAfter == 0UL) return;
    unsafe {
        while (k < K) {
            if (v.idle->data[k] >= (float)v.cfg.resetDeadAfter) vq_set_code_(v, k, ze, v.rng.below(m));
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

&Tensor VqVae::loss(const &Tensor x, const &Tensor y) {
    &Tensor ze = self.enc.forward(x);
    vq_maintain_(self, ze);
    &Tensor idx = vq_nearest_(self, ze);
    vq_count_use_(self, idx);
    &Tensor zq = tensor_gather_rows(self.codebook, idx);
    // straight-through: forward zq, backward d/dze = identity
    &Tensor st = tensor_add(ze, tensor_detach(tensor_sub(zq, ze)));
    &Tensor out = self.dec.forward(st);
    &Tensor recon = ae_recon_mean_(out, x, self.cfg.recon);
    &Tensor book = tensor_mse_loss(zq, tensor_detach(ze));
    &Tensor commit = tensor_mse_loss(ze, tensor_detach(zq));
    &Tensor vq = tensor_add(book, tensor_scale(commit, self.cfg.commitment));
    self.lastRecon = ae_value_(recon);
    self.lastVq = ae_value_(vq);
    self.lastPerplexity = vq_perplexity_(idx, self.cfg.codes);
    return tensor_add(recon, vq);
}

&Tensor VqVae::encode(const &Tensor x) { return vq_nearest_(self, self.enc.forward(x)); }

&Tensor VqVae::decode(const &Tensor codes) {
    return ae_output_(self.dec.forward(tensor_gather_rows(self.codebook, codes)), self.cfg.recon);
}

&Tensor VqVae::reconstruct(const &Tensor x) { return self.decode(self.encode(x)); }

void VqVae::free() {
    self.enc.free();
    self.dec.free();
    self.codebook->free();
    self.idle->free();
    unsafe { free((void*)self.codebook); free((void*)self.idle); }
}

// ══ Convolutional autoencoder / VAE ══════════════════════════════════════════

static void ae_net_check_(int ok, const &char what) {
    if (!ok) panic_at(what, "autoencoder.sc", __LINE__);
}

// Runs 'net' forward in inference mode.
static &Tensor ae_infer_(&ConvNet net, const &Tensor x) {
    int was = net.set_training(0);
    &Tensor out = net.forward(x);
    net.set_training(was);
    return out;
}

struct ConvAutoencoder conv_autoencoder_new(struct ConvNet enc, struct ConvNet dec, int recon) {
    struct ConvAutoencoder ae;
    ae.enc = enc;
    ae.dec = dec;
    ae_net_check_(ae.dec.in_dim() == ae.enc.out_dim()
                  && ae.dec.out_dim() == ae.enc.in_dim(),
                  "conv_autoencoder_new: the decoder must map the code back to the encoder's input size");
    ae.recon = recon;
    ae.lastRecon = (float)0.0;
    return ae;
}

&Tensor ConvAutoencoder::encode(const &Tensor x) { return ae_infer_(&self.enc, x); }

&Tensor ConvAutoencoder::decode(const &Tensor z) {
    return ae_output_(ae_infer_(&self.dec, z), self.recon);
}

&Tensor ConvAutoencoder::reconstruct(const &Tensor x) {
    return self.decode(self.encode(x));
}

&Tensor ConvAutoencoder::loss(const &Tensor x, const &Tensor y) {
    &Tensor out = self.dec.forward(self.enc.forward(x));
    &Tensor loss = ae_recon_mean_(out, x, self.recon);
    self.lastRecon = ae_value_(loss);
    return loss;
}

void ConvAutoencoder::free() {
    self.enc.free();
    self.dec.free();
}

struct ConvVae conv_vae_new(struct ConvNet enc, struct ConvNet dec, const &VaeConfig cfg) {
    struct ConvVae v;
    v.cfg = *cfg;
    v.rng = rng_new(cfg.seed);
    v.enc = enc;
    v.dec = dec;
    unsigned long c = cfg.classes;
    ae_net_check_(v.enc.inC > c, "conv_vae_new: the encoder needs the image's channels plus one per class");
    unsigned long dataDim = (v.enc.inC - c) * v.enc.inH * v.enc.inW;
    ae_net_check_(v.dec.in_dim() == cfg.latent + c && v.dec.out_dim() == dataDim,
                  "conv_vae_new: the decoder must map latent (+ classes) inputs to the image's size");
    unsigned long feat = v.enc.out_dim();
    v.muHead = mlp_new(feat, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.logvarHead = mlp_new(feat, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.lastRecon = (float)0.0;
    v.lastKl = (float)0.0;
    return v;
}

// The encoder's input: x, plus one label plane per class for a CVAE.
static &Tensor cvae_enc_in_(const &ConvVae v, const &Tensor x, const &Tensor y) {
    if (v.cfg.classes == 0UL) return x;
    return tensor_append_label_channels(x, y, v.cfg.classes, v.enc.inH, v.enc.inW);
}

// The decoder's input: z, plus the one-hot label for a CVAE.
static &Tensor cvae_dec_in_(const &ConvVae v, const &Tensor z, const &Tensor y) {
    if (v.cfg.classes == 0UL) return z;
    return tensor_concat_cols(z, tensor_one_hot(y, v.cfg.classes));
}

&Tensor ConvVae::loss(const &Tensor x, const &Tensor y) {
    &Tensor h = self.enc.forward(cvae_enc_in_(self, x, y));
    &Tensor mu = self.muHead.forward(h);
    &Tensor logvar = self.logvarHead.forward(h);
    unsigned long m = ae_rows_(x);
    &Tensor z = vae_reparam_(&self.rng, mu, logvar, m, self.cfg.latent);
    &Tensor out = self.dec.forward(cvae_dec_in_(self, z, y));
    &Tensor recon = tensor_scale(ae_recon_mean_(out, x, self.cfg.recon), ae_cols_(x));
    &Tensor kl = vae_kl_(mu, logvar, (float)m);
    self.lastRecon = ae_value_(recon);
    self.lastKl = ae_value_(kl);
    return tensor_add(recon, tensor_scale(kl, self.cfg.beta));
}

&Tensor ConvVae::encode(const &Tensor x, const &Tensor y) {
    return self.muHead.forward(ae_infer_(&self.enc, cvae_enc_in_(self, x, y)));
}

&Tensor ConvVae::decode(const &Tensor z, const &Tensor y) {
    return ae_output_(ae_infer_(&self.dec, cvae_dec_in_(self, z, y)), self.cfg.recon);
}

&Tensor ConvVae::reconstruct(const &Tensor x, const &Tensor y) {
    return self.decode(self.encode(x, y), y);
}

&Tensor ConvVae::sample(unsigned long n, ?&Tensor labels) {
    &Tensor z = self.rng.normal_tensor(n, self.cfg.latent);
    match (labels) {
        case some(l): { return self.decode(z, l); }
        case null: { }
    }
    &Tensor none = tensor_new_1d(n, 0);   // class 0 if a CVAE is sampled without labels
    return self.decode(z, none);
}

void ConvVae::free() {
    self.enc.free(); self.muHead.free(); self.logvarHead.free(); self.dec.free();
}

void Autoencoder::collect_params(&Vec<&Tensor> out) const {
    self.enc.collect_params(out);
    self.dec.collect_params(out);
}

void Autoencoder::collect_state(&Vec<&Tensor> out) const {
    self.enc.collect_state(out);
    self.dec.collect_state(out);
}

void Vae::collect_params(&Vec<&Tensor> out) const {
    self.enc.collect_params(out);
    self.muHead.collect_params(out);
    self.logvarHead.collect_params(out);
    self.dec.collect_params(out);
}

void Vae::collect_state(&Vec<&Tensor> out) const {
    self.enc.collect_state(out);
    self.muHead.collect_state(out);
    self.logvarHead.collect_state(out);
    self.dec.collect_state(out);
}

void VqVae::collect_params(&Vec<&Tensor> out) const {
    self.enc.collect_params(out);
    self.dec.collect_params(out);
    out.push(self.codebook);
}

void VqVae::collect_state(&Vec<&Tensor> out) const {
    self.enc.collect_state(out);
    self.dec.collect_state(out);
    out.push(self.codebook);
}

void ConvAutoencoder::collect_params(&Vec<&Tensor> out) const {
    self.enc.collect_params(out);
    self.dec.collect_params(out);
}

void ConvAutoencoder::collect_state(&Vec<&Tensor> out) const {
    self.enc.collect_state(out);
    self.dec.collect_state(out);
}

void ConvVae::collect_params(&Vec<&Tensor> out) const {
    self.enc.collect_params(out);
    self.muHead.collect_params(out);
    self.logvarHead.collect_params(out);
    self.dec.collect_params(out);
}

void ConvVae::collect_state(&Vec<&Tensor> out) const {
    self.enc.collect_state(out);
    self.muHead.collect_state(out);
    self.logvarHead.collect_state(out);
    self.dec.collect_state(out);
}

} // namespace std
