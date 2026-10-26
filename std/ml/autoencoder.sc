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

static float ae_cols_(const &FloatTensor t) {
    unsafe { tensor_host_raw_((FloatTensor*)t); }
    unsafe { return (float)t->shape[t->ndim - 1UL]; }
}

static unsigned long ae_rows_(const &FloatTensor t) {
    unsafe { tensor_host_raw_((FloatTensor*)t); }
    unsafe { return t->ndim >= 2UL ? t->shape[0] : 1UL; }
}

// Reconstruction loss averaged over elements.
static &FloatTensor ae_recon_mean_(const &FloatTensor out, const &FloatTensor x, int recon) {
    if (recon == RECON_BCE) return tensor_bce_logits_loss(out, x);
    return tensor_mse_loss(out, x);
}

// The decoder's output as data: sigmoid of the logits for BCE.
static &FloatTensor ae_output_(&FloatTensor out, int recon) {
    if (recon == RECON_BCE) return out.sigmoid();
    return out;
}

static float ae_value_(const &FloatTensor t) {
    unsafe { tensor_host_raw_((FloatTensor*)t); }
    unsafe { return t->data[0]; }
}

// ══ Autoencoder ══════════════════════════════════════════════════════════════

struct AutoencoderConfig AutoencoderConfig::default(unsigned long latent) {
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

struct Autoencoder Autoencoder::new(unsigned long inDim, const &AutoencoderConfig cfg) {
    struct Autoencoder ae;
    ae.cfg = *cfg;
    ae.rng = Rng::new(cfg.seed);
    ae.enc = Mlp::new(inDim, cfg.hidden, cfg.hiddenLayers, cfg.latent, cfg.act, cfg.latentAct, &ae.rng);
    ae.dec = Mlp::new(cfg.latent, cfg.hidden, cfg.hiddenLayers, inDim, cfg.act, ACT_NONE, &ae.rng);
    ae.lastRecon = (float)0.0;
    ae.lastPenalty = (float)0.0;
    return ae;
}

&FloatTensor Autoencoder::encode(const &FloatTensor x) { return self.enc.forward(x); }

&FloatTensor Autoencoder::decode(const &FloatTensor z) {
    return ae_output_(self.dec.forward(z), self.cfg.recon);
}

&FloatTensor Autoencoder::reconstruct(const &FloatTensor x) {
    return self.decode(self.encode(x));
}

&FloatTensor Autoencoder::loss(const &FloatTensor x, const &FloatTensor y) {
    // the encoder's input: a copy of x, corrupted for a denoising AE
    &FloatTensor noisy = x.detach();
    unsafe { tensor_host_write_raw_((FloatTensor*)noisy); }
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
    &FloatTensor z = self.enc.forward(noisy);
    &FloatTensor out = self.dec.forward(z);
    &FloatTensor loss = ae_recon_mean_(out, x, self.cfg.recon);
    self.lastRecon = ae_value_(loss);
    self.lastPenalty = (float)0.0;
    if (self.cfg.sparsity > (float)0.0) {
        &FloatTensor pen = z.abs().mean().scale(self.cfg.sparsity);
        self.lastPenalty = ae_value_(pen);
        loss = tensor_add(loss, pen);
    }
    return loss;
}

void Autoencoder::free() { self.drop(); }

void Autoencoder::drop() {
    self.enc.free();
    self.dec.free();
}

// ══ VAE ══════════════════════════════════════════════════════════════════════

struct VaeConfig VaeConfig::default(unsigned long latent) {
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

struct Vae Vae::new(unsigned long inDim, const &VaeConfig cfg) {
    struct Vae v;
    v.cfg = *cfg;
    if (v.cfg.hiddenLayers == 0UL) v.cfg.hiddenLayers = 1UL;
    v.rng = Rng::new(cfg.seed);
    unsigned long c = cfg.classes;
    v.enc = Mlp::new(inDim + c, cfg.hidden, v.cfg.hiddenLayers - 1UL, cfg.hidden, cfg.act, cfg.act, &v.rng);
    v.muHead = Mlp::new(cfg.hidden, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.logvarHead = Mlp::new(cfg.hidden, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.dec = Mlp::new(cfg.latent + c, cfg.hidden, v.cfg.hiddenLayers, inDim, cfg.act, ACT_NONE, &v.rng);
    v.lastRecon = (float)0.0;
    v.lastKl = (float)0.0;
    return v;
}

// Appends the one-hot label for a CVAE.
static &FloatTensor vae_cond_(const &Vae v, const &FloatTensor x, const &FloatTensor y) {
    if (v.cfg.classes == 0UL) return x.add_scalar((float)0.0);
    return tensor_concat_cols(x, y.one_hot(v.cfg.classes));
}

// z = mu + exp(logvar / 2) * eps
static &FloatTensor vae_reparam_(&Rng rng, const &FloatTensor mu, const &FloatTensor logvar, unsigned long m, unsigned long latent) {
    &FloatTensor eps = rng.normal_tensor(m, latent);
    return tensor_add(mu, tensor_mul(logvar.scale((float)0.5).exp(), eps));
}

// KL(N(mu, sigma^2) || N(0, 1)) = -1/2 sum(1 + logvar - mu^2 - exp(logvar)),
// averaged over the batch.
static &FloatTensor vae_kl_(const &FloatTensor mu, const &FloatTensor logvar, float rows) {
    &FloatTensor inner = tensor_sub(logvar.add_scalar((float)1.0),
                               tensor_add(tensor_mul(mu, mu), logvar.exp()));
    return inner.sum().scale((float)-0.5 / rows);
}

&FloatTensor Vae::loss(const &FloatTensor x, const &FloatTensor y) {
    &FloatTensor h = self.enc.forward(vae_cond_(self, x, y));
    &FloatTensor mu = self.muHead.forward(h);
    &FloatTensor logvar = self.logvarHead.forward(h);
    unsigned long m = ae_rows_(x);
    float rows = (float)m;
    &FloatTensor z = vae_reparam_(&self.rng, mu, logvar, m, self.cfg.latent);
    &FloatTensor out = self.dec.forward(vae_cond_(self, z, y));
    // reconstruction summed over features, averaged over the batch
    &FloatTensor recon = ae_recon_mean_(out, x, self.cfg.recon).scale(ae_cols_(x));
    &FloatTensor kl = vae_kl_(mu, logvar, rows);
    self.lastRecon = ae_value_(recon);
    self.lastKl = ae_value_(kl);
    return tensor_add(recon, kl.scale(self.cfg.beta));
}

&FloatTensor Vae::encode(const &FloatTensor x, const &FloatTensor y) {
    return self.muHead.forward(self.enc.forward(vae_cond_(self, x, y)));
}

&FloatTensor Vae::decode(const &FloatTensor z, const &FloatTensor y) {
    return ae_output_(self.dec.forward(vae_cond_(self, z, y)), self.cfg.recon);
}

&FloatTensor Vae::reconstruct(const &FloatTensor x, const &FloatTensor y) {
    return self.decode(self.encode(x, y), y);
}

&FloatTensor Vae::sample(unsigned long n, ?&FloatTensor labels) {
    &FloatTensor z = self.rng.normal_tensor(n, self.cfg.latent);
    match (labels) {
        case some(l): { return self.decode(z, l); }
        case null: { }
    }
    &FloatTensor none = FloatTensor::new_1d(n, 0);   // class 0 if a CVAE is sampled without labels
    return self.decode(z, none);
}

void Vae::free() { self.drop(); }

void Vae::drop() {
    self.enc.free(); self.muHead.free(); self.logvarHead.free(); self.dec.free();
}

// ══ VQ-VAE ═══════════════════════════════════════════════════════════════════

struct VqVaeConfig VqVaeConfig::default(unsigned long latent) {
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

struct VqVae VqVae::new(unsigned long inDim, const &VqVaeConfig cfg) {
    struct VqVae v;
    v.cfg = *cfg;
    v.rng = Rng::new(cfg.seed);
    v.enc = Mlp::new(inDim, cfg.hidden, cfg.hiddenLayers, cfg.latent, cfg.act, ACT_NONE, &v.rng);
    v.dec = Mlp::new(cfg.latent, cfg.hidden, cfg.hiddenLayers, inDim, cfg.act, ACT_NONE, &v.rng);
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    v.codebook = FloatTensor::new_2d(cfg.codes, cfg.latent, 1);
    v.idle = FloatTensor::new_1d(cfg.codes, 0);
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
static &FloatTensor vq_nearest_(const &VqVae v, const &FloatTensor ze) {
    unsafe { tensor_host_raw_((FloatTensor*)ze); }
    unsigned long m = ae_rows_(ze);
    unsigned long d = v.cfg.latent;
    unsigned long K = v.cfg.codes;
    &FloatTensor idx = FloatTensor::new_1d(m, 0);
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

static float vq_perplexity_(const &FloatTensor idx, unsigned long K) {
    unsafe { tensor_host_raw_((FloatTensor*)idx); }
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
        free((unsigned char*)counts);
    }
    return exp_f(H);
}

// Copies encoder output row 'row' into code 'k'.
static void vq_set_code_(&VqVae v, unsigned long k, const &FloatTensor ze, unsigned long row) {
    unsafe { tensor_host_raw_((FloatTensor*)ze); }
    unsigned long d = v.cfg.latent;
    unsafe {
        unsigned long c = 0UL;
        while (c < d) { v.codebook->data[k * d + c] = ze->data[row * d + c]; c = c + 1UL; }
        v.idle->data[k] = (float)0.0;
    }
}

// Data-dependent init on the first call; afterwards restart codes that
// have been idle for resetDeadAfter batches.
static void vq_maintain_(&VqVae v, const &FloatTensor ze) {
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

static void vq_count_use_(&VqVae v, const &FloatTensor idx) {
    unsafe { tensor_host_raw_((FloatTensor*)idx); }
    unsafe {
        unsigned long k = 0UL;
        while (k < v.cfg.codes) { v.idle->data[k] = v.idle->data[k] + (float)1.0; k = k + 1UL; }
        unsigned long i = 0UL;
        while (i < idx->size) { v.idle->data[(unsigned long)idx->data[i]] = (float)0.0; i = i + 1UL; }
    }
}

&FloatTensor VqVae::loss(const &FloatTensor x, const &FloatTensor y) {
    &FloatTensor ze = self.enc.forward(x);
    vq_maintain_(self, ze);
    &FloatTensor idx = vq_nearest_(self, ze);
    vq_count_use_(self, idx);
    &FloatTensor zq = tensor_gather_rows(self.codebook, idx);
    // straight-through: forward zq, backward d/dze = identity
    &FloatTensor st = tensor_add(ze, tensor_sub(zq, ze).detach());
    &FloatTensor out = self.dec.forward(st);
    &FloatTensor recon = ae_recon_mean_(out, x, self.cfg.recon);
    &FloatTensor book = tensor_mse_loss(zq, ze.detach());
    &FloatTensor commit = tensor_mse_loss(ze, zq.detach());
    &FloatTensor vq = tensor_add(book, commit.scale(self.cfg.commitment));
    self.lastRecon = ae_value_(recon);
    self.lastVq = ae_value_(vq);
    self.lastPerplexity = vq_perplexity_(idx, self.cfg.codes);
    return tensor_add(recon, vq);
}

&FloatTensor VqVae::encode(const &FloatTensor x) { return vq_nearest_(self, self.enc.forward(x)); }

&FloatTensor VqVae::decode(const &FloatTensor codes) {
    return ae_output_(self.dec.forward(tensor_gather_rows(self.codebook, codes)), self.cfg.recon);
}

&FloatTensor VqVae::reconstruct(const &FloatTensor x) { return self.decode(self.encode(x)); }

void VqVae::free() { self.drop(); }

void VqVae::drop() {
    self.enc.free();
    self.dec.free();
    self.codebook->free();
    self.idle->free();
    unsafe { free((unsigned char*)self.codebook); free((unsigned char*)self.idle); }
}

// ══ Convolutional autoencoder / VAE ══════════════════════════════════════════

static void ae_net_check_(int ok, const &char what) {
    if (!ok) panic_at(what, "autoencoder.sc", __LINE__);
}

// Runs 'net' forward in inference mode.
static &FloatTensor ae_infer_(&ConvNet net, const &FloatTensor x) {
    int was = net.set_training(0);
    &FloatTensor out = net.forward(x);
    net.set_training(was);
    return out;
}

struct ConvAutoencoder ConvAutoencoder::new(struct ConvNet enc, struct ConvNet dec, int recon) {
    struct ConvAutoencoder ae;
    ae.enc = enc;
    ae.dec = dec;
    ae_net_check_(ae.dec.in_dim() == ae.enc.out_dim()
                  && ae.dec.out_dim() == ae.enc.in_dim(),
                  "ConvAutoencoder::new: the decoder must map the code back to the encoder's input size");
    ae.recon = recon;
    ae.lastRecon = (float)0.0;
    return ae;
}

&FloatTensor ConvAutoencoder::encode(const &FloatTensor x) { return ae_infer_(&self.enc, x); }

&FloatTensor ConvAutoencoder::decode(const &FloatTensor z) {
    return ae_output_(ae_infer_(&self.dec, z), self.recon);
}

&FloatTensor ConvAutoencoder::reconstruct(const &FloatTensor x) {
    return self.decode(self.encode(x));
}

&FloatTensor ConvAutoencoder::loss(const &FloatTensor x, const &FloatTensor y) {
    &FloatTensor out = self.dec.forward(self.enc.forward(x));
    &FloatTensor loss = ae_recon_mean_(out, x, self.recon);
    self.lastRecon = ae_value_(loss);
    return loss;
}

void ConvAutoencoder::free() { self.drop(); }

void ConvAutoencoder::drop() {
    self.enc.free();
    self.dec.free();
}

struct ConvVae ConvVae::new(struct ConvNet enc, struct ConvNet dec, const &VaeConfig cfg) {
    struct ConvVae v;
    v.cfg = *cfg;
    v.rng = Rng::new(cfg.seed);
    v.enc = enc;
    v.dec = dec;
    unsigned long c = cfg.classes;
    ae_net_check_(v.enc.inC > c, "ConvVae::new: the encoder needs the image's channels plus one per class");
    unsigned long dataDim = (v.enc.inC - c) * v.enc.inH * v.enc.inW;
    ae_net_check_(v.dec.in_dim() == cfg.latent + c && v.dec.out_dim() == dataDim,
                  "ConvVae::new: the decoder must map latent (+ classes) inputs to the image's size");
    unsigned long feat = v.enc.out_dim();
    v.muHead = Mlp::new(feat, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.logvarHead = Mlp::new(feat, 0UL, 0UL, cfg.latent, ACT_NONE, ACT_NONE, &v.rng);
    v.lastRecon = (float)0.0;
    v.lastKl = (float)0.0;
    return v;
}

// The encoder's input: x, plus one label plane per class for a CVAE.
static &FloatTensor cvae_enc_in_(const &ConvVae v, const &FloatTensor x, const &FloatTensor y) {
    if (v.cfg.classes == 0UL) return x;
    return tensor_append_label_channels(x, y, v.cfg.classes, v.enc.inH, v.enc.inW);
}

// The decoder's input: z, plus the one-hot label for a CVAE.
static &FloatTensor cvae_dec_in_(const &ConvVae v, const &FloatTensor z, const &FloatTensor y) {
    if (v.cfg.classes == 0UL) return z;
    return tensor_concat_cols(z, y.one_hot(v.cfg.classes));
}

&FloatTensor ConvVae::loss(const &FloatTensor x, const &FloatTensor y) {
    &FloatTensor h = self.enc.forward(cvae_enc_in_(self, x, y));
    &FloatTensor mu = self.muHead.forward(h);
    &FloatTensor logvar = self.logvarHead.forward(h);
    unsigned long m = ae_rows_(x);
    &FloatTensor z = vae_reparam_(&self.rng, mu, logvar, m, self.cfg.latent);
    &FloatTensor out = self.dec.forward(cvae_dec_in_(self, z, y));
    &FloatTensor recon = ae_recon_mean_(out, x, self.cfg.recon).scale(ae_cols_(x));
    &FloatTensor kl = vae_kl_(mu, logvar, (float)m);
    self.lastRecon = ae_value_(recon);
    self.lastKl = ae_value_(kl);
    return tensor_add(recon, kl.scale(self.cfg.beta));
}

&FloatTensor ConvVae::encode(const &FloatTensor x, const &FloatTensor y) {
    return self.muHead.forward(ae_infer_(&self.enc, cvae_enc_in_(self, x, y)));
}

&FloatTensor ConvVae::decode(const &FloatTensor z, const &FloatTensor y) {
    return ae_output_(ae_infer_(&self.dec, cvae_dec_in_(self, z, y)), self.cfg.recon);
}

&FloatTensor ConvVae::reconstruct(const &FloatTensor x, const &FloatTensor y) {
    return self.decode(self.encode(x, y), y);
}

&FloatTensor ConvVae::sample(unsigned long n, ?&FloatTensor labels) {
    &FloatTensor z = self.rng.normal_tensor(n, self.cfg.latent);
    match (labels) {
        case some(l): { return self.decode(z, l); }
        case null: { }
    }
    &FloatTensor none = FloatTensor::new_1d(n, 0);   // class 0 if a CVAE is sampled without labels
    return self.decode(z, none);
}

void ConvVae::free() { self.drop(); }

void ConvVae::drop() {
    self.enc.free(); self.muHead.free(); self.logvarHead.free(); self.dec.free();
}

void Autoencoder::collect_params(&Vec<&FloatTensor> out) const {
    self.enc.collect_params(out);
    self.dec.collect_params(out);
}

void Autoencoder::collect_state(&Vec<&FloatTensor> out) const {
    self.enc.collect_state(out);
    self.dec.collect_state(out);
}

void Vae::collect_params(&Vec<&FloatTensor> out) const {
    self.enc.collect_params(out);
    self.muHead.collect_params(out);
    self.logvarHead.collect_params(out);
    self.dec.collect_params(out);
}

void Vae::collect_state(&Vec<&FloatTensor> out) const {
    self.enc.collect_state(out);
    self.muHead.collect_state(out);
    self.logvarHead.collect_state(out);
    self.dec.collect_state(out);
}

void VqVae::collect_params(&Vec<&FloatTensor> out) const {
    self.enc.collect_params(out);
    self.dec.collect_params(out);
    out.push(self.codebook);
}

void VqVae::collect_state(&Vec<&FloatTensor> out) const {
    self.enc.collect_state(out);
    self.dec.collect_state(out);
    out.push(self.codebook);
}

void ConvAutoencoder::collect_params(&Vec<&FloatTensor> out) const {
    self.enc.collect_params(out);
    self.dec.collect_params(out);
}

void ConvAutoencoder::collect_state(&Vec<&FloatTensor> out) const {
    self.enc.collect_state(out);
    self.dec.collect_state(out);
}

void ConvVae::collect_params(&Vec<&FloatTensor> out) const {
    self.enc.collect_params(out);
    self.muHead.collect_params(out);
    self.logvarHead.collect_params(out);
    self.dec.collect_params(out);
}

void ConvVae::collect_state(&Vec<&FloatTensor> out) const {
    self.enc.collect_state(out);
    self.muHead.collect_state(out);
    self.logvarHead.collect_state(out);
    self.dec.collect_state(out);
}

} // namespace std
