#pragma once
// SafeC Standard Library — autoencoders: plain, denoising, sparse,
// variational (VAE, beta-VAE, conditional VAE) and vector-quantized
// (VQ-VAE). MLP encoders and decoders over flattened rows, trained with
// Autograd::get -- or convolutional ones (ConvAutoencoder, ConvVae) built from
// convnet.h's ConvNet.
//
// Each model's loss(x, y) method is its training loss, so each is a
// LossModel (module.h) and module.h's generic functions train any of them:
//   struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.001);
//   model_add_params(&vae, &opt);
//   model_fit(&vae, &opt, &data, &cfg);
// 'y' is ignored except by a conditional VAE, which reads class indices
// from it.
//
// Reconstruction losses: RECON_MSE for real-valued data, RECON_BCE for data
// in [0, 1] (images). With RECON_BCE the decoder outputs logits and the
// *_decode / *_reconstruct functions apply the sigmoid.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/convnet.h>

namespace std {

#define RECON_MSE 0
#define RECON_BCE 1

// ── Autoencoder (plain, denoising, sparse) ───────────────────────────────────
// x -> enc -> z (latent) -> dec -> x'. Variants are switched on in the
// config and can be combined:
//   noiseStd > 0 or dropProb > 0  denoising AE (Vincent et al. 2008): the
//                                 encoder sees x + N(0, noiseStd^2) with a
//                                 'dropProb' fraction of inputs zeroed, the
//                                 loss compares against the clean x.
//   sparsity > 0                  sparse AE: + sparsity * mean |z| (L1).
struct AutoencoderConfig {
    unsigned long hidden;        // width of the hidden layers
    unsigned long hiddenLayers;  // in encoder and decoder each
    unsigned long latent;
    int   act;                   // ACT_* between layers
    int   latentAct;             // ACT_* on the code (ACT_NONE for a linear code)
    int   recon;                 // RECON_MSE / RECON_BCE
    float noiseStd;
    float dropProb;
    float sparsity;
    unsigned long seed;

    // hidden 128, 1 hidden layer, ReLU, linear code, MSE, no noise or sparsity.
    static struct AutoencoderConfig default(unsigned long latent);
};

struct Autoencoder {
    struct Mlp enc;
    struct Mlp dec;
    struct AutoencoderConfig cfg;
    struct Rng rng;
    float lastRecon;     // last loss call's reconstruction term
    float lastPenalty;   // ... and its sparsity term

    &Tensor encode(const &Tensor x);         // [m, latent]
    &Tensor decode(const &Tensor z);         // [m, inDim]
    &Tensor reconstruct(const &Tensor x);
    // Parameterized (module.h).
    void collect_params(&Vec<&Tensor> out) const;
    void collect_state(&Vec<&Tensor> out) const;
    void free();

    &Tensor loss(const &Tensor x, const &Tensor y);

    static struct Autoencoder new(unsigned long inDim, const &AutoencoderConfig cfg);
};

// ── Variational autoencoder (VAE, beta-VAE, CVAE) ────────────────────────────
// Kingma & Welling 2014. The encoder outputs mu and log sigma^2 of a
// diagonal Gaussian q(z|x); z = mu + sigma * eps (the reparameterization
// trick) is decoded back. Loss per sample:
//   recon(x, x') + beta * KL(q(z|x) || N(0, I))
// with the reconstruction summed over features. beta = 1 is the VAE,
// beta > 1 the beta-VAE (Higgins et al. 2017, more disentangled codes);
// raise beta from 0 over the first epochs for KL warm-up. classes > 0
// makes it a conditional VAE (Sohn et al. 2015): the one-hot label is
// appended to the encoder's and the decoder's inputs.
struct VaeConfig {
    unsigned long hidden;
    unsigned long hiddenLayers;   // >= 1
    unsigned long latent;
    int   act;
    int   recon;
    float beta;
    unsigned long classes;        // 0: unconditional
    unsigned long seed;

    // hidden 256, 1 hidden layer, ReLU, BCE, beta 1, unconditional.
    static struct VaeConfig default(unsigned long latent);
};

struct Vae {
    struct Mlp enc;        // trunk: x (+ label) -> hidden
    struct Mlp muHead;     // hidden -> latent
    struct Mlp logvarHead; // hidden -> latent
    struct Mlp dec;        // z (+ label) -> x
    struct VaeConfig cfg;
    struct Rng rng;
    float lastRecon;
    float lastKl;

    // mu of q(z|x) (y: class indices, read only by a CVAE).
    &Tensor encode(const &Tensor x, const &Tensor y);
    &Tensor decode(const &Tensor z, const &Tensor y);
    &Tensor reconstruct(const &Tensor x, const &Tensor y);   // decode(mu)
    // n new samples from the prior; 'labels' (n class indices) for a CVAE,
    // null for an unconditional VAE.
    &Tensor sample(unsigned long n, ?&Tensor labels);
    // Parameterized (module.h).
    void collect_params(&Vec<&Tensor> out) const;
    void collect_state(&Vec<&Tensor> out) const;
    void free();

    &Tensor loss(const &Tensor x, const &Tensor y);

    static struct Vae new(unsigned long inDim, const &VaeConfig cfg);
};

// ── Vector-quantized VAE ─────────────────────────────────────────────────────
// van den Oord et al. 2017. The encoder output z_e is replaced by its
// nearest codebook vector z_q (a discrete code per row); gradients pass
// from z_q back to z_e unchanged (straight-through). Loss:
//   recon + ||sg(z_e) - z_q||^2 + commitment * ||z_e - sg(z_q)||^2
// The codebook (codes x latent) is a trained parameter. Two standard fixes
// against codebook collapse: the first loss call initializes the codebook
// from that batch's encoder outputs, and a code no batch has picked for
// 'resetDeadAfter' calls is moved onto a random encoder output.
struct VqVaeConfig {
    unsigned long hidden;
    unsigned long hiddenLayers;
    unsigned long latent;         // code vector size
    unsigned long codes;          // codebook size
    float commitment;             // 0.25 in the paper
    unsigned long resetDeadAfter; // 0: never restart unused codes
    int   act;
    int   recon;
    unsigned long seed;

    // hidden 128, 1 hidden layer, 32 codes, commitment 0.25, dead codes
    // restarted after 20 batches, ReLU, MSE.
    static struct VqVaeConfig default(unsigned long latent);
};

struct VqVae {
    struct Mlp enc;
    struct Mlp dec;
    &Tensor codebook;             // [codes, latent]
    &Tensor idle;                 // [codes] batches since each code was last used
    int     initialized;          // codebook set from encoder outputs yet
    struct Rng rng;
    struct VqVaeConfig cfg;
    float lastRecon;
    float lastVq;                 // codebook + commitment terms
    float lastPerplexity;         // exp(entropy) of code use in the last batch

    &Tensor encode(const &Tensor x);         // [m] code indices
    &Tensor decode(const &Tensor codes);     // indices -> [m, inDim]
    &Tensor reconstruct(const &Tensor x);
    // Parameterized (module.h).
    void collect_params(&Vec<&Tensor> out) const;
    void collect_state(&Vec<&Tensor> out) const;
    void free();

    &Tensor loss(const &Tensor x, const &Tensor y);

    static struct VqVae new(unsigned long inDim, const &VqVaeConfig cfg);
};

// ── Convolutional autoencoder and VAE ────────────────────────────────────────
// The same models with a ConvNet encoder and decoder you build (convnet.h):
// images stay [batch, C*H*W] rows, as datasets.h loads them. The encoder
// maps the image to features; the decoder takes the code as a
// latent x 1 x 1 input (start it with ConvNet::dense + ConvNet::reshape, or
// a 1x1-input ConvNet::deconv) and must output the image's size. With
// RECON_BCE the decoder outputs logits, as above. The model takes
// ownership of both nets: free the model, not the nets. Loss functions run
// the nets in training mode (batch norm on batch statistics); encode /
// decode / reconstruct / sample in inference mode (running statistics).
//
//   struct ConvNet enc = ConvNet::new(1UL, 16UL, 16UL);
//   enc.conv(16UL, 4UL, 2UL, 1UL, ACT_LEAKY, &rng);   // 16 x 8 x 8
//   enc.conv(32UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);    // 32 x 4 x 4
//   enc.batchnorm(ACT_LEAKY);
//   struct ConvNet dec = ConvNet::new(latent, 1UL, 1UL);
//   dec.dense(32UL * 16UL, ACT_RELU, &rng);
//   dec.reshape(32UL, 4UL, 4UL);
//   dec.deconv(16UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);  // 16 x 8 x 8
//   dec.deconv(1UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);   // 1 x 16 x 16
//   struct ConvVae v = ConvVae::new(enc, dec, &cfg);

// Plain convolutional autoencoder: the encoder's output is the code (end
// it with enc.dense(latent, ...) for a vector code).
struct ConvAutoencoder {
    struct ConvNet enc;
    struct ConvNet dec;
    int   recon;           // RECON_MSE / RECON_BCE
    float lastRecon;

    &Tensor encode(const &Tensor x);
    &Tensor decode(const &Tensor z);
    &Tensor reconstruct(const &Tensor x);
    // Parameterized (module.h).
    void collect_params(&Vec<&Tensor> out) const;
    void collect_state(&Vec<&Tensor> out) const;
    void free();

    &Tensor loss(const &Tensor x, const &Tensor y);

    static struct ConvAutoencoder new(struct ConvNet enc, struct ConvNet dec, int recon);
};

// Convolutional VAE: linear mu / log sigma^2 heads on the flattened encoder
// features, the loss of Vae. Reads latent, recon, beta, classes and seed
// from the config (hidden, hiddenLayers and act describe the nets you built
// and are ignored). classes > 0 makes it a conditional VAE: the label goes
// to the encoder as 'classes' extra input channels (one constant plane per
// class, see tensor_append_label_channels) and to the decoder as a one-hot
// appended to the code. So for C-channel images the encoder takes C +
// classes channels, the decoder latent + classes inputs and outputs C
// channels. 'y' holds the class indices, as for Vae.
struct ConvVae {
    struct ConvNet enc;
    struct Mlp muHead;     // features -> latent
    struct Mlp logvarHead;
    struct ConvNet dec;    // latent (+ label) -> image
    struct VaeConfig cfg;
    struct Rng rng;
    float lastRecon;
    float lastKl;

    &Tensor encode(const &Tensor x, const &Tensor y);        // mu
    &Tensor decode(const &Tensor z, const &Tensor y);
    &Tensor reconstruct(const &Tensor x, const &Tensor y);
    // n samples from the prior; 'labels' (n class indices) for a conditional
    // VAE, null otherwise.
    &Tensor sample(unsigned long n, ?&Tensor labels);
    // Parameterized (module.h).
    void collect_params(&Vec<&Tensor> out) const;
    void collect_state(&Vec<&Tensor> out) const;
    void free();

    &Tensor loss(const &Tensor x, const &Tensor y);

    static struct ConvVae new(struct ConvNet enc, struct ConvNet dec, const &VaeConfig cfg);
};

} // namespace std
