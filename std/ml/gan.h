#pragma once
// SafeC Standard Library — generative adversarial networks.
//
// An MLP generator G maps latent noise z ~ N(0, I) to data rows; an MLP
// discriminator (critic) D scores rows as real or generated. They train in
// alternation: D on a real and a generated batch, then G through D. The
// objective is picked in the config:
//
//   GAN_STANDARD  Goodfellow et al. 2014, with the non-saturating generator
//                 loss: D minimizes BCE(D(x), 1) + BCE(D(G(z)), 0), G
//                 minimizes BCE(D(G(z)), 1).
//   GAN_LSGAN     least squares (Mao et al. 2017): D minimizes
//                 (D(x) - 1)^2 / 2 + D(G(z))^2 / 2, G (D(G(z)) - 1)^2 / 2.
//   GAN_WGAN      Wasserstein (Arjovsky et al. 2017): the critic maximizes
//                 D(x) - D(G(z)) with its weights clipped to [-clip, clip],
//                 'nCritic' critic steps per generator step, RMSProp.
//   GAN_HINGE     hinge loss (Miyato et al. 2018): D minimizes
//                 relu(1 - D(x)) + relu(1 + D(G(z))), G -D(G(z)).
//   GAN_WGAN_GP   Wasserstein with gradient penalty (Gulrajani et al.
//                 2017): the critic minimizes D(G(z)) - D(x) + gpWeight *
//                 (||grad_x D(x^)|| - 1)^2 on random interpolates x^ of
//                 the real and generated rows, no clipping, Adam with
//                 betas (0, 0.9); use nCritic 5. The penalty's parameter
//                 gradient is a mixed second derivative; it is computed as
//                 a central difference of the critic along each row's
//                 normalized input gradient (two extra critic passes, exact
//                 to O(eps^2), eps 1e-2) rather than by differentiating the
//                 backward pass. The critic must treat rows independently:
//                 no batch norm in it (group / layer norm are fine).
//
// classes > 0 makes it a conditional GAN (Mirza & Osindero 2014): the
// one-hot label is appended to G's noise and to D's input, and G learns
// the class-conditional distribution.
//
// emaDecay > 0 keeps an exponential moving average of the generator's
// weights (Karras et al. 2018) and Gan::generate samples from it: the
// adversarial game oscillates, and the average sits nearer its centre
// than any single step's weights. The decay ramps up as
// min(emaDecay, (1 + t) / (10 + t)) over the first steps t.
//
// The GAN owns both optimizers (Adam, beta1 0.5, as DCGAN recommends;
// RMSProp for GAN_WGAN) and a tensor arena for its steps.
//
// Dcgan (below) is the same game with convolutional networks from
// convnet.h (Radford et al. 2016).
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/convnet.h>
#include <std/ml/datasets.h>

namespace std {

#define GAN_STANDARD 0
#define GAN_LSGAN    1
#define GAN_WGAN     2
#define GAN_HINGE    3
#define GAN_WGAN_GP  4

struct GanConfig {
    unsigned long latent;
    unsigned long hidden;
    unsigned long hiddenLayers;
    int   loss;              // GAN_*
    int   genAct;            // ACT_* in G (D always uses leaky ReLU)
    int   outAct;            // ACT_* on G's output: ACT_NONE, ACT_TANH ([-1,1]) or ACT_SIGMOID ([0,1])
    float lrG;
    float lrD;
    unsigned long nCritic;   // D steps per G step
    float clip;              // GAN_WGAN weight clipping
    float gpWeight;          // GAN_WGAN_GP penalty weight (lambda)
    unsigned long classes;   // 0: unconditional
    float emaDecay;          // generator weight averaging, 0 = off
    unsigned long seed;
};
// latent as given, hidden 128, 2 hidden layers, GAN_STANDARD, ReLU in G,
// linear output, lr 2e-4 for both, 1 critic step, clip 0.01, gradient-penalty
// weight 10, EMA 0.999.
struct GanConfig gan_config_default(unsigned long latent);

struct Gan {
    struct Mlp gen;
    struct Mlp genEma;       // averaged generator (a copy of gen when emaDecay is 0)
    struct Mlp disc;
    struct Optimizer optG;
    struct Optimizer optD;
    struct GanConfig cfg;
    struct Rng rng;
    struct TensorArena stepArena;
    unsigned long steps;     // generator steps taken

    // One generator step after cfg.nCritic discriminator steps on 'real'
    // ([m, dataDim]); 'labels' (m class indices) for a conditional GAN, else null.
    struct GanLosses train_step(const &Tensor real, ?&Tensor labels);
    // Epochs over a dataset in shuffled batches (labels taken from the dataset
    // when the GAN is conditional). Returns the last step's losses.
    struct GanLosses train(const &Dataset data, unsigned long epochs, unsigned long batchSize);
    // n generated rows from the averaged generator (heap tensor, caller
    // frees); 'labels' as above.
    &Tensor generate(unsigned long n, ?&Tensor labels);
    // D's raw score for rows (higher = more real).
    &Tensor score(const &Tensor x, ?&Tensor labels);
    void free();
};

struct GanLosses {
    float d;                 // last discriminator / critic loss
    float g;                 // generator loss
};

struct Gan gan_new(unsigned long dataDim, const &GanConfig cfg);
// Internal (exposed for tests): GAN_WGAN_GP's penalty term for g's critic
// at the interpolates xh. tensor_backward of the result accumulates the
// penalty's gradient into D's parameters (after it has already added the
// input-gradient pass's, so zero them in between, as Gan::train_step does);
// 'value' receives gpWeight * mean (||grad_x D|| - 1)^2.
&Tensor gan_gradient_penalty_(&Gan g, const &Tensor xh, ?&Tensor labels, &float value);

// ── DCGAN ────────────────────────────────────────────────────────────────────
// Gan with a ConvNet generator and discriminator you build (convnet.h). G
// takes the latent batch as a latent x 1 x 1 input and must output the
// data's size; D takes a data row ([C*H*W], CHW order) and must output one
// score. Reads latent, loss, lrG, lrD, nCritic, clip, classes, emaDecay and
// seed from the config (hidden, hiddenLayers, genAct and outAct describe
// the nets you built and are ignored). classes > 0 makes it a conditional
// DCGAN: G gets the one-hot label appended to its noise (latent + classes
// inputs), D gets it as 'classes' extra input channels, one constant plane
// per class (tensor_append_label_channels) -- so for C-channel images D
// takes C + classes channels. Takes ownership of both nets. Training steps
// run the nets in training mode (batch norm on batch statistics);
// Dcgan::generate and Dcgan::score in inference mode.
//
//   struct ConvNet gen = convnet_new(latent, 1UL, 1UL);
//   gen.deconv(64UL, 4UL, 1UL, 0UL, ACT_NONE, &rng);   // 64 x 4 x 4
//   gen.batchnorm(ACT_RELU);
//   gen.deconv(32UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);   // 32 x 8 x 8
//   gen.batchnorm(ACT_RELU);
//   gen.deconv(1UL, 4UL, 2UL, 1UL, ACT_TANH, &rng);    // 1 x 16 x 16
//   struct ConvNet disc = convnet_new(1UL, 16UL, 16UL);
//   disc.conv(32UL, 4UL, 2UL, 1UL, ACT_LEAKY, &rng);   // 32 x 8 x 8
//   disc.conv(64UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);    // 64 x 4 x 4
//   disc.batchnorm(ACT_LEAKY);
//   disc.conv(1UL, 4UL, 1UL, 0UL, ACT_NONE, &rng);     // 1 x 1 x 1
//   struct Dcgan g = dcgan_new(gen, disc, &cfg);
struct Dcgan {
    struct ConvNet gen;
    struct ConvNet genEma;
    struct ConvNet disc;
    struct Optimizer optG;
    struct Optimizer optD;
    struct GanConfig cfg;
    struct Rng rng;
    struct TensorArena stepArena;
    unsigned long steps;

    // 'labels' (m class indices) for a conditional DCGAN, else null -- as for Gan.
    struct GanLosses train_step(const &Tensor real, ?&Tensor labels);
    struct GanLosses train(const &Dataset data, unsigned long epochs, unsigned long batchSize);
    &Tensor generate(unsigned long n, ?&Tensor labels);   // heap tensor, caller frees
    &Tensor score(const &Tensor x, ?&Tensor labels);
    void free();
};
struct Dcgan dcgan_new(struct ConvNet gen, struct ConvNet disc, const &GanConfig cfg);
// Internal (exposed for tests): gan_gradient_penalty_ for a Dcgan's critic.
&Tensor dcgan_gradient_penalty_(&Dcgan g, const &Tensor xh, ?&Tensor labels, &float value);

} // namespace std
