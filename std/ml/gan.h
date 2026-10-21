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
//
// classes > 0 makes it a conditional GAN (Mirza & Osindero 2014): the
// one-hot label is appended to G's noise and to D's input, and G learns
// the class-conditional distribution.
//
// emaDecay > 0 keeps an exponential moving average of the generator's
// weights (Karras et al. 2018) and gan_generate samples from it: the
// adversarial game oscillates, and the average sits nearer its centre
// than any single step's weights. The decay ramps up as
// min(emaDecay, (1 + t) / (10 + t)) over the first steps t.
//
// The GAN owns both optimizers (Adam, beta1 0.5, as DCGAN recommends;
// RMSProp for GAN_WGAN) and a tensor arena for its steps.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>

namespace std {

#define GAN_STANDARD 0
#define GAN_LSGAN    1
#define GAN_WGAN     2
#define GAN_HINGE    3

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
    unsigned long classes;   // 0: unconditional
    float emaDecay;          // generator weight averaging, 0 = off
    unsigned long seed;
};
// latent as given, hidden 128, 2 hidden layers, GAN_STANDARD, ReLU in G,
// linear output, lr 2e-4 for both, 1 critic step, clip 0.01, EMA 0.999.
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
};

struct GanLosses {
    float d;                 // last discriminator / critic loss
    float g;                 // generator loss
};

struct Gan gan_new(unsigned long dataDim, const &GanConfig cfg);
// One generator step after cfg.nCritic discriminator steps on 'real'
// ([m, dataDim]); 'labels' (m class indices) for a conditional GAN, else null.
struct GanLosses gan_train_step(&Gan g, const &Tensor real, ?&Tensor labels);
// Epochs over a dataset in shuffled batches (labels taken from the dataset
// when the GAN is conditional). Returns the last step's losses.
struct GanLosses gan_train(&Gan g, const &Dataset data, unsigned long epochs, unsigned long batchSize);
// n generated rows from the averaged generator (heap tensor, caller
// frees); 'labels' as above.
&Tensor gan_generate(&Gan g, unsigned long n, ?&Tensor labels);
// D's raw score for rows (higher = more real).
&Tensor gan_score(&Gan g, const &Tensor x, ?&Tensor labels);
void gan_free(&Gan g);

} // namespace std
