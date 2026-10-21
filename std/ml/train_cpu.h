#pragma once
// SafeC Standard Library — the CPU backend of train_traits.h's interface.
//
// Thin wrappers that make the CPU models satisfy the traits: CpuTrainer
// (Trainable: trainer_fit's loop and arithmetic, one epoch per fit_epoch),
// CpuGanTrainer (GanTrainable: Gan::train / Dcgan::train), CpuDqnAgent
// (ValueAgent: rl.h's Dqn) and CpuPgAgent (PolicyLearner: rl.h's
// PolicyAgent). Each takes its model by value into 'host' and trains it in
// place, so sync() has nothing to do. train.h names these without the Cpu
// prefix when the build has no device feature.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/train_traits.h>
#include <std/ml/convnet.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/ml/rl.h>

namespace std {

// A model M, its optimizer and its loss: trainer_fit as a Trainable.
generic<M>
struct CpuTrainer {
    M host;
    struct Optimizer opt;     // made by the constructor from a configuration Optimizer
    fn &Tensor(&M, const &Tensor, const &Tensor) lossFn;
    struct DataLoader dl;     // between fit_begin and fit_end
    struct TensorArena stepArena; // each step's tensors
    float clipNorm;           // from the last fit_begin (0: none)
    int   active;

    int   fit_begin(const &Dataset data, const &TrainConfig cfg);
    struct EpochLoss fit_epoch();
    void  fit_end();
    float train_step(const &Tensor x, const &Tensor y);
    float learning_rate();
    void  set_learning_rate(float lr);
    int   sync();
    void  free();
};

generic<M>
int CpuTrainer::fit_begin(const &Dataset data, const &TrainConfig cfg) {
    if (self.active) self.dl.free();
    self.dl = dataloader_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    self.clipNorm = cfg.clipNorm;
    self.active = 1;
    return 1;
}

generic<M>
struct EpochLoss CpuTrainer::fit_epoch() {
    struct EpochLoss el;
    el.steps = 0UL;
    el.ok = self.active;
    el.loss = (float)0.0;
    if (self.active)
        el.loss = trainer_epoch_(&self.host, self.lossFn, &self.opt, &self.dl, &self.stepArena, self.clipNorm, &el.steps);
    return el;
}

generic<M>
void CpuTrainer::fit_end() {
    if (self.active) self.dl.free();
    self.active = 0;
}

generic<M>
float CpuTrainer::train_step(const &Tensor x, const &Tensor y) {
    self.stepArena.begin();
    float l = trainer_step_(&self.opt, self.lossFn(&self.host, x, y), self.clipNorm);
    tensor_arena_end();
    self.stepArena.reset();
    return l;
}

generic<M>
float CpuTrainer::learning_rate() { return self.opt.lr; }

generic<M>
void CpuTrainer::set_learning_rate(float lr) { self.opt.lr = lr; }

generic<M>
int CpuTrainer::sync() { return 1; }

generic<M>
void CpuTrainer::free() {
    self.fit_end();
    self.opt.free();
    self.stepArena.free();
}

// 'opt' supplies the optimizer's kind and hyperparameters (optimizer_from);
// the trainer adds the model's parameters to its own. loss: TRAIN_LOSS_*.
struct CpuTrainer<struct Mlp> cpu_mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt);
struct CpuTrainer<struct ConvNet> cpu_convnet_trainer_new(struct ConvNet n, int loss, const &Optimizer opt);
// Autoencoders train on their own losses (vae_loss, conv_vae_loss, conv_autoencoder_loss).
struct CpuTrainer<struct Vae> cpu_vae_trainer_new(struct Vae v, const &Optimizer opt);
struct CpuTrainer<struct ConvVae> cpu_conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt);
struct CpuTrainer<struct ConvAutoencoder> cpu_conv_ae_trainer_new(struct ConvAutoencoder ae, const &Optimizer opt);

// A Gan or Dcgan (H) with its own optimizers: GanTrainable.
generic<H>
struct CpuGanTrainer {
    H host;
    fn struct GanLosses(&H, const &Tensor, ?&Tensor) stepFn;
    fn struct GanLosses(&H, const &Dataset, unsigned long, unsigned long) trainFn;

    struct GanLosses train_step(const &Tensor real, ?&Tensor labels);
    struct GanLosses train(const &Dataset data, unsigned long epochs, unsigned long batchSize);
    int  sync();
    void free();
};

generic<H>
struct GanLosses CpuGanTrainer::train_step(const &Tensor real, ?&Tensor labels) {
    return self.stepFn(&self.host, real, labels);
}

generic<H>
struct GanLosses CpuGanTrainer::train(const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    return self.trainFn(&self.host, data, epochs, batchSize);
}

generic<H>
int CpuGanTrainer::sync() { return 1; }

generic<H>
void CpuGanTrainer::free() { }

struct CpuGanTrainer<struct Gan> cpu_gan_trainer_new(struct Gan g);
struct CpuGanTrainer<struct Dcgan> cpu_dcgan_trainer_new(struct Dcgan g);

// rl.h's Dqn as a ValueAgent.
struct CpuDqnAgent {
    struct Dqn host;

    float epsilon();
    unsigned long act(const &Tensor obs, int explore);
    int   observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal);
    float last_loss();
    unsigned long updates();
    int   sync();
    void  free();
};
struct CpuDqnAgent cpu_dqn_agent_new(struct Dqn d);

// rl.h's PolicyAgent as a PolicyLearner.
struct CpuPgAgent {
    struct PolicyAgent host;

    unsigned long act(const &Tensor obs, int sample);
    void  record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                 int truncated);
    float update();
    int   sync();
    void  free();
};
struct CpuPgAgent cpu_pg_agent_new(struct PolicyAgent a);

} // namespace std
