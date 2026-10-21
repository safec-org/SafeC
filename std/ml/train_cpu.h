#pragma once
// SafeC Standard Library — the CPU backend of train_traits.h's interface.
//
// Thin wrappers that make the CPU models satisfy the training traits, each a
// generic struct bounded by the model capability it needs (module.h):
// CpuTrainer<M: Module> trains any Module on a built-in loss (TRAIN_LOSS_*),
// CpuModelTrainer<M: LossModel> any model on its own loss (autoencoders,
// VAEs), CpuGanTrainer<H: GanModel> a Gan or Dcgan; CpuDqnAgent and
// CpuPgAgent wrap rl.h's agents. Each takes its model by value into 'host'
// and trains it in place, so sync() has nothing to do. train.h names these
// without the Cpu prefix when the build has no device feature.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/module.h>
#include <std/ml/train_traits.h>
#include <std/ml/convnet.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/ml/rl.h>

namespace std {

// ── Module + built-in loss: Trainable ───────────────────────────────────────
generic<M: Module>
struct CpuTrainer {
    M host;
    struct Optimizer opt;     // made by the constructor from a configuration Optimizer
    int   loss;               // TRAIN_LOSS_*
    struct DataLoader dl;     // between fit_begin and fit_end
    struct TensorArena stepArena;
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

generic<M: Module>
int CpuTrainer::fit_begin(const &Dataset data, const &TrainConfig cfg) {
    if (self.active) self.dl.free();
    self.dl = dataloader_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    self.clipNorm = cfg.clipNorm;
    self.active = 1;
    return 1;
}

generic<M: Module>
struct EpochLoss CpuTrainer::fit_epoch() {
    struct EpochLoss el;
    el.steps = 0UL;
    el.ok = self.active;
    el.loss = (float)0.0;
    if (self.active)
        el.loss = module_epoch_(&self.host, self.loss, &self.opt, &self.dl, &self.stepArena, self.clipNorm, &el.steps);
    return el;
}

generic<M: Module>
void CpuTrainer::fit_end() {
    if (self.active) self.dl.free();
    self.active = 0;
}

generic<M: Module>
float CpuTrainer::train_step(const &Tensor x, const &Tensor y) {
    self.stepArena.begin();
    float l = trainer_step_(&self.opt, module_loss(&self.host, self.loss, x, y), self.clipNorm);
    tensor_arena_end();
    self.stepArena.reset();
    return l;
}

generic<M: Module>
float CpuTrainer::learning_rate() { return self.opt.lr; }

generic<M: Module>
void CpuTrainer::set_learning_rate(float lr) { self.opt.lr = lr; }

generic<M: Module>
int CpuTrainer::sync() { return 1; }

generic<M: Module>
void CpuTrainer::free() {
    self.fit_end();
    self.opt.free();
    self.stepArena.free();
}

// 'opt' supplies the optimizer's kind and hyperparameters (optimizer_from);
// the trainer adds the model's parameters to its own. loss: TRAIN_LOSS_*.
generic<M: Module>
struct CpuTrainer<M> cpu_module_trainer_new(M m, int loss, const &Optimizer opt) {
    struct CpuTrainer<M> t;
    t.host = m;
    t.opt = optimizer_from(opt);
    model_add_params(&t.host, &t.opt);
    t.loss = loss;
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

// ── LossModel: Trainable ────────────────────────────────────────────────────
generic<M: LossModel>
struct CpuModelTrainer {
    M host;
    struct Optimizer opt;
    struct DataLoader dl;
    struct TensorArena stepArena;
    float clipNorm;
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

generic<M: LossModel>
int CpuModelTrainer::fit_begin(const &Dataset data, const &TrainConfig cfg) {
    if (self.active) self.dl.free();
    self.dl = dataloader_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    self.clipNorm = cfg.clipNorm;
    self.active = 1;
    return 1;
}

generic<M: LossModel>
struct EpochLoss CpuModelTrainer::fit_epoch() {
    struct EpochLoss el;
    el.steps = 0UL;
    el.ok = self.active;
    el.loss = (float)0.0;
    if (self.active)
        el.loss = model_epoch_(&self.host, &self.opt, &self.dl, &self.stepArena, self.clipNorm, &el.steps);
    return el;
}

generic<M: LossModel>
void CpuModelTrainer::fit_end() {
    if (self.active) self.dl.free();
    self.active = 0;
}

generic<M: LossModel>
float CpuModelTrainer::train_step(const &Tensor x, const &Tensor y) {
    self.stepArena.begin();
    float l = trainer_step_(&self.opt, self.host.loss(x, y), self.clipNorm);
    tensor_arena_end();
    self.stepArena.reset();
    return l;
}

generic<M: LossModel>
float CpuModelTrainer::learning_rate() { return self.opt.lr; }

generic<M: LossModel>
void CpuModelTrainer::set_learning_rate(float lr) { self.opt.lr = lr; }

generic<M: LossModel>
int CpuModelTrainer::sync() { return 1; }

generic<M: LossModel>
void CpuModelTrainer::free() {
    self.fit_end();
    self.opt.free();
    self.stepArena.free();
}

generic<M: LossModel>
struct CpuModelTrainer<M> cpu_model_trainer_new(M m, const &Optimizer opt) {
    struct CpuModelTrainer<M> t;
    t.host = m;
    t.opt = optimizer_from(opt);
    model_add_params(&t.host, &t.opt);
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

// ── GanModel: GanTrainable ──────────────────────────────────────────────────
generic<H: GanModel>
struct CpuGanTrainer {
    H host;

    struct GanLosses train_step(const &Tensor real, ?&Tensor labels);
    struct GanLosses train(const &Dataset data, unsigned long epochs, unsigned long batchSize);
    int  sync();
    void free();
};

generic<H: GanModel>
struct GanLosses CpuGanTrainer::train_step(const &Tensor real, ?&Tensor labels) {
    return self.host.train_step(real, labels);
}

generic<H: GanModel>
struct GanLosses CpuGanTrainer::train(const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    return self.host.train(data, epochs, batchSize);
}

generic<H: GanModel>
int CpuGanTrainer::sync() { return 1; }

generic<H: GanModel>
void CpuGanTrainer::free() { }

generic<H: GanModel>
struct CpuGanTrainer<H> cpu_gan_trainer_new(H g) {
    struct CpuGanTrainer<H> t;
    t.host = g;
    return t;
}

// ── rl.h's agents ───────────────────────────────────────────────────────────

// Dqn as a ValueAgent.
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

// PolicyAgent as a PolicyLearner.
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
