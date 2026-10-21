// SafeC Standard Library — the CPU backend of the training traits (see train_cpu.h).
#pragma once
#include <std/ml/train_cpu.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

static &Tensor tc_mlp_ce_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(m.forward(x), y);
}
static &Tensor tc_mlp_mse_(&Mlp m, const &Tensor x, const &Tensor y) { return tensor_mse_loss(m.forward(x), y); }
static &Tensor tc_mlp_bce_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_bce_logits_loss(m.forward(x), y);
}
static &Tensor tc_conv_ce_(&ConvNet n, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(n.forward(x), y);
}
static &Tensor tc_conv_mse_(&ConvNet n, const &Tensor x, const &Tensor y) {
    return tensor_mse_loss(n.forward(x), y);
}
static &Tensor tc_conv_bce_(&ConvNet n, const &Tensor x, const &Tensor y) {
    return tensor_bce_logits_loss(n.forward(x), y);
}

struct CpuTrainer<struct Mlp> cpu_mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt) {
    struct CpuTrainer<struct Mlp> t;
    t.host = m;
    t.opt = optimizer_from(opt);
    t.host.add_params(&t.opt);
    t.lossFn = tc_mlp_ce_;
    if (loss == TRAIN_LOSS_MSE) t.lossFn = tc_mlp_mse_;
    if (loss == TRAIN_LOSS_BCE) t.lossFn = tc_mlp_bce_;
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

struct CpuTrainer<struct ConvNet> cpu_convnet_trainer_new(struct ConvNet n, int loss, const &Optimizer opt) {
    struct CpuTrainer<struct ConvNet> t;
    t.host = n;
    t.opt = optimizer_from(opt);
    t.host.add_params(&t.opt);
    t.lossFn = tc_conv_ce_;
    if (loss == TRAIN_LOSS_MSE) t.lossFn = tc_conv_mse_;
    if (loss == TRAIN_LOSS_BCE) t.lossFn = tc_conv_bce_;
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

struct CpuTrainer<struct Vae> cpu_vae_trainer_new(struct Vae v, const &Optimizer opt) {
    struct CpuTrainer<struct Vae> t;
    t.host = v;
    t.opt = optimizer_from(opt);
    t.host.add_params(&t.opt);
    t.lossFn = vae_loss;
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

struct CpuTrainer<struct ConvVae> cpu_conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt) {
    struct CpuTrainer<struct ConvVae> t;
    t.host = v;
    t.opt = optimizer_from(opt);
    t.host.add_params(&t.opt);
    t.lossFn = conv_vae_loss;
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

struct CpuTrainer<struct ConvAutoencoder> cpu_conv_ae_trainer_new(struct ConvAutoencoder ae, const &Optimizer opt) {
    struct CpuTrainer<struct ConvAutoencoder> t;
    t.host = ae;
    t.opt = optimizer_from(opt);
    t.host.add_params(&t.opt);
    t.lossFn = conv_autoencoder_loss;
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

// CpuGanTrainer calls these through its fn fields.
static struct GanLosses tc_gan_step_(&Gan g, const &Tensor real, ?&Tensor labels) { return g.train_step(real, labels); }
static struct GanLosses tc_gan_train_(&Gan g, const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    return g.train(data, epochs, batchSize);
}
static struct GanLosses tc_dcgan_step_(&Dcgan g, const &Tensor real, ?&Tensor labels) {
    return g.train_step(real, labels);
}
static struct GanLosses tc_dcgan_train_(&Dcgan g, const &Dataset data, unsigned long epochs,
                                        unsigned long batchSize) {
    return g.train(data, epochs, batchSize);
}

struct CpuGanTrainer<struct Gan> cpu_gan_trainer_new(struct Gan g) {
    struct CpuGanTrainer<struct Gan> t;
    t.host = g;
    t.stepFn = tc_gan_step_;
    t.trainFn = tc_gan_train_;
    return t;
}

struct CpuGanTrainer<struct Dcgan> cpu_dcgan_trainer_new(struct Dcgan g) {
    struct CpuGanTrainer<struct Dcgan> t;
    t.host = g;
    t.stepFn = tc_dcgan_step_;
    t.trainFn = tc_dcgan_train_;
    return t;
}

// ── DQN ──────────────────────────────────────────────────────────────────────

struct CpuDqnAgent cpu_dqn_agent_new(struct Dqn d) {
    struct CpuDqnAgent a;
    a.host = d;
    return a;
}

float CpuDqnAgent::epsilon() { return self.host.epsilon(); }

unsigned long CpuDqnAgent::act(const &Tensor obs, int explore) { return self.host.act(obs, explore); }

int CpuDqnAgent::observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal) {
    unsigned long before = self.host.updates;
    self.host.observe(obs, action, reward, next, terminal);
    return self.host.updates > before ? 1 : 0;
}

float CpuDqnAgent::last_loss() { return self.host.lastLoss; }

unsigned long CpuDqnAgent::updates() { return self.host.updates; }

int CpuDqnAgent::sync() { return 1; }

void CpuDqnAgent::free() { }

// ── Policy gradient ──────────────────────────────────────────────────────────

struct CpuPgAgent cpu_pg_agent_new(struct PolicyAgent a) {
    struct CpuPgAgent p;
    p.host = a;
    return p;
}

unsigned long CpuPgAgent::act(const &Tensor obs, int sample) { return self.host.act(obs, sample); }

void CpuPgAgent::record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                        int truncated) {
    self.host.record(obs, action, reward, next, terminated, truncated);
}

float CpuPgAgent::update() { return self.host.update(); }

int CpuPgAgent::sync() { return 1; }

void CpuPgAgent::free() { }

} // namespace std
