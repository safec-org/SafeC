// SafeC Standard Library — the CPU backend of the training traits (see train_cpu.h).
#pragma once
#include <std/ml/train_cpu.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

static &Tensor tc_mlp_ce_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(mlp_forward(m, x), y);
}
static &Tensor tc_mlp_mse_(&Mlp m, const &Tensor x, const &Tensor y) { return tensor_mse_loss(mlp_forward(m, x), y); }
static &Tensor tc_mlp_bce_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_bce_logits_loss(mlp_forward(m, x), y);
}
static &Tensor tc_conv_ce_(&ConvNet n, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(convnet_forward(n, x), y);
}
static &Tensor tc_conv_mse_(&ConvNet n, const &Tensor x, const &Tensor y) {
    return tensor_mse_loss(convnet_forward(n, x), y);
}
static &Tensor tc_conv_bce_(&ConvNet n, const &Tensor x, const &Tensor y) {
    return tensor_bce_logits_loss(convnet_forward(n, x), y);
}

struct CpuTrainer<struct Mlp> cpu_mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt) {
    struct CpuTrainer<struct Mlp> t;
    t.host = m;
    t.opt = optimizer_from(opt);
    optimizer_add_mlp(&t.opt, &t.host);
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
    optimizer_add_convnet(&t.opt, &t.host);
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
    vae_add_params(&t.opt, &t.host);
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
    conv_vae_add_params(&t.opt, &t.host);
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
    conv_autoencoder_add_params(&t.opt, &t.host);
    t.lossFn = conv_autoencoder_loss;
    t.stepArena = tensor_arena_new(0UL);
    t.clipNorm = (float)0.0;
    t.active = 0;
    return t;
}

struct CpuGanTrainer<struct Gan> cpu_gan_trainer_new(struct Gan g) {
    struct CpuGanTrainer<struct Gan> t;
    t.host = g;
    t.stepFn = gan_train_step;
    t.trainFn = gan_train;
    return t;
}

struct CpuGanTrainer<struct Dcgan> cpu_dcgan_trainer_new(struct Dcgan g) {
    struct CpuGanTrainer<struct Dcgan> t;
    t.host = g;
    t.stepFn = dcgan_train_step;
    t.trainFn = dcgan_train;
    return t;
}

// ── DQN ──────────────────────────────────────────────────────────────────────

struct CpuDqnAgent cpu_dqn_agent_new(struct Dqn d) {
    struct CpuDqnAgent a;
    a.host = d;
    return a;
}

float CpuDqnAgent::epsilon() { return dqn_epsilon(&self.host); }

unsigned long CpuDqnAgent::act(const &Tensor obs, int explore) { return dqn_act(&self.host, obs, explore); }

int CpuDqnAgent::observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal) {
    unsigned long before = self.host.updates;
    dqn_observe(&self.host, obs, action, reward, next, terminal);
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

unsigned long CpuPgAgent::act(const &Tensor obs, int sample) { return pg_act(&self.host, obs, sample); }

void CpuPgAgent::record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                        int truncated) {
    pg_record(&self.host, obs, action, reward, next, terminated, truncated);
}

float CpuPgAgent::update() { return pg_update(&self.host); }

int CpuPgAgent::sync() { return 1; }

void CpuPgAgent::free() { }

} // namespace std
