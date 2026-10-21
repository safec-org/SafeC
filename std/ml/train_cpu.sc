// SafeC Standard Library — the CPU backend of the training traits (see train_cpu.h).
#pragma once
#include <std/ml/train_cpu.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

// ── DQN ──────────────────────────────────────────────────────────────────────

struct CpuDqnAgent CpuDqnAgent::new(struct Dqn d) {
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

struct CpuPgAgent CpuPgAgent::new(struct PolicyAgent a) {
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
