#pragma once
// SafeC Standard Library — Diffusion: noise schedules (EDM/Karras, DDPM
// linear) and samplers (DDPM ancestral, DDIM, DPM-Solver, DPM-Solver++).
//
// Operates on plain std::Tensor samples (1D or 2D — a sampler step is
// just elementwise arithmetic over whatever shape the caller's data/
// noise tensors already are; nothing here is image-shape-specific).
// Forward-only orchestration around a caller-supplied denoiser callback,
// like the rest of std/ml's newer additions (see attention.h).
//
// Convention: DDPM/DDIM/DPM-Solver/DPM-Solver++ all use the VP
// (variance-preserving) parametrization: alpha_t = sqrt(alphaBar_t),
// sigma_t = sqrt(1 - alphaBar_t), alpha_t^2 + sigma_t^2 = 1. EDM's Karras
// schedule is a separate, VE-flavored (variance-exploding) convention
// used by a different family of samplers not implemented here — it's
// included as a standalone schedule generator (edm_karras_sigmas) rather
// than force-fit into the VP sampler functions below.
//
// Scope note: DPM-Solver-v3 (Zheng et al. 2023) is NOT implemented — its
// distinguishing idea is an affine reparametrization of the ODE using
// "empirical model statistics" (EMS) calibrated from a real trained
// model over real data, which isn't something this environment has
// access to; a version that skipped the calibration step wouldn't
// actually implement the paper's contribution, so it's left out rather
// than faked. std/ml/ROADMAP.md tracks this.
#include <std/ml/tensor.h>

namespace std {

// Type-erased forms behind dpm_solver_2_step / dpm_solver_pp_2_step.
typedef fn &Tensor(const &Tensor x, float sigma, void* userData) EpsModelFn;
&Tensor dpm_solver_2_step_raw_(EpsModelFn model, void* userData,
                               const &Tensor x_cur, const &Tensor epsPred,
                               float alpha_cur, float sigma_cur,
                               float alpha_next, float sigma_next);
&Tensor dpm_solver_pp_2_step_raw_(EpsModelFn model, void* userData,
                                  const &Tensor x_cur, const &Tensor x0Pred,
                                  float alpha_cur, float sigma_cur,
                                  float alpha_next, float sigma_next);

// ── EDM (Karras et al., "Elucidating the Design Space...") schedule ────────
// sigma(i) = (sigmaMax^(1/rho) + i/(numSteps-1)*(sigmaMin^(1/rho)-sigmaMax^(1/rho)))^rho
// for i in [0,numSteps), followed by sigma(numSteps)=0 (the final,
// noise-free point). 'outSigmas' must have numSteps+1 slots (caller-
// allocated) — the standard Karras et al. sigma sequence used by EDM/
// k-diffusion-style samplers (e.g. Heun/Euler ancestral, not
// reimplemented here — this schedule alone is the deferred-scope-item
// std/ml/ROADMAP.md previously listed).
void edm_karras_sigmas(unsigned long numSteps, float sigmaMin, float sigmaMax, float rho,
                        &float outSigmas);

// ── DDPM linear beta schedule ────────────────────────────────────────────────
// beta_t = betaStart + (betaEnd-betaStart)*t/(numSteps-1), alpha_t = 1-beta_t,
// alphaBar_t = prod_{s<=t} alpha_s. 'outBeta'/'outAlphaBar' must each have
// numSteps slots (caller-allocated).
void ddpm_linear_schedule(unsigned long numSteps, float betaStart, float betaEnd,
                           &float outBeta, &float outAlphaBar);

// ── DDPM ancestral sampler step ──────────────────────────────────────────────
// x_{t-1} = (1/sqrt(1-beta_t)) * (x_t - beta_t/sqrt(1-alphaBar_t) * epsPred) + sqrt(beta_t) * noise
// Pass a zero tensor for 'noise' to get the deterministic (posterior-
// mean-only) step — useful for verification, matching how the rest of
// std/ml avoids baking in an RNG (see rnn.h's header comment).
&Tensor ddpm_sampler_step(const &Tensor x_t, const &Tensor epsPred,
                           float beta_t, float alphaBar_t, const &Tensor noise);

// ── DDIM deterministic sampler step (eta=0) ─────────────────────────────────
// x0Pred = (x_t - sqrt(1-alphaBar_t)*epsPred) / sqrt(alphaBar_t)
// x_{t-1} = sqrt(alphaBar_prev)*x0Pred + sqrt(1-alphaBar_prev)*epsPred
&Tensor ddim_sampler_step(const &Tensor x_t, const &Tensor epsPred,
                           float alphaBar_t, float alphaBar_prev);

// ── DPM-Solver (Lu et al. 2022): eps-prediction exponential integrator ─────
// in log-SNR space. lambda_t = log(alpha_t/sigma_t); order-1
// (dpm_solver_1_step) is the paper's eq. 4.3 first-order update. Note
// this is NOT the same update as ddim_sampler_step — that equivalence
// holds for the *data*-prediction form below (dpm_solver_pp_1_step), not
// this eps-prediction one; see diffusion.sc's header comment.
&Tensor dpm_solver_1_step(const &Tensor x_cur, const &Tensor epsPred,
                           float alpha_cur, float sigma_cur,
                           float alpha_next, float sigma_next);

// Order-2 (midpoint variant, r1=1/2): evaluates the eps-model a second
// time at the log-SNR midpoint between the current and next step for a
// higher-accuracy update. 'model' is called internally as
// model(x_s, sigma_s, userData) at the midpoint sample/noise-level.
// 'userData' stays 'void*' — deliberately type-erased caller context,
// same as std/gui's callback userData (see gui_widget.h's field comment).
// 'userData' is passed through to 'model' unchanged; the callback and its
// context are checked against each other at compile time.
generic<T>
&Tensor dpm_solver_2_step(fn &Tensor(const &Tensor, float, ?&T) model, ?&T userData,
                           const &Tensor x_cur, const &Tensor epsPred,
                           float alpha_cur, float sigma_cur,
                           float alpha_next, float sigma_next) {
    unsafe {
        return dpm_solver_2_step_raw_((EpsModelFn)model, (void*)userData, x_cur, epsPred,
                                      alpha_cur, sigma_cur, alpha_next, sigma_next);
    }
}

// ── DPM-Solver++ (Lu et al. 2022b): data(x0)-prediction parametrization ────
// Same exponential-integrator idea as DPM-Solver, reformulated around a
// predicted x0 (the denoised sample) instead of a predicted eps (the
// noise). Order-1 is provably identical to ddim_sampler_step — see
// diffusion.sc's header comment for the derivation.
&Tensor dpm_solver_pp_1_step(const &Tensor x_cur, const &Tensor x0Pred,
                              float alpha_cur, float sigma_cur,
                              float alpha_next, float sigma_next);

// Order-2 midpoint variant, x0-prediction form. 'model' is called
// internally as model(x_s, sigma_s, userData) at the log-SNR midpoint,
// returning a predicted x0 (not eps).
generic<T>
&Tensor dpm_solver_pp_2_step(fn &Tensor(const &Tensor, float, ?&T) model, ?&T userData,
                              const &Tensor x_cur, const &Tensor x0Pred,
                              float alpha_cur, float sigma_cur,
                              float alpha_next, float sigma_next) {
    unsafe {
        return dpm_solver_pp_2_step_raw_((EpsModelFn)model, (void*)userData, x_cur, x0Pred,
                                         alpha_cur, sigma_cur, alpha_next, sigma_next);
    }
}

} // namespace std
