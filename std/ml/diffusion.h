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

// A denoiser the order-2 solvers call at their midpoint: an eps-prediction
// model for dpm_solver_2_step, an x0-prediction one for
// dpm_solver_pp_2_step. Any struct with this method conforms.
trait DenoiseModel {
    &FloatTensor predict(const &FloatTensor x, float sigma);
}

// Internal: the order-2 solvers' midpoint sample and noise level (pp: the
// x0-prediction form), and the update from the midpoint prediction.
struct DpmMidpoint {
    &FloatTensor x_s;
    float sigma_s;
    float h;            // lambda_next - lambda_cur
};
struct DpmMidpoint dpm_midpoint_(const &FloatTensor x_cur, const &FloatTensor pred, float alpha_cur, float sigma_cur,
                                 float alpha_next, float sigma_next, int pp);
&FloatTensor dpm_finish_(const &FloatTensor x_cur, const &FloatTensor predMid, float sigma_cur, float alpha_next,
                    float sigma_next, float h, int pp);
void dpm_drop_(&FloatTensor t);

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
&FloatTensor ddpm_sampler_step(const &FloatTensor x_t, const &FloatTensor epsPred,
                           float beta_t, float alphaBar_t, const &FloatTensor noise);

// ── DDIM deterministic sampler step (eta=0) ─────────────────────────────────
// x0Pred = (x_t - sqrt(1-alphaBar_t)*epsPred) / sqrt(alphaBar_t)
// x_{t-1} = sqrt(alphaBar_prev)*x0Pred + sqrt(1-alphaBar_prev)*epsPred
&FloatTensor ddim_sampler_step(const &FloatTensor x_t, const &FloatTensor epsPred,
                           float alphaBar_t, float alphaBar_prev);

// ── DPM-Solver (Lu et al. 2022): eps-prediction exponential integrator ─────
// in log-SNR space. lambda_t = log(alpha_t/sigma_t); order-1
// (dpm_solver_1_step) is the paper's eq. 4.3 first-order update. Note
// this is NOT the same update as ddim_sampler_step — that equivalence
// holds for the *data*-prediction form below (dpm_solver_pp_1_step), not
// this eps-prediction one; see diffusion.sc's header comment.
&FloatTensor dpm_solver_1_step(const &FloatTensor x_cur, const &FloatTensor epsPred,
                           float alpha_cur, float sigma_cur,
                           float alpha_next, float sigma_next);

// Order-2 (midpoint variant, r1=1/2): evaluates the eps-model a second
// time at the log-SNR midpoint between the current and next step for a
// higher-accuracy update: model.predict(x_s, sigma_s) at the midpoint
// sample / noise level, returning the predicted eps.
generic<M: DenoiseModel>
&FloatTensor dpm_solver_2_step(&M model, const &FloatTensor x_cur, const &FloatTensor epsPred,
                          float alpha_cur, float sigma_cur, float alpha_next, float sigma_next) {
    struct DpmMidpoint mp = dpm_midpoint_(x_cur, epsPred, alpha_cur, sigma_cur, alpha_next, sigma_next, 0);
    &FloatTensor eps_s = model.predict(mp.x_s, mp.sigma_s);
    &FloatTensor out = dpm_finish_(x_cur, eps_s, sigma_cur, alpha_next, sigma_next, mp.h, 0);
    dpm_drop_(mp.x_s);
    dpm_drop_(eps_s);
    return out;
}

// ── DPM-Solver++ (Lu et al. 2022b): data(x0)-prediction parametrization ────
// Same exponential-integrator idea as DPM-Solver, reformulated around a
// predicted x0 (the denoised sample) instead of a predicted eps (the
// noise). Order-1 is provably identical to ddim_sampler_step — see
// diffusion.sc's header comment for the derivation.
&FloatTensor dpm_solver_pp_1_step(const &FloatTensor x_cur, const &FloatTensor x0Pred,
                              float alpha_cur, float sigma_cur,
                              float alpha_next, float sigma_next);

// Order-2 midpoint variant, x0-prediction form: model.predict(x_s, sigma_s)
// at the log-SNR midpoint returns a predicted x0 (not eps).
generic<M: DenoiseModel>
&FloatTensor dpm_solver_pp_2_step(&M model, const &FloatTensor x_cur, const &FloatTensor x0Pred,
                             float alpha_cur, float sigma_cur, float alpha_next, float sigma_next) {
    struct DpmMidpoint mp = dpm_midpoint_(x_cur, x0Pred, alpha_cur, sigma_cur, alpha_next, sigma_next, 1);
    &FloatTensor x0_s = model.predict(mp.x_s, mp.sigma_s);
    &FloatTensor out = dpm_finish_(x_cur, x0_s, sigma_cur, alpha_next, sigma_next, mp.h, 1);
    dpm_drop_(mp.x_s);
    dpm_drop_(x0_s);
    return out;
}

} // namespace std
