// SafeC Standard Library — Diffusion implementation (see diffusion.h).
//
// Derivation note (DPM-Solver++ order 1 == DDIM): substituting
// eps = (x_cur - alpha_cur*x0Pred)/sigma_cur into DDIM's update
// x_next = alpha_next*x0Pred + sigma_next*eps and simplifying with
// lambda_t = log(alpha_t/sigma_t), h = lambda_next - lambda_cur gives
// exactly x_next = (sigma_next/sigma_cur)*x_cur - alpha_next*(exp(-h)-1)*x0Pred
// — dpm_solver_pp_1_step below — so the two must agree bit-for-bit
// (mod float rounding) for the same (x0Pred, eps) pair. Verified in
// this library's ml test suite.
#pragma once
#include <std/ml/diffusion.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor.h>
#include <std/math.h>
#include <std/math.h>
#include <std/mem.h>

namespace std {

void edm_karras_sigmas(unsigned long numSteps, float sigmaMin, float sigmaMax, float rho,
                        &float outSigmas) {
    float minInvRho = pow_f(sigmaMin, (float)1.0 / rho);
    float maxInvRho = pow_f(sigmaMax, (float)1.0 / rho);
    unsafe {
        unsigned long i = 0UL;
        while (i < numSteps) {
            float frac = (float)i / (float)(numSteps - 1UL);
            float s = maxInvRho + frac * (minInvRho - maxInvRho);
            outSigmas[i] = pow_f(s, rho);
            i = i + 1UL;
        }
        outSigmas[numSteps] = (float)0.0;
    }
}

void ddpm_linear_schedule(unsigned long numSteps, float betaStart, float betaEnd,
                           &float outBeta, &float outAlphaBar) {
    unsafe {
        float alphaBarAcc = (float)1.0;
        unsigned long t = 0UL;
        while (t < numSteps) {
            float frac = (float)t / (float)(numSteps - 1UL);
            float beta = betaStart + frac * (betaEnd - betaStart);
            outBeta[t] = beta;
            alphaBarAcc = alphaBarAcc * ((float)1.0 - beta);
            outAlphaBar[t] = alphaBarAcc;
            t = t + 1UL;
        }
    }
}

&FloatTensor ddpm_sampler_step(const &FloatTensor x_t, const &FloatTensor epsPred,
                           float beta_t, float alphaBar_t, const &FloatTensor noise) {
    float invSqrtAlpha = (float)1.0 / sqrt_f((float)1.0 - beta_t);
    float coef = beta_t / sqrt_f((float)1.0 - alphaBar_t);
    &FloatTensor scaledEps = epsPred.scale(coef);
    &FloatTensor diff = tensor_sub(x_t, scaledEps);
    &FloatTensor mean = diff.scale(invSqrtAlpha);
    float sigma_t = sqrt_f(beta_t);
    &FloatTensor scaledNoise = noise.scale(sigma_t);
    &FloatTensor out = tensor_add(mean, scaledNoise);

    scaledEps->free(); diff->free(); mean->free(); scaledNoise->free();
    unsafe { free((unsigned char*)scaledEps); free((unsigned char*)diff); free((unsigned char*)mean); free((unsigned char*)scaledNoise); }
    return out;
}

&FloatTensor ddim_sampler_step(const &FloatTensor x_t, const &FloatTensor epsPred,
                           float alphaBar_t, float alphaBar_prev) {
    float sqrtAlphaBar_t = sqrt_f(alphaBar_t);
    float sqrtOneMinusAlphaBar_t = sqrt_f((float)1.0 - alphaBar_t);
    &FloatTensor scaledEps1 = epsPred.scale(sqrtOneMinusAlphaBar_t);
    &FloatTensor numer = tensor_sub(x_t, scaledEps1);
    &FloatTensor x0Pred = numer.scale((float)1.0 / sqrtAlphaBar_t);

    float sqrtAlphaBarPrev = sqrt_f(alphaBar_prev);
    float sqrtOneMinusAlphaBarPrev = sqrt_f((float)1.0 - alphaBar_prev);
    &FloatTensor term1 = x0Pred.scale(sqrtAlphaBarPrev);
    &FloatTensor term2 = epsPred.scale(sqrtOneMinusAlphaBarPrev);
    &FloatTensor out = tensor_add(term1, term2);

    scaledEps1->free(); numer->free(); x0Pred->free(); term1->free(); term2->free();
    unsafe {
        free((unsigned char*)scaledEps1); free((unsigned char*)numer); free((unsigned char*)x0Pred);
        free((unsigned char*)term1); free((unsigned char*)term2);
    }
    return out;
}

&FloatTensor dpm_solver_1_step(const &FloatTensor x_cur, const &FloatTensor epsPred,
                           float alpha_cur, float sigma_cur,
                           float alpha_next, float sigma_next) {
    float lambda_cur = log_f(alpha_cur / sigma_cur);
    float lambda_next = log_f(alpha_next / sigma_next);
    float h = lambda_next - lambda_cur;
    float expm1h = exp_f(h) - (float)1.0;

    &FloatTensor term1 = x_cur.scale(sigma_next / sigma_cur);
    &FloatTensor term2 = epsPred.scale(alpha_next * expm1h);
    &FloatTensor out = tensor_sub(term1, term2);

    term1->free(); term2->free();
    unsafe { free((unsigned char*)term1); free((unsigned char*)term2); }
    return out;
}

struct DpmMidpoint dpm_midpoint_(const &FloatTensor x_cur, const &FloatTensor pred, float alpha_cur, float sigma_cur,
                                 float alpha_next, float sigma_next, int pp) {
    float lambda_cur = log_f(alpha_cur / sigma_cur);
    float lambda_next = log_f(alpha_next / sigma_next);
    float h = lambda_next - lambda_cur;
    float lambda_s = lambda_cur + (float)0.5 * h;
    float sigma_s = (float)1.0 / sqrt_f(exp_f((float)2.0 * lambda_s) + (float)1.0);
    float alpha_s = exp_f(lambda_s) * sigma_s;
    // eps form: x_s = (sigma_s/sigma) x - alpha_s (e^{h/2} - 1) eps
    // x0 form:  x_s = (sigma_s/sigma) x - alpha_s (e^{-h/2} - 1) x0
    float expm1Half = pp ? exp_f((float)-0.5 * h) - (float)1.0 : exp_f((float)0.5 * h) - (float)1.0;
    &FloatTensor sTerm1 = x_cur.scale(sigma_s / sigma_cur);
    &FloatTensor sTerm2 = pred.scale(alpha_s * expm1Half);
    struct DpmMidpoint mp;
    mp.x_s = tensor_sub(sTerm1, sTerm2);
    mp.sigma_s = sigma_s;
    mp.h = h;
    dpm_drop_(sTerm1);
    dpm_drop_(sTerm2);
    return mp;
}

&FloatTensor dpm_finish_(const &FloatTensor x_cur, const &FloatTensor predMid, float sigma_cur, float alpha_next,
                    float sigma_next, float h, int pp) {
    float expm1 = pp ? exp_f(-h) - (float)1.0 : exp_f(h) - (float)1.0;
    &FloatTensor nTerm1 = x_cur.scale(sigma_next / sigma_cur);
    &FloatTensor nTerm2 = predMid.scale(alpha_next * expm1);
    &FloatTensor out = tensor_sub(nTerm1, nTerm2);
    dpm_drop_(nTerm1);
    dpm_drop_(nTerm2);
    return out;
}

void dpm_drop_(&FloatTensor t) {
    // A heap tensor: destroyed in place through the reference, then its
    // memory released (a Box the checker can't see).
    unsafe { t->free(); }
    unsafe { free((unsigned char*)t); }
}

&FloatTensor dpm_solver_pp_1_step(const &FloatTensor x_cur, const &FloatTensor x0Pred,
                              float alpha_cur, float sigma_cur,
                              float alpha_next, float sigma_next) {
    float lambda_cur = log_f(alpha_cur / sigma_cur);
    float lambda_next = log_f(alpha_next / sigma_next);
    float h = lambda_next - lambda_cur;
    float expm1NegH = exp_f(-h) - (float)1.0;

    &FloatTensor term1 = x_cur.scale(sigma_next / sigma_cur);
    &FloatTensor term2 = x0Pred.scale(alpha_next * expm1NegH);
    &FloatTensor out = tensor_sub(term1, term2);

    term1->free(); term2->free();
    unsafe { free((unsigned char*)term1); free((unsigned char*)term2); }
    return out;
}

} // namespace std
