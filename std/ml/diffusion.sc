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
#include <std/ml/tensor.sc>
#include <std/math.h>
#include <std/math.sc>
#include <std/mem.sc>

namespace std {

void edm_karras_sigmas(unsigned long numSteps, float sigmaMin, float sigmaMax, float rho,
                        float* outSigmas) {
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
                           float* outBeta, float* outAlphaBar) {
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

&Tensor ddpm_sampler_step(const &Tensor x_t, const &Tensor epsPred,
                           float beta_t, float alphaBar_t, const &Tensor noise) {
    float invSqrtAlpha = (float)1.0 / sqrt_f((float)1.0 - beta_t);
    float coef = beta_t / sqrt_f((float)1.0 - alphaBar_t);
    struct Tensor* scaledEps = tensor_scale(epsPred, coef);
    struct Tensor* diff = tensor_sub(x_t, scaledEps);
    struct Tensor* mean = tensor_scale(diff, invSqrtAlpha);
    float sigma_t = sqrt_f(beta_t);
    struct Tensor* scaledNoise = tensor_scale(noise, sigma_t);
    struct Tensor* out = tensor_add(mean, scaledNoise);

    scaledEps->free(); diff->free(); mean->free(); scaledNoise->free();
    unsafe { free((void*)scaledEps); free((void*)diff); free((void*)mean); free((void*)scaledNoise); }
    return out;
}

&Tensor ddim_sampler_step(const &Tensor x_t, const &Tensor epsPred,
                           float alphaBar_t, float alphaBar_prev) {
    float sqrtAlphaBar_t = sqrt_f(alphaBar_t);
    float sqrtOneMinusAlphaBar_t = sqrt_f((float)1.0 - alphaBar_t);
    struct Tensor* scaledEps1 = tensor_scale(epsPred, sqrtOneMinusAlphaBar_t);
    struct Tensor* numer = tensor_sub(x_t, scaledEps1);
    struct Tensor* x0Pred = tensor_scale(numer, (float)1.0 / sqrtAlphaBar_t);

    float sqrtAlphaBarPrev = sqrt_f(alphaBar_prev);
    float sqrtOneMinusAlphaBarPrev = sqrt_f((float)1.0 - alphaBar_prev);
    struct Tensor* term1 = tensor_scale(x0Pred, sqrtAlphaBarPrev);
    struct Tensor* term2 = tensor_scale(epsPred, sqrtOneMinusAlphaBarPrev);
    struct Tensor* out = tensor_add(term1, term2);

    scaledEps1->free(); numer->free(); x0Pred->free(); term1->free(); term2->free();
    unsafe {
        free((void*)scaledEps1); free((void*)numer); free((void*)x0Pred);
        free((void*)term1); free((void*)term2);
    }
    return out;
}

&Tensor dpm_solver_1_step(const &Tensor x_cur, const &Tensor epsPred,
                           float alpha_cur, float sigma_cur,
                           float alpha_next, float sigma_next) {
    float lambda_cur = log_f(alpha_cur / sigma_cur);
    float lambda_next = log_f(alpha_next / sigma_next);
    float h = lambda_next - lambda_cur;
    float expm1h = exp_f(h) - (float)1.0;

    struct Tensor* term1 = tensor_scale(x_cur, sigma_next / sigma_cur);
    struct Tensor* term2 = tensor_scale(epsPred, alpha_next * expm1h);
    struct Tensor* out = tensor_sub(term1, term2);

    term1->free(); term2->free();
    unsafe { free((void*)term1); free((void*)term2); }
    return out;
}

&Tensor dpm_solver_2_step(EpsModelFn model, void* userData,
                           const &Tensor x_cur, const &Tensor epsPred,
                           float alpha_cur, float sigma_cur,
                           float alpha_next, float sigma_next) {
    float lambda_cur = log_f(alpha_cur / sigma_cur);
    float lambda_next = log_f(alpha_next / sigma_next);
    float h = lambda_next - lambda_cur;
    float lambda_s = lambda_cur + (float)0.5 * h;
    float sigma_s = (float)1.0 / sqrt_f(exp_f((float)2.0 * lambda_s) + (float)1.0);
    float alpha_s = exp_f(lambda_s) * sigma_s;
    float expm1Half = exp_f((float)0.5 * h) - (float)1.0;

    struct Tensor* sTerm1 = tensor_scale(x_cur, sigma_s / sigma_cur);
    struct Tensor* sTerm2 = tensor_scale(epsPred, alpha_s * expm1Half);
    struct Tensor* x_s = tensor_sub(sTerm1, sTerm2);

    struct Tensor* eps_s;
    unsafe { eps_s = model(x_s, sigma_s, userData); }

    float expm1h = exp_f(h) - (float)1.0;
    struct Tensor* nTerm1 = tensor_scale(x_cur, sigma_next / sigma_cur);
    struct Tensor* nTerm2 = tensor_scale(eps_s, alpha_next * expm1h);
    struct Tensor* out = tensor_sub(nTerm1, nTerm2);

    sTerm1->free(); sTerm2->free(); x_s->free(); eps_s->free(); nTerm1->free(); nTerm2->free();
    unsafe {
        free((void*)sTerm1); free((void*)sTerm2); free((void*)x_s);
        free((void*)eps_s); free((void*)nTerm1); free((void*)nTerm2);
    }
    return out;
}

&Tensor dpm_solver_pp_1_step(const &Tensor x_cur, const &Tensor x0Pred,
                              float alpha_cur, float sigma_cur,
                              float alpha_next, float sigma_next) {
    float lambda_cur = log_f(alpha_cur / sigma_cur);
    float lambda_next = log_f(alpha_next / sigma_next);
    float h = lambda_next - lambda_cur;
    float expm1NegH = exp_f(-h) - (float)1.0;

    struct Tensor* term1 = tensor_scale(x_cur, sigma_next / sigma_cur);
    struct Tensor* term2 = tensor_scale(x0Pred, alpha_next * expm1NegH);
    struct Tensor* out = tensor_sub(term1, term2);

    term1->free(); term2->free();
    unsafe { free((void*)term1); free((void*)term2); }
    return out;
}

&Tensor dpm_solver_pp_2_step(DataModelFn model, void* userData,
                              const &Tensor x_cur, const &Tensor x0Pred,
                              float alpha_cur, float sigma_cur,
                              float alpha_next, float sigma_next) {
    float lambda_cur = log_f(alpha_cur / sigma_cur);
    float lambda_next = log_f(alpha_next / sigma_next);
    float h = lambda_next - lambda_cur;
    float lambda_s = lambda_cur + (float)0.5 * h;
    float sigma_s = (float)1.0 / sqrt_f(exp_f((float)2.0 * lambda_s) + (float)1.0);
    float alpha_s = exp_f(lambda_s) * sigma_s;
    float expm1NegHalf = exp_f((float)-0.5 * h) - (float)1.0;

    struct Tensor* sTerm1 = tensor_scale(x_cur, sigma_s / sigma_cur);
    struct Tensor* sTerm2 = tensor_scale(x0Pred, alpha_s * expm1NegHalf);
    struct Tensor* x_s = tensor_sub(sTerm1, sTerm2);

    struct Tensor* x0_s;
    unsafe { x0_s = model(x_s, sigma_s, userData); }

    float expm1NegH = exp_f(-h) - (float)1.0;
    struct Tensor* nTerm1 = tensor_scale(x_cur, sigma_next / sigma_cur);
    struct Tensor* nTerm2 = tensor_scale(x0_s, alpha_next * expm1NegH);
    struct Tensor* out = tensor_sub(nTerm1, nTerm2);

    sTerm1->free(); sTerm2->free(); x_s->free(); x0_s->free(); nTerm1->free(); nTerm2->free();
    unsafe {
        free((void*)sTerm1); free((void*)sTerm2); free((void*)x_s);
        free((void*)x0_s); free((void*)nTerm1); free((void*)nTerm2);
    }
    return out;
}

} // namespace std
