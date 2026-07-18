// SafeC Standard Library — Minimum-phase implementation (see minphase.h)
#pragma once
#include <std/dsp/minphase.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/math.h>
#include <std/mem.h>

namespace std {

void minimum_phase(const double* x, unsigned long n, unsigned long fftLen, double* outMinPhase) {
    struct Complex* spec;
    unsafe { spec = (struct Complex*)alloc(fftLen * (unsigned long)sizeof(struct Complex)); }

    unsigned long i = 0UL;
    while (i < fftLen) {
        double v = 0.0;
        if (i < n) {
            unsafe { v = x[i]; }
        }
        unsafe { spec[i] = complex_new(v, 0.0); }
        i = i + 1UL;
    }
    fft(spec, fftLen);

    // Log-magnitude spectrum, in place.
    i = 0UL;
    while (i < fftLen) {
        struct Complex xi;
        unsafe { xi = spec[i]; }
        double mag = xi.abs();
        double logMag = log_d(mag + 1e-9);
        unsafe { spec[i] = complex_new(logMag, 0.0); }
        i = i + 1UL;
    }

    // Real cepstrum = IFFT(log-magnitude), in place.
    ifft(spec, fftLen);

    // Homomorphic ("minimum-phase") window: fold the anti-causal half of
    // the cepstrum onto the causal half by doubling it, and zero the
    // anti-causal half entirely — this is what makes the reconstructed
    // signal minimum-phase.
    unsigned long half = fftLen / 2UL;
    i = 0UL;
    while (i < fftLen) {
        struct Complex ci;
        unsafe { ci = spec[i]; }
        double cre;
        unsafe { cre = ci.re; }
        double wv = 0.0;
        if (i == 0UL) {
            wv = cre;
        } else if (i < half) {
            wv = 2.0 * cre;
        } else if (i == half) {
            wv = cre;
        }
        unsafe { spec[i] = complex_new(wv, 0.0); }
        i = i + 1UL;
    }

    fft(spec, fftLen);

    // Exponentiate the (now complex) cepstrum spectrum back into the
    // minimum-phase spectrum: exp(a+bi) = e^a * (cos(b) + i*sin(b)).
    i = 0UL;
    while (i < fftLen) {
        struct Complex wi;
        unsafe { wi = spec[i]; }
        double a;
        double b;
        unsafe { a = wi.re; b = wi.im; }
        double ea = exp_d(a);
        unsafe { spec[i] = complex_new(ea * cos_d(b), ea * sin_d(b)); }
        i = i + 1UL;
    }

    ifft(spec, fftLen);
    i = 0UL;
    while (i < fftLen) {
        struct Complex wi;
        unsafe { wi = spec[i]; }
        double re;
        unsafe { re = wi.re; }
        unsafe { outMinPhase[i] = re; }
        i = i + 1UL;
    }

    unsafe { dealloc((void*)spec); }
}

} // namespace std
