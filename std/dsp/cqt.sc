// SafeC Standard Library — CQT implementation (see cqt.h)
#pragma once
#include <std/dsp/cqt.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>

#define SAFEC_DSP_PI_CQT 3.14159265358979323846

namespace std {

void cqt_forward(const double* signal, unsigned long signalLen, double fs,
                  double fmin, unsigned long numBins, unsigned long binsPerOctave,
                  struct Complex* out) {
    double q = 1.0 / (pow_d(2.0, 1.0 / (double)binsPerOctave) - 1.0);

    unsigned long k = 0UL;
    while (k < numBins) {
        double fk = fmin * pow_d(2.0, (double)k / (double)binsPerOctave);
        unsigned long nk = (unsigned long)(q * fs / fk + 0.5);
        if (nk < 1UL) { nk = 1UL; }
        if (nk > signalLen) { nk = signalLen; }

        double denom = 1.0;
        if (nk > 1UL) { denom = (double)(nk - 1UL); }

        double sumRe = 0.0;
        double sumIm = 0.0;
        unsigned long n = 0UL;
        while (n < nk) {
            double s;
            unsafe { s = signal[n]; }
            double w = 0.5 * (1.0 - cos_d(2.0 * SAFEC_DSP_PI_CQT * (double)n / denom));
            double angle = -2.0 * SAFEC_DSP_PI_CQT * q * (double)n / (double)nk;
            double kre = w * cos_d(angle) / (double)nk;
            double kim = w * sin_d(angle) / (double)nk;
            sumRe = sumRe + s * kre;
            sumIm = sumIm + s * kim;
            n = n + 1UL;
        }
        unsafe { out[k] = complex_new(sumRe, sumIm); }
        k = k + 1UL;
    }
}

} // namespace std
