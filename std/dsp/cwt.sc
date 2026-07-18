// SafeC Standard Library — CWT implementation (see cwt.h)
#pragma once
#include <std/dsp/cwt.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>

#define SAFEC_DSP_PI_CWT 3.14159265358979323846

namespace std {

void cwt_morlet(const double* signal, unsigned long signalLen,
                 const double* scales, unsigned long numScales,
                 double w0, struct Complex* out) {
    double normConst = pow_d(SAFEC_DSP_PI_CWT, 0.0 - 0.25);

    unsigned long si = 0UL;
    while (si < numScales) {
        double s;
        unsafe { s = scales[si]; }
        unsigned long halfWidth = (unsigned long)(4.0 * s) + 1UL;
        double invSqrtS = 1.0 / sqrt_d(s);

        unsigned long b = 0UL;
        while (b < signalLen) {
            unsigned long nStart = 0UL;
            if (b >= halfWidth) { nStart = b - halfWidth; }
            unsigned long nEnd = b + halfWidth;
            if (nEnd >= signalLen) { nEnd = signalLen - 1UL; }

            double sumRe = 0.0;
            double sumIm = 0.0;
            unsigned long n = nStart;
            while (n <= nEnd) {
                double t = ((double)n - (double)b) / s;
                double env = normConst * exp_d(-0.5 * t * t);
                double c = cos_d(w0 * t);
                double sn = sin_d(w0 * t);
                double xn;
                unsafe { xn = signal[n]; }
                // conj(psi(t)) = env * (cos(w0*t) - i*sin(w0*t))
                sumRe = sumRe + xn * env * c;
                sumIm = sumIm - xn * env * sn;
                n = n + 1UL;
            }
            unsafe { out[si * signalLen + b] = complex_new(sumRe * invSqrtS, sumIm * invSqrtS); }
            b = b + 1UL;
        }
        si = si + 1UL;
    }
}

} // namespace std
