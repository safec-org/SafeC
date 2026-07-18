// SafeC Standard Library — General bilinear transform + Z-domain response
// implementation (see ztransform.h)
#pragma once
#include <std/dsp/ztransform.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>
#include <std/mem.h>

#define SAFEC_DSP_TWO_PI_ZT 6.28318530717958647692

namespace std {

// Multiply the ascending-power-of-z polynomial 'poly' (length polyLen) by
// the linear factor (z + factor), writing the length-(polyLen+1) result
// into 'out'.
static void poly_mul_linear_(const double* poly, unsigned long polyLen, double factor, double* out) {
    unsigned long j = 0UL;
    while (j <= polyLen) {
        unsafe { out[j] = 0.0; }
        j = j + 1UL;
    }
    j = 0UL;
    while (j < polyLen) {
        double c;
        unsafe { c = poly[j]; }
        unsafe {
            out[j] = out[j] + c * factor;
            out[j + 1UL] = out[j + 1UL] + c;
        }
        j = j + 1UL;
    }
}

void bilinear_nth_order(const double* B, const double* A, unsigned long order,
                         double fs, double* bOut, double* aOut) {
    unsigned long m = order;
    double c = 2.0 * fs;

    double* numPoly;
    double* denPoly;
    double* bufA;
    double* bufB;
    unsafe {
        numPoly = (double*)alloc((m + 1UL) * (unsigned long)sizeof(double));
        denPoly = (double*)alloc((m + 1UL) * (unsigned long)sizeof(double));
        bufA    = (double*)alloc((m + 1UL) * (unsigned long)sizeof(double));
        bufB    = (double*)alloc((m + 1UL) * (unsigned long)sizeof(double));
    }
    unsigned long j = 0UL;
    while (j <= m) {
        unsafe { numPoly[j] = 0.0; denPoly[j] = 0.0; }
        j = j + 1UL;
    }

    double ck = 1.0; // c^k, built up incrementally
    unsigned long k = 0UL;
    while (k <= m) {
        // Build (z-1)^k * (z+1)^(m-k), ascending powers, into bufA.
        unsigned long curLen = 1UL;
        unsafe { bufA[0] = 1.0; }
        unsigned long t = 0UL;
        while (t < k) {
            unsafe { poly_mul_linear_(bufA, curLen, -1.0, bufB); }
            curLen = curLen + 1UL;
            unsigned long cpy = 0UL;
            while (cpy < curLen) {
                unsafe { bufA[cpy] = bufB[cpy]; }
                cpy = cpy + 1UL;
            }
            t = t + 1UL;
        }
        unsigned long remain = m - k;
        t = 0UL;
        while (t < remain) {
            unsafe { poly_mul_linear_(bufA, curLen, 1.0, bufB); }
            curLen = curLen + 1UL;
            unsigned long cpy = 0UL;
            while (cpy < curLen) {
                unsafe { bufA[cpy] = bufB[cpy]; }
                cpy = cpy + 1UL;
            }
            t = t + 1UL;
        }

        double bk;
        double ak;
        unsafe { bk = B[k]; ak = A[k]; }
        j = 0UL;
        while (j <= m) {
            double coeff;
            unsafe { coeff = bufA[j]; }
            unsafe {
                numPoly[j] = numPoly[j] + bk * ck * coeff;
                denPoly[j] = denPoly[j] + ak * ck * coeff;
            }
            j = j + 1UL;
        }

        ck = ck * c;
        k = k + 1UL;
    }

    // numPoly/denPoly are ascending powers of z; reversing index j -> m-j
    // converts z^j into the negative-power-of-z term z^-(m-j), and
    // dividing by denPoly[m] normalizes aOut[0] to 1.
    double a0;
    unsafe { a0 = denPoly[m]; }
    j = 0UL;
    while (j <= m) {
        double nb;
        double na;
        unsafe { nb = numPoly[m - j]; na = denPoly[m - j]; }
        unsafe {
            bOut[j] = nb / a0;
            aOut[j] = na / a0;
        }
        j = j + 1UL;
    }

    unsafe {
        dealloc((void*)numPoly);
        dealloc((void*)denPoly);
        dealloc((void*)bufA);
        dealloc((void*)bufB);
    }
}

static struct Complex ztransform_response_(const double* b, const double* a, unsigned long order,
                                            double fs, double freqHz) {
    double omega = SAFEC_DSP_TWO_PI_ZT * freqHz / fs;
    struct Complex zInv = complex_new(cos_d(-omega), sin_d(-omega));
    struct Complex numAcc = complex_new(0.0, 0.0);
    struct Complex denAcc = complex_new(0.0, 0.0);
    struct Complex zPow = complex_new(1.0, 0.0);

    unsigned long i = 0UL;
    while (i <= order) {
        double bi;
        double ai;
        unsafe { bi = b[i]; ai = a[i]; }
        struct Complex bCoef = complex_new(bi, 0.0);
        struct Complex aCoef = complex_new(ai, 0.0);
        struct Complex bTerm = bCoef * zPow;
        struct Complex aTerm = aCoef * zPow;
        struct Complex newNum = numAcc + bTerm;
        struct Complex newDen = denAcc + aTerm;
        numAcc = newNum;
        denAcc = newDen;
        struct Complex nextZPow = zPow * zInv;
        zPow = nextZPow;
        i = i + 1UL;
    }

    return numAcc / denAcc;
}

double ztransform_response_mag(const double* b, const double* a, unsigned long order,
                                double fs, double freqHz) {
    struct Complex h = ztransform_response_(b, a, order, fs, freqHz);
    return h.abs();
}

double ztransform_response_phase(const double* b, const double* a, unsigned long order,
                                  double fs, double freqHz) {
    struct Complex h = ztransform_response_(b, a, order, fs, freqHz);
    return h.arg();
}

} // namespace std
