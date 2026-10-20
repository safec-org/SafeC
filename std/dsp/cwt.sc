// SafeC Standard Library — CWT implementation (see cwt.h)
#pragma once
#include <std/dsp/cwt.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/dsp/dsp.h>
#include <std/sync/parallel.h>

#define SAFEC_DSP_PI_CWT 3.14159265358979323846

namespace std {

// The wavelet at scale s depends only on (n - b), so each scale's kernel
// (envelope * cos and * -sin, 2 * halfWidth + 1 taps) is computed once, and
// every output is two SIMD dot products of the signal with it (dsp_dot_f64).
// Rows of (scale, position) run on the thread pool.

struct CwtJob {
    const double*  signal;
    unsigned long  signalLen;
    const double*  scales;
    double* const* kRe;    // per scale: 2 * halfWidth + 1 taps
    double* const* kIm;
    double*        out;    // interleaved complex, numScales x signalLen
};

static unsigned long cwt_half_(double s) { return (unsigned long)(4.0 * s) + 1UL; }

static void cwt_rows_(&CwtJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long r = lo;
        while (r < hi) {
            unsigned long si = r / j.signalLen;
            unsigned long b = r % j.signalLen;
            unsigned long hw = cwt_half_(j.scales[si]);
            unsigned long nStart = b >= hw ? b - hw : 0UL;
            unsigned long nEnd = b + hw;
            if (nEnd >= j.signalLen) { nEnd = j.signalLen - 1UL; }
            unsigned long len = nEnd - nStart + 1UL;
            unsigned long t0 = nStart + hw - b;      // tap index of n = nStart
            double re = dsp_dot_f64(&j.signal[nStart], (const double*)&j.kRe[si][t0], len);
            double im = dsp_dot_f64(&j.signal[nStart], (const double*)&j.kIm[si][t0], len);
            j.out[2UL * r] = re;
            j.out[2UL * r + 1UL] = im;
            r = r + 1UL;
        }
    }
}

void cwt_morlet(const &double signal, unsigned long signalLen,
                 const &double scales, unsigned long numScales,
                 double w0, &Complex out) {
    if (signalLen == 0UL || numScales == 0UL) return;
    double normConst = pow_d(SAFEC_DSP_PI_CWT, 0.0 - 0.25);
    unsafe {
        double** kRe = (double**)alloc_raw_(sizeof(double*) * numScales);
        double** kIm = (double**)alloc_raw_(sizeof(double*) * numScales);
        if (kRe == (double**)0 || kIm == (double**)0) return;
        unsigned long si = 0UL;
        while (si < numScales) {
            double s = scales[si];
            unsigned long hw = cwt_half_(s);
            unsigned long taps = 2UL * hw + 1UL;
            double invSqrtS = 1.0 / sqrt_d(s);
            kRe[si] = (double*)alloc_raw_(sizeof(double) * taps);
            kIm[si] = (double*)alloc_raw_(sizeof(double) * taps);
            if (kRe[si] == (double*)0 || kIm[si] == (double*)0) return;
            unsigned long t = 0UL;
            while (t < taps) {
                double tt = ((double)t - (double)hw) / s;
                double env = normConst * exp_d(-0.5 * tt * tt) * invSqrtS;
                kRe[si][t] = env * cos_d(w0 * tt);
                kIm[si][t] = -env * sin_d(w0 * tt);
                t = t + 1UL;
            }
            si = si + 1UL;
        }
        struct CwtJob j;
        j.signal = (const double*)signal;
        j.signalLen = signalLen;
        j.scales = (const double*)scales;
        j.kRe = (double* const*)kRe;
        j.kIm = (double* const*)kIm;
        j.out = (double*)out;
        unsigned long rows = numScales * signalLen;
        if (rows >= 4096UL) parallel_for(rows, 1024UL, cwt_rows_, &j);
        else cwt_rows_(&j, 0UL, rows);
        si = 0UL;
        while (si < numScales) { dealloc_raw_((void*)kRe[si]); dealloc_raw_((void*)kIm[si]); si = si + 1UL; }
        dealloc_raw_((void*)kRe);
        dealloc_raw_((void*)kIm);
    }
}

} // namespace std
