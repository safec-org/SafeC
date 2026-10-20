// SafeC Standard Library — CQT implementation (see cqt.h)
#pragma once
#include <std/dsp/cqt.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>
#include <std/sync/parallel.h>

#define SAFEC_DSP_PI_CQT 3.14159265358979323846

namespace std {

// Each bin's kernel is a Hann window times a complex exponential. Both are
// rotations by a fixed angle per sample, so they are advanced with complex
// multiplications and recomputed exactly every 64 samples, instead of three
// cos/sin calls per sample. Bins are independent and run on the thread pool
// (the low bins, with the longest kernels, balance through dynamic chunks).

struct CqtJob {
    const double* signal;
    unsigned long signalLen;
    double        fs;
    double        fmin;
    double        q;
    unsigned long binsPerOctave;
    double*       out;    // interleaved complex
};

static void cqt_bins_(&CqtJob j, unsigned long lo, unsigned long hi) {
    unsigned long k = lo;
    while (k < hi) {
        double fk = j.fmin * pow_d(2.0, (double)k / (double)j.binsPerOctave);
        unsigned long nk = (unsigned long)(j.q * j.fs / fk + 0.5);
        if (nk < 1UL) { nk = 1UL; }
        if (nk > j.signalLen) { nk = j.signalLen; }
        double denom = nk > 1UL ? (double)(nk - 1UL) : 1.0;
        double wStep = 2.0 * SAFEC_DSP_PI_CQT / denom;              // window phase per sample
        double aStep = -2.0 * SAFEC_DSP_PI_CQT * j.q / (double)nk;  // kernel phase per sample
        double cw = cos_d(wStep);
        double sw = sin_d(wStep);
        double ca = cos_d(aStep);
        double sa = sin_d(aStep);
        double sumRe = 0.0;
        double sumIm = 0.0;
        double wr = 1.0; double wi = 0.0;    // exp(i * wStep * n)
        double er = 1.0; double ei = 0.0;    // exp(i * aStep * n)
        unsigned long n = 0UL;
        while (n < nk) {
            if ((n & 63UL) == 0UL) {
                wr = cos_d(wStep * (double)n); wi = sin_d(wStep * (double)n);
                er = cos_d(aStep * (double)n); ei = sin_d(aStep * (double)n);
            }
            double s;
            unsafe { s = j.signal[n]; }
            double w = 0.5 * (1.0 - wr);
            double sw_ = s * w;
            sumRe = __builtin_fmuladd(sw_, er, sumRe);
            sumIm = __builtin_fmuladd(sw_, ei, sumIm);
            double t = __builtin_fmuladd(wr, cw, -wi * sw);
            wi = __builtin_fmuladd(wr, sw, wi * cw);
            wr = t;
            t = __builtin_fmuladd(er, ca, -ei * sa);
            ei = __builtin_fmuladd(er, sa, ei * ca);
            er = t;
            n = n + 1UL;
        }
        unsafe {
            j.out[2UL * k] = sumRe / (double)nk;
            j.out[2UL * k + 1UL] = sumIm / (double)nk;
        }
        k = k + 1UL;
    }
}

void cqt_forward(const &double signal, unsigned long signalLen, double fs,
                  double fmin, unsigned long numBins, unsigned long binsPerOctave,
                  &Complex out) {
    struct CqtJob j;
    unsafe {
        j.signal = (const double*)signal;
        j.out = (double*)out;
    }
    j.signalLen = signalLen;
    j.fs = fs;
    j.fmin = fmin;
    j.q = 1.0 / (pow_d(2.0, 1.0 / (double)binsPerOctave) - 1.0);
    j.binsPerOctave = binsPerOctave;
    if (numBins > 1UL && signalLen >= 1024UL) parallel_for(numBins, 1UL, cqt_bins_, &j);
    else cqt_bins_(&j, 0UL, numBins);
}

} // namespace std
