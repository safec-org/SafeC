// SafeC Standard Library — DCT implementation (see dct.h)
#pragma once
#include <std/simd/cpu.h>
#include <std/dsp/dct.h>
#include <std/dsp/fixed.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/sync/parallel.h>

#define SAFEC_DSP_PI_DCT 3.14159265358979323846

namespace std {

// Both transforms read cos(pi m / (2n)) from a table of 4n entries, with
// the index (2i + 1) k stepped mod 4n, instead of a cos call per term, and
// split their outputs across the thread pool when n is large. Without
// memory for the table they compute each cosine as before.

struct DctJob {
    const double* in;
    double*       out;
    const double* cosTab;   // cos(pi m / (2n)), m < 4n
    unsigned long n;
    int           inverse;
};

static void dct_outs_body_(const &DctJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsafe {
        unsigned long n = j.n;
        unsigned long period = 4UL * n;
        unsigned long o = lo;
        while (o < hi) {
            if (!j.inverse) {
                // out[k] = sum_i x[i] cos(pi (2i + 1) k / 2n): index starts at k, steps 2k
                unsigned long k = o;
                unsigned long idx = k % period;
                unsigned long step = (2UL * k) % period;
                double sum = 0.0;
                unsigned long i = 0UL;
                while (i < n) {
                    sum = __builtin_fmuladd(j.in[i], j.cosTab[idx], sum);
                    idx = idx + step;
                    idx = idx >= period ? idx - period : idx;
                    i = i + 1UL;
                }
                j.out[k] = sum;
            } else {
                // out[i] = (x[0] + 2 sum_{k >= 1} x[k] cos(pi (2i + 1) k / 2n)) / n
                unsigned long i = o;
                unsigned long step = (2UL * i + 1UL) % period;
                unsigned long idx = step;
                double sum = j.in[0];
                unsigned long k = 1UL;
                while (k < n) {
                    sum = __builtin_fmuladd(2.0 * j.in[k], j.cosTab[idx], sum);
                    idx = idx + step;
                    idx = idx >= period ? idx - period : idx;
                    k = k + 1UL;
                }
                j.out[i] = sum / (double)n;
            }
            o = o + 1UL;
        }
    }
}

static void dct_outs_base_(const &DctJob j, unsigned long lo, unsigned long hi) { dct_outs_body_(j, lo, hi); }
#if defined(__x86_64__)
static void dct_outs_fma_(const &DctJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { dct_outs_body_(j, lo, hi); }
#endif

static void dct_outs_(&DctJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    if (simd_level() >= SIMD_AVX2) { dct_outs_fma_(j, lo, hi); return; }
#endif
    dct_outs_base_(j, lo, hi);
}

static void dct_slow_(const double* in, double* out, unsigned long n, int inverse) {
    double nD = (double)n;
    unsigned long o = 0UL;
    while (o < n) {
        unsafe {
            if (!inverse) {
                double sum = 0.0;
                unsigned long i = 0UL;
                while (i < n) {
                    sum = sum + in[i] * cos_d(SAFEC_DSP_PI_DCT / nD * ((double)i + 0.5) * (double)o);
                    i = i + 1UL;
                }
                out[o] = sum;
            } else {
                double sum = in[0];
                unsigned long k = 1UL;
                while (k < n) {
                    sum = sum + 2.0 * in[k] * cos_d(SAFEC_DSP_PI_DCT / nD * ((double)o + 0.5) * (double)k);
                    k = k + 1UL;
                }
                out[o] = sum / nD;
            }
        }
        o = o + 1UL;
    }
}

static void dct_run_(const double* in, double* out, unsigned long n, int inverse) {
    if (n == 0UL) return;
    unsafe {
        double* tab = (double*)alloc_raw_(sizeof(double) * 4UL * n);
        if (tab == (double*)0) { dct_slow_(in, out, n, inverse); return; }
        unsigned long m = 0UL;
        while (m < 4UL * n) { tab[m] = cos_d(SAFEC_DSP_PI_DCT * (double)m / (2.0 * (double)n)); m = m + 1UL; }
        struct DctJob j;
        j.in = in;
        j.out = out;
        j.cosTab = (const double*)tab;
        j.n = n;
        j.inverse = inverse;
        if (n >= 512UL) parallel_for(n, 8192UL / n + 1UL, dct_outs_, &j);
        else dct_outs_(&j, 0UL, n);
        dealloc_raw_((void*)tab);
    }
}

void dct2_n_(const &double in, &double out, unsigned long n) {
    unsafe { dct_run_((const double*)in, (double*)out, n, 0); }
}

void idct3_n_(const &double in, &double out, unsigned long n) {
    unsafe { dct_run_((const double*)in, (double*)out, n, 1); }
}

// Float in and out, double inside: these O(n^2) sums are where float
// rounding would add up, and the conversion is O(n).
static void dct_run_f32_(const float* in, float* out, unsigned long n, int inverse) {
    if (n == 0UL) return;
    unsafe {
        double* buf = (double*)alloc_raw_(sizeof(double) * 2UL * n);
        if (buf == (double*)0) return;
        unsigned long i = 0UL;
        while (i < n) { buf[i] = (double)in[i]; i = i + 1UL; }
        dct_run_((const double*)buf, &buf[n], n, inverse);
        i = 0UL;
        while (i < n) { out[i] = (float)buf[n + i]; i = i + 1UL; }
        dealloc_raw_((void*)buf);
    }
}

void dct2_f32_n_(const &float in, &float out, unsigned long n) {
    unsafe { dct_run_f32_((const float*)in, (float*)out, n, 0); }
}

void idct3_f32_n_(const &float in, &float out, unsigned long n) {
    unsafe { dct_run_f32_((const float*)in, (float*)out, n, 1); }
}

void fdct2_n_(const &Fixed in, &Fixed out, unsigned long n) {
    if (n == 0UL) { return; }
    double nD = (double)n;
    unsigned long k = 0UL;
    while (k < n) {
        double sum = 0.0;
        unsigned long i = 0UL;
        while (i < n) {
            Fixed xiF;
            unsafe { xiF = in[i]; }
            double xi = fixed_to_float(xiF);
            double angle = SAFEC_DSP_PI_DCT / nD * ((double)i + 0.5) * (double)k;
            sum = sum + xi * cos_d(angle);
            i = i + 1UL;
        }
        unsafe { out[k] = fixed_from_float(sum); }
        k = k + 1UL;
    }
}

void fidct3_n_(const &Fixed in, &Fixed out, unsigned long n) {
    if (n == 0UL) { return; }
    double nD = (double)n;
    Fixed x0F;
    unsafe { x0F = in[0]; }
    double x0 = fixed_to_float(x0F);
    unsigned long i = 0UL;
    while (i < n) {
        double sum = x0;
        unsigned long k = 1UL;
        while (k < n) {
            Fixed xkF;
            unsafe { xkF = in[k]; }
            double xk = fixed_to_float(xkF);
            double angle = SAFEC_DSP_PI_DCT / nD * ((double)i + 0.5) * (double)k;
            sum = sum + 2.0 * xk * cos_d(angle);
            k = k + 1UL;
        }
        unsafe { out[i] = fixed_from_float(sum / nD); }
        i = i + 1UL;
    }
}

// ── Slice API (the public signatures; the (reference, length) cores above
// are the *_n_ functions) ─────────────────────────────────────────────────────

int dct2([]const double in, []double out) {
    if (out.len < in.len) return 0;
    if (in.len == 0UL) return 1;
    unsafe { dct2_n_((const &double)in.ptr, (&double)out.ptr, in.len); }
    return 1;
}

int idct3([]const double in, []double out) {
    if (out.len < in.len) return 0;
    if (in.len == 0UL) return 1;
    unsafe { idct3_n_((const &double)in.ptr, (&double)out.ptr, in.len); }
    return 1;
}

int dct2_f32([]const float in, []float out) {
    if (out.len < in.len) return 0;
    if (in.len == 0UL) return 1;
    unsafe { dct2_f32_n_((const &float)in.ptr, (&float)out.ptr, in.len); }
    return 1;
}

int idct3_f32([]const float in, []float out) {
    if (out.len < in.len) return 0;
    if (in.len == 0UL) return 1;
    unsafe { idct3_f32_n_((const &float)in.ptr, (&float)out.ptr, in.len); }
    return 1;
}

int fdct2([]const Fixed in, []Fixed out) {
    if (out.len < in.len) return 0;
    if (in.len == 0UL) return 1;
    unsafe { fdct2_n_((const &Fixed)in.ptr, (&Fixed)out.ptr, in.len); }
    return 1;
}

int fidct3([]const Fixed in, []Fixed out) {
    if (out.len < in.len) return 0;
    if (in.len == 0UL) return 1;
    unsafe { fidct3_n_((const &Fixed)in.ptr, (&Fixed)out.ptr, in.len); }
    return 1;
}

} // namespace std
