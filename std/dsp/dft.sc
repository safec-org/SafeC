// SafeC Standard Library — DFT implementation (see dft.h)
#pragma once
#include <std/simd/cpu.h>
#include <std/dsp/dft.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/sync/parallel.h>

#define SAFEC_DSP_TWO_PI 6.28318530717958647692

namespace std {

// The double-precision transforms read their twiddles from a table of the
// n exact roots of unity, w[m] = exp(-2 pi i m / n), stepping the index by
// k mod n, instead of calling cos/sin for every term. Output bins are
// independent, so long transforms split them across the thread pool. If
// the table can't be allocated they fall back to computing each twiddle.

struct DftJob {
    const double* in;     // interleaved complex, or real when 'real' is set
    double*       out;    // interleaved complex
    const double* tab;    // tab[2m], tab[2m + 1] = cos, -sin (2 pi m / n)
    unsigned long n;
    int           inv;    // conjugate twiddles and scale by 1/n
    int           real;
};

static void dft_bins_body_(const &DftJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsafe {
        unsigned long n = j.n;
        double sgn = j.inv ? -1.0 : 1.0;
        double scale = j.inv ? 1.0 / (double)n : 1.0;
        unsigned long k = lo;
        while (k < hi) {
            double sr = 0.0;
            double si = 0.0;
            unsigned long idx = 0UL;
            unsigned long i = 0UL;
            if (j.real) {
                while (i < n) {
                    double x = j.in[i];
                    sr = __builtin_fmuladd(x, j.tab[2UL * idx], sr);
                    si = __builtin_fmuladd(x, j.tab[2UL * idx + 1UL], si);
                    idx = idx + k;
                    idx = idx >= n ? idx - n : idx;
                    i = i + 1UL;
                }
            } else {
                while (i < n) {
                    double xr = j.in[2UL * i];
                    double xi = j.in[2UL * i + 1UL];
                    double wr = j.tab[2UL * idx];
                    double wi = sgn * j.tab[2UL * idx + 1UL];
                    sr = __builtin_fmuladd(xr, wr, __builtin_fmuladd(-xi, wi, sr));
                    si = __builtin_fmuladd(xr, wi, __builtin_fmuladd(xi, wr, si));
                    idx = idx + k;
                    idx = idx >= n ? idx - n : idx;
                    i = i + 1UL;
                }
            }
            j.out[2UL * k] = sr * scale;
            j.out[2UL * k + 1UL] = si * scale;
            k = k + 1UL;
        }
    }
}

static void dft_bins_base_(const &DftJob j, unsigned long lo, unsigned long hi) { dft_bins_body_(j, lo, hi); }
#if defined(__x86_64__)
static void dft_bins_fma_(const &DftJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { dft_bins_body_(j, lo, hi); }
#endif

static void dft_bins_(&DftJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    if (simd_level() >= SIMD_AVX2) { dft_bins_fma_(j, lo, hi); return; }
#endif
    dft_bins_base_(j, lo, hi);
}

// Old per-term path, used only when the table can't be allocated.
static void dft_slow_(const double* in, double* out, unsigned long n, int inv, int real) {
    double sign = inv ? 1.0 : -1.0;
    double scale = inv ? 1.0 / (double)n : 1.0;
    unsigned long k = 0UL;
    while (k < n) {
        double sr = 0.0;
        double si = 0.0;
        unsigned long i = 0UL;
        while (i < n) {
            double theta = sign * SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            double wr = cos_d(theta);
            double wi = sin_d(theta);
            unsafe {
                double xr = real ? in[i] : in[2UL * i];
                double xi = real ? 0.0 : in[2UL * i + 1UL];
                sr = sr + xr * wr - xi * wi;
                si = si + xr * wi + xi * wr;
            }
            i = i + 1UL;
        }
        unsafe { out[2UL * k] = sr * scale; out[2UL * k + 1UL] = si * scale; }
        k = k + 1UL;
    }
}

static void dft_run_(const double* in, double* out, unsigned long n, int inv, int real) {
    if (n == 0UL) return;
    unsafe {
        double* tab = (double*)alloc_raw_(sizeof(double) * 2UL * n);
        if (tab == (double*)0) { dft_slow_(in, out, n, inv, real); return; }
        unsigned long m = 0UL;
        while (m < n) {
            double th = SAFEC_DSP_TWO_PI * (double)m / (double)n;
            tab[2UL * m] = cos_d(th);
            tab[2UL * m + 1UL] = -sin_d(th);
            m = m + 1UL;
        }
        struct DftJob j;
        j.in = in;
        j.out = out;
        j.tab = (const double*)tab;
        j.n = n;
        j.inv = inv;
        j.real = real;
        if (n >= 256UL) parallel_for(n, 4096UL / n + 1UL, dft_bins_, &j);
        else dft_bins_(&j, 0UL, n);
        dealloc_raw_((void*)tab);
    }
}

void dft(const &Complex in, &Complex out, unsigned long n) {
    unsafe { dft_run_((const double*)in, (double*)out, n, 0, 0); }
}

void dft_real(const &double in, &Complex out, unsigned long n) {
    unsafe { dft_run_((const double*)in, (double*)out, n, 0, 1); }
}

void idft(const &Complex in, &Complex out, unsigned long n) {
    unsafe { dft_run_((const double*)in, (double*)out, n, 1, 0); }
}

// Float in and out, double inside (as for dct2_f32). real: 'in' holds n
// floats instead of n complex values.
static void dft_run_f32_(const float* in, float* out, unsigned long n, int inv, int real) {
    if (n == 0UL) return;
    unsigned long inVals = real ? n : 2UL * n;
    unsafe {
        double* buf = (double*)alloc_raw_(sizeof(double) * (inVals + 2UL * n));
        if (buf == (double*)0) return;
        unsigned long i = 0UL;
        while (i < inVals) { buf[i] = (double)in[i]; i = i + 1UL; }
        dft_run_((const double*)buf, &buf[inVals], n, inv, real);
        i = 0UL;
        while (i < 2UL * n) { out[i] = (float)buf[inVals + i]; i = i + 1UL; }
        dealloc_raw_((void*)buf);
    }
}

void dft_f32(const &Complex32 in, &Complex32 out, unsigned long n) {
    unsafe { dft_run_f32_((const float*)in, (float*)out, n, 0, 0); }
}

void dft_real_f32(const &float in, &Complex32 out, unsigned long n) {
    unsafe { dft_run_f32_((const float*)in, (float*)out, n, 0, 1); }
}

void idft_f32(const &Complex32 in, &Complex32 out, unsigned long n) {
    unsafe { dft_run_f32_((const float*)in, (float*)out, n, 1, 0); }
}

void fdft(const &FComplex in, &FComplex out, unsigned long n) {
    unsigned long k = 0UL;
    while (k < n) {
        struct FComplex sum = FComplex::new((Fixed)0, (Fixed)0);
        unsigned long i = 0UL;
        while (i < n) {
            double theta = -SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            struct FComplex twiddle = FComplex::from(Complex::polar(1.0, theta));
            struct FComplex xi;
            unsafe { xi = in[i]; }
            sum = sum + xi * twiddle;
            i = i + 1UL;
        }
        unsafe { out[k] = sum; }
        k = k + 1UL;
    }
}

void fidft(const &FComplex in, &FComplex out, unsigned long n) {
    if (n == 0UL) { return; }
    Fixed invN = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)n));
    unsigned long k = 0UL;
    while (k < n) {
        struct FComplex sum = FComplex::new((Fixed)0, (Fixed)0);
        unsigned long i = 0UL;
        while (i < n) {
            double theta = SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            struct FComplex twiddle = FComplex::from(Complex::polar(1.0, theta));
            struct FComplex xi;
            unsafe { xi = in[i]; }
            sum = sum + xi * twiddle;
            i = i + 1UL;
        }
        unsafe {
            out[k] = FComplex::new(fixed_mul(sum.re, invN), fixed_mul(sum.im, invN));
        }
        k = k + 1UL;
    }
}

} // namespace std
