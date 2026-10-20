#pragma once
#include <std/dsp/dsp.h>
#include <std/simd/simd.h>
#include <std/simd/cpu.h>

// ── dsp_dot ──────────────────────────────────────────────────────────────────
// Accumulate fixed-point dot product.
namespace std {

Fixed dsp_dot(&stack Fixed a, &stack Fixed b, unsigned long n) {
    Fixed acc = (Fixed)0;
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            Fixed* pa = (Fixed*)a;
            Fixed* pb = (Fixed*)b;
            acc = fixed_add(acc, fixed_mul(pa[i], pb[i]));
        }
        i = i + 1UL;
    }
    return acc;
}

// ── dsp_scale ─────────────────────────────────────────────────────────────────
// Multiply every element by scale in-place.
void dsp_scale(&stack Fixed a, Fixed scale, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            Fixed* pa = (Fixed*)a;
            pa[i] = fixed_mul(pa[i], scale);
        }
        i = i + 1UL;
    }
}

// ── dsp_add ───────────────────────────────────────────────────────────────────
// Element-wise addition: out[i] = a[i] + b[i].
void dsp_add(&stack Fixed a, &stack Fixed b, &stack Fixed out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        unsafe {
            Fixed* pa = (Fixed*)a;
            Fixed* pb = (Fixed*)b;
            Fixed* po = (Fixed*)out;
            po[i] = fixed_add(pa[i], pb[i]);
        }
        i = i + 1UL;
    }
}

// ── dsp_clip ──────────────────────────────────────────────────────────────────
// Clamp every element to [lo, hi].
void dsp_clip(&stack Fixed a, Fixed lo, Fixed hi, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        Fixed v;
        unsafe {
            Fixed* pa = (Fixed*)a;
            v = pa[i];
        }
        if ((int)v < (int)lo) {
            v = lo;
        } else if ((int)v > (int)hi) {
            v = hi;
        }
        unsafe {
            Fixed* pa = (Fixed*)a;
            pa[i] = v;
        }
        i = i + 1UL;
    }
}

// ── dsp_peak ──────────────────────────────────────────────────────────────────
// Return maximum absolute value in array.
Fixed dsp_peak(&stack Fixed a, unsigned long n) {
    Fixed peak = (Fixed)0;
    unsigned long i = 0UL;
    while (i < n) {
        Fixed v;
        unsafe {
            Fixed* pa = (Fixed*)a;
            v = pa[i];
        }
        Fixed av = fixed_abs(v);
        if ((int)av > (int)peak) {
            peak = av;
        }
        i = i + 1UL;
    }
    return peak;
}

// ── dsp_rms ───────────────────────────────────────────────────────────────────
// Root mean square: sqrt( sum(x[i]^2) / n ).
Fixed dsp_rms(&stack Fixed a, unsigned long n) {
    if (n == 0UL) {
        return (Fixed)0;
    }
    Fixed sum = (Fixed)0;
    unsigned long i = 0UL;
    while (i < n) {
        Fixed v;
        unsafe {
            Fixed* pa = (Fixed*)a;
            v = pa[i];
        }
        sum = fixed_add(sum, fixed_mul(v, v));
        i = i + 1UL;
    }
    // Divide by n.
    Fixed inv_n = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)n));
    Fixed mean_sq = fixed_mul(sum, inv_n);
    return fixed_sqrt(mean_sq);
}

// ── dsp_dot_f64 ──────────────────────────────────────────────────────────────
// SIMD FMA dot product (see dsp.h for why this stays float64-only).
static double dsp_dot_body_(const double* a, const double* b, unsigned long n) __attribute__((always_inline)) {
    // Two independent f64x4 accumulators hide the multiply-add latency.
    f64x4 acc0;
    f64x4 acc1;
    unsigned long i = 0UL;
    unsafe {
        while (i + 8UL <= n) {
            acc0 = __builtin_fmuladd(*(const f64x4*)&a[i], *(const f64x4*)&b[i], acc0);
            acc1 = __builtin_fmuladd(*(const f64x4*)&a[i + 4UL], *(const f64x4*)&b[i + 4UL], acc1);
            i = i + 8UL;
        }
        if (i + 4UL <= n) {
            acc0 = __builtin_fmuladd(*(const f64x4*)&a[i], *(const f64x4*)&b[i], acc0);
            i = i + 4UL;
        }
    }
    f64x4 acc = acc0 + acc1;
    double sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while (i < n) {
        unsafe { sum = sum + a[i] * b[i]; }
        i = i + 1UL;
    }
    return sum;
}

// The same body per ISA level (std/simd/cpu.h): FMA from AVX2 on.
static double dsp_dot_base_(const double* a, const double* b, unsigned long n) { return dsp_dot_body_(a, b, n); }
#if defined(__x86_64__)
static double dsp_dot_avx_(const double* a, const double* b, unsigned long n)
    __attribute__((target("avx"))) { return dsp_dot_body_(a, b, n); }
static double dsp_dot_avx2_(const double* a, const double* b, unsigned long n)
    __attribute__((target("avx2,fma"))) { return dsp_dot_body_(a, b, n); }
#endif

double dsp_dot_f64(const &double a_ref, const &double b_ref, unsigned long n) {
    unsafe {
        const double* a = (const double*)a_ref;
        const double* b = (const double*)b_ref;
#if defined(__x86_64__)
        int l = simd_level();
        if (l >= SIMD_AVX2) return dsp_dot_avx2_(a, b, n);
        if (l >= SIMD_AVX) return dsp_dot_avx_(a, b, n);
#endif
        return dsp_dot_base_(a, b, n);
    }
}

// ── dsp_dot_f32 ──────────────────────────────────────────────────────────────
// The float version: f32x8 accumulators, so twice the values per register.
static float dsp_dot32_body_(const float* a, const float* b, unsigned long n) __attribute__((always_inline)) {
    f32x8 acc0;
    f32x8 acc1;
    unsigned long i = 0UL;
    unsafe {
        while (i + 16UL <= n) {
            acc0 = __builtin_fmuladd(*(const f32x8*)&a[i], *(const f32x8*)&b[i], acc0);
            acc1 = __builtin_fmuladd(*(const f32x8*)&a[i + 8UL], *(const f32x8*)&b[i + 8UL], acc1);
            i = i + 16UL;
        }
        if (i + 8UL <= n) {
            acc0 = __builtin_fmuladd(*(const f32x8*)&a[i], *(const f32x8*)&b[i], acc0);
            i = i + 8UL;
        }
    }
    f32x8 acc = acc0 + acc1;
    float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    while (i < n) {
        unsafe { sum = __builtin_fmuladd(a[i], b[i], sum); }
        i = i + 1UL;
    }
    return sum;
}

static float dsp_dot32_base_(const float* a, const float* b, unsigned long n) { return dsp_dot32_body_(a, b, n); }
#if defined(__x86_64__)
static float dsp_dot32_avx_(const float* a, const float* b, unsigned long n)
    __attribute__((target("avx"))) { return dsp_dot32_body_(a, b, n); }
static float dsp_dot32_avx2_(const float* a, const float* b, unsigned long n)
    __attribute__((target("avx2,fma"))) { return dsp_dot32_body_(a, b, n); }
#endif

float dsp_dot_f32(const &float a_ref, const &float b_ref, unsigned long n) {
    unsafe {
        const float* a = (const float*)a_ref;
        const float* b = (const float*)b_ref;
#if defined(__x86_64__)
        int l = simd_level();
        if (l >= SIMD_AVX2) return dsp_dot32_avx2_(a, b, n);
        if (l >= SIMD_AVX) return dsp_dot32_avx_(a, b, n);
#endif
        return dsp_dot32_base_(a, b, n);
    }
}

// ── dsp_sliding_dot ──────────────────────────────────────────────────────────
// out[q] = sum_t a[t] * x[q + t] for q < nOut: 32 (double) or 64 (float)
// outputs at a time, each tap broadcast once and multiplied into four
// accumulator vectors of consecutive outputs. No horizontal sums, and one
// load per multiply-add instead of two; the leftovers use dsp_dot.
typedef vec<double, 8> dsp_f64x8_;
typedef vec<float, 16> dsp_f32x16_;

static dsp_f64x8_ dsp_splat8_(double v) __attribute__((always_inline)) {
    dsp_f64x8_ r;
    r[0] = v; r[1] = v; r[2] = v; r[3] = v; r[4] = v; r[5] = v; r[6] = v; r[7] = v;
    return r;
}

static dsp_f32x16_ dsp_splat16_(float v) __attribute__((always_inline)) {
    dsp_f32x16_ r;
    r[0] = v; r[1] = v; r[2] = v; r[3] = v; r[4] = v; r[5] = v; r[6] = v; r[7] = v;
    r[8] = v; r[9] = v; r[10] = v; r[11] = v; r[12] = v; r[13] = v; r[14] = v; r[15] = v;
    return r;
}

static unsigned long dsp_sdot64_body_(const double* a, const double* x, double* out, unsigned long nOut, unsigned long n)
    __attribute__((always_inline)) {
    unsigned long q = 0UL;
    unsafe {
        while (q + 32UL <= nOut) {
            const double* xp = &x[q];
            dsp_f64x8_ s0 = dsp_splat8_(0.0);
            dsp_f64x8_ s1 = s0;
            dsp_f64x8_ s2 = s0;
            dsp_f64x8_ s3 = s0;
            unsigned long t = 0UL;
            while (t < n) {
                dsp_f64x8_ c = dsp_splat8_(a[t]);
                s0 = __builtin_fmuladd(c, *(const dsp_f64x8_*)&xp[t], s0);
                s1 = __builtin_fmuladd(c, *(const dsp_f64x8_*)&xp[t + 8UL], s1);
                s2 = __builtin_fmuladd(c, *(const dsp_f64x8_*)&xp[t + 16UL], s2);
                s3 = __builtin_fmuladd(c, *(const dsp_f64x8_*)&xp[t + 24UL], s3);
                t = t + 1UL;
            }
            *(dsp_f64x8_*)&out[q] = s0;
            *(dsp_f64x8_*)&out[q + 8UL] = s1;
            *(dsp_f64x8_*)&out[q + 16UL] = s2;
            *(dsp_f64x8_*)&out[q + 24UL] = s3;
            q = q + 32UL;
        }
    }
    return q;
}

static unsigned long dsp_sdot32_body_(const float* a, const float* x, float* out, unsigned long nOut, unsigned long n)
    __attribute__((always_inline)) {
    unsigned long q = 0UL;
    unsafe {
        while (q + 64UL <= nOut) {
            const float* xp = &x[q];
            dsp_f32x16_ s0 = dsp_splat16_((float)0.0);
            dsp_f32x16_ s1 = s0;
            dsp_f32x16_ s2 = s0;
            dsp_f32x16_ s3 = s0;
            unsigned long t = 0UL;
            while (t < n) {
                dsp_f32x16_ c = dsp_splat16_(a[t]);
                s0 = __builtin_fmuladd(c, *(const dsp_f32x16_*)&xp[t], s0);
                s1 = __builtin_fmuladd(c, *(const dsp_f32x16_*)&xp[t + 16UL], s1);
                s2 = __builtin_fmuladd(c, *(const dsp_f32x16_*)&xp[t + 32UL], s2);
                s3 = __builtin_fmuladd(c, *(const dsp_f32x16_*)&xp[t + 48UL], s3);
                t = t + 1UL;
            }
            *(dsp_f32x16_*)&out[q] = s0;
            *(dsp_f32x16_*)&out[q + 16UL] = s1;
            *(dsp_f32x16_*)&out[q + 32UL] = s2;
            *(dsp_f32x16_*)&out[q + 48UL] = s3;
            q = q + 64UL;
        }
    }
    return q;
}

static unsigned long dsp_sdot64_base_(const double* a, const double* x, double* out, unsigned long nOut, unsigned long n) {
    return dsp_sdot64_body_(a, x, out, nOut, n);
}
static unsigned long dsp_sdot32_base_(const float* a, const float* x, float* out, unsigned long nOut, unsigned long n) {
    return dsp_sdot32_body_(a, x, out, nOut, n);
}
#if defined(__x86_64__)
static unsigned long dsp_sdot64_avx2_(const double* a, const double* x, double* out, unsigned long nOut, unsigned long n)
    __attribute__((target("avx2,fma"))) { return dsp_sdot64_body_(a, x, out, nOut, n); }
static unsigned long dsp_sdot64_avx512_(const double* a, const double* x, double* out, unsigned long nOut, unsigned long n)
    __attribute__((target("avx512f,avx512vl,avx2,fma"))) { return dsp_sdot64_body_(a, x, out, nOut, n); }
static unsigned long dsp_sdot32_avx2_(const float* a, const float* x, float* out, unsigned long nOut, unsigned long n)
    __attribute__((target("avx2,fma"))) { return dsp_sdot32_body_(a, x, out, nOut, n); }
static unsigned long dsp_sdot32_avx512_(const float* a, const float* x, float* out, unsigned long nOut, unsigned long n)
    __attribute__((target("avx512f,avx512vl,avx2,fma"))) { return dsp_sdot32_body_(a, x, out, nOut, n); }
#endif

void dsp_sliding_dot_f64(const &double a_ref, const &double x_ref, &double out_ref, unsigned long nOut, unsigned long n) {
    unsafe {
        const double* a = (const double*)a_ref;
        const double* x = (const double*)x_ref;
        double* out = (double*)out_ref;
        unsigned long q = 0UL;
#if defined(__x86_64__)
        int l = simd_level();
        if (l >= SIMD_AVX512) q = dsp_sdot64_avx512_(a, x, out, nOut, n);
        else if (l >= SIMD_AVX2) q = dsp_sdot64_avx2_(a, x, out, nOut, n);
        else
#endif
        q = dsp_sdot64_base_(a, x, out, nOut, n);
        while (q < nOut) { out[q] = dsp_dot_f64(a, &x[q], n); q = q + 1UL; }
    }
}

void dsp_sliding_dot_f32(const &float a_ref, const &float x_ref, &float out_ref, unsigned long nOut, unsigned long n) {
    unsafe {
        const float* a = (const float*)a_ref;
        const float* x = (const float*)x_ref;
        float* out = (float*)out_ref;
        unsigned long q = 0UL;
#if defined(__x86_64__)
        int l = simd_level();
        if (l >= SIMD_AVX512) q = dsp_sdot32_avx512_(a, x, out, nOut, n);
        else if (l >= SIMD_AVX2) q = dsp_sdot32_avx2_(a, x, out, nOut, n);
        else
#endif
        q = dsp_sdot32_base_(a, x, out, nOut, n);
        while (q < nOut) { out[q] = dsp_dot_f32(a, &x[q], n); q = q + 1UL; }
    }
}

} // namespace std
