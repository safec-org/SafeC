#pragma once
#include <std/dsp/dsp.h>
#include <std/simd/simd.h>

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
double dsp_dot_f64(const &double a_ref, const &double b_ref, unsigned long n) { const double* a = a_ref; const double* b = b_ref;
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

} // namespace std
