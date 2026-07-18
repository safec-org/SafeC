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
double dsp_dot_f64(const double* a, const double* b, unsigned long n) {
    f64x4 vacc = simd_splat_f64x4(0.0);
    unsigned long i = 0UL;
    while (i + 4UL <= n) {
        f64x4 va;
        f64x4 vb;
        unsafe {
            va = simd_load_f64x4(a + i);
            vb = simd_load_f64x4(b + i);
        }
        vacc = simd_fma_f64x4(va, vb, vacc);
        i = i + 4UL;
    }
    double acc = simd_hsum_f64x4(vacc);
    while (i < n) {
        unsafe { acc = acc + a[i] * b[i]; }
        i = i + 1UL;
    }
    return acc;
}

} // namespace std
