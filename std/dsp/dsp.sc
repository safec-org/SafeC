#pragma once
#include "dsp.h"

// ── dsp_dot ──────────────────────────────────────────────────────────────────
// Accumulate fixed-point dot product.
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

// ── dsp_moving_avg ────────────────────────────────────────────────────────────
// Causal moving average of order `order`.
// state[0..order-1] holds the history ring buffer (oldest first).
// Simple O(n*order) reference implementation — suitable for small order values.
void dsp_moving_avg(&stack Fixed in, &stack Fixed out, unsigned long n,
                    &stack Fixed state, unsigned long order) {
    if (order == 0UL) {
        return;
    }
    // Divisor in Q8.24: 1/order.
    Fixed inv_order = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)order));

    unsigned long i = 0UL;
    while (i < n) {
        // Shift state left (drop oldest) and insert new sample.
        unsigned long k = 0UL;
        while (k < order - 1UL) {
            unsafe {
                Fixed* ps = (Fixed*)state;
                ps[k] = ps[k + 1UL];
            }
            k = k + 1UL;
        }
        unsafe {
            Fixed* ps = (Fixed*)state;
            Fixed* pi = (Fixed*)in;
            ps[order - 1UL] = pi[i];
        }

        // Sum all state elements.
        Fixed sum = (Fixed)0;
        k = 0;
        while (k < order) {
            unsafe {
                Fixed* ps = (Fixed*)state;
                sum = fixed_add(sum, ps[k]);
            }
            k = k + 1UL;
        }

        // Divide by order using Q8.24 multiply by 1/order.
        unsafe {
            Fixed* po = (Fixed*)out;
            po[i] = fixed_mul(sum, inv_order);
        }
        i = i + 1UL;
    }
}

// ── dsp_iir_lp ───────────────────────────────────────────────────────────────
// First-order IIR low-pass: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
void dsp_iir_lp(&stack Fixed in, &stack Fixed out, unsigned long n,
                Fixed alpha, &stack Fixed prev_y) {
    Fixed one_minus_alpha = fixed_sub((Fixed)FIXED_ONE, alpha);
    unsigned long i = 0UL;
    while (i < n) {
        Fixed xn;
        unsafe {
            Fixed* pi = (Fixed*)in;
            xn = pi[i];
        }
        Fixed yn = fixed_add(fixed_mul(alpha, xn),
                             fixed_mul(one_minus_alpha, *prev_y));
        unsafe {
            Fixed* po = (Fixed*)out;
            po[i] = yn;
        }
        *prev_y = yn;
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
