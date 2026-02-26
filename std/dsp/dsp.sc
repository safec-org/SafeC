#pragma once
#include "dsp.h"

// ── dsp_dot ──────────────────────────────────────────────────────────────────
// Accumulate fixed-point dot product.
Fixed dsp_dot(const Fixed* a, const Fixed* b, unsigned long n) {
    Fixed acc = (Fixed)0;
    unsigned long i = 0;
    while (i < n) {
        unsafe {
            acc = fixed_add(acc, fixed_mul(a[i], b[i]));
        }
        i = i + 1;
    }
    return acc;
}

// ── dsp_scale ─────────────────────────────────────────────────────────────────
// Multiply every element by scale in-place.
void dsp_scale(Fixed* a, Fixed scale, unsigned long n) {
    unsigned long i = 0;
    while (i < n) {
        unsafe {
            a[i] = fixed_mul(a[i], scale);
        }
        i = i + 1;
    }
}

// ── dsp_add ───────────────────────────────────────────────────────────────────
// Element-wise addition: out[i] = a[i] + b[i].
void dsp_add(const Fixed* a, const Fixed* b, Fixed* out, unsigned long n) {
    unsigned long i = 0;
    while (i < n) {
        unsafe {
            out[i] = fixed_add(a[i], b[i]);
        }
        i = i + 1;
    }
}

// ── dsp_moving_avg ────────────────────────────────────────────────────────────
// Causal moving average of order `order`.
// state[0..order-1] holds the history ring buffer (oldest first).
// Simple O(n*order) reference implementation — suitable for small order values.
void dsp_moving_avg(const Fixed* in, Fixed* out, unsigned long n,
                    Fixed* state, unsigned long order) {
    if (order == 0) {
        return;
    }
    // Divisor in Q16.16: 1/order.
    Fixed inv_order = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)order));

    unsigned long i = 0;
    while (i < n) {
        // Shift state left (drop oldest) and insert new sample.
        unsigned long k = 0;
        while (k < order - 1) {
            unsafe {
                state[k] = state[k + 1];
            }
            k = k + 1;
        }
        unsafe {
            state[order - 1] = in[i];
        }

        // Sum all state elements.
        Fixed sum = (Fixed)0;
        k = 0;
        while (k < order) {
            unsafe {
                sum = fixed_add(sum, state[k]);
            }
            k = k + 1;
        }

        // Divide by order using Q16.16 multiply by 1/order.
        unsafe {
            out[i] = fixed_mul(sum, inv_order);
        }
        i = i + 1;
    }
}

// ── dsp_iir_lp ───────────────────────────────────────────────────────────────
// First-order IIR low-pass: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
void dsp_iir_lp(const Fixed* in, Fixed* out, unsigned long n,
                Fixed alpha, &stack Fixed prev_y) {
    Fixed one_minus_alpha = fixed_sub((Fixed)FIXED_ONE, alpha);
    unsigned long i = 0;
    while (i < n) {
        Fixed xn;
        unsafe {
            xn = in[i];
        }
        Fixed yn = fixed_add(fixed_mul(alpha, xn),
                             fixed_mul(one_minus_alpha, prev_y));
        unsafe {
            out[i] = yn;
        }
        prev_y = yn;
        i = i + 1;
    }
}

// ── dsp_clip ──────────────────────────────────────────────────────────────────
// Clamp every element to [lo, hi].
void dsp_clip(Fixed* a, Fixed lo, Fixed hi, unsigned long n) {
    unsigned long i = 0;
    while (i < n) {
        Fixed v;
        unsafe {
            v = a[i];
        }
        if ((int)v < (int)lo) {
            v = lo;
        } else if ((int)v > (int)hi) {
            v = hi;
        }
        unsafe {
            a[i] = v;
        }
        i = i + 1;
    }
}

// ── dsp_peak ──────────────────────────────────────────────────────────────────
// Return maximum absolute value in array.
Fixed dsp_peak(const Fixed* a, unsigned long n) {
    Fixed peak = (Fixed)0;
    unsigned long i = 0;
    while (i < n) {
        Fixed v;
        unsafe {
            v = a[i];
        }
        Fixed av = fixed_abs(v);
        if ((int)av > (int)peak) {
            peak = av;
        }
        i = i + 1;
    }
    return peak;
}

// ── dsp_rms ───────────────────────────────────────────────────────────────────
// Root mean square: sqrt( sum(x[i]^2) / n ).
Fixed dsp_rms(const Fixed* a, unsigned long n) {
    if (n == 0) {
        return (Fixed)0;
    }
    Fixed sum = (Fixed)0;
    unsigned long i = 0;
    while (i < n) {
        Fixed v;
        unsafe {
            v = a[i];
        }
        sum = fixed_add(sum, fixed_mul(v, v));
        i = i + 1;
    }
    // Divide by n.
    Fixed inv_n = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)n));
    Fixed mean_sq = fixed_mul(sum, inv_n);
    return fixed_sqrt(mean_sq);
}
