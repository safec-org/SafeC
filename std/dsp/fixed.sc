#pragma once
#include "fixed.h"

newtype Fixed = int;

inline const Fixed fixed_from_int(int x) {
    return (Fixed)(x << 24);
}

inline const Fixed fixed_from_float(double f) {
    // Multiply by 2^24 and cast to int to get Q8.24 representation.
    int raw = (int)(f * 16777216.0);
    return (Fixed)raw;
}

inline const int fixed_to_int(Fixed x) {
    // Arithmetic right shift drops the fractional bits.
    return (int)x >> 24;
}

inline const double fixed_to_float(Fixed x) {
    return (double)(int)x / 16777216.0;
}

inline const Fixed fixed_add(Fixed a, Fixed b) {
    return (Fixed)((int)a + (int)b);
}

inline const Fixed fixed_sub(Fixed a, Fixed b) {
    return (Fixed)((int)a - (int)b);
}

inline const Fixed fixed_mul(Fixed a, Fixed b) {
    // Promote to 64-bit to avoid intermediate overflow.
    long long product = (long long)(int)a * (long long)(int)b;
    return (Fixed)(int)(product >> 24);
}

inline const Fixed fixed_div(Fixed a, Fixed b) {
    // Shift a up by 24 before dividing so the result stays in Q8.24.
    long long num = (long long)(int)a << 24;
    return (Fixed)(int)(num / (long long)(int)b);
}

inline const Fixed fixed_abs(Fixed x) {
    int raw = (int)x;
    if (raw < 0) {
        return (Fixed)(-raw);
    }
    return (Fixed)raw;
}

inline const Fixed fixed_neg(Fixed x) {
    return (Fixed)(-(int)x);
}

// Newton-Raphson integer square root on Q8.24.
// We compute sqrt(x) in Q8.24 by treating the underlying int as a Q32 value
// and performing 4 Newton iterations: r = (r + x/r) / 2.
inline const Fixed fixed_sqrt(Fixed x) {
    if ((int)x <= 0) {
        return (Fixed)0;
    }

    // Initial guess: shift raw right 12 bits (midpoint of the
    // Q24 fractional scale).
    long long raw = (long long)(int)x;
    long long r = raw >> 12;
    if (r == 0) {
        r = 1;
    }

    // 4 Newton-Raphson iterations for Q8.24 precision.
    r = (r + raw / r) / (long long)2;
    r = (r + raw / r) / (long long)2;
    r = (r + raw / r) / (long long)2;
    r = (r + raw / r) / (long long)2;

    // Newton-Raphson above converges 'r' to the plain integer sqrt(raw), not
    // the Q8.24 result: sqrt(x) = sqrt(raw/2^24) = sqrt(raw)/2^12, so
    // sqrt(x) in Q8.24 is sqrt(raw)/2^12 * 2^24 = sqrt(raw) * 2^12.
    return (Fixed)(int)(r << 12);
}
