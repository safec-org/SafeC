#pragma once
#include <std/dsp/fixed.h>

namespace std {

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
    // sqrt(x) in Q8.24 is sqrt(raw / 2^24) * 2^24 = sqrt(raw * 2^24): take
    // the exact integer square root of the widened value directly (raw <
    // 2^31, so raw << 24 < 2^55 fits). Bit-by-bit method: 28 fixed
    // iterations, exact floor result for every input — no initial-guess
    // dependence like Newton's method, which needs input-dependent
    // iteration counts to converge.
    unsigned long long v = (unsigned long long)(long long)(int)x << 24;
    unsigned long long res = 0ULL;
    unsigned long long bit = 1ULL << 54;
    while (bit > v) { bit = bit >> 2; }
    while (bit != 0ULL) {
        if (v >= res + bit) {
            v = v - (res + bit);
            res = (res >> 1) + bit;
        } else {
            res = res >> 1;
        }
        bit = bit >> 2;
    }
    return (Fixed)(int)res;
}

} // namespace std
