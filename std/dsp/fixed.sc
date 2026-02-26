#pragma once
#include "fixed.h"

newtype Fixed = int;

Fixed fixed_from_int(int x) {
    return (Fixed)(x << 16);
}

Fixed fixed_from_float(double f) {
    // Multiply by 65536.0 and cast to int to get Q16.16 representation.
    int raw = (int)(f * 65536.0);
    return (Fixed)raw;
}

int fixed_to_int(Fixed x) {
    // Arithmetic right shift drops the fractional bits.
    return (int)x >> 16;
}

double fixed_to_float(Fixed x) {
    return (double)(int)x / 65536.0;
}

Fixed fixed_add(Fixed a, Fixed b) {
    return (Fixed)((int)a + (int)b);
}

Fixed fixed_sub(Fixed a, Fixed b) {
    return (Fixed)((int)a - (int)b);
}

Fixed fixed_mul(Fixed a, Fixed b) {
    // Promote to 64-bit to avoid intermediate overflow.
    long long product = (long long)(int)a * (long long)(int)b;
    return (Fixed)(int)(product >> 16);
}

Fixed fixed_div(Fixed a, Fixed b) {
    // Shift a up by 16 before dividing so the result stays in Q16.16.
    long long num = (long long)(int)a << 16;
    return (Fixed)(int)(num / (long long)(int)b);
}

Fixed fixed_abs(Fixed x) {
    int raw = (int)x;
    if (raw < 0) {
        return (Fixed)(-raw);
    }
    return (Fixed)raw;
}

Fixed fixed_neg(Fixed x) {
    return (Fixed)(-(int)x);
}

// Newton-Raphson integer square root on Q16.16.
// We compute sqrt(x) in Q16.16 by treating the underlying int as a Q32 value
// and performing 3 Newton iterations: r = (r + x/r) / 2.
Fixed fixed_sqrt(Fixed x) {
    if ((int)x <= 0) {
        return (Fixed)0;
    }

    // Initial guess: use the integer sqrt of the raw value as a starting point
    // then scale appropriately.  A safe initial estimate is (raw >> 8) which
    // gives a value in roughly the right Q16.16 ballpark.
    long long raw = (long long)(int)x;

    // Start with a coarse estimate: shift raw right 8 bits (midpoint of the
    // Q16 fractional scale).
    long long r = raw >> 8;
    if (r == 0) {
        r = 1;
    }

    // 3 Newton-Raphson iterations: r = (r + raw/r) / 2
    r = (r + raw / r) / 2;
    r = (r + raw / r) / 2;
    r = (r + raw / r) / 2;

    return (Fixed)(int)r;
}
