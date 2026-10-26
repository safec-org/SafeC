// SafeC Standard Library — S24 packed 24-bit samples (see s24.h).
#pragma once
#include <std/audio/s24.h>

namespace std {

struct S24 S24::of(int v) {
    struct S24 s;
    s.set(v);
    return s;
}

struct S24 S24::from_f32(float v) {
    double x = (double)v * 8388607.0;
    int i;
    if (x != x) i = 0;                       // NaN
    else if (x <= -8388608.0) i = S24_MIN;
    else if (x >= 8388607.0) i = S24_MAX;
    else if (x >= 0.0) i = (int)(x + 0.5);
    else i = 0 - (int)(0.5 - x);
    return S24::of(i);
}

int S24::value() const {
    int v = (int)self.b[0] | ((int)self.b[1] << 8) | ((int)self.b[2] << 16);
    if ((v & 0x800000) != 0) v = v - 0x1000000;
    return v;
}

float S24::to_f32() const {
    return (float)((double)self.value() / 8388608.0);
}

void S24::set(int v) {
    if (v < S24_MIN) v = S24_MIN;
    if (v > S24_MAX) v = S24_MAX;
    unsigned int u = (unsigned int)v & 0xFFFFFFU;
    self.b[0] = (unsigned char)(u & 0xFFU);
    self.b[1] = (unsigned char)((u >> 8) & 0xFFU);
    self.b[2] = (unsigned char)((u >> 16) & 0xFFU);
}

} // namespace std
