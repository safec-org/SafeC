// SafeC Standard Library — fp16/bf16 conversion implementation (see
// float16.h). Bit-level float32 access goes through memcpy into/out of an
// unsigned int, the same reinterpret-via-memcpy pattern gpu_webgpu.sc
// already uses for packing scalars into a push-constant/uniform buffer --
// avoids relying on pointer-punning through mismatched types, which
// safec (like C) doesn't guarantee is well-defined.
#pragma once
#include <std/ml/float16.h>

namespace std {

extern void* memcpy(void* dst, const void* src, unsigned long n);

static unsigned int __f32_bits(float v) {
    unsigned int bits = 0U;
    unsafe { memcpy((void*)&bits, (const void*)&v, 4UL); }
    return bits;
}

static float __bits_f32(unsigned int bits) {
    float v = 0.0f;
    unsafe { memcpy((void*)&v, (const void*)&bits, 4UL); }
    return v;
}

unsigned short f32_to_fp16(float v) {
    unsigned int bits = __f32_bits(v);
    unsigned int sign = (bits >> 16) & 0x8000U;
    unsigned int expField = (bits >> 23) & 0xFFU;
    unsigned int mant = bits & 0x7FFFFFU;
    int exp = (int)expField - 127 + 15; // rebias float32's exponent into fp16's

    if (expField == 0xFFU) {
        // Infinity (mant == 0) or NaN (mant != 0) -- force a nonzero
        // mantissa bit for NaN so it can't collapse into infinity.
        unsigned int m = (mant != 0U) ? 0x200U : 0U;
        return (unsigned short)(sign | 0x7C00U | m);
    }
    if (exp >= 0x1F) {
        // Overflow -- magnitude too large for fp16, rounds to infinity.
        return (unsigned short)(sign | 0x7C00U);
    }
    if (exp <= 0) {
        // Subnormal (or underflows to zero) in fp16.
        if (exp < -10) {
            return (unsigned short)sign; // too small to round up to even the smallest subnormal
        }
        unsigned int sig = mant | 0x800000U; // restore the implicit leading 1 -> 24-bit significand
        unsigned int shift = (unsigned int)(14 - exp);
        unsigned int roundedMant = sig >> shift;
        unsigned int remainder = sig & ((1U << shift) - 1U);
        unsigned int halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (roundedMant & 1U) == 1U)) {
            roundedMant = roundedMant + 1U;
        }
        return (unsigned short)(sign | roundedMant);
    }

    // Normal range: round the 23-bit mantissa down to 10 bits,
    // round-to-nearest-even, with carry-out bumping the exponent (and
    // possibly overflowing to infinity) on a mantissa round-up to 0x400.
    unsigned int roundedMant = mant >> 13;
    unsigned int remainder = mant & 0x1FFFU;
    unsigned int halfway = 0x1000U;
    if (remainder > halfway || (remainder == halfway && (roundedMant & 1U) == 1U)) {
        roundedMant = roundedMant + 1U;
        if (roundedMant == 0x400U) {
            roundedMant = 0U;
            exp = exp + 1;
            if (exp >= 0x1F) {
                return (unsigned short)(sign | 0x7C00U);
            }
        }
    }
    return (unsigned short)(sign | ((unsigned int)exp << 10) | roundedMant);
}

float fp16_to_f32(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000U) << 16;
    unsigned int exp = ((unsigned int)h >> 10) & 0x1FU;
    unsigned int mant = (unsigned int)h & 0x3FFU;

    if (exp == 0U) {
        if (mant == 0U) {
            return __bits_f32(sign); // signed zero
        }
        // Subnormal: normalize by left-shifting until the implicit
        // leading bit (bit 10) is set, tracking the shift count so the
        // float32 exponent can be rebased to match.
        unsigned int m = mant;
        unsigned int s = 0U;
        while ((m & 0x400U) == 0U) {
            m = m << 1;
            s = s + 1U;
        }
        m = m & 0x3FFU; // drop the now-explicit leading bit
        unsigned int outExp = 113U - s; // = (-14 - 1 - s) rebiased to float32's 127 bias -- see float16.sc's derivation
        unsigned int outMant = m << 13;
        return __bits_f32(sign | (outExp << 23) | outMant);
    }
    if (exp == 0x1FU) {
        return __bits_f32(sign | 0x7F800000U | (mant << 13)); // infinity or NaN
    }
    unsigned int outExp = exp - 15U + 127U;
    unsigned int outMant = mant << 13;
    return __bits_f32(sign | (outExp << 23) | outMant);
}

unsigned short f32_to_bf16(float v) {
    unsigned int bits = __f32_bits(v);
    unsigned int expField = (bits >> 23) & 0xFFU;
    if (expField == 0xFFU && (bits & 0x7FFFFFU) != 0U) {
        // NaN -- truncating could zero out the mantissa entirely (turning
        // it into infinity), so force one mantissa bit set.
        unsigned int top = (bits >> 16) & 0xFFFFU;
        return (unsigned short)(top | 0x0040U);
    }
    unsigned int roundBit = (bits >> 15) & 1U;
    unsigned int sticky = ((bits & 0x7FFFU) != 0U) ? 1U : 0U;
    unsigned int lsb = (bits >> 16) & 1U;
    unsigned int roundUp = (roundBit != 0U && (sticky != 0U || lsb != 0U)) ? 1U : 0U;
    unsigned int result = (bits >> 16) + roundUp; // plain integer add -- a mantissa carry correctly ripples into the (shared-width) exponent field
    return (unsigned short)(result & 0xFFFFU);
}

float bf16_to_f32(unsigned short b) {
    unsigned int bits = ((unsigned int)b) << 16;
    return __bits_f32(bits);
}

void f32_to_fp16_bulk(const float* src, unsigned short* dst, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { dst[i] = f32_to_fp16(src[i]); i = i + 1UL; }
    }
}

void fp16_to_f32_bulk(const unsigned short* src, float* dst, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { dst[i] = fp16_to_f32(src[i]); i = i + 1UL; }
    }
}

void f32_to_bf16_bulk(const float* src, unsigned short* dst, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { dst[i] = f32_to_bf16(src[i]); i = i + 1UL; }
    }
}

void bf16_to_f32_bulk(const unsigned short* src, float* dst, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { dst[i] = bf16_to_f32(src[i]); i = i + 1UL; }
    }
}

} // namespace std
