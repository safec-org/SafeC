#pragma once
// SafeC Standard Library — tensor element types.
//
// A DenseTensor<E> (tensor.h) stores its values as E and computes in
// float32: an op loads its operands to float32, runs the float32 kernel and
// stores the result as E, and every gradient is float32. E is 'float'
// itself for FloatTensor, whose values are already float32 (no conversion,
// no copy), or one of the 16-bit storage types below for HalfTensor and
// BF16Tensor. A storage type provides two static conversions, which
// generic code reaches through the type parameter ('E::load_n(...)'):
//
//   load_n   n stored values -> float32
//   store_n  n float32 values -> stored values, rounded to nearest even
//
// Both use float16.h's bulk conversions (hardware where the CPU has it).
#include <std/ml/float16.h>

namespace std {

// IEEE binary16: 1 sign, 5 exponent, 10 mantissa bits.
struct F16 {
    unsigned short bits;

    static void load_n(const &F16 src, &float dst, unsigned long n);
    static void store_n(const &float src, &F16 dst, unsigned long n);
};

// bfloat16: float32's sign and 8 exponent bits, 7 mantissa bits.
struct BF16 {
    unsigned short bits;

    static void load_n(const &BF16 src, &float dst, unsigned long n);
    static void store_n(const &float src, &BF16 dst, unsigned long n);
};

} // namespace std
