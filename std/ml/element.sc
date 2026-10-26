// SafeC Standard Library — tensor element types (see element.h).
#pragma once
#include <std/ml/element.h>
#include <std/ml/float16.h>

namespace std {

void F16::load_n(const &F16 src, &float dst, unsigned long n) {
    unsafe { fp16_to_f32_bulk_n_((const &unsigned short)(const unsigned short*)src, dst, n); }
}

void F16::store_n(const &float src, &F16 dst, unsigned long n) {
    unsafe { f32_to_fp16_bulk_n_(src, (&unsigned short)(unsigned short*)dst, n); }
}

void BF16::load_n(const &BF16 src, &float dst, unsigned long n) {
    unsafe { bf16_to_f32_bulk_n_((const &unsigned short)(const unsigned short*)src, dst, n); }
}

void BF16::store_n(const &float src, &BF16 dst, unsigned long n) {
    unsafe { f32_to_bf16_bulk_n_(src, (&unsigned short)(unsigned short*)dst, n); }
}

} // namespace std
