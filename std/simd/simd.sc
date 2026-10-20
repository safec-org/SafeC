// SafeC Standard Library — std::simd (portable core implementation)
#pragma once
#include <std/simd/simd.h>

namespace std {

// ── Load ───────────────────────────────────────────────────────────────────
// One unaligned vector load/store each: dereferencing a vec pointer into an
// element array only assumes the element's alignment.
f32x4 simd_load_f32x4(const &float p) {
    unsafe { return *(const f32x4*)p; }
}

f32x8 simd_load_f32x8(const &float p) {
    unsafe { return *(const f32x8*)p; }
}

f64x2 simd_load_f64x2(const &double p) {
    unsafe { return *(const f64x2*)p; }
}

f64x4 simd_load_f64x4(const &double p) {
    unsafe { return *(const f64x4*)p; }
}

i32x4 simd_load_i32x4(const &int p) {
    unsafe { return *(const i32x4*)p; }
}

i32x8 simd_load_i32x8(const &int p) {
    unsafe { return *(const i32x8*)p; }
}

u32x4 simd_load_u32x4(const &unsigned int p) {
    unsafe { return *(const u32x4*)p; }
}

u32x8 simd_load_u32x8(const &unsigned int p) {
    unsafe { return *(const u32x8*)p; }
}

i8x16 simd_load_i8x16(const &signed char p) {
    unsafe { return *(const i8x16*)p; }
}

u8x16 simd_load_u8x16(const &unsigned char p) {
    unsafe { return *(const u8x16*)p; }
}

// ── Store ──────────────────────────────────────────────────────────────────
void simd_store_f32x4(f32x4 v, &float p) {
    unsafe { *(f32x4*)p = v; }
}

void simd_store_f32x8(f32x8 v, &float p) {
    unsafe { *(f32x8*)p = v; }
}

void simd_store_f64x2(f64x2 v, &double p) {
    unsafe { *(f64x2*)p = v; }
}

void simd_store_f64x4(f64x4 v, &double p) {
    unsafe { *(f64x4*)p = v; }
}

void simd_store_i32x4(i32x4 v, &int p) {
    unsafe { *(i32x4*)p = v; }
}

void simd_store_i32x8(i32x8 v, &int p) {
    unsafe { *(i32x8*)p = v; }
}

void simd_store_u32x4(u32x4 v, &unsigned int p) {
    unsafe { *(u32x4*)p = v; }
}

void simd_store_u32x8(u32x8 v, &unsigned int p) {
    unsafe { *(u32x8*)p = v; }
}

void simd_store_i8x16(i8x16 v, &signed char p) {
    unsafe { *(i8x16*)p = v; }
}

void simd_store_u8x16(u8x16 v, &unsigned char p) {
    unsafe { *(u8x16*)p = v; }
}

// ── Broadcast (splat) ──────────────────────────────────────────────────────
f32x4 simd_splat_f32x4(float x) {
    f32x4 v;
    v[0] = x;
        v[1] = x;
        v[2] = x;
        v[3] = x;
    return v;
}

f32x8 simd_splat_f32x8(float x) {
    f32x8 v;
    v[0] = x;
        v[1] = x;
        v[2] = x;
        v[3] = x;
        v[4] = x;
        v[5] = x;
        v[6] = x;
        v[7] = x;
    return v;
}

f64x2 simd_splat_f64x2(double x) {
    f64x2 v;
    v[0] = x;
        v[1] = x;
    return v;
}

f64x4 simd_splat_f64x4(double x) {
    f64x4 v;
    v[0] = x;
        v[1] = x;
        v[2] = x;
        v[3] = x;
    return v;
}

i32x4 simd_splat_i32x4(int x) {
    i32x4 v;
    v[0] = x;
        v[1] = x;
        v[2] = x;
        v[3] = x;
    return v;
}

i32x8 simd_splat_i32x8(int x) {
    i32x8 v;
    v[0] = x;
        v[1] = x;
        v[2] = x;
        v[3] = x;
        v[4] = x;
        v[5] = x;
        v[6] = x;
        v[7] = x;
    return v;
}

u32x4 simd_splat_u32x4(unsigned int x) {
    u32x4 v;
    v[0] = x;
        v[1] = x;
        v[2] = x;
        v[3] = x;
    return v;
}

// ── Fused multiply-add: a*b + c ───────────────────────────────────────────
f32x4 simd_fma_f32x4(f32x4 a, f32x4 b, f32x4 c) {
    return __builtin_fmuladd(a, b, c);
}

f32x8 simd_fma_f32x8(f32x8 a, f32x8 b, f32x8 c) {
    return __builtin_fmuladd(a, b, c);
}

f64x2 simd_fma_f64x2(f64x2 a, f64x2 b, f64x2 c) {
    return __builtin_fmuladd(a, b, c);
}

f64x4 simd_fma_f64x4(f64x4 a, f64x4 b, f64x4 c) {
    return __builtin_fmuladd(a, b, c);
}

// ── Elementwise min/max (per-lane; LLVM's SLP vectorizer commonly
// re-fuses this unrolled form back into a single vector min/max
// instruction at -O2+, same as it does for the load/store loops above) ──
f32x4 simd_min_f32x4(f32x4 a, f32x4 b) {
    f32x4 r;
    r[0] = a[0] < b[0] ? a[0] : b[0];
    r[1] = a[1] < b[1] ? a[1] : b[1];
    r[2] = a[2] < b[2] ? a[2] : b[2];
    r[3] = a[3] < b[3] ? a[3] : b[3];
    return r;
}

f32x4 simd_max_f32x4(f32x4 a, f32x4 b) {
    f32x4 r;
    r[0] = a[0] > b[0] ? a[0] : b[0];
    r[1] = a[1] > b[1] ? a[1] : b[1];
    r[2] = a[2] > b[2] ? a[2] : b[2];
    r[3] = a[3] > b[3] ? a[3] : b[3];
    return r;
}

f32x8 simd_min_f32x8(f32x8 a, f32x8 b) {
    f32x8 r;
    r[0] = a[0] < b[0] ? a[0] : b[0];
    r[1] = a[1] < b[1] ? a[1] : b[1];
    r[2] = a[2] < b[2] ? a[2] : b[2];
    r[3] = a[3] < b[3] ? a[3] : b[3];
    r[4] = a[4] < b[4] ? a[4] : b[4];
    r[5] = a[5] < b[5] ? a[5] : b[5];
    r[6] = a[6] < b[6] ? a[6] : b[6];
    r[7] = a[7] < b[7] ? a[7] : b[7];
    return r;
}

f32x8 simd_max_f32x8(f32x8 a, f32x8 b) {
    f32x8 r;
    r[0] = a[0] > b[0] ? a[0] : b[0];
    r[1] = a[1] > b[1] ? a[1] : b[1];
    r[2] = a[2] > b[2] ? a[2] : b[2];
    r[3] = a[3] > b[3] ? a[3] : b[3];
    r[4] = a[4] > b[4] ? a[4] : b[4];
    r[5] = a[5] > b[5] ? a[5] : b[5];
    r[6] = a[6] > b[6] ? a[6] : b[6];
    r[7] = a[7] > b[7] ? a[7] : b[7];
    return r;
}

i32x4 simd_min_i32x4(i32x4 a, i32x4 b) {
    i32x4 r;
    r[0] = a[0] < b[0] ? a[0] : b[0];
    r[1] = a[1] < b[1] ? a[1] : b[1];
    r[2] = a[2] < b[2] ? a[2] : b[2];
    r[3] = a[3] < b[3] ? a[3] : b[3];
    return r;
}

i32x4 simd_max_i32x4(i32x4 a, i32x4 b) {
    i32x4 r;
    r[0] = a[0] > b[0] ? a[0] : b[0];
    r[1] = a[1] > b[1] ? a[1] : b[1];
    r[2] = a[2] > b[2] ? a[2] : b[2];
    r[3] = a[3] > b[3] ? a[3] : b[3];
    return r;
}

// ── Horizontal reductions ──────────────────────────────────────────────────
float simd_hsum_f32x4(f32x4 v) {
    return v[0] + v[1] + v[2] + v[3];
}

float simd_hmin_f32x4(f32x4 v) {
    float acc = v[0];
    if (v[1] < acc) { acc = v[1]; }
    if (v[2] < acc) { acc = v[2]; }
    if (v[3] < acc) { acc = v[3]; }
    return acc;
}

float simd_hmax_f32x4(f32x4 v) {
    float acc = v[0];
    if (v[1] > acc) { acc = v[1]; }
    if (v[2] > acc) { acc = v[2]; }
    if (v[3] > acc) { acc = v[3]; }
    return acc;
}

float simd_hsum_f32x8(f32x8 v) {
    return v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
}

double simd_hsum_f64x2(f64x2 v) {
    return v[0] + v[1];
}

double simd_hsum_f64x4(f64x4 v) {
    return v[0] + v[1] + v[2] + v[3];
}

int simd_hsum_i32x4(i32x4 v) {
    return v[0] + v[1] + v[2] + v[3];
}

int simd_hsum_i32x8(i32x8 v) {
    return v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
}

unsigned int simd_hsum_u32x4(u32x4 v) {
    return v[0] + v[1] + v[2] + v[3];
}

} // namespace std
