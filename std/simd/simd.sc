// SafeC Standard Library — std::simd (portable core implementation)
#pragma once
#include <std/simd/simd.h>

namespace std {

// ── Load ───────────────────────────────────────────────────────────────────
f32x4 simd_load_f32x4(const float* p) {
    f32x4 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
    }
    return v;
}

f32x8 simd_load_f32x8(const float* p) {
    f32x8 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
        v[4] = p[4];
        v[5] = p[5];
        v[6] = p[6];
        v[7] = p[7];
    }
    return v;
}

f64x2 simd_load_f64x2(const double* p) {
    f64x2 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
    }
    return v;
}

f64x4 simd_load_f64x4(const double* p) {
    f64x4 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
    }
    return v;
}

i32x4 simd_load_i32x4(const int* p) {
    i32x4 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
    }
    return v;
}

i32x8 simd_load_i32x8(const int* p) {
    i32x8 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
        v[4] = p[4];
        v[5] = p[5];
        v[6] = p[6];
        v[7] = p[7];
    }
    return v;
}

u32x4 simd_load_u32x4(const unsigned int* p) {
    u32x4 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
    }
    return v;
}

u32x8 simd_load_u32x8(const unsigned int* p) {
    u32x8 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
        v[4] = p[4];
        v[5] = p[5];
        v[6] = p[6];
        v[7] = p[7];
    }
    return v;
}

i8x16 simd_load_i8x16(const signed char* p) {
    i8x16 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
        v[4] = p[4];
        v[5] = p[5];
        v[6] = p[6];
        v[7] = p[7];
        v[8] = p[8];
        v[9] = p[9];
        v[10] = p[10];
        v[11] = p[11];
        v[12] = p[12];
        v[13] = p[13];
        v[14] = p[14];
        v[15] = p[15];
    }
    return v;
}

u8x16 simd_load_u8x16(const unsigned char* p) {
    u8x16 v;
    unsafe {
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        v[3] = p[3];
        v[4] = p[4];
        v[5] = p[5];
        v[6] = p[6];
        v[7] = p[7];
        v[8] = p[8];
        v[9] = p[9];
        v[10] = p[10];
        v[11] = p[11];
        v[12] = p[12];
        v[13] = p[13];
        v[14] = p[14];
        v[15] = p[15];
    }
    return v;
}

// ── Store ──────────────────────────────────────────────────────────────────
void simd_store_f32x4(f32x4 v, float* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }
}

void simd_store_f32x8(f32x8 v, float* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
        p[4] = v[4];
        p[5] = v[5];
        p[6] = v[6];
        p[7] = v[7];
    }
}

void simd_store_f64x2(f64x2 v, double* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
    }
}

void simd_store_f64x4(f64x4 v, double* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }
}

void simd_store_i32x4(i32x4 v, int* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }
}

void simd_store_i32x8(i32x8 v, int* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
        p[4] = v[4];
        p[5] = v[5];
        p[6] = v[6];
        p[7] = v[7];
    }
}

void simd_store_u32x4(u32x4 v, unsigned int* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }
}

void simd_store_u32x8(u32x8 v, unsigned int* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
        p[4] = v[4];
        p[5] = v[5];
        p[6] = v[6];
        p[7] = v[7];
    }
}

void simd_store_i8x16(i8x16 v, signed char* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
        p[4] = v[4];
        p[5] = v[5];
        p[6] = v[6];
        p[7] = v[7];
        p[8] = v[8];
        p[9] = v[9];
        p[10] = v[10];
        p[11] = v[11];
        p[12] = v[12];
        p[13] = v[13];
        p[14] = v[14];
        p[15] = v[15];
    }
}

void simd_store_u8x16(u8x16 v, unsigned char* p) {
    unsafe {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
        p[4] = v[4];
        p[5] = v[5];
        p[6] = v[6];
        p[7] = v[7];
        p[8] = v[8];
        p[9] = v[9];
        p[10] = v[10];
        p[11] = v[11];
        p[12] = v[12];
        p[13] = v[13];
        p[14] = v[14];
        p[15] = v[15];
    }
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
    return a * b + c;
}

f32x8 simd_fma_f32x8(f32x8 a, f32x8 b, f32x8 c) {
    return a * b + c;
}

f64x2 simd_fma_f64x2(f64x2 a, f64x2 b, f64x2 c) {
    return a * b + c;
}

f64x4 simd_fma_f64x4(f64x4 a, f64x4 b, f64x4 c) {
    return a * b + c;
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
