// SafeC Standard Library — vectorized math (see vmath.h).
//
// Cephes-style single precision: exp reduces x = n ln2 + r with a two-part
// ln2 and evaluates a degree-6 polynomial on |r| <= ln2/2, then scales by
// 2^n built from the exponent bits. Every branch is a select, so the loops
// vectorize.
#pragma once
#include <std/simd/vmath.h>
#include <std/sync/parallel.h>
#include <std/simd/cpu.h>

namespace std {

static float vm_bits_(int bits) __attribute__((always_inline)) {
    float f;
    unsafe { f = *(float*)&bits; }
    return f;
}

static float vm_exp1_(float xo) __attribute__((always_inline)) {
    float x = xo > (float)88.7228390 ? (float)88.7228390 : xo;
    x = x < (float)-103.0 ? (float)-103.0 : x;
    float fx = x * (float)1.44269504088896341 + (float)0.5;
    int n = (int)fx;
    n = (float)n > fx ? n - 1 : n;                       // floor
    float nf = (float)n;
    float r = x - nf * (float)0.693359375;
    r = r + nf * (float)2.12194440e-4;
    float z = r * r;
    float y = (float)1.9875691500e-4;
    y = __builtin_fmuladd(y, r, (float)1.3981999507e-3);
    y = __builtin_fmuladd(y, r, (float)8.3334519073e-3);
    y = __builtin_fmuladd(y, r, (float)4.1665795894e-2);
    y = __builtin_fmuladd(y, r, (float)1.6666665459e-1);
    y = __builtin_fmuladd(y, r, (float)5.0000001201e-1);
    y = __builtin_fmuladd(y, z, r + (float)1.0);
    // 2^n as two factors with valid exponents, so n = 128 (results just
    // below FLT_MAX) and n down to -149 (subnormal results) both work.
    int n1 = n < -126 ? -126 : (n > 127 ? 127 : n);
    int n2 = n - n1;
    float res = y * vm_bits_((n1 + 127) << 23) * vm_bits_((n2 + 127) << 23);
    res = xo > (float)88.7228390 ? vm_bits_(2139095040) : res;   // +inf
    res = xo < (float)-103.972077 ? (float)0.0 : res;
    return xo != xo ? xo : res;                                 // NaN
}

static float vm_tanh1_(float x) __attribute__((always_inline)) {
    float ax = x < (float)0.0 ? -x : x;
    // |x| < 0.625: odd polynomial (no cancellation near 0)
    float z = x * x;
    float p = (float)-5.70498872745e-3;
    p = __builtin_fmuladd(p, z, (float)2.06390887954e-2);
    p = __builtin_fmuladd(p, z, (float)-5.37397155531e-2);
    p = __builtin_fmuladd(p, z, (float)1.33314422036e-1);
    p = __builtin_fmuladd(p, z, (float)-3.33332819422e-1);
    float small = __builtin_fmuladd(x * z, p, x);
    // otherwise 1 - 2 / (exp(2|x|) + 1), with the sign of x
    float e = vm_exp1_((float)2.0 * (ax > (float)9.0 ? (float)9.0 : ax));
    float big = (float)1.0 - (float)2.0 / (e + (float)1.0);
    big = ax > (float)9.0 ? (float)1.0 : big;
    big = x < (float)0.0 ? -big : big;
    float res = ax < (float)0.625 ? small : big;
    return x != x ? x : res;
}

static float vm_sigmoid1_(float x) __attribute__((always_inline)) {
    return (float)1.0 / ((float)1.0 + vm_exp1_(-x));
}

#define VM_EXP 0
#define VM_TANH 1
#define VM_SIGMOID 2

struct VmJob {
    const float* x;
    float*       y;
    int          op;
};

// One loop body, compiled per ISA level (std/simd/cpu.h) by the target()
// wrappers; the compiler vectorizes it at each one's width.
static void vm_loop_(const float* x, float* y, unsigned long lo, unsigned long hi, int op)
    __attribute__((always_inline)) {
    unsafe {
        unsigned long i = lo;
        if (op == VM_EXP) {
            while (i < hi) { y[i] = vm_exp1_(x[i]); i = i + 1UL; }
        } else if (op == VM_TANH) {
            while (i < hi) { y[i] = vm_tanh1_(x[i]); i = i + 1UL; }
        } else {
            while (i < hi) { y[i] = vm_sigmoid1_(x[i]); i = i + 1UL; }
        }
    }
}

static void vm_loop_base_(const float* x, float* y, unsigned long lo, unsigned long hi, int op) {
    vm_loop_(x, y, lo, hi, op);
}
#if defined(__x86_64__)
static void vm_loop_sse41_(const float* x, float* y, unsigned long lo, unsigned long hi, int op)
    __attribute__((target("sse4.1"))) { vm_loop_(x, y, lo, hi, op); }
static void vm_loop_avx_(const float* x, float* y, unsigned long lo, unsigned long hi, int op)
    __attribute__((target("avx"))) { vm_loop_(x, y, lo, hi, op); }
static void vm_loop_avx2_(const float* x, float* y, unsigned long lo, unsigned long hi, int op)
    __attribute__((target("avx2,fma"))) { vm_loop_(x, y, lo, hi, op); }
static void vm_loop_avx512_(const float* x, float* y, unsigned long lo, unsigned long hi, int op)
    __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma"))) { vm_loop_(x, y, lo, hi, op); }
#elif defined(__aarch64__) && defined(__linux__) && !defined(__SAFEC_FREESTANDING__)
static void vm_loop_sve_(const float* x, float* y, unsigned long lo, unsigned long hi, int op)
    __attribute__((target("sve"))) { vm_loop_(x, y, lo, hi, op); }
#endif

static void vm_range_(&VmJob j, unsigned long lo, unsigned long hi) {
    int l = simd_level();
#if defined(__x86_64__)
    if (l >= SIMD_AVX512) { vm_loop_avx512_(j.x, j.y, lo, hi, j.op); return; }
    if (l >= SIMD_AVX2) { vm_loop_avx2_(j.x, j.y, lo, hi, j.op); return; }
    if (l >= SIMD_AVX) { vm_loop_avx_(j.x, j.y, lo, hi, j.op); return; }
    if (l >= SIMD_SSE41) { vm_loop_sse41_(j.x, j.y, lo, hi, j.op); return; }
#elif defined(__aarch64__) && defined(__linux__) && !defined(__SAFEC_FREESTANDING__)
    if (l >= SIMD_SVE) { vm_loop_sve_(j.x, j.y, lo, hi, j.op); return; }
#endif
    vm_loop_base_(j.x, j.y, lo, hi, j.op);
}

static void vm_run_(int op, const &float x, &float y, unsigned long n) {
    struct VmJob j;
    unsafe {
        j.x = (const float*)x;
        j.y = (float*)y;
    }
    j.op = op;
    if (n >= 32768UL) parallel_for(n, 8192UL, vm_range_, &j);
    else vm_range_(&j, 0UL, n);
}

void vm_exp_f32(const &float x, &float y, unsigned long n) { vm_run_(VM_EXP, x, y, n); }
void vm_tanh_f32(const &float x, &float y, unsigned long n) { vm_run_(VM_TANH, x, y, n); }
void vm_sigmoid_f32(const &float x, &float y, unsigned long n) { vm_run_(VM_SIGMOID, x, y, n); }

} // namespace std
