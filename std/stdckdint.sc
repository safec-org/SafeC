// SafeC Standard Library — Checked integer arithmetic (C23)
// Pure SafeC implementation — overflow is detected before the operation.
#pragma once
#include "stdckdint.h"

// ── 32-bit signed ─────────────────────────────────────────────────────────────

int ckd_add_i32(int* result, int a, int b) {
    int overflow = 0;
    if (b > 0 && a > 2147483647 - b)         { overflow = 1; }
    if (b < 0 && a < (-2147483647 - 1) - b) { overflow = 1; }
    unsafe { *result = a + b; }
    return overflow;
}

int ckd_sub_i32(int* result, int a, int b) {
    int overflow = 0;
    if (b < 0 && a > 2147483647 + b)         { overflow = 1; }
    if (b > 0 && a < (-2147483647 - 1) + b) { overflow = 1; }
    unsafe { *result = a - b; }
    return overflow;
}

int ckd_mul_i32(int* result, int a, int b) {
    long long wide = (long long)a * (long long)b;
    int overflow = 0;
    if (wide > 2147483647LL || wide < (long long)(-2147483647 - 1)) {
        overflow = 1;
    }
    unsafe { *result = (int)wide; }
    return overflow;
}

// ── 64-bit signed ─────────────────────────────────────────────────────────────

int ckd_add_i64(long long* result, long long a, long long b) {
    int overflow = 0;
    if (b > 0LL && a > 9223372036854775807LL - b)                          { overflow = 1; }
    if (b < 0LL && a < (-9223372036854775807LL - (long long)1) - b)       { overflow = 1; }
    unsafe { *result = a + b; }
    return overflow;
}

int ckd_sub_i64(long long* result, long long a, long long b) {
    int overflow = 0;
    if (b < 0LL && a > 9223372036854775807LL + b)                          { overflow = 1; }
    if (b > 0LL && a < (-9223372036854775807LL - (long long)1) + b)       { overflow = 1; }
    unsafe { *result = a - b; }
    return overflow;
}

int ckd_mul_i64(long long* result, long long a, long long b) {
    long long llong_min = -9223372036854775807LL - (long long)1;
    long long llong_max = 9223372036854775807LL;
    int overflow = 0;
    if (a != 0LL && b != 0LL) {
        if (a > 0LL && b > 0LL && a > llong_max / b)             { overflow = 1; }
        if (a < 0LL && b < 0LL && a < llong_max / b)             { overflow = 1; }
        if (a > 0LL && b < 0LL && b < llong_min / a)             { overflow = 1; }
        if (a < 0LL && b > 0LL && a < llong_min / b)             { overflow = 1; }
    }
    unsafe { *result = a * b; }
    return overflow;
}

// ── 32-bit unsigned ───────────────────────────────────────────────────────────

int ckd_add_u32(unsigned int* result, unsigned int a, unsigned int b) {
    unsigned int r = a + b;
    unsafe { *result = r; }
    if (r < a) { return 1; }
    return 0;
}

int ckd_sub_u32(unsigned int* result, unsigned int a, unsigned int b) {
    unsafe { *result = a - b; }
    if (b > a) { return 1; }
    return 0;
}

int ckd_mul_u32(unsigned int* result, unsigned int a, unsigned int b) {
    unsigned long long wide = (unsigned long long)a * (unsigned long long)b;
    int overflow = 0;
    if (wide > 4294967295ULL) { overflow = 1; }
    unsafe { *result = (unsigned int)wide; }
    return overflow;
}

// ── 64-bit unsigned ───────────────────────────────────────────────────────────

int ckd_add_u64(unsigned long long* result, unsigned long long a, unsigned long long b) {
    unsigned long long r = a + b;
    unsafe { *result = r; }
    if (r < a) { return 1; }
    return 0;
}

int ckd_sub_u64(unsigned long long* result, unsigned long long a, unsigned long long b) {
    unsafe { *result = a - b; }
    if (b > a) { return 1; }
    return 0;
}

int ckd_mul_u64(unsigned long long* result, unsigned long long a, unsigned long long b) {
    int overflow = 0;
    if (a != 0ULL && b > 18446744073709551615ULL / a) { overflow = 1; }
    unsafe { *result = a * b; }
    return overflow;
}
