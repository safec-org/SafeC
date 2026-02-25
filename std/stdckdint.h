// SafeC Standard Library — Checked integer arithmetic (C23 <stdckdint.h>)
// Each function performs op(a, b), stores the wrapped result in *result,
// and returns 1 if the mathematical result overflowed, 0 otherwise.
// This is the SafeC equivalent of C23's type-generic ckd_add/ckd_sub/ckd_mul.
#pragma once

// ── Checked int (32-bit signed) ───────────────────────────────────────────────
int ckd_add_i32(int* result, int a, int b);
int ckd_sub_i32(int* result, int a, int b);
int ckd_mul_i32(int* result, int a, int b);

// ── Checked long long (64-bit signed) ────────────────────────────────────────
int ckd_add_i64(long long* result, long long a, long long b);
int ckd_sub_i64(long long* result, long long a, long long b);
int ckd_mul_i64(long long* result, long long a, long long b);

// ── Checked unsigned int (32-bit) ────────────────────────────────────────────
int ckd_add_u32(unsigned int* result, unsigned int a, unsigned int b);
int ckd_sub_u32(unsigned int* result, unsigned int a, unsigned int b);
int ckd_mul_u32(unsigned int* result, unsigned int a, unsigned int b);

// ── Checked unsigned long long (64-bit) ──────────────────────────────────────
int ckd_add_u64(unsigned long long* result, unsigned long long a, unsigned long long b);
int ckd_sub_u64(unsigned long long* result, unsigned long long a, unsigned long long b);
int ckd_mul_u64(unsigned long long* result, unsigned long long a, unsigned long long b);

// ── Convenience constants ─────────────────────────────────────────────────────
// CKD_OK: operation succeeded with no overflow.
#define CKD_OK        0
// CKD_OVERFLOW: mathematical result overflowed; wrapped value is stored.
#define CKD_OVERFLOW  1
