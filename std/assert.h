// SafeC Standard Library — Runtime assertions (C89–C23)
// SafeC has compile-time `static_assert` as a built-in keyword.
// This module provides runtime assertion with file/line diagnostics.
#pragma once

// ── Runtime assertion ─────────────────────────────────────────────────────────

// Internal failure handler — do not call directly.
void assert_fail_(const char* file, int line, const char* msg);

// Assert that `cond` is true at runtime.
// On failure: prints "Assertion failed: <msg> at <file>:<line>" to stderr
// and calls abort().  Pass a description string as `msg`.
// Example: runtime_assert(ptr != 0, "pointer must not be null");
void runtime_assert(int cond, const char* msg);

// Assert that `cond` is true; uses a fixed message ("assertion failed").
// Lighter weight — no message string required.
void assert_true(int cond);

// ── NDEBUG support ────────────────────────────────────────────────────────────
// Define NDEBUG before including this header to disable runtime_assert()
// and assert_true() (they become no-ops, compiled away).
// static_assert is never affected by NDEBUG (it is always compile-time).

#ifdef NDEBUG
#define runtime_assert(cond, msg)  ((void)0)
#define assert_true(cond)          ((void)0)
#endif
