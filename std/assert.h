// SafeC Standard Library — Runtime assertions (C89–C23)
// SafeC has compile-time `static_assert` as a built-in keyword.
// This module provides runtime assertion with file/line diagnostics.
#pragma once

// ── Runtime assertion ─────────────────────────────────────────────────────────

// Internal failure handler — do not call directly. 'file' may be null
// (no location known), 'msg' may be null (no description).
namespace std {
void assert_fail_(const ?&char file, int line, const ?&char msg);

// Assert that `cond` is true at runtime, reporting the given location.
// On failure: prints "Assertion failed: <msg> at <file>:<line>" to stderr
// and calls abort(). Pass __FILE__ and __LINE__ for the call site:
//   runtime_assert_at(ptr != 0, "pointer must not be null", __FILE__, __LINE__);
void runtime_assert_at(int cond, const &char msg, const &char file, int line);

// Assert that `cond` is true; on failure prints "Assertion failed: <msg>"
// and aborts. A function can't learn its caller's file/line, so these
// report no location — in compat mode (--compat-preprocessor) the macros
// below route them through runtime_assert_at with the caller's location.
void runtime_assert(int cond, const &char msg);

// Assert that `cond` is true; no message string required.
void assert_true(int cond);

// ── Macros (compat mode only: function-like macros) ──────────────────────────
// runtime_assert / assert_true pick up the caller's __FILE__/__LINE__.
// Define NDEBUG before including this header to compile them away.
// static_assert is never affected by NDEBUG (it is always compile-time).
#ifndef SAFEC_ASSERT_IMPL_
#ifdef __SAFEC_COMPAT_PREPROCESSOR__
#ifdef NDEBUG
#define runtime_assert(cond, msg)  ((void)0)
#define assert_true(cond)          ((void)0)
#else
#define runtime_assert(cond, msg)  runtime_assert_at((cond), (msg), __FILE__, __LINE__)
#define assert_true(cond)          runtime_assert_at((cond), "assertion failed", __FILE__, __LINE__)
#endif
#endif
#endif

} // namespace std
