// SafeC Standard Library — Result Type
// Explicit error-propagation type.  Heap-allocates the contained value.
//
// Usage pattern (mirrors ?T optional for the ok-path):
//
//   struct Result r = result_ok(42);          // ok containing int 42
//   if (r.ok()) { int* v = result_get_ok_t(&r, int); }
//   r.free();
//
// Generic accessor macros are provided because SafeC generic functions
// cannot return a typed pointer without a wrapper.
//
// result_get_ok_t(r, T)  — expands to (T*)result_get_ok(r)
// result_get_err_t(r, T) — expands to (T*)result_get_err(r)
#pragma once

#define result_get_ok_t(r, T)  ((T*)result_get_ok(r))
#define result_get_err_t(r, T) ((T*)result_get_err(r))

struct Result {
    void* data;    // heap pointer to ok value or error value
    int   is_ok;   // 1 = ok, 0 = err

    // Return 1 if this is an ok result.
    int   ok() const;

    // Return 1 if this is an error result.
    int   err() const;

    // Free the heap-allocated data.  Safe to call on null results.
    void  free();
};

// Construct an ok Result from a generic value (copied to heap).
generic<T> struct Result result_ok(T val);

// Construct an err Result from a generic error value (copied to heap).
generic<T> struct Result result_err(T err);

// Return a pointer to the ok value, or NULL if this is an error.
void* result_get_ok(&stack Result r);

// Return a pointer to the error value, or NULL if this is an ok result.
void* result_get_err(&stack Result r);

// Construct an empty/null Result (no allocation, is_ok = 0, data = NULL).
struct Result result_none();
