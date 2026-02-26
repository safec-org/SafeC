// SafeC Standard Library — Result Type
#pragma once
#include "result.h"

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, unsigned long n);

// ── Result methods ────────────────────────────────────────────────────────────

int Result::ok() const {
    return self.is_ok;
}

int Result::err() const {
    return self.is_ok == 0 ? 1 : 0;
}

void Result::free() {
    if (self.data != (void*)0) {
        unsafe { free(self.data); }
        self.data = (void*)0;
    }
}

// ── Constructors (free functions) ─────────────────────────────────────────────

generic<T>
struct Result result_ok(T val) {
    struct Result r;
    unsafe {
        r.data = malloc(sizeof(T));
        if (r.data != (void*)0) {
            memcpy(r.data, (const void*)&val, sizeof(T));
        }
    }
    r.is_ok = 1;
    return r;
}

generic<T>
struct Result result_err(T err) {
    struct Result r;
    unsafe {
        r.data = malloc(sizeof(T));
        if (r.data != (void*)0) {
            memcpy(r.data, (const void*)&err, sizeof(T));
        }
    }
    r.is_ok = 0;
    return r;
}

struct Result result_none() {
    struct Result r;
    r.data  = (void*)0;
    r.is_ok = 0;
    return r;
}

// ── Accessors ─────────────────────────────────────────────────────────────────

void* result_get_ok(&stack Result r) {
    if (r.is_ok == 0) { return (void*)0; }
    return r.data;
}

void* result_get_err(&stack Result r) {
    if (r.is_ok != 0) { return (void*)0; }
    return r.data;
}
