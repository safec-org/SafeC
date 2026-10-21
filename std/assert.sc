// SafeC Standard Library — Runtime assertions
// runtime_assert() — checked condition with diagnostic message.
// assert_true()    — checked condition, minimal overhead.
#pragma once
#define SAFEC_ASSERT_IMPL_   // the function definitions below, not the macros
#include <std/assert.h>
#include <std/io.h>
#include <std/stderr_compat.h>

// ── Explicit extern declarations ─────────────────────────────────────────────
namespace std {

extern int   fputs(const char* s, void* stream);
extern int   fputc(int c, void* stream);
extern int   fflush(void* stream);
extern void  abort();

// Print failure diagnostics to stderr and abort the process.
void assert_fail_(const char* file, int line, const char* msg) {
    unsafe {
        fputs("Assertion failed", SAFEC_STDERR_);
        if (msg != (const char*)0) {
            fputs(": ", SAFEC_STDERR_);
            fputs(msg, SAFEC_STDERR_);
        }
        if (file == (const char*)0) {      // location unknown
            fputc('\n', SAFEC_STDERR_);
            fflush(SAFEC_STDERR_);
            abort();
        }
        fputs(" at ", SAFEC_STDERR_);
        fputs(file, SAFEC_STDERR_);
        fputc(':', SAFEC_STDERR_);
        // Print line number manually (no printf here to avoid dependency).
        int tmp = line;
        if (tmp <= 0) {
            fputc('0', SAFEC_STDERR_);
        } else {
            char buf[12];
            int i = 11;
            buf[i] = '\0';
            while (tmp > 0) {
                i = i - 1;
                buf[i] = (char)('0' + (tmp % 10));
                tmp = tmp / 10;
            }
            fputs((char*)(buf + i), SAFEC_STDERR_);
        }
        fputc('\n', SAFEC_STDERR_);
        fflush(SAFEC_STDERR_);
        abort();
    }
}

inline void runtime_assert_at(int cond, const &char msg, const &char file, int line) {
    if (cond == 0) {
        unsafe { assert_fail_(file, line, msg); }
    }
}

// No caller location: a function can't see its call site (see assert.h).
inline void runtime_assert(int cond, const &char msg) {
    if (cond == 0) {
        unsafe { assert_fail_((const char*)0, 0, msg); }
    }
}

inline void assert_true(int cond) {
    if (cond == 0) {
        assert_fail_((const char*)0, 0, (const char*)0);
    }
}

} // namespace std
