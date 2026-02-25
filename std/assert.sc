// SafeC Standard Library — Runtime assertions
// runtime_assert() — checked condition with diagnostic message.
// assert_true()    — checked condition, minimal overhead.
#pragma once
#include "assert.h"
#include "io.h"

// ── Explicit extern declarations ─────────────────────────────────────────────
extern int   fputs(const char* s, void* stream);
extern int   fputc(int c, void* stream);
extern int   fflush(void* stream);
extern void  abort();
extern void* __stderrp;

// Print failure diagnostics to stderr and abort the process.
void assert_fail_(const char* file, int line, const char* msg) {
    unsafe {
        fputs("Assertion failed", __stderrp);
        if (msg != (const char*)0) {
            fputs(": ", __stderrp);
            fputs(msg, __stderrp);
        }
        fputs(" at ", __stderrp);
        fputs(file, __stderrp);
        fputc(':', __stderrp);
        // Print line number manually (no printf here to avoid dependency).
        int tmp = line;
        if (tmp == 0) {
            fputc('0', __stderrp);
        } else {
            char buf[12];
            int i = 11;
            buf[i] = '\0';
            while (tmp > 0) {
                i = i - 1;
                buf[i] = (char)('0' + (tmp % 10));
                tmp = tmp / 10;
            }
            fputs((char*)(buf + i), __stderrp);
        }
        fputc('\n', __stderrp);
        fflush(__stderrp);
        abort();
    }
}

// Assert that `cond` is true; on failure call assert_fail_ with message.
void runtime_assert(int cond, const char* msg) {
    if (cond == 0) {
        assert_fail_(__FILE__, __LINE__, msg);
    }
}

// Assert that `cond` is true; minimal message on failure.
void assert_true(int cond) {
    if (cond == 0) {
        assert_fail_(__FILE__, __LINE__, (const char*)0);
    }
}
