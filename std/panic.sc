#pragma once
#include <std/panic.h>

namespace std {

extern void abort();
extern int fprintf(void* stream, const char* fmt, ...);
extern void* stderr;
#define NULL ((void*)0)

// Stored as a raw pointer, not 'PanicHandler' directly: a '&static fn(...)'
// reference is non-nullable, but "no handler installed" needs a null
// sentinel — cast back to 'PanicHandler' at the call site instead.
static void* current_handler_ = NULL;

inline void panic_set_handler(PanicHandler handler) {
    unsafe { current_handler_ = (void*)handler; }
}

inline noreturn void panic_at(const char* msg, const char* file, int line) {
    if (current_handler_ != NULL) {
        unsafe {
            PanicHandler cb = (PanicHandler)current_handler_;
            cb(msg, file, line);
        }
    }

#ifdef __SAFEC_FREESTANDING__
    // In freestanding mode: spin forever (no hosted runtime available).
    for (;;) {}
#else
    // Hosted mode: print diagnostics to stderr then abort.
    unsafe {
        fprintf(stderr, "panic at %s:%d: %s\n", file, line, msg);
    }
    abort();
#endif
}

} // namespace std
