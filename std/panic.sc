#pragma once
#include "panic.h"

extern void abort();
extern int fprintf(void* stream, const char* fmt, ...);
extern void* stderr;

static PanicHandler current_handler_ = NULL;

void panic_set_handler(PanicHandler handler) {
    current_handler_ = handler;
}

void panic_at(const char* msg, const char* file, int line) noreturn {
    if (current_handler_ != NULL) {
        current_handler_(msg, file, line);
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
