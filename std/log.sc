#pragma once
#include <std/log.h>

namespace std {

extern int fprintf(void* stream, const char* fmt, ...);
extern void* stderr;
#define NULL ((void*)0)

// Stored as a raw pointer (not 'LogBackend' directly): a '&static fn(...)'
// reference is non-nullable, but "no custom backend installed" needs a null
// sentinel here. Cast back to 'LogBackend' at the call site instead.
static void* backend_ = NULL;

inline void log_set_backend(LogBackend backend) {
    unsafe { backend_ = (void*)backend; }
}

void log_write(int level, const char* tag, const char* msg,
               const char* file, int line) {
    // Compile-time threshold: discard messages below LOG_LEVEL.
    if (level > LOG_LEVEL) {
        return;
    }

    // Dispatch to custom backend if installed.
    if (backend_ != NULL) {
        unsafe {
            LogBackend cb = (LogBackend)backend_;
            cb(level, tag, msg, file, line);
        }
        return;
    }

    // Default backend.
#ifdef __SAFEC_FREESTANDING__
    // No I/O available in freestanding mode — silently drop.
    return;
#else
    // Hosted: emit to stderr with level prefix. A match-expression here
    // (vs. the previous if/else-if chain reassigning one variable) lets
    // CodeGen merge the four cases with a PHI instead of a stack
    // alloca+store+load for 'prefix' — fewer instructions per log call
    // even before the optimizer's own mem2reg would erase the difference.
    const char* prefix = match (level) {
        case LOG_LEVEL_ERROR: "E",
        case LOG_LEVEL_WARN:  "W",
        case LOG_LEVEL_INFO:  "I",
        case LOG_LEVEL_DEBUG: "D",
        default:              "?",
    };
    unsafe {
        fprintf(stderr, "[%s] %s (%s:%d): %s\n", prefix, tag, file, line, msg);
    }
#endif
}

} // namespace std
