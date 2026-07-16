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

void log_set_backend(LogBackend backend) {
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
    // Hosted: emit to stderr with level prefix.
    const char* prefix = "?";
    if (level == LOG_LEVEL_ERROR) {
        prefix = "E";
    } else if (level == LOG_LEVEL_WARN) {
        prefix = "W";
    } else if (level == LOG_LEVEL_INFO) {
        prefix = "I";
    } else if (level == LOG_LEVEL_DEBUG) {
        prefix = "D";
    }
    unsafe {
        fprintf(stderr, "[%s] %s (%s:%d): %s\n", prefix, tag, file, line, msg);
    }
#endif
}

} // namespace std
