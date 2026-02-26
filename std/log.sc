#pragma once
#include "log.h"

extern int fprintf(void* stream, const char* fmt, ...);
extern void* stderr;

static LogBackend backend_ = NULL;

void log_set_backend(LogBackend backend) {
    backend_ = backend;
}

void log_write(int level, const char* tag, const char* msg,
               const char* file, int line) {
    // Compile-time threshold: discard messages below LOG_LEVEL.
    if (level > LOG_LEVEL) {
        return;
    }

    // Dispatch to custom backend if installed.
    if (backend_ != NULL) {
        backend_(level, tag, msg, file, line);
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
