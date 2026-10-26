#pragma once
#include <std/log.h>
#include <std/stderr_compat.h>

namespace std {

extern int fprintf(CFile_* stream, const char* fmt, ...);
// Nullable: "no custom backend installed" is null.
static ?fn void(int, const &char, const &char, const &char, int) backend_ = null;

inline void log_set_backend(LogBackend backend) {
    backend_ = backend;
}

void log_write(int level, const &char tag, const &char msg,
               const &char file, int line) {
    // Compile-time threshold: discard messages below LOG_LEVEL.
    if (level > LOG_LEVEL) {
        return;
    }

    // Dispatch to custom backend if installed.
    match (backend_) {
        case some(cb): { cb(level, tag, msg, file, line); return; }
        case null: {}
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
        fprintf(SAFEC_STDERR_, "[%s] %s (%s:%d): %s\n", prefix, tag, file, line, msg);
    }
#endif
}

} // namespace std
