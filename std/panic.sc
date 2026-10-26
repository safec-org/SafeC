#pragma once
#include <std/panic.h>
#include <std/stderr_compat.h>

namespace std {

extern void abort();
extern int fprintf(CFile_* stream, const char* fmt, ...);
// Nullable: "no handler installed" is null.
static ?fn void(const &char, const &char, int) current_handler_ = null;

inline void panic_set_handler(PanicHandler handler) {
    current_handler_ = handler;
}

inline noreturn void panic_at(const &char msg, const &char file, int line) {
    match (current_handler_) {
        case some(cb): { cb(msg, file, line); }
        case null: {}
    }

#if defined(__wasm__)
    // WebAssembly: trap (std/wasm/wasm_rt's abort()) — spinning would hang
    // the host's event loop instead of surfacing a RuntimeError.
    abort();
#elif defined(__SAFEC_FREESTANDING__)
    // In freestanding mode: spin forever (no hosted runtime available).
    for (;;) {}
#else
    // Hosted mode: print diagnostics to stderr then abort.
    unsafe {
        fprintf(SAFEC_STDERR_, "panic at %s:%d: %s\n", file, line, msg);
    }
    abort();
#endif
}

} // namespace std
