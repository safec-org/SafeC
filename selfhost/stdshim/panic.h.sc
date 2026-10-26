// stdshim/panic.h.sc — playground implementation of <std/panic.h>: prints
// "panic at file:line: msg" to stderr and exits 134, like the hosted std.
namespace std {

void* __scc_panic_handler_ = null;

void panic_set_handler(PanicHandler handler) {
    unsafe { __scc_panic_handler_ = (void*)handler; }
}

void panic_at(const &char msg, const &char file, int line) {
    if (__scc_panic_handler_ != null) {
        unsafe {
            PanicHandler cb = (PanicHandler)__scc_panic_handler_;
            cb(msg, file, line);
        }
    }
    unsafe { fprintf(&__rt_files[2], "panic at %s:%d: %s\n", file, line, msg); }
    abort();
}

} // namespace std
