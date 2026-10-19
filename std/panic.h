// SafeC Standard Library — Optional Panic System
// Opt-in; does nothing unless panic_set_handler is called.
// In freestanding mode defaults to infinite loop; hosted mode calls abort().
#pragma once

// Panic handler function type: fn(message, file, line) → noreturn
namespace std {

typedef fn void(const char* msg, const char* file, int line) PanicHandler;

// Install a custom panic handler. Pass NULL to reset to default.
void panic_set_handler(PanicHandler handler);

// Trigger a panic with message, file, line.
// Never returns.
void panic_at(const char* msg, const char* file, int line);

// Checked narrowing of a nullable reference: returns 'r' as '&T', or
// panics with 'msg' if it is null. The explicit form of "this can't be
// null here" — a cast '(&T)r' is rejected because it would skip the check.
// For recoverable absence, 'match (r) { case null: ... case some(x): ... }'.
generic<T>
&T expect_ref(?&T r, const char* msg) {
    match (r) {
        case some(x): return x;
        case null: { panic_at(msg, "expect_ref", 0); }
    }
    panic_at(msg, "expect_ref", 0);
    return expect_ref(r, msg);   // unreachable: panic_at never returns
}

// Convenience macro.
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define PANIC(msg)  panic_at((msg), __FILE__, __LINE__)
#endif

// Assert that condition is true; panics with expression string if false.
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define PANIC_ASSERT(cond)  do { if (!(cond)) { PANIC(#cond); } } while(0)
#endif

} // namespace std
