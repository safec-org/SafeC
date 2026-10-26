// SafeC Standard Library — Result<T,E>: a value or an error, stored inline
//
//   struct Result<int, long long> r = std::Result::new_ok(42);   // E taken from the target type
//   if (r.ok()) { int v = r.get_ok().default(0); }
//   struct Result<int, long long> e = std::Result::new_err(7LL);
//   match (e.get_err()) { case some(code): ...; case null: ... }
//
// No heap allocation: both slots live in the struct and only the one named
// by 'is_ok' is meaningful. When T or E owns something (a Drop type), so
// does the Result: drop() drops whichever slot is live, and unwrap_or()
// consumes it, as do unwrap() / unwrap_err() (which panic on the other kind).
#pragma once
#include <std/panic.h>

namespace std {

generic<T, E>
struct Result {
    T   value;   // valid when is_ok == 1
    E   error;   // valid when is_ok == 0
    int is_ok;

    static struct Result<T, E> new_ok(T val);    // (ok/err are the queries below)
    static struct Result<T, E> new_err(E err);

    int ok() const;
    int err() const;
    ?&T get_ok();             // null for an error result
    ?&E get_err();            // null for an ok result
    T   unwrap_or(T fallback) consume;   // the value, or 'fallback' (the other one is dropped)
    T   unwrap() consume;                // the value; panics for an error result
    E   unwrap_err() consume;            // the error; panics for an ok result
    void drop() if (needs_drop(T) || needs_drop(E));
};

generic<T, E>
struct Result<T, E> Result::new_ok(T val) {
    struct Result<T, E> r;
    r.value = val;
    r.is_ok = 1;
    return r;
}

generic<T, E>
struct Result<T, E> Result::new_err(E err) {
    struct Result<T, E> r;
    r.error = err;
    r.is_ok = 0;
    return r;
}

generic<T, E>
int Result::ok() const { return self.is_ok; }

generic<T, E>
int Result::err() const { return !self.is_ok; }

generic<T, E>
?&T Result::get_ok() {
    if (!self.is_ok) return null;
    unsafe { return (?&T)&self.value; }
}

generic<T, E>
?&E Result::get_err() {
    if (self.is_ok) return null;
    unsafe { return (?&E)&self.error; }
}

generic<T, E>
T Result::unwrap_or(T fallback) consume {
    if (self.is_ok) { drop(fallback); return self.value; }
    drop(self.error);
    return fallback;
}

generic<T, E>
T Result::unwrap() consume {
    if (!self.is_ok) { drop(self.error); panic_at("Result::unwrap on an error result", "result.h", 0); }
    return self.value;
}

generic<T, E>
E Result::unwrap_err() consume {
    if (self.is_ok) { drop(self.value); panic_at("Result::unwrap_err on an ok result", "result.h", 0); }
    return self.error;
}

generic<T, E>
void Result::drop() {
    if (self.is_ok) drop(self.value);
    else drop(self.error);
}

} // namespace std
