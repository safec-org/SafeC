// SafeC Standard Library — Result<T,E>: a value or an error, stored inline
//
//   struct Result<int, long long> r = std::result_ok(42);   // E taken from the target type
//   if (r.ok()) { int v = r.get_ok().default(0); }
//   struct Result<int, long long> e = std::result_err(7LL);
//   match (e.get_err()) { case some(code): ...; case null: ... }
//
// No heap allocation: both slots live in the struct and only the one named
// by 'is_ok' is meaningful.
#pragma once

namespace std {

generic<T, E>
struct Result {
    T   value;   // valid when is_ok == 1
    E   error;   // valid when is_ok == 0
    int is_ok;

    int ok() const;
    int err() const;
    ?&T get_ok();             // null for an error result
    ?&E get_err();            // null for an ok result
    T   unwrap_or(T fallback) const;
};

generic<T, E>
struct Result<T, E> result_ok(T val) {
    struct Result<T, E> r;
    r.value = val;
    r.is_ok = 1;
    return r;
}

generic<T, E>
struct Result<T, E> result_err(E err) {
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
T Result::unwrap_or(T fallback) const { return self.is_ok ? self.value : fallback; }

} // namespace std
