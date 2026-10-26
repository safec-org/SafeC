#pragma once
// SafeC Standard Library — ownership helpers
//
// SafeC has no implicit destructors, but it does check them: a value whose
// type has a 'void drop()' method (the builtin Drop trait), or an aggregate
// holding one, must be dropped — 'x.drop()', 'defer x.drop()', or a method
// declared 'consume' — or moved (returned, passed by value, stored into
// something that owns it, sent to another thread) exactly once on every
// path. The compiler rejects a leak, a use after drop or move, a double
// drop, and overwriting a live value. See the language reference's
// "Ownership & Drop" page.
//
// Memory reached through a reference is borrowed: a Drop value can be
// dropped in place there, but not moved out. These move a value out of a
// place by putting another one in:
//
//   struct Waker w = std::replace(&self.w, Waker::noop());   // take self.w
//   std::swap(&a, &b);
//
// forget(x) (a compiler built-in, only inside 'unsafe { }') gives up a
// value without dropping it.

namespace std {

// Stores 'value' in *place and returns what was there.
generic<T>
T replace(&T place, T value) {
    unsafe {
        T old = *place;
        *place = value;
        return old;
    }
}

// Exchanges *a and *b.
generic<T>
void swap(&T a, &T b) {
    unsafe {
        T t = *a;
        *a = *b;
        *b = t;
    }
}

} // namespace std
