#pragma once
// SafeC Standard Library — Erased: the referent of a type-erased reference.
//
// Some std types hold a value whose type only the generic code that stored
// it knows: a Waker's task, a Task's argument, a widget's click-handler
// context. SafeC has no trait objects, so such a type keeps a typed
// function and its context as '?&Erased' / 'fn R(?&Erased, ...)' fields,
// written only by a generic constructor that takes the matching
// 'fn R(?&T, ...)' and '?&T' — so every call through the erased pair is
// well-typed — and read back by calling the function with its context:
//
//   generic<T> struct Waker Waker::new(fn void(?&T, unsigned long) wake, ?&T data, unsigned long tag);
//   ...
//   self.wake_(self.data_, self.tag_);   // calls wake(data, tag) as T
//
// Erasing (T -> Erased) is a cast inside 'unsafe' in that constructor;
// nothing outside it ever turns an Erased back into a T. An Erased is never
// constructed: it only appears behind a reference. This replaces C's
// 'void* userdata' everywhere in std.

namespace std {

struct Erased {
    char opaque_;   // never constructed or read; only referenced
};

} // namespace std
