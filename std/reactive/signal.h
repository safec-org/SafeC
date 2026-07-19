#pragma once
// SafeC Standard Library — Signal<T>: fine-grained reactive state.
//
// The client-side reactivity primitive for WASM-hydrated scx pages (see
// std/scx/scx.h for the *server-side* templating half — scx itself is
// pure compile-time string building with no runtime state). A Signal
// owns one value; std/reactive/hydrate.h's DOM bindings subscribe()
// listeners onto it so that signal_set_t() both updates the value and
// pushes the change straight to the specific DOM node(s) that depend on
// it — no virtual DOM, no diffing, matching the "SolidJS-style" signal
// model rather than a React-style re-render-the-tree one, since a from-
// scratch vdom differ is a lot of machinery for what a WASM binary
// linked with -nostdlib doesn't otherwise need.
//
// Like std::Result (see result.h), Signal is type-erased at the struct
// level (SafeC structs cannot themselves be generic — only free
// functions can be 'generic<T>') and heap-backs its value.
//
//   struct Signal count = signal_new(0);
//   count.subscribe(on_count_changed, someUserdata);
//   ...
//   int cur = signal_get_t(&count, int);      // macro: (*(int*)signal_get_raw(&count))
//   signal_set_t(&count, cur + 1);             // updates + bumps version + notifies
//   count.free();
#include <std/collections/vec.h>

#define signal_get_t(s, T) (*(T*)std::signal_get_raw(s))

namespace std {

typedef fn void(void* userdata) SignalListener;

struct SignalSub {
    SignalListener listener;
    void* userdata;
};

struct Signal {
    &heap void    data;      // heap pointer to the current value
    unsigned long elemSize;  // size of the held value, in bytes
    unsigned long version;   // bumped on every set_raw()/signal_set_t()
    struct Vec    subs;      // Vec<struct SignalSub>

    // Current version number — compare against a previously-read value to
    // detect whether this signal has changed since (see signal_changed()).
    unsigned long get_version() const;

    // Registers 'listener' to be called (with 'userdata') on every future
    // set_raw()/signal_set_t(). Not called for the signal's initial value —
    // read it directly after construction if you need the starting state.
    void subscribe(SignalListener listener, void* userdata);

    // Removes every subscriber (does not free their userdata).
    void unsubscribe_all();

    // Calls every subscriber once, in registration order. Called
    // automatically by set_raw()/signal_set_t(); exposed directly for
    // cases that mutate the pointed-to value in place (via signal_get_raw())
    // and need to announce the change without a fresh set_raw() copy.
    void notify();

    // Frees the held value and the subscriber list. Does not call
    // subscribers' free logic — unsubscribe_all() them first if needed.
    void free();
};

// Constructs a Signal holding a copy of 'val' (version starts at 0, no
// subscribers).
generic<T> struct Signal signal_new(T val);

// Raw (type-erased) read: pointer to the currently-held value, or NULL if
// 's' is a freed/default-initialized Signal. Prefer signal_get_t(s, T).
void* signal_get_raw(&stack Signal s);

// Raw (type-erased) write: memcpy's elemSize bytes from 'val' over the
// held value, bumps the version, and calls notify(). Prefer
// signal_set_t(s, val).
void signal_set_raw(&stack Signal s, const void* val);

generic<T> void signal_set_t(&stack Signal s, T val);

// Compares 's's current version against '*lastSeenVersion'; if different,
// updates '*lastSeenVersion' to match and returns 1, else returns 0. Lets
// polling-style code (rather than subscribe()'s push style) detect changes.
int signal_changed(&stack Signal s, unsigned long* lastSeenVersion);

} // namespace std
