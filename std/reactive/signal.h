#pragma once
// SafeC Standard Library — Signal<T>: fine-grained reactive state.
//
// The client-side reactivity primitive for WASM-hydrated scx pages (see
// std/scx/scx.h for the server-side templating half). A Signal owns one
// value; std/wasm/hydrate.h's DOM bindings subscribe listeners onto it so
// that set() both updates the value and pushes the change straight to the
// DOM node(s) that depend on it — SolidJS-style signals, no virtual DOM.
//
//   struct Signal<int> count = std::Signal::new(0);
//   std::signal_subscribe(&count, on_count_changed, (?&Label)&label);
//   count.set(count.get() + 1);      // updates, bumps version, notifies
//   count.free();
#include <std/collections/vec.h>

namespace std {

// One subscriber. Listeners on the same signal carry unrelated context
// types, so each is stored type-erased; signal_subscribe() only ever pairs
// a fn void(?&U) with a ?&U, so the erased call is always well-typed.
struct SignalSub {
    void* listener;   // fn void(?&U)
    void* userdata;   // ?&U
};

generic<T>
struct Signal {
    T             value;
    unsigned long version;   // bumped on every set()
    struct Vec<struct SignalSub> subs;

    static struct Signal<T> new(T val);   // holds val, version 0, no subscribers

    T             get() const;
    &T            get_ref();            // in-place access; call notify() after mutating
    void          set(T val);           // stores val, bumps version, notifies
    unsigned long get_version() const;
    // 1 (and updates *last_seen) if the version moved since *last_seen —
    // polling-style change detection instead of subscribe()'s push style.
    int           changed(&unsigned long last_seen) const;
    void          notify();             // calls every subscriber in registration order
    void          unsubscribe_all();
    void          free();               // frees the subscriber list
};

generic<T>
struct Signal<T> Signal::new(T val) {
    struct Signal<T> s;
    s.value = val;
    s.version = 0UL;
    s.subs = Vec::new();
    return s;
}

// Registers listener(userdata) to run on every future set()/notify(). Not
// called for the initial value — read it directly after subscribing.
generic<T, U>
void signal_subscribe(&Signal<T> s, fn void(?&U) listener, ?&U userdata) {
    struct SignalSub sub;
    unsafe {
        sub.listener = (void*)listener;
        sub.userdata = (void*)userdata;
    }
    s.subs.push(sub);
}

generic<T>
T Signal::get() const { return self.value; }

generic<T>
&T Signal::get_ref() {
    unsafe { return &self.value; }
}

generic<T>
void Signal::set(T val) {
    self.value = val;
    self.version = self.version + 1UL;
    self.notify();
}

generic<T>
unsigned long Signal::get_version() const { return self.version; }

generic<T>
int Signal::changed(&unsigned long last_seen) const {
    if (*last_seen == self.version) return 0;
    *last_seen = self.version;
    return 1;
}

generic<T>
void Signal::notify() {
    unsigned long i = 0UL;
    while (i < self.subs.length()) {
        &SignalSub sub = self.subs.at(i);
        unsafe {
            fn void(void*) f = (fn void(void*))sub.listener;
            f(sub.userdata);
        }
        i = i + 1UL;
    }
}

generic<T>
void Signal::unsubscribe_all() { self.subs.clear(); }

generic<T>
void Signal::free() { self.subs.free(); }

} // namespace std
