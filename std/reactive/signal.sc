// SafeC Standard Library — Signal<T> implementation (see signal.h).
#pragma once
#include <std/reactive/signal.h>
#include <std/collections/vec.h>
#include <std/collections/vec.sc>
#include <std/mem.sc>

namespace std {

inline unsigned long Signal::get_version() const {
    return self.version;
}

inline void Signal::subscribe(SignalListener listener, void* userdata) {
    struct SignalSub sub;
    sub.listener = listener;
    sub.userdata = userdata;
    unsafe { self.subs.push((const void*)&sub); }
}

inline void Signal::unsubscribe_all() {
    self.subs.clear();
}

inline void Signal::notify() {
    unsigned long i = 0UL;
    unsigned long n;
    unsafe { n = self.subs.length(); }
    while (i < n) {
        struct SignalSub* sub;
        unsafe { sub = (struct SignalSub*)self.subs.get_raw(i); }
        if (sub != (struct SignalSub*)0) {
            unsafe { sub->listener(sub->userdata); }
        }
        i = i + 1UL;
    }
}

inline void Signal::free() {
    unsafe {
        if (self.data != (void*)0) {
            free(self.data);
            self.data = (void*)0;
        }
        self.subs.free();
    }
}

generic<T>
struct Signal signal_new(T val) {
    struct Signal s;
    unsafe {
        s.data = malloc(sizeof(T));
        if (s.data != (void*)0) {
            memcpy(s.data, (const void*)&val, sizeof(T));
        }
    }
    s.elemSize = sizeof(T);
    s.version = 0UL;
    unsafe { s.subs = vec_new(sizeof(struct SignalSub)); }
    return s;
}

inline void* signal_get_raw(&stack Signal s) {
    return s.data;
}

inline void signal_set_raw(&stack Signal s, const void* val) {
    unsafe {
        if (s.data != (void*)0 && val != (const void*)0) {
            memcpy(s.data, val, s.elemSize);
        }
    }
    s.version = s.version + 1UL;
    s.notify();
}

generic<T>
void signal_set_t(&stack Signal s, T val) {
    unsafe { signal_set_raw(s, (const void*)&val); }
}

inline int signal_changed(&stack Signal s, unsigned long* lastSeenVersion) {
    unsigned long cur = s.version;
    unsafe {
        if (*lastSeenVersion != cur) {
            *lastSeenVersion = cur;
            return 1;
        }
    }
    return 0;
}

} // namespace std
