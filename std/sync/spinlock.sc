// SafeC Standard Library — Spinlock
#pragma once
#include <std/sync/spinlock.h>

namespace std {

inline struct Spinlock spinlock_init() {
    struct Spinlock s;
    s.locked = 0;
    return s;
}

inline void Spinlock::lock() {
    unsafe {
        while (atomic_exchange(&self.locked, 1) != 0) {
            // Spin — compiler will emit a busy-wait loop
            // Platform pause hint would go here (x86: pause, ARM: yield)
            while (atomic_load(&self.locked) != 0) {
                // Read-only spin to reduce bus traffic
            }
        }
    }
}

inline int Spinlock::trylock() {
    unsafe {
        if (atomic_exchange(&self.locked, 1) == 0) {
            return 1;
        }
        return 0;
    }
}

inline void Spinlock::unlock() {
    unsafe {
        atomic_store(&self.locked, 0);
    }
}

inline int Spinlock::is_locked() const {
    return self.locked;
}

} // namespace std
