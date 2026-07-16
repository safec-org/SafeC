// SafeC Standard Library — Spinlock
// Atomic test-and-set spinlock with platform-specific pause hints.
#pragma once

namespace std {

struct Spinlock {
    int locked;    // 0 = unlocked, 1 = locked (accessed atomically)

    // Acquire the lock. Spins until successful.
    void lock();

    // Try to acquire the lock. Returns 1 on success, 0 if already held.
    int  trylock();

    // Release the lock.
    void unlock();

    // Check if the lock is currently held (non-atomic peek).
    int  is_locked() const;
};

// Initialize a spinlock (unlocked).
struct Spinlock spinlock_init();

} // namespace std
