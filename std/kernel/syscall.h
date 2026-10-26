// SafeC Standard Library — Syscall Table
// Syscall registration and dispatch. Freestanding-safe.
#pragma once

#define SYSCALL_MAX 256

namespace std {

struct SyscallTable {
    ?fn long long(long long, long long, long long) handlers[SYSCALL_MAX];   // null = none
    int   count;                  // number of registered syscalls

    // Register a handler for syscall number `num`.
    // Returns 1 on success, 0 if out of range.
    int       register_(int num, fn long long(long long, long long, long long) handler);

    // Unregister a syscall handler.
    void      unregister_(int num);

    // Dispatch a syscall. Returns the handler's return value, or -1 if not registered.
    long long dispatch(int num, long long arg0, long long arg1, long long arg2);

    // Check if a syscall number is registered.
    int       is_registered(int num) const;

    // Initialize syscall table (all entries NULL).
    static struct SyscallTable new();
};

} // namespace std
