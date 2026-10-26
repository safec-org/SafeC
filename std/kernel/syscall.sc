// SafeC Standard Library — Syscall Table
#pragma once
#include <std/kernel/syscall.h>

namespace std {

inline struct SyscallTable SyscallTable::new() {
    struct SyscallTable t;
    t.count = 0;
    int i = 0;
    while (i < 256) {
        t.handlers[i] = null;
        i = i + 1;
    }
    return t;
}

inline int SyscallTable::register_(int num, fn long long(long long, long long, long long) handler) {
    if (num < 0 || num >= SYSCALL_MAX) { return 0; }
    if (self.handlers[num].is_null()) {
        self.count = self.count + 1;
    }
    self.handlers[num] = handler;
    return 1;
}

inline void SyscallTable::unregister_(int num) {
    if (num >= 0 && num < 256) {
        if (!self.handlers[num].is_null()) {
            self.handlers[num] = null;
            self.count = self.count - 1;
        }
    }
}

inline long long SyscallTable::dispatch(int num, long long arg0, long long arg1, long long arg2) {
    if (num < 0 || num >= 256) { return (long long)-1; }
    match (self.handlers[num]) {
        case some(handler): return handler(arg0, arg1, arg2);
        case null: return (long long)-1;
    }
    return (long long)-1;
}

inline int SyscallTable::is_registered(int num) const {
    if (num < 0 || num >= 256) { return 0; }
    if (!self.handlers[num].is_null()) { return 1; }
    return 0;
}

} // namespace std
