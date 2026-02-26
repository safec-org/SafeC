// SafeC Standard Library — Syscall Table
#pragma once
#include "syscall.h"

struct SyscallTable syscall_init() {
    struct SyscallTable t;
    t.count = 0;
    int i = 0;
    while (i < 256) {
        t.handlers[i] = (void*)0;
        i = i + 1;
    }
    return t;
}

int SyscallTable::register_(int num, void* handler) {
    if (num < 0 || num >= 256) { return 0; }
    if (self.handlers[num] == (void*)0 && handler != (void*)0) {
        self.count = self.count + 1;
    }
    self.handlers[num] = handler;
    return 1;
}

void SyscallTable::unregister_(int num) {
    if (num >= 0 && num < 256) {
        if (self.handlers[num] != (void*)0) {
            self.handlers[num] = (void*)0;
            self.count = self.count - 1;
        }
    }
}

long long SyscallTable::dispatch(int num, long long arg0, long long arg1, long long arg2) {
    if (num < 0 || num >= 256) { return (long long)-1; }
    if (self.handlers[num] == (void*)0) { return (long long)-1; }
    unsafe {
        return ((long long(*)(long long, long long, long long))self.handlers[num])(arg0, arg1, arg2);
    }
}

int SyscallTable::is_registered(int num) const {
    if (num < 0 || num >= 256) { return 0; }
    if (self.handlers[num] != (void*)0) { return 1; }
    return 0;
}
