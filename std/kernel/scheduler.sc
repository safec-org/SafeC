// SafeC Standard Library — Priority Round-Robin Scheduler
#pragma once
#include <std/kernel/scheduler.h>

namespace std {

inline struct Scheduler sched_init() {
    struct Scheduler s;
    s.count   = 0;
    s.current = -1;
    return s;
}

inline int Scheduler::spawn_proc(&PCB proc) {
    if (self.count >= SCHED_MAX_PROCS) { return -1; }
    int idx = self.count;
    self.procs[idx] = *proc;
    self.count = self.count + 1;
    return idx;
}

int Scheduler::next() {
    // Highest READY priority wins; among equals, round-robin starting just
    // after the current process.
    int best     = -1;
    int best_pri = 0;
    int start    = 0;
    if (self.current >= 0) { start = self.current + 1; }
    int k = 0;
    while (k < self.count) {
        int i = (start + k) % self.count;
        if (self.procs[i].state == PROC_READY &&
            (best < 0 || self.procs[i].priority > best_pri)) {
            best_pri = self.procs[i].priority;
            best = i;
        }
        k = k + 1;
    }
    if (best >= 0) {
        self.current = best;
        self.procs[best].state = PROC_RUNNING;
    }
    return best;
}

inline int Scheduler::yield() {
    if (self.current >= 0 && self.procs[self.current].state == 1) {
        self.procs[self.current].state = 0; // PROC_READY
    }
    return self.next();
}

inline void Scheduler::block_current() {
    if (self.current >= 0) {
        self.procs[self.current].state = 2; // PROC_BLOCKED
    }
}

inline void Scheduler::unblock(int idx) {
    if (idx >= 0 && idx < self.count) {
        if (self.procs[idx].state == 2) { // PROC_BLOCKED
            self.procs[idx].state = 0; // PROC_READY
        }
    }
}

inline void Scheduler::remove(int idx) {
    if (idx < 0 || idx >= self.count) { return; }
    int i = idx;
    while (i < self.count - 1) {
        self.procs[i] = self.procs[i + 1];
        i = i + 1;
    }
    self.count = self.count - 1;
    // Keep 'current' on the same process: it shifted down if it was after
    // idx, and there is none if it was the one removed.
    if (self.current == idx) { self.current = -1; }
    else if (self.current > idx) { self.current = self.current - 1; }
}

inline int Scheduler::ready_count() const {
    int count = 0;
    int i = 0;
    while (i < self.count) {
        if (self.procs[i].state == 0) { count = count + 1; }
        i = i + 1;
    }
    return count;
}

} // namespace std
