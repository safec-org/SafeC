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

inline int Scheduler::spawn_proc(&stack PCB proc) {
    if (self.count >= 256) { return -1; }
    int idx = self.count;
    self.procs[idx] = *proc;
    self.count = self.count + 1;
    return idx;
}

int Scheduler::next() {
    int best     = -1;
    int best_pri = -1;
    int start    = 0;

    if (self.current >= 0) { start = self.current + 1; }

    // Two passes: from start→end, then 0→start (round-robin fairness)
    int pass = 0;
    while (pass < 2) {
        int begin = 0;
        int end   = 0;
        if (pass == 0) {
            begin = start;
            end   = self.count;
        } else {
            begin = 0;
            end   = start;
        }
        int i = begin;
        while (i < end) {
            if (self.procs[i].state == 0) { // PROC_READY
                if (self.procs[i].priority > best_pri) {
                    best_pri = self.procs[i].priority;
                    best = i;
                }
            }
            i = i + 1;
        }
        if (best >= 0 && pass == 0) {
            self.current = best;
            self.procs[best].state = 1; // PROC_RUNNING
            return best;
        }
        pass = pass + 1;
    }

    if (best >= 0) {
        self.current = best;
        self.procs[best].state = 1; // PROC_RUNNING
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
    if (self.current >= self.count) { self.current = self.count - 1; }
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
