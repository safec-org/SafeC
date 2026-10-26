// SafeC Standard Library — Freestanding Cooperative Threads
#pragma once
#include <std/sync/thread_bare.h>

namespace std {

inline struct ThreadSched ThreadSched::new() {
    struct ThreadSched s;
    s.inner = TaskScheduler::new();
    int i = 0;
    while (i < THREAD_MAX) {
        s.priority[i] = 0;
        i = i + 1;
    }
    return s;
}

inline Thread ThreadSched::spawn_raw_(fn int(?&Erased, int) func, ?&Erased arg, int priority) {
    int id = self.inner.spawn_raw_(func, arg);
    if (id == -1) { return (Thread)THREAD_NONE; }
    self.priority[id] = priority;
    return (Thread)id;
}

int ThreadSched::tick() {
    int n     = self.inner.count;
    int active = 0;

    // Build a simple priority-sorted run order (insertion sort on indices).
    int order[THREAD_MAX];
    int i = 0;
    while (i < n) {
        order[i] = i;
        i = i + 1;
    }
    // Sort descending by priority (bubble sort).
    int swapped = 1;
    while (swapped != 0) {
        swapped = 0;
        int j = 0;
        while (j < n - 1) {
            if (self.priority[order[j]] < self.priority[order[j + 1]]) {
                int tmp      = order[j];
                order[j]     = order[j + 1];
                order[j + 1] = tmp;
                swapped      = 1;
            }
            j = j + 1;
        }
    }

    i = 0;
    while (i < n) {
        int idx = order[i];
        if (self.inner.tasks[idx].state != TASK_DONE) {
            self.inner.current          = idx;
            self.inner.tasks[idx].state = TASK_RUNNING;
            unsafe {
                int result = self.inner.tasks[idx].func_(
                    self.inner.tasks[idx].arg_,
                    self.inner.tasks[idx].resume_point);
                if (result == 0) {
                    self.inner.tasks[idx].state = TASK_DONE;
                } else {
                    self.inner.tasks[idx].state        = TASK_READY;
                    self.inner.tasks[idx].resume_point = result;
                    active = active + 1;
                }
            }
        }
        i = i + 1;
    }
    return active;
}

inline void ThreadSched::run_all() {
    while (self.tick() > 0) { }
}

inline int ThreadSched::is_active(Thread t) const {
    int idx = (int)t;
    if (idx < 0 || idx >= self.inner.count) { return 0; }
    return self.inner.tasks[idx].state != TASK_DONE ? 1 : 0;
}

inline void ThreadSched::join_thread(Thread t) {
    while (self.is_active(t) != 0) {
        self.tick();
    }
}

inline int ThreadSched::active_count() const {
    return self.inner.active_count();
}

} // namespace std
