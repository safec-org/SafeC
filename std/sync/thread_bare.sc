// SafeC Standard Library — Freestanding Cooperative Threads
#pragma once
#include "thread_bare.h"

struct ThreadSched thread_sched_init() {
    struct ThreadSched s;
    s.inner = task_sched_init();
    int i = 0;
    while (i < THREAD_MAX) {
        s.priority[i] = 0;
        i = i + 1;
    }
    return s;
}

Thread ThreadSched::spawn(void* fn, void* arg, int priority) {
    int id = self.inner.spawn(fn, arg);
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
    // Sort descending by priority (bubble sort — n <= 64, acceptable).
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
                int result = ((int(*)(void*, int))self.inner.tasks[idx].fn)(
                    self.inner.tasks[idx].arg,
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

void ThreadSched::run_all() {
    while (self.tick() > 0) { }
}

int ThreadSched::is_active(Thread t) const {
    int idx = (int)t;
    if (idx < 0 || idx >= self.inner.count) { return 0; }
    return self.inner.tasks[idx].state != TASK_DONE ? 1 : 0;
}

void ThreadSched::join(Thread t) {
    while (self.is_active(t) != 0) {
        self.tick();
    }
}

int ThreadSched::active_count() const {
    return self.inner.active_count();
}
