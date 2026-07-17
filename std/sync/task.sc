// SafeC Standard Library — Cooperative Task Scheduler
#pragma once
#include <std/sync/task.h>

namespace std {

inline struct TaskScheduler task_sched_init() {
    struct TaskScheduler s;
    s.count   = 0;
    s.current = 0;
    return s;
}

inline int TaskScheduler::spawn_task(void* func, void* arg) {
    if (self.count >= 64) {
        return -1;
    }
    int idx = self.count;
    self.tasks[idx].func           = func;
    self.tasks[idx].arg          = arg;
    self.tasks[idx].state        = 0; // TASK_READY
    self.tasks[idx].resume_point = 0;
    self.tasks[idx].blocked      = 0;
    self.tasks[idx].wait_fd      = -1;
    self.tasks[idx].wait_filter  = 0;
    self.count = self.count + 1;
    return idx;
}

int TaskScheduler::tick() {
    int active = 0;
    int i = 0;
    while (i < self.count) {
        if (self.tasks[i].state != 2) { // not TASK_DONE
            active = active + 1;
            // Blocked tasks are skipped entirely — not invoked, not
            // counted as making progress this round — until unblock_fd()
            // clears them (normally the reactor reporting fd readiness).
            // This is what turns tick() from a busy-poll-every-task loop
            // into something reactor_run() can drive efficiently: it only
            // ever calls a task that can actually do something.
            if (self.tasks[i].blocked == 0) {
                self.current = i;
                self.tasks[i].state = 1; // TASK_RUNNING
                unsafe {
                    // Call: int func(void* arg, int resume_point)
                    fn int(void*, int) taskfn = (fn int(void*, int))self.tasks[i].func;
                    int result = taskfn(
                        self.tasks[i].arg, self.tasks[i].resume_point);
                    if (result == 0) {
                        self.tasks[i].state = 2; // TASK_DONE
                        self.tasks[i].blocked = 0;
                        active = active - 1; // just finished, no longer active
                    } else {
                        self.tasks[i].state = 0; // TASK_READY (yielded)
                        self.tasks[i].resume_point = result;
                        // 'blocked' may have just been set by this task's
                        // own await_fd() call during the step above — left
                        // as-is either way; tick() will skip it from here
                        // on until unblock_fd() clears it.
                    }
                }
            }
        }
        i = i + 1;
    }
    return active;
}

inline void TaskScheduler::run_all() {
    while (self.tick() > 0) {
        // Keep running until all tasks are done
    }
}

inline int TaskScheduler::active_count() const {
    int count = 0;
    int i = 0;
    while (i < self.count) {
        if (self.tasks[i].state != 2) {
            count = count + 1;
        }
        i = i + 1;
    }
    return count;
}

inline void TaskScheduler::await_fd(int fd, int filter) {
    if (self.current < 0 || self.current >= self.count) {
        return;
    }
    self.tasks[self.current].blocked     = 1;
    self.tasks[self.current].wait_fd     = fd;
    self.tasks[self.current].wait_filter = filter;
}

inline int TaskScheduler::unblock_fd(int fd, int filter) {
    int n = 0;
    int i = 0;
    while (i < self.count) {
        if (self.tasks[i].blocked == 1 && self.tasks[i].wait_fd == fd &&
            self.tasks[i].wait_filter == filter) {
            self.tasks[i].blocked = 0;
            n = n + 1;
        }
        i = i + 1;
    }
    return n;
}

inline int TaskScheduler::has_blocked() const {
    int i = 0;
    while (i < self.count) {
        if (self.tasks[i].state != 2 && self.tasks[i].blocked == 1) {
            return 1;
        }
        i = i + 1;
    }
    return 0;
}

inline int TaskScheduler::has_ready() const {
    int i = 0;
    while (i < self.count) {
        if (self.tasks[i].state != 2 && self.tasks[i].blocked == 0) {
            return 1;
        }
        i = i + 1;
    }
    return 0;
}

} // namespace std
