// SafeC Standard Library — Cooperative Task Scheduler
#pragma once
#include "task.h"

struct TaskScheduler task_sched_init() {
    struct TaskScheduler s;
    s.count   = 0;
    s.current = 0;
    return s;
}

int TaskScheduler::spawn(void* fn, void* arg) {
    if (self.count >= 64) {
        return -1;
    }
    int idx = self.count;
    self.tasks[idx].fn           = fn;
    self.tasks[idx].arg          = arg;
    self.tasks[idx].state        = 0; // TASK_READY
    self.tasks[idx].resume_point = 0;
    self.count = self.count + 1;
    return idx;
}

int TaskScheduler::tick() {
    int active = 0;
    int i = 0;
    while (i < self.count) {
        if (self.tasks[i].state != 2) { // not TASK_DONE
            self.current = i;
            self.tasks[i].state = 1; // TASK_RUNNING
            unsafe {
                // Call: int fn(void* arg, int resume_point)
                int result = ((int(*)(void*, int))self.tasks[i].fn)(
                    self.tasks[i].arg, self.tasks[i].resume_point);
                if (result == 0) {
                    self.tasks[i].state = 2; // TASK_DONE
                } else {
                    self.tasks[i].state = 0; // TASK_READY (yielded)
                    self.tasks[i].resume_point = result;
                    active = active + 1;
                }
            }
        }
        i = i + 1;
    }
    return active;
}

void TaskScheduler::run_all() {
    while (self.tick() > 0) {
        // Keep running until all tasks are done
    }
}

int TaskScheduler::active_count() const {
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
