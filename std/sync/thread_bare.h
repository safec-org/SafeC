// SafeC Standard Library — Freestanding Cooperative Threads
// Priority-aware wrapper over TaskScheduler.  Freestanding-safe.
//
// Thread functions use the same coroutine protocol as TaskScheduler:
//   int fn(void* arg, int resume_point)
//   Returns > 0 to yield (value = new resume_point), 0 when done.
//
// Threads with higher priority values run before lower-priority ones
// within each scheduling tick.
#pragma once
#include "task.h"

#define THREAD_MAX  TASK_MAX
#define THREAD_NONE (-1)

// Thread identifier: index into the ThreadSched table.
newtype Thread = int;

struct ThreadSched {
    struct TaskScheduler inner;             // underlying round-robin scheduler
    int                  priority[THREAD_MAX];  // per-thread priority (higher = first)

    // Spawn a new thread.  fn: int(*)(void* arg, int resume_point).
    // priority >= 0; higher values run earlier in each tick.
    // Returns Thread id (>= 0) or THREAD_NONE if table is full.
    Thread  spawn(void* fn, void* arg, int priority);

    // Run one scheduling pass: threads are served in descending priority order.
    // Returns the number of still-active threads.
    int     tick();

    // Run until all threads complete.
    void    run_all();

    // Return 1 if thread `t` is still active (READY or RUNNING), 0 if DONE.
    int     is_active(Thread t) const;

    // Cooperative join: calls tick() until thread `t` is done.
    void    join(Thread t);

    // Return number of active (non-DONE) threads.
    int     active_count() const;
};

// Initialize a thread scheduler (all slots empty, no threads running).
struct ThreadSched thread_sched_init();
