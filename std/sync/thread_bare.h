// SafeC Standard Library — Freestanding Cooperative Threads
// Priority-aware wrapper over TaskScheduler.  Freestanding-safe.
//
// Thread functions use the same coroutine protocol as TaskScheduler:
//   int func(?&T arg, int resume_point)
//   Returns > 0 to yield (value = new resume_point), 0 when done.
//
// Threads with higher priority values run before lower-priority ones
// within each scheduling tick.
#pragma once
#include <std/sync/task.h>

#define THREAD_MAX  TASK_MAX
#define THREAD_NONE (-1)

// Thread identifier: index into the ThreadSched table.
namespace std {

newtype Thread = int;

struct ThreadSched {
    struct TaskScheduler inner;             // underlying round-robin scheduler
    int                  priority[THREAD_MAX];  // per-thread priority (higher = first)

    // Type-erased spawn behind thread_spawn() (below).
    Thread  spawn_raw_(void* func, void* arg, int priority);

    // Run one scheduling pass: threads are served in descending priority order.
    // Returns the number of still-active threads.
    int     tick();

    // Run until all threads complete.
    void    run_all();

    // Return 1 if thread `t` is still active (READY or RUNNING), 0 if DONE.
    int     is_active(Thread t) const;

    // Cooperative join: calls tick() until thread `t` is done.
    void    join_thread(Thread t);

    // Return number of active (non-DONE) threads.
    int     active_count() const;
};

// Initialize a thread scheduler (all slots empty, no threads running).
struct ThreadSched thread_sched_init();

// Spawn a thread running func(arg, resume_point) (same protocol as
// task_spawn). priority >= 0; higher values run earlier in each tick.
// Returns the Thread id (>= 0) or THREAD_NONE if the table is full.
generic<T>
Thread thread_spawn(&ThreadSched s, fn int(?&T, int) func, ?&T arg, int priority) {
    unsafe { return s.spawn_raw_((void*)func, (void*)arg, priority); }
}

} // namespace std
