// SafeC Standard Library — Cooperative Task Scheduler
// Stackless cooperative multitasking. Tasks yield voluntarily via resume_point.
#pragma once

// Task states
#define TASK_READY    0
#define TASK_RUNNING  1
#define TASK_DONE     2

#define TASK_MAX 64

namespace std {

struct Task {
    void*  func;            // task function: int(*)(void* arg, int resume_point)
    void*  arg;           // user argument
    int    state;         // TASK_READY, TASK_RUNNING, or TASK_DONE
    int    resume_point;  // where to resume (0 = start)
    // I/O-readiness blocking (std/sched/reactor.h drives these): a task
    // blocks itself by calling TaskScheduler::await_fd(fd, filter) from
    // *within* its own function body (self.current identifies it while
    // running) before returning its yield value. tick() then skips it
    // entirely — no re-invocation, no busy-polling — until something calls
    // unblock_fd() with a matching (fd, filter), normally the reactor
    // reporting that fd became ready.
    int    blocked;       // 0 = runnable, 1 = blocked awaiting I/O
    int    wait_fd;       // fd (or signal number) this task is waiting on
    int    wait_filter;   // SCHED_READ / SCHED_WRITE / SCHED_SIGNAL (see reactor.h)
};

struct TaskScheduler {
    struct Task   tasks[TASK_MAX];
    int           count;     // number of registered tasks
    int           current;   // index of currently running task

    // Spawn a new task. func signature: int func(void* arg, int resume_point)
    //   - func returns >0 to yield with a new resume_point
    //   - func returns 0 to indicate completion
    // Returns task index, or -1 if scheduler is full.
    int           spawn_task(void* func, void* arg);

    // Run one scheduling round: every non-blocked, non-done task gets one
    // turn (blocked tasks are skipped without being invoked — see
    // await_fd). Returns the count of still-active (blocked or ready,
    // i.e. not yet TASK_DONE) tasks.
    int           tick();

    // Run all tasks in round-robin until all are TASK_DONE. Note: a task
    // that calls await_fd() and is never unblocked (no reactor driving
    // this scheduler) will spin run_all() forever — use
    // std::reactor_run() instead whenever any task may block on I/O.
    void          run_all();

    // Return number of active (non-DONE) tasks.
    int           active_count() const;

    // Marks the *currently running* task (self.current) as blocked until
    // 'fd' becomes ready for 'filter'. Must be called from within a task
    // function's own body during its turn (i.e. before that turn's
    // 'return'), not from outside the scheduler.
    void          await_fd(int fd, int filter);

    // Unblocks every task waiting on the given (fd, filter) pair. Returns
    // how many tasks were unblocked. Called by std::Reactor::poll when the
    // OS reports readiness — not normally called directly.
    int           unblock_fd(int fd, int filter);

    // True if at least one non-done task is currently blocked.
    int           has_blocked() const;
    // True if at least one non-done task is immediately runnable (not
    // blocked). Together with has_blocked(), this is what reactor_run()
    // uses to decide whether to poll with a zero or infinite timeout.
    int           has_ready() const;
};

// Initialize a task scheduler.
struct TaskScheduler task_sched_init();

} // namespace std
