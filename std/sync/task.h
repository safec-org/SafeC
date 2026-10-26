// SafeC Standard Library — Cooperative Task Scheduler
// Stackless cooperative multitasking. Tasks yield voluntarily via resume_point.
#pragma once
#include <std/erased.h>

// Task states
#define TASK_READY    0
#define TASK_RUNNING  1
#define TASK_DONE     2

// Was 64 -- too low for a reactor thread under real concurrent load: once
// more than TASK_MAX connections are simultaneously in flight on one
// TaskScheduler, TaskScheduler::spawn() returns -1 and the caller (see
// std/http/http.sc's http_accept_task_) closes the freshly-accepted fd
// before reading its pending request bytes, which the kernel turns into
// a TCP RST to the client. Verified: `ab -c 128` against a single
// http_serve_reactor thread reliably reproduced exactly this ("Connection
// reset by peer" partway through the run); `ab -c 100` (under the old 64)
// did not. 1024 gives real headroom above realistic load-test/production
// concurrency; each Task is small (a few pointers/ints), so the resulting
// per-scheduler array is trivial (~tens of KB, not the old few KB, but
// still nothing).
#define TASK_MAX 1024

namespace std {

// What a cooperative scheduler offers to the code driving it: one round,
// run to completion, and how many tasks are left. TaskScheduler and
// ThreadSched (thread_bare.h, priority-ordered) both conform, so a driver
// loop written once against TaskRunner runs either. Spawning isn't part of
// it: ThreadSched::spawn takes a priority that TaskScheduler::spawn has no
// place for. (Not called Scheduler: that's kernel/scheduler.h's struct.)
trait TaskRunner {
    int  tick();                 // one round; returns the still-active count
    void run_all();              // until every task is done
    int  active_count() const;
}

struct Task {
    // The task function int(?&T arg, int resume_point) and its argument,
    // erased (see std/erased.h) — both set only by TaskScheduler::spawn.
    fn int(?&Erased, int) func_;
    ?&Erased              arg_;
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

    // An empty scheduler (no tasks).
    static struct TaskScheduler new();

    // Spawn a task running func(arg, resume_point):
    //   - func returns >0 to yield, with that value as the next resume_point
    //   - func returns 0 when done
    // Returns the task index, or -1 if the scheduler is full.
    //   static int blink(?&Led led, int step) { ...; return step < 3 ? step + 1 : 0; }
    //   sched.spawn(blink, (?&Led)&led);
    generic<T> int spawn(fn int(?&T, int) func, ?&T arg);

    // Type-erased spawn behind spawn() and the compiler's bare-metal spawn
    // runtime. Returns task index, or -1 if full.
    int           spawn_raw_(fn int(?&Erased, int) func, ?&Erased arg);

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

generic<T>
int TaskScheduler::spawn(fn int(?&T, int) func, ?&T arg) {
    unsafe { return self.spawn_raw_((fn int(?&Erased, int))func, (?&Erased)arg); }
}

// Runs at most max_rounds rounds of s (a bounded run_all, e.g. for a
// watchdog-fed main loop). Returns the tasks still active afterwards.
generic<S: TaskRunner>
int run_rounds(&S s, int max_rounds) {
    int left = s.active_count();
    int r = 0;
    while (r < max_rounds && left > 0) { left = s.tick(); r = r + 1; }
    return left;
}

} // namespace std
