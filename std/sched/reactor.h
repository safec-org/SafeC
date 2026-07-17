// SafeC Standard Library — Real-time I/O scheduler (reactor)
//
// A single-thread event loop that lets many concurrent network, file, and
// signal operations make progress on one real OS thread, without ever
// blocking the whole program on any single one of them — "async" in the
// same sense as epoll/kqueue-based runtimes elsewhere, but built directly
// on SafeC's existing cooperative TaskScheduler (std/sync/task.h) rather
// than introducing a second, competing concurrency primitive.
//
// This module is a *new*, hosted-only (macOS/Linux userspace) reactor
// operating on real OS file descriptors from real OS syscalls (open/socket
// + kqueue). It's deliberately separate from std/net/tcp.sc, which is a
// from-scratch, bare-metal-oriented packet-level TCP/IP stack with no OS
// socket syscalls at all — the two serve different targets (hosted
// userspace vs. no-OS-underneath embedded) and neither is a drop-in
// replacement for the other.
//
// Usage shape:
//   1. Spawn tasks on a std::TaskScheduler as usual (see task.h) — each
//      task follows the same 'int(void* arg, int resume_point)' yield
//      protocol.
//   2. A task that would otherwise block on I/O calls
//      'sched->await_fd(fd, SCHED_READ)' (or SCHED_WRITE/SCHED_SIGNAL)
//      before returning its yield value, instead of actually blocking.
//   3. Drive the whole thing with std::reactor_run(sched, reactor) instead
//      of TaskScheduler::run_all() — it ticks every runnable task, and
//      whenever nothing is immediately runnable, blocks in the OS's event
//      queue (zero CPU spin) until the next fd becomes ready, then resumes
//      exactly the task(s) waiting on it.
#pragma once
#include <std/sync/task.h>

// Portable readiness filter ids used by Reactor::add/remove and
// TaskScheduler::await_fd/unblock_fd — translated internally to
// OS-specific values (e.g. EVFILT_READ/WRITE/SIGNAL for the kqueue
// backend). For SCHED_SIGNAL, the 'fd' parameter of add/remove/await_fd is
// actually a signal number, not a file descriptor.
#define SCHED_READ    1
#define SCHED_WRITE   2
#define SCHED_SIGNAL  3

namespace std {

struct Reactor {
    int kq;   // underlying OS event queue fd (kqueue fd on this backend)

    // Opens the underlying OS event queue. Returns 0 on success, -1 on
    // failure (check errno).
    int  init();

    // Registers interest in 'fd' becoming ready for 'filter'. Must be
    // re-added after a EV_ONESHOT-style consumption if the backend uses
    // one-shot semantics — this backend re-registers automatically on
    // every await_fd() call, so callers never need to think about it.
    void add(int fd, int filter);
    void remove(int fd, int filter);

    // Waits up to 'timeout_ms' for events (0 = return immediately if none
    // are pending, negative = wait indefinitely), then calls
    // sched->unblock_fd() for every task waiting on a now-ready
    // (fd, filter). Returns the number of OS events processed (not the
    // number of tasks unblocked, which may differ if several tasks await
    // the same fd).
    int  poll(struct TaskScheduler* sched, long long timeout_ms);

    void close_();
};

struct Reactor reactor_init();

// Drives 'sched' to completion: repeatedly ticks every runnable task, and
// whenever none are immediately runnable, polls the reactor — with a
// zero timeout if some tasks are merely mid-round (so newly-ready I/O is
// picked up without stalling), or an indefinite wait if literally
// everything remaining is blocked on I/O, so the process consumes zero
// CPU until the OS actually has something to report. This is the
// scheduler itself: replaces TaskScheduler::run_all() for any workload
// where tasks call await_fd().
void reactor_run(struct TaskScheduler* sched, struct Reactor* r);

} // namespace std
