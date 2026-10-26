// SafeC Standard Library — Reactor: portable scheduling loop
//
// reactor_run's body, and Reactor::poll (a callback over each backend's
// poll_events_raw_), are entirely backend-agnostic (it only calls through
// the portable std::Reactor/std::TaskScheduler interface declared in
// reactor.h/task.h), so it lives here once rather than being duplicated
// verbatim in every backend file (reactor_kqueue.sc, reactor_epoll.sc,
// reactor_win32.sc) — each of those includes this file instead of
// redefining it, the same way collections/*.sc share one implementation
// regardless of which platform ends up linking it in.
#pragma once
#include <std/sched/reactor.h>
#include <std/sync/task.h>

namespace std {

// poll()'s callback: what readiness means to a TaskScheduler.
static void reactor_unblock_(?&TaskScheduler sq, int fd, int filter) {
    match (sq) {
        case some(s): s.unblock_fd(fd, filter);
        case null: { }
    }
}

int Reactor::poll(&TaskScheduler sched, long long timeout_ms) {
    fn void(?&TaskScheduler, int, int) cb = reactor_unblock_;
    unsafe { return self.poll_events_raw_((fn void(?&Erased, int, int))cb, (?&Erased)sched, timeout_ms); }
}

void reactor_run(&TaskScheduler sched, &Reactor r) {
    while (sched->active_count() > 0) {
        int active = sched->tick();
        if (active == 0) {
            break;
        }
        if (sched->has_ready() != 0) {
            // Some task is still immediately runnable this round — drain
            // any already-pending events without stalling, then loop back
            // to tick() right away.
            r->poll(sched, 0LL);
        } else {
            // Everything remaining is blocked on I/O: nothing to do until
            // the OS reports something, so wait for it instead of spinning.
            r->poll(sched, -1LL);
        }
    }
}

} // namespace std
