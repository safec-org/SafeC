// SafeC Standard Library — Reactor backend: kqueue (macOS / iOS / BSD)
//
// See reactor.h for the portable API and usage shape, and for the full
// list of backend files (reactor_epoll.sc for Linux/Android,
// reactor_win32.sc for Windows) — all three implement the exact same
// std::Reactor/reactor_init/reactor_run surface, since SCHED_READ/
// SCHED_WRITE/SCHED_SIGNAL are already backend-agnostic; nothing in this
// module's callers needs to know which one is actually compiled in.
//
// struct kevent's real layout (from <sys/event.h>) is 32 bytes on 64-bit
// Darwin: ident(8) filter(2) flags(2) fflags(4) data(8) udata(8), each
// field naturally aligned with no padding — KEvent below mirrors it field
// for field rather than using an opaque byte blob, since the layout is a
// stable, documented part of the kqueue syscall ABI.
#pragma once
#include <std/sched/reactor.h>
#include <std/sync/task.sc>
#include <std/collections/vec.sc>

namespace std {

struct KEvent {
    unsigned long   ident;    // uintptr_t: fd, or signal number for EVFILT_SIGNAL
    short           filter;
    unsigned short  flags;
    unsigned int    fflags;
    long            data;     // intptr_t
    void*           udata;
};

struct TimeSpec {
    long tv_sec;
    long tv_nsec;
};

extern int   kqueue();
extern int   kevent(int kq, const void* changelist, int nchanges,
                     void* eventlist, int nevents, const void* timeout);
extern int   close(int fd);
extern void* signal(int sig, void* handler);

} // namespace std

// kqueue filter/flag values (stable BSD/Darwin syscall ABI constants).
#define SAFEC_EVFILT_READ    (-1)
#define SAFEC_EVFILT_WRITE   (-2)
#define SAFEC_EVFILT_SIGNAL  (-6)
#define SAFEC_EV_ADD         0x0001
#define SAFEC_EV_DELETE      0x0002
#define SAFEC_EV_ENABLE      0x0004
#define SAFEC_SIG_IGN        1

namespace std {

static short reactor_kq_filter_(int portableFilter) {
    if (portableFilter == SCHED_WRITE)  { return (short)SAFEC_EVFILT_WRITE; }
    if (portableFilter == SCHED_SIGNAL) { return (short)SAFEC_EVFILT_SIGNAL; }
    return (short)SAFEC_EVFILT_READ; // SCHED_READ, and the default
}

static int reactor_portable_filter_(short kqFilter) {
    if (kqFilter == (short)SAFEC_EVFILT_WRITE)  { return SCHED_WRITE; }
    if (kqFilter == (short)SAFEC_EVFILT_SIGNAL) { return SCHED_SIGNAL; }
    return SCHED_READ;
}

inline struct Reactor reactor_init() {
    struct Reactor r;
    r.kq = -1;
    // Unused on this backend (kqueue has native EVFILT_SIGNAL and
    // persistent kernel-side registration) — only initialized to satisfy
    // definite-initialization for these shared struct Reactor fields; see
    // reactor.h's comments on sigfds/watched.
    r.sigfds = vec_new(sizeof(struct SigFdEntry));
    r.watched = vec_new(sizeof(struct WatchedFd));
    return r;
}

inline int Reactor::init() {
    unsafe { self.kq = kqueue(); }
    if (self.kq < 0) {
        return -1;
    }
    return 0;
}

inline void Reactor::add(int fd, int filter) {
    // EVFILT_SIGNAL requires the process to ignore the signal's default
    // disposition first, or the default action (often terminate) happens
    // before kqueue ever gets a chance to report it — standard, documented
    // kqueue behavior, not optional.
    if (filter == SCHED_SIGNAL) {
        unsafe { signal(fd, (void*)SAFEC_SIG_IGN); }
    }
    struct KEvent ev;
    ev.ident  = (unsigned long)fd;
    ev.filter = reactor_kq_filter_(filter);
    ev.flags  = (unsigned short)(SAFEC_EV_ADD | SAFEC_EV_ENABLE);
    ev.fflags = 0U;
    ev.data   = 0L;
    unsafe { ev.udata = (void*)0; }
    unsafe {
        kevent(self.kq, (const void*)&ev, 1, (void*)0, 0, (const void*)0);
    }
}

inline void Reactor::remove(int fd, int filter) {
    struct KEvent ev;
    ev.ident  = (unsigned long)fd;
    ev.filter = reactor_kq_filter_(filter);
    ev.flags  = (unsigned short)SAFEC_EV_DELETE;
    ev.fflags = 0U;
    ev.data   = 0L;
    unsafe { ev.udata = (void*)0; }
    unsafe {
        kevent(self.kq, (const void*)&ev, 1, (void*)0, 0, (const void*)0);
    }
}

inline int Reactor::poll(struct TaskScheduler* sched, long long timeout_ms) {
    struct KEvent events[64];
    struct TimeSpec ts;
    const void* toPtr;
    if (timeout_ms < 0LL) {
        unsafe { toPtr = (const void*)0; } // NULL timeout = block indefinitely
    } else {
        ts.tv_sec  = timeout_ms / 1000LL;
        ts.tv_nsec = (timeout_ms % 1000LL) * 1000000LL;
        unsafe { toPtr = (const void*)&ts; }
    }
    int n;
    unsafe {
        n = kevent(self.kq, (const void*)0, 0, (void*)&events[0], 64, toPtr);
    }
    if (n <= 0) {
        return 0;
    }
    int i = 0;
    while (i < n) {
        int filter = reactor_portable_filter_(events[i].filter);
        sched->unblock_fd((int)events[i].ident, filter);
        i = i + 1;
    }
    return n;
}

inline void Reactor::close_() {
    unsafe { close(self.kq); }
}

} // namespace std

#include <std/sched/reactor.sc>
