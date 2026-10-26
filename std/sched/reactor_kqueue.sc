// SafeC Standard Library — Reactor backend: kqueue (macOS / iOS / BSD)
//
// See reactor.h for the portable API and usage shape, and for the full
// list of backend files (reactor_epoll.sc for Linux/Android,
// reactor_win32.sc for Windows) — all three implement the exact same
// std::Reactor (Reactor::new)/reactor_run surface, since SCHED_READ/
// SCHED_WRITE/SCHED_SIGNAL are already backend-agnostic; nothing in this
// module's callers needs to know which one is actually compiled in.
//
// struct kevent's real layout (from <sys/event.h>) is 32 bytes on 64-bit
// Darwin: ident(8) filter(2) flags(2) fflags(4) data(8) udata(8), each
// field naturally aligned with no padding — KEvent below mirrors it field
// for field rather than using an opaque byte blob, since the layout is a
// stable, documented part of the kqueue syscall ABI.
#pragma once
// Marks that a reactor backend is present in this translation unit — lets
// callers that need a reactor conditionally on it (std::http_serve_reactor
// in std/http/http.sc) without forcing every other http.sc caller to also
// link one just to satisfy Reactor::add/etc.'s symbol references. See that
// file's own comment on this macro.
#define SAFEC_REACTOR_BACKEND_INCLUDED_
#include <std/sched/reactor.h>
#include <std/sync/task.h>
#include <std/collections/vec.h>

namespace std {

struct KEvent {
    unsigned long   ident;    // uintptr_t: fd, or signal number for EVFILT_SIGNAL
    short           filter;
    unsigned short  flags;
    unsigned int    fflags;
    long            data;     // intptr_t
    unsigned char*           udata;
};

struct TimeSpec {
    long tv_sec;
    long tv_nsec;
};

extern int   kqueue();
extern int   kevent(int kq, const unsigned char* changelist, int nchanges,
                     unsigned char* eventlist, int nevents, const unsigned char* timeout);
extern int   close(int fd);
extern unsigned char* signal(int sig, unsigned char* handler);

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

inline struct Reactor Reactor::new() {
    struct Reactor r;
    r.kq = -1;
    // Unused on this backend (kqueue has native EVFILT_SIGNAL and
    // persistent kernel-side registration) — only initialized to satisfy
    // definite-initialization for these shared struct Reactor fields; see
    // reactor.h's comments on sigfds/watched.
    r.sigfds = Vec::new();
    r.watched = Vec::new();
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
        unsafe { signal(fd, (unsigned char*)SAFEC_SIG_IGN); }
    }
    struct KEvent ev;
    ev.ident  = (unsigned long)fd;
    ev.filter = reactor_kq_filter_(filter);
    ev.flags  = (unsigned short)(SAFEC_EV_ADD | SAFEC_EV_ENABLE);
    ev.fflags = 0U;
    ev.data   = 0L;
    unsafe { ev.udata = (unsigned char*)0; }
    unsafe {
        kevent(self.kq, (const unsigned char*)&ev, 1, (unsigned char*)0, 0, (const unsigned char*)0);
    }
}

inline void Reactor::remove(int fd, int filter) {
    struct KEvent ev;
    ev.ident  = (unsigned long)fd;
    ev.filter = reactor_kq_filter_(filter);
    ev.flags  = (unsigned short)SAFEC_EV_DELETE;
    ev.fflags = 0U;
    ev.data   = 0L;
    unsafe { ev.udata = (unsigned char*)0; }
    unsafe {
        kevent(self.kq, (const unsigned char*)&ev, 1, (unsigned char*)0, 0, (const unsigned char*)0);
    }
}

// Reactor::poll(sched, ...) is reactor.sc's, over this.
inline int Reactor::poll_events_raw_(fn void(?&Erased, int, int) on_ready, ?&Erased ctx, long long timeout_ms) {
    struct KEvent events[64];
    struct TimeSpec ts;
    const unsigned char* toPtr;
    if (timeout_ms < 0LL) {
        unsafe { toPtr = (const unsigned char*)0; } // NULL timeout = block indefinitely
    } else {
        ts.tv_sec  = timeout_ms / 1000LL;
        ts.tv_nsec = (timeout_ms % 1000LL) * 1000000LL;
        unsafe { toPtr = (const unsigned char*)&ts; }
    }
    int n;
    unsafe {
        n = kevent(self.kq, (const unsigned char*)0, 0, (unsigned char*)&events[0], 64, toPtr);
    }
    if (n <= 0) {
        return 0;
    }
    int i = 0;
    while (i < n) {
        int filter = reactor_portable_filter_(events[i].filter);
        on_ready(ctx, (int)events[i].ident, filter);
        i = i + 1;
    }
    return n;
}

inline void Reactor::close_() {
    unsafe { close(self.kq); }
}

} // namespace std

#include <std/sched/reactor.h>
