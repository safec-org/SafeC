// SafeC Standard Library — Reactor backend: epoll (Linux / Android)
//
// See reactor.h for the portable API, usage shape, and the full backend
// file list. Implements the same std::Reactor/reactor_init/reactor_run
// surface as reactor_kqueue.sc, against epoll_create1/epoll_ctl/
// epoll_wait instead of kqueue/kevent.
//
// epoll has no native "wake on signal" filter the way kqueue's
// EVFILT_SIGNAL does. SCHED_SIGNAL here uses the standard Linux idiom
// instead: block the signal (so its default disposition never fires),
// convert it into a readable fd via signalfd(), and epoll-poll that fd
// like any other — see Reactor::add's SCHED_SIGNAL branch below. The
// signum -> signalfd mapping lives in struct Reactor::sigfds (reactor.h)
// since epoll_wait only ever reports back the fd that became ready, not
// which signal (if any) it corresponds to.
//
// struct epoll_event is __attribute__((packed)) in <sys/epoll.h> — 12
// bytes (events:4 + data:8) on both 32- and 64-bit Linux, NOT the 16 bytes
// a naturally-aligned '{ unsigned int; unsigned long; }' would compute
// (the 8-byte data field would otherwise pull in 4 bytes of padding after
// events to reach an 8-byte boundary). EpollEvent below is declared
// 'packed' to match the kernel's actual wire layout — getting this wrong
// would silently misalign every element past the first in a multi-event
// epoll_wait() array, the same class of bug reactor_kqueue.sc's KEvent
// comment describes for kevent's layout.
//
// sigset_t is 128 bytes in glibc on every Linux architecture it supports
// (1024 signal bits / 8 = 128, independent of word width) — SigSet below
// is a flat byte array rather than a word array so its *total* size stays
// correct on both 32- and 64-bit targets without needing a separate
// element count for each (an 'unsigned long bits[16]' declaration would
// only be 64 bytes on a 32-bit target, where 'unsigned long' is 4 bytes
// instead of 8).
#pragma once
#include <std/sched/reactor.h>
#include <std/sched/reactor.sc>
#include <std/sync/task.sc>
#include <std/collections/vec.sc>

namespace std {

packed struct EpollEvent {
    unsigned int  events;   // EPOLLIN / EPOLLOUT / ...
    unsigned long data;     // epoll_data_t — we only ever use its fd-sized slot
};

struct SigSet {
    unsigned char bytes[128]; // sizeof(sigset_t) — see file header comment
};

struct SignalfdSiginfo {
    unsigned char bytes[128]; // sizeof(struct signalfd_siginfo); contents unused, only drained
};

extern int  epoll_create1(int flags);
extern int  epoll_ctl(int epfd, int op, int fd, const void* event);
extern int  epoll_wait(int epfd, void* events, int maxevents, int timeout_ms);
extern int  close(int fd);
extern long read(int fd, void* buf, unsigned long count);
extern int  sigemptyset(void* set);
extern int  sigaddset(void* set, int signum);
extern int  sigprocmask(int how, const void* set, void* oldset);
extern int  signalfd(int fd, const void* mask, int flags);

} // namespace std

// epoll_ctl operations, event bits, and sigprocmask 'how' (stable Linux
// syscall ABI constants — same values across every glibc architecture).
#define SAFEC_EPOLL_CTL_ADD  1
#define SAFEC_EPOLL_CTL_DEL  2
#define SAFEC_EPOLLIN        0x00000001
#define SAFEC_EPOLLOUT       0x00000004
#define SAFEC_SIG_BLOCK      0

namespace std {

static unsigned int reactor_epoll_events_(int portableFilter) {
    if (portableFilter == SCHED_WRITE) { return (unsigned int)SAFEC_EPOLLOUT; }
    return (unsigned int)SAFEC_EPOLLIN; // SCHED_READ, and the default
}

// Finds the signal number registered under 'fd' in r->sigfds, if any —
// used by poll() to tell a ready signalfd apart from a ready ordinary fd
// (both arrive through the same epoll_wait() array with no other marker).
// Returns -1 if 'fd' isn't one of ours.
static int reactor_signum_for_fd_(const struct Vec* sigfds, int fd) {
    unsigned long n;
    unsafe { n = sigfds->length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct SigFdEntry* e;
        unsafe { e = (struct SigFdEntry*)sigfds->get_raw(i); }
        int entryFd;
        int entrySignum;
        unsafe { entryFd = e->fd; entrySignum = e->signum; }
        if (entryFd == fd) { return entrySignum; }
        i = i + 1UL;
    }
    return -1;
}

inline struct Reactor reactor_init() {
    struct Reactor r;
    r.kq = -1;
    r.sigfds = vec_new(sizeof(struct SigFdEntry));
    // Unused on this backend (epoll has persistent kernel-side
    // registration) — only initialized to satisfy definite-initialization
    // for this shared struct Reactor field; see reactor.h's comment on
    // watched.
    r.watched = vec_new(sizeof(struct WatchedFd));
    return r;
}

inline int Reactor::init() {
    unsafe { self.kq = epoll_create1(0); }
    if (self.kq < 0) {
        return -1;
    }
    return 0;
}

inline void Reactor::add(int fd, int filter) {
    if (filter == SCHED_SIGNAL) {
        // 'fd' is actually a signal number here (see reactor.h) — block it
        // first (signalfd() requires the signal to already be blocked, or
        // its default disposition — often terminating the process — still
        // fires before the fd ever sees it), then convert it into a
        // readable fd and epoll-poll that instead.
        int signum = fd;
        struct SigSet set;
        unsafe {
            sigemptyset((void*)&set);
            sigaddset((void*)&set, signum);
            sigprocmask(SAFEC_SIG_BLOCK, (const void*)&set, (void*)0);
        }
        int sfd;
        unsafe { sfd = signalfd(-1, (const void*)&set, 0); }
        if (sfd < 0) {
            return;
        }

        struct SigFdEntry e;
        e.signum = signum;
        e.fd = sfd;
        unsafe { self.sigfds.push((const void*)&e); }

        struct EpollEvent ev;
        ev.events = (unsigned int)SAFEC_EPOLLIN;
        ev.data = (unsigned long)sfd;
        unsafe { epoll_ctl(self.kq, SAFEC_EPOLL_CTL_ADD, sfd, (const void*)&ev); }
        return;
    }

    struct EpollEvent ev;
    ev.events = reactor_epoll_events_(filter);
    ev.data = (unsigned long)fd;
    unsafe { epoll_ctl(self.kq, SAFEC_EPOLL_CTL_ADD, fd, (const void*)&ev); }
}

inline void Reactor::remove(int fd, int filter) {
    if (filter == SCHED_SIGNAL) {
        int signum = fd;
        unsigned long n = self.sigfds.length();
        unsigned long i = 0UL;
        while (i < n) {
            struct SigFdEntry* e;
            unsafe { e = (struct SigFdEntry*)self.sigfds.get_raw(i); }
            int entrySignum;
            unsafe { entrySignum = e->signum; }
            if (entrySignum == signum) {
                int sfd;
                unsafe { sfd = e->fd; }
                unsafe { epoll_ctl(self.kq, SAFEC_EPOLL_CTL_DEL, sfd, (const void*)0); }
                unsafe { close(sfd); }
                struct SigFdEntry removed;
                unsafe { self.sigfds.remove(i, (void*)&removed); }
                return;
            }
            i = i + 1UL;
        }
        return;
    }

    struct EpollEvent ev; // epoll_ctl(DEL) ignores this, but still wants a valid pointer
    ev.events = 0U;
    ev.data = 0UL;
    unsafe { epoll_ctl(self.kq, SAFEC_EPOLL_CTL_DEL, fd, (const void*)&ev); }
}

inline int Reactor::poll(&TaskScheduler sched, long long timeout_ms) {
    struct EpollEvent events[64];
    int timeout;
    if (timeout_ms < 0LL) {
        timeout = -1; // epoll_wait: negative = block indefinitely
    } else if (timeout_ms > 2147483647LL) {
        timeout = 2147483647; // clamp — epoll_wait's timeout is a plain 'int' (ms)
    } else {
        timeout = (int)timeout_ms;
    }
    int n;
    unsafe {
        n = epoll_wait(self.kq, (void*)&events[0], 64, timeout);
    }
    if (n <= 0) {
        return 0;
    }
    int i = 0;
    while (i < n) {
        int fd = (int)events[i].data;
        int signum = reactor_signum_for_fd_(&self.sigfds, fd);
        if (signum >= 0) {
            // Drain the pending signalfd_siginfo so this fd doesn't stay
            // "ready" forever (epoll is level-triggered by default here) —
            // its contents are unused, only the fact that it arrived
            // matters.
            struct SignalfdSiginfo info;
            unsafe { read(fd, (void*)&info, sizeof(struct SignalfdSiginfo)); }
            sched->unblock_fd(signum, SCHED_SIGNAL);
        } else {
            int filter = (events[i].events & (unsigned int)SAFEC_EPOLLOUT) != 0U ? SCHED_WRITE : SCHED_READ;
            sched->unblock_fd(fd, filter);
        }
        i = i + 1;
    }
    return n;
}

inline void Reactor::close_() {
    // Close every outstanding signalfd we own before the epoll fd itself.
    unsigned long n = self.sigfds.length();
    unsigned long i = 0UL;
    while (i < n) {
        struct SigFdEntry* e;
        unsafe { e = (struct SigFdEntry*)self.sigfds.get_raw(i); }
        unsafe { close(e->fd); }
        i = i + 1UL;
    }
    self.sigfds.clear();
    unsafe { close(self.kq); }
}

} // namespace std
