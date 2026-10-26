// SafeC Standard Library — Reactor backend: Win32 (WSAPoll)
//
// See reactor.h for the portable API, usage shape, and the full backend
// file list. Implements the same std::Reactor (Reactor::new)/reactor_run
// surface as reactor_kqueue.sc/reactor_epoll.sc, against WSAPoll instead
// of kqueue/kevent or epoll.
//
// Two structural differences from the other two backends, both handled
// here rather than by changing the portable API:
//
//   - No persistent OS-side registration. kqueue/epoll let you register a
//     fd once and keep getting notified about it; WSAPoll (like plain
//     POSIX poll()) takes a flat array of every fd you currently care
//     about and re-scans all of them on every single call — there is no
//     "add this to the kernel's watch set" primitive. So Reactor::add/
//     remove maintain that array themselves in struct Reactor::watched
//     (see reactor.h), and poll() rebuilds a WSAPOLLFD array from it
//     before calling WSAPoll.
//
//   - No SCHED_SIGNAL support. Windows has no POSIX-signal-equivalent
//     mechanism that integrates with a socket/handle readiness loop the
//     way kqueue's EVFILT_SIGNAL or Linux's signalfd do — Reactor::add
//     silently no-ops for SCHED_SIGNAL (poll() will simply never report
//     one), rather than shipping a half-correct emulation. A program
//     that needs Windows console-control-event handling (Ctrl+C, etc.)
//     alongside this reactor should use SetConsoleCtrlHandler directly
//     and have that handler signal the scheduler some other way (e.g. by
//     closing/writing to a fd already registered here) — out of scope
//     for this backend to provide.
//
// WSAPOLLFD's real layout (from <winsock2.h>) is the ordinary (non-packed)
// C struct { SOCKET fd; SHORT events; SHORT revents; } — SOCKET is
// UINT_PTR (8 bytes on Win64), so natural alignment already places the two
// 2-byte SHORTs immediately after it with no gap, then pads the overall
// struct up to 16 bytes (a multiple of the 8-byte SOCKET's alignment) —
// WSAPollFd below just declares the same three fields in the same order
// and lets ordinary (non-packed) struct layout reproduce that.
//
// NOTE: this backend has been compiled (host and cross-target) but not
// runtime-verified on an actual Windows host in this environment — treat
// it the way reactor_kqueue.sc's own header comment once described the
// (now-implemented) epoll backend: implemented against the documented
// Win32 API surface and struct layouts, but not yet exercised against a
// real WSAPoll call.
// safeguard: link windows -lws2_32
#pragma once
// See reactor_kqueue.sc's identical comment on this macro.
#define SAFEC_REACTOR_BACKEND_INCLUDED_
#include <std/sched/reactor.h>
#include <std/sched/reactor.h>
#include <std/sync/task.h>
#include <std/collections/vec.h>

namespace std {

struct WSAPollFd {
    unsigned long long fd;      // SOCKET (UINT_PTR)
    short               events;
    short               revents;
};

// Winsock must be initialized (WSAStartup) before any other Winsock call
// in the process, including WSAPoll. Its contents are never read here,
// but WSAStartup writes the whole WSADATA (408 bytes on Win64: the two
// strings szDescription[257] and szSystemStatus[129] dominate), so the
// buffer must be at least that big. It once was a 32-byte array, and
// every WSAStartup call smashed ~380 bytes of its caller's stack frames
// (Wine's "Running" status string landing on a saved pointer is what
// crashed examples/async_demo in ConnectFut::poll); the static_assert
// keeps the real layout pinned.
// Guarded — io_nb_win32.sc declares this same struct (same purpose, same
// shape) and a caller may include both (e.g. std::http_serve_reactor
// does). See io_nb_win32.sc's identical guard comment.
#ifndef SAFEC_WSADATA_DEFINED_
#define SAFEC_WSADATA_DEFINED_
struct WSAData_ {
    unsigned short     wVersion;
    unsigned short     wHighVersion;
    unsigned short     iMaxSockets;          // Win64 order: these two precede
    unsigned short     iMaxUdpDg;            // the strings (Win32 puts them after)
    unsigned long long lpVendorInfo;         // char*
    char               szDescription[257];   // WSADESCRIPTION_LEN + 1
    char               szSystemStatus[129];  // WSASYS_STATUS_LEN + 1
};
static_assert(sizeof(struct WSAData_) == 408, "WSADATA is 408 bytes on 64-bit Windows");
#endif

extern int   WSAStartup(unsigned short versionRequested, WSAData_* wsaData);
extern int   WSACleanup();
extern int   WSAPoll(WSAPollFd* fdArray, unsigned long long fds, int timeout);
extern int   closesocket(unsigned long long s);

} // namespace std

// WSAPoll event bits — stable Winsock2 ABI constants, deliberately mirror
// POSIX poll()'s values.
#define SAFEC_POLLRDNORM  0x0100
#define SAFEC_POLLRDBAND  0x0200
#define SAFEC_POLLIN      0x0300  // POLLRDNORM | POLLRDBAND
#define SAFEC_POLLWRNORM  0x0010
#define SAFEC_POLLOUT     0x0010  // POLLWRNORM
#define SAFEC_WSA_VERSION 0x0202  // MAKEWORD(2, 2) — request Winsock 2.2

namespace std {

static short reactor_wsapoll_events_(int portableFilter) {
    if (portableFilter == SCHED_WRITE) { return (short)SAFEC_POLLOUT; }
    return (short)SAFEC_POLLIN; // SCHED_READ, and the default
}

inline struct Reactor Reactor::new() {
    struct Reactor r;
    r.kq = -1;
    // Unused on this backend — only initialized to satisfy definite-
    // initialization for this shared struct Reactor field; see
    // reactor.h's comment on sigfds. SCHED_SIGNAL is a no-op here (see
    // this file's header comment), so nothing ever populates it.
    r.sigfds = Vec::new();
    r.watched = Vec::new();
    return r;
}

inline int Reactor::init() {
    struct WSAData_ wsaData;
    int rc;
    unsafe { rc = WSAStartup((unsigned short)SAFEC_WSA_VERSION, &wsaData); }
    if (rc != 0) {
        return -1;
    }
    self.kq = 0; // unused by this backend; 0 just signals "initialized" for symmetry
    return 0;
}

inline void Reactor::add(int fd, int filter) {
    if (filter == SCHED_SIGNAL) {
        return; // not supported on this backend — see file header comment
    }
    // Replace any existing entry for the same (fd, filter) rather than
    // growing the table unboundedly if a caller re-adds the same pair.
    unsigned long n;
    unsafe { n = self.watched.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct WatchedFd* e;
        unsafe { e = self.watched.at(i); }
        int entryFd;
        int entryFilter;
        unsafe { entryFd = e->fd; entryFilter = e->filter; }
        if (entryFd == fd && entryFilter == filter) {
            return; // already watching this exact (fd, filter)
        }
        i = i + 1UL;
    }
    struct WatchedFd w;
    w.fd = fd;
    w.filter = filter;
    unsafe { self.watched.push(w); }
}

inline void Reactor::remove(int fd, int filter) {
    if (filter == SCHED_SIGNAL) {
        return;
    }
    unsigned long n;
    unsafe { n = self.watched.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct WatchedFd* e;
        unsafe { e = self.watched.at(i); }
        int entryFd;
        int entryFilter;
        unsafe { entryFd = e->fd; entryFilter = e->filter; }
        if (entryFd == fd && entryFilter == filter) {
            struct WatchedFd removed;
            unsafe { self.watched.remove(i); }
            return;
        }
        i = i + 1UL;
    }
}

// Reactor::poll(sched, ...) is reactor.sc's, over this.
inline int Reactor::poll_events_raw_(fn void(?&Erased, int, int) on_ready, ?&Erased ctx, long long timeout_ms) {
    unsigned long n;
    unsafe { n = self.watched.length(); }
    if (n == 0UL) {
        // WSAPoll with zero fds still respects the timeout (unlike some
        // platforms' poll() returning immediately) — but there is
        // nothing useful to wait for here, so just return.
        return 0;
    }

    // WSAPOLLFD array sized to the current watch set — kept at TASK_MAX
    // (std/sync/task.h) so this cap is never the bottleneck below whatever
    // the scheduler itself can already hold; a real caller registering
    // more concurrent fds than that would need a heap-allocated array
    // instead, not something this reactor currently does.
    struct WSAPollFd fds[TASK_MAX];
    // Each entry's filter, captured with its fd: the callbacks below may
    // remove (or add) registrations — std/async's IoDriver unregisters an
    // fd as soon as its waiter is woken — so 'watched' can shrink or shift
    // while the ready entries are dispatched, and index i of fds[] stops
    // naming index i of 'watched' (reading it there was an out-of-range
    // Vec::at as soon as two fds were ready in one WSAPoll).
    int filters[TASK_MAX];
    unsigned long cap = n;
    if (cap > (unsigned long)TASK_MAX) { cap = (unsigned long)TASK_MAX; }
    unsigned long i = 0UL;
    while (i < cap) {
        struct WatchedFd* w;
        unsafe { w = self.watched.at(i); }
        int wFd;
        int wFilter;
        unsafe { wFd = w->fd; wFilter = w->filter; }
        unsafe {
            fds[i].fd = (unsigned long long)wFd;
            fds[i].events = reactor_wsapoll_events_(wFilter);
            fds[i].revents = (short)0;
            filters[i] = wFilter;
        }
        i = i + 1UL;
    }

    int timeout;
    if (timeout_ms < 0LL) {
        timeout = -1; // WSAPoll: negative = block indefinitely
    } else if (timeout_ms > 2147483647LL) {
        timeout = 2147483647; // clamp — WSAPoll's timeout is a plain 'int' (ms)
    } else {
        timeout = (int)timeout_ms;
    }

    int rc;
    unsafe { rc = WSAPoll(&fds[0], (unsigned long long)cap, timeout); }
    if (rc <= 0) {
        return 0;
    }

    int processed = 0;
    i = 0UL;
    while (i < cap) {
        short revents;
        unsigned long long readyFd;
        unsafe { revents = fds[i].revents; readyFd = fds[i].fd; }
        if (revents != (short)0) {
            int wFilter;
            unsafe { wFilter = filters[i]; }
            on_ready(ctx, (int)readyFd, wFilter);
            processed = processed + 1;
        }
        i = i + 1UL;
    }
    return processed;
}

inline void Reactor::close_() {
    self.watched.clear();
    self.sigfds.clear();
    unsafe { WSACleanup(); }
}

} // namespace std
