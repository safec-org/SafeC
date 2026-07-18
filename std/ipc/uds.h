// SafeC Standard Library — Unix Domain Sockets
//
// Named, addressable IPC between unrelated processes (unlike pipe.h's
// anonymous pipes, which only work between a process and something that
// already has the fds — typically a fork()'d child): any process that
// knows the filesystem path can connect, the same way any process that
// knows a TCP port can connect to std/sched/io_nb.h's tcp_listen_nb.
// Deliberately the same non-blocking, std::Reactor-pairable shape as
// io_nb.h's tcp_*_nb functions — a task calls one of these, gets
// EAGAIN/EWOULDBLOCK immediately instead of blocking the whole program,
// and awaits readiness via TaskScheduler::await_fd(fd, SCHED_READ/
// SCHED_WRITE) instead.
//
// Windows has no AF_UNIX-domain-socket equivalent in the traditional
// BSD-sockets sense (Windows 10 1803+ does support AF_UNIX, but not
// universally across the target versions this project otherwise
// supports), so there's no uds_win32.sc — named pipes (a different API
// shape entirely: CreateNamedPipe/ConnectNamedPipe, not socket/bind/
// listen/accept/connect) are Windows' native equivalent for this use
// case, and would need their own, differently-shaped module rather than
// slotting into this one's function signatures.
//
// Backends:
//   uds_bsd.sc    — macOS, iOS, FreeBSD (struct sockaddr_un has a leading
//                   1-byte sun_len field, matching sockaddr_in's BSD
//                   quirk in std/sched/io_nb_bsd.sc)
//   uds_linux.sc  — Linux, Android (no sun_len; plain 2-byte sun_family)
#pragma once

namespace std {

// socket()+bind()+listen() on 'path', non-blocking. Returns the listening
// fd, or -1 on failure. Fails with EADDRINUSE if 'path' already exists as
// a socket file from a previous run that didn't clean up — call
// uds_unlink(path) first if that's a possibility.
int uds_listen_nb(const char* path);

// Non-blocking accept(): returns a connected client fd (itself set
// non-blocking), or -1 with errno EAGAIN/EWOULDBLOCK if none is pending
// yet — the caller should await_fd(listenfd, SCHED_READ) and retry.
int uds_accept_nb(int listenfd);

// socket()+connect() to 'path', non-blocking: connect() on a non-blocking
// socket returns immediately with EINPROGRESS rather than waiting for the
// handshake — the caller should await_fd(fd, SCHED_WRITE) and treat the
// fd becoming writable as "connect finished". Returns the socket fd, or
// -1 on immediate failure.
int uds_connect_nb(const char* path);

// Removes a stale socket file at 'path' (unlink(2)) — a Unix domain
// socket's bind() fails EADDRINUSE if the filesystem path from a
// previous, uncleanly-terminated run still exists, unlike a TCP port
// which is simply released by the OS. Returns 0 on success (including
// "path didn't exist"), -1 on a real failure.
int uds_unlink(const char* path);

} // namespace std
