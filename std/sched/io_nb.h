// SafeC Standard Library — Non-blocking file/socket helpers for the reactor
//
// Thin wrappers around real OS syscalls (open/socket/accept/connect) that
// enable non-blocking mode, meant to be paired with std::Reactor/
// std::TaskScheduler (see reactor.h): a task calls one of these, gets
// EAGAIN/EWOULDBLOCK back immediately instead of blocking the whole
// program, and awaits readiness via
// TaskScheduler::await_fd(fd, SCHED_READ/SCHED_WRITE) instead.
//
// This header declares only the genuinely portable pieces (AF_INET/
// SOCK_STREAM/O_RDONLY/O_WRONLY/O_RDWR share the same values across every
// backend below — a historical BSD-sockets accident every later API,
// including Winsock, deliberately preserved for source compatibility) and
// the function signatures every backend implements identically. Anything
// that actually differs by platform — sockaddr_in's layout (BSD has an
// extra sin_len byte Linux/Windows don't), O_NONBLOCK's value (or lack of
// a direct equivalent at all, on Windows), the non-blocking-mode syscall
// itself (fcntl vs. ioctlsocket) — lives in the matching backend file
// instead, picked the same way as reactor.h's own backends:
//   io_nb_bsd.sc    — macOS, iOS, FreeBSD
//   io_nb_linux.sc  — Linux, Android
//   io_nb_win32.sc  — Windows (fd_open_nb is best-effort there — see its
//                     own comment; Windows has no non-blocking-file-open
//                     equivalent to O_NONBLOCK, only non-blocking sockets)
#pragma once
#include <std/sched/reactor.h>

#define SCHED_AF_INET     2
#define SCHED_SOCK_STREAM 1

#define SCHED_O_RDONLY    0
#define SCHED_O_WRONLY    1
#define SCHED_O_RDWR      2

namespace std {

unsigned short sched_htons(unsigned short host16);
unsigned int   sched_htonl(unsigned int host32);
// Packs four host-order octets (e.g. 127,0,0,1) into a network-order
// address suitable for a backend's sockaddr_in sin_addr field — avoids
// callers needing to hand-compute the byte order themselves for the
// common "literal IP" case.
unsigned int   sched_ipv4(unsigned char a, unsigned char b,
                           unsigned char c, unsigned char d);

// Puts an already-open socket fd into non-blocking mode (fcntl+O_NONBLOCK
// on BSD/Linux, ioctlsocket+FIONBIO on Windows). Returns 0 on success.
int fd_set_nonblocking(int fd);

// open() in non-blocking mode where the platform supports it for regular
// files (BSD/Linux: O_NONBLOCK; Windows: no equivalent — see
// io_nb_win32.sc). Returns fd, or -1 on failure.
int fd_open_nb(const char* path, int flags, int mode);

// socket()+bind()+listen(), non-blocking. Returns listening fd, or -1.
int tcp_listen_nb(unsigned short port);

// Non-blocking accept(): returns a connected client fd (itself set
// non-blocking), or -1 with errno EAGAIN/EWOULDBLOCK if none is pending
// yet — the caller should await_fd(listenfd, SCHED_READ) and retry.
int tcp_accept_nb(int listenfd);

// socket()+connect(), non-blocking: connect() on a non-blocking socket
// returns immediately with EINPROGRESS rather than waiting for the
// handshake — the caller should await_fd(fd, SCHED_WRITE) and treat the
// fd becoming writable as "connect finished (successfully or not; check
// SO_ERROR if you need to distinguish)". Returns the socket fd, or -1 on
// immediate failure.
int tcp_connect_nb(unsigned int addr_network_order, unsigned short port);

} // namespace std
