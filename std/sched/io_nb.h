// SafeC Standard Library — Non-blocking file/socket helpers for the reactor
//
// Thin wrappers around real OS syscalls (open/socket/accept/connect) that
// set O_NONBLOCK, meant to be paired with std::Reactor/std::TaskScheduler
// (see reactor.h): a task calls one of these, gets EAGAIN/EWOULDBLOCK back
// immediately instead of blocking the whole program, and awaits readiness
// via TaskScheduler::await_fd(fd, SCHED_READ/SCHED_WRITE) instead.
//
// macOS/BSD-only constant values below (O_NONBLOCK, sockaddr_in layout) —
// matches the kqueue backend this module is meant to pair with
// (reactor_kqueue.sc). A Linux build needs the equivalent Linux constants
// (O_NONBLOCK is 0x800 there, not 0x0004; sockaddr_in has no sin_len byte)
// wired up alongside a future epoll reactor backend.
#pragma once
#include <std/sched/reactor.h>

#define SCHED_AF_INET     2
#define SCHED_SOCK_STREAM 1

#define SCHED_O_RDONLY    0
#define SCHED_O_WRONLY    1
#define SCHED_O_RDWR      2
#define SCHED_O_CREAT     0x0200  // macOS value
#define SCHED_O_NONBLOCK  0x0004  // macOS value

namespace std {

// BSD/macOS 'struct sockaddr_in' (16 bytes): sin_len(1) sin_family(1)
// sin_port(2) sin_addr(4) sin_zero(8) — note macOS has the extra 1-byte
// sin_len field Linux's sockaddr_in doesn't, ahead of sin_family (which is
// also only 1 byte here, not 2 as on Linux).
struct SockAddrIn {
    unsigned char  sin_len;
    unsigned char  sin_family;
    unsigned short sin_port;   // network byte order — see sched_htons
    unsigned int   sin_addr;   // network byte order — see sched_htonl
    unsigned long  sin_zero;   // 8 bytes, always zero
};

unsigned short sched_htons(unsigned short host16);
unsigned int   sched_htonl(unsigned int host32);
// Packs four host-order octets (e.g. 127,0,0,1) into a network-order
// address suitable for SockAddrIn::sin_addr — avoids callers needing to
// hand-compute the byte order themselves for the common "literal IP"
// case.
unsigned int   sched_ipv4(unsigned char a, unsigned char b,
                           unsigned char c, unsigned char d);

// Sets O_NONBLOCK on an already-open fd. Returns 0 on success.
int fd_set_nonblocking(int fd);

// open() + O_NONBLOCK. Returns fd, or -1 on failure.
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
