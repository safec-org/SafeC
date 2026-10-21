// SafeC Standard Library — Non-blocking file/socket helpers: Linux backend
// (Linux / Android — see io_nb.h for the portable API and the full
// backend file list)
#pragma once
#include <std/sched/io_nb.h>
#include <std/errno.h>
#include <std/time.h>

#define SCHED_O_CREAT     0x40
#define SCHED_O_NONBLOCK  0x800
#define SCHED_F_GETFL     3
#define SCHED_F_SETFL     4

namespace std {

// Linux 'struct sockaddr_in' (16 bytes): sin_family(2) sin_port(2)
// sin_addr(4) sin_zero(8) — unlike BSD/macOS, there's no leading sin_len
// byte; sin_family is a plain 2-byte field, not split into a 1-byte
// length + 1-byte family pair.
struct SockAddrIn {
    unsigned short sin_family;
    unsigned short sin_port;   // network byte order — see sched_htons
    unsigned int   sin_addr;   // network byte order — see sched_htonl
    unsigned long  sin_zero;   // 8 bytes, always zero
};

extern int open(const char* path, int flags, int mode);
extern int socket(int domain, int type, int protocol);
extern int bind(int fd, const void* addr, unsigned int addrlen);
extern int listen(int fd, int backlog);
extern int accept(int fd, void* addr, void* addrlen);
extern int connect(int fd, const void* addr, unsigned int addrlen);
// Genuinely variadic in the real libc — see io_nb_bsd.sc's identical
// declaration for why this matters (a real ARM64/AAPCS64 ABI mismatch
// there; x86_64 happens to tolerate the fixed-arg form, but that's an
// accident of this specific ABI, not something to depend on).
extern int fcntl(int fd, int cmd, ...);
extern int setsockopt(int fd, int level, int optname, const void* optval, unsigned int optlen);

#define SCHED_SOL_SOCKET    1
#define SCHED_SO_REUSEADDR  2
#define SCHED_IPPROTO_TCP   6
#define SCHED_TCP_NODELAY   1
#define SCHED_SO_ERROR      4
#define SCHED_SO_RCVTIMEO   20
#define SCHED_SO_SNDTIMEO   21
#define SCHED_POLLOUT       0x004
#define SCHED_EINTR         4
#define SCHED_EINPROGRESS   115

extern int poll(void* fds, unsigned long nfds, int timeout);
extern int getsockopt(int fd, int level, int optname, void* optval, void* optlen);
extern int close(int fd);

struct SchedPollFd {
    int   fd;
    short events;
    short revents;
};

// 'struct timeval' for SO_RCVTIMEO/SO_SNDTIMEO: two longs on every 64-bit
// Linux ABI.
struct SchedTimeval {
    long tv_sec;
    long tv_usec;
};

inline unsigned short sched_htons(unsigned short host16) {
    return (unsigned short)(((host16 & (unsigned short)0xFF) << 8) |
                             ((host16 >> 8) & (unsigned short)0xFF));
}

inline unsigned int sched_htonl(unsigned int host32) {
    return ((host32 & 0xFFU) << 24) | ((host32 & 0xFF00U) << 8) |
           ((host32 >> 8) & 0xFF00U) | ((host32 >> 24) & 0xFFU);
}

inline unsigned int sched_ipv4(unsigned char a, unsigned char b,
                                unsigned char c, unsigned char d) {
    unsigned int host = ((unsigned int)a << 24) | ((unsigned int)b << 16) |
                         ((unsigned int)c << 8)  |  (unsigned int)d;
    return sched_htonl(host);
}

inline int fd_set_nonblocking(int fd) {
    int flags;
    unsafe { flags = fcntl(fd, SCHED_F_GETFL, 0); }
    if (flags < 0) {
        return -1;
    }
    int rc;
    unsafe { rc = fcntl(fd, SCHED_F_SETFL, flags | SCHED_O_NONBLOCK); }
    return rc;
}

inline int fd_set_blocking(int fd) {
    int flags;
    unsafe { flags = fcntl(fd, SCHED_F_GETFL, 0); }
    if (flags < 0) {
        return -1;
    }
    int rc;
    unsafe { rc = fcntl(fd, SCHED_F_SETFL, flags & ~SCHED_O_NONBLOCK); }
    return rc;
}

inline int sock_would_block() {
    int e = errno_get();
    return (e == ERRNO_EAGAIN()) || (e == ERRNO_EWOULDBLOCK());
}

inline int fd_open_nb(const &char path, int flags, int mode) {
    int fd;
    unsafe { fd = open(path, flags | SCHED_O_NONBLOCK, mode); }
    return fd;
}

inline int tcp_listen_nb(unsigned short port) {
    int fd;
    unsafe { fd = socket(SCHED_AF_INET, SCHED_SOCK_STREAM, 0); }
    if (fd < 0) {
        return -1;
    }
    if (fd_set_nonblocking(fd) != 0) {
        return -1;
    }
    // See io_nb_bsd.sc's tcp_listen_nb for why: without this, rebinding
    // the port fails for as long as a just-served connection's socket
    // sits in TIME_WAIT on it -- the normal case after any real traffic,
    // not an edge case.
    unsafe {
        int reuseOpt = 1;
        setsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_REUSEADDR,
                   (const void*)&reuseOpt, 4U);
    }

    struct SockAddrIn addr;
    addr.sin_family = (unsigned short)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = 0U; // INADDR_ANY
    addr.sin_zero   = 0UL;

    int rc;
    unsafe { rc = bind(fd, (const void*)&addr, 16U); }
    if (rc != 0) {
        return -1;
    }
    // See io_nb_bsd.sc's tcp_listen_nb for why 512, not a small literal
    // like 16: verified that a small backlog causes real TCP
    // SYN-retransmit tail latency once concurrent connection bursts
    // exceed it, not just a theoretical concern.
    unsafe { rc = listen(fd, 512); }
    if (rc != 0) {
        return -1;
    }
    return fd;
}

inline int tcp_accept_nb(int listenfd) {
    int fd;
    unsafe { fd = accept(listenfd, (void*)0, (void*)0); }
    if (fd < 0) {
        return -1;
    }
    if (fd_set_nonblocking(fd) != 0) {
        return -1;
    }
    // See io_nb_bsd.sc's tcp_accept_nb for why: request/response protocols
    // like HTTP get nothing from Nagle's write-batching and only pay its
    // latency against the peer's delayed-ACK timer.
    unsafe {
        int nodelayOpt = 1;
        setsockopt(fd, SCHED_IPPROTO_TCP, SCHED_TCP_NODELAY,
                   (const void*)&nodelayOpt, 4U);
    }
    return fd;
}

// socket()+non-blocking connect() shared by tcp_connect_nb and
// tcp_connect_timeout. Returns the fd (or -1, with nothing left open) and
// stores 0 in 'err_out' if connect() succeeded immediately, otherwise
// connect()'s errno (normally EINPROGRESS).
static int sched_connect_begin_(unsigned int addr_network_order, unsigned short port,
                                &int err_out) {
    int fd;
    unsafe { fd = socket(SCHED_AF_INET, SCHED_SOCK_STREAM, 0); }
    if (fd < 0) {
        return -1;
    }
    if (fd_set_nonblocking(fd) != 0) {
        unsafe { close(fd); }
        return -1;
    }

    struct SockAddrIn addr;
    addr.sin_family = (unsigned short)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = addr_network_order;
    addr.sin_zero   = 0UL;

    int rc;
    unsafe { rc = connect(fd, (const void*)&addr, 16U); }
    int e = (rc == 0) ? 0 : errno_get();
    unsafe { *err_out = e; }
    return fd;
}

inline int tcp_connect_nb(unsigned int addr_network_order, unsigned short port) {
    // Non-blocking connect(): a non-zero return here almost always just
    // means EINPROGRESS (the handshake hasn't finished yet), not failure —
    // the caller awaits SCHED_WRITE on 'fd' and treats writability as
    // "connect attempt finished" per the header comment, so the return
    // value here isn't itself meaningful the way it is for a blocking
    // connect().
    int err = 0;
    return sched_connect_begin_(addr_network_order, port, &err);
}

inline int tcp_connect_timeout(unsigned int addr_network_order, unsigned short port,
                               int timeout_ms) {
    int err = 0;
    int fd = sched_connect_begin_(addr_network_order, port, &err);
    if (fd < 0) {
        return -1;
    }
    if (err == SCHED_EINPROGRESS) {
        long long deadline = 0LL;
        if (timeout_ms > 0) {
            deadline = time_mono_ns() + (long long)timeout_ms * 1000000LL;
        }
        while (1) {
            int wait_ms = -1;
            if (timeout_ms > 0) {
                long long left = deadline - time_mono_ns();
                wait_ms = (left <= 0LL) ? 0 : (int)((left + 999999LL) / 1000000LL);
            }
            struct SchedPollFd pfd;
            pfd.fd = fd;
            pfd.events = (short)SCHED_POLLOUT;
            pfd.revents = (short)0;
            int n;
            unsafe { n = poll((void*)&pfd, 1UL, wait_ms); }
            if (n > 0) {
                int soerr = 0;
                unsigned int len = 4U;
                int grc;
                unsafe { grc = getsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_ERROR,
                                          (void*)&soerr, (void*)&len); }
                err = (grc != 0) ? errno_get() : soerr;
                break;
            }
            if (n == 0) {
                err = ERRNO_ETIMEDOUT();
                break;
            }
            int perr = errno_get();
            if (perr != SCHED_EINTR) {
                err = perr;
                break;
            }
        }
    }
    if (err == 0 && fd_set_blocking(fd) != 0) {
        err = errno_get();
    }
    if (err != 0) {
        unsafe { close(fd); }
        errno_set(err);
        return -1;
    }
    return fd;
}

inline int sock_set_io_timeout(int fd, int timeout_ms) {
    struct SchedTimeval tv;
    int ms = (timeout_ms > 0) ? timeout_ms : 0;
    tv.tv_sec  = (long)(ms / 1000);
    tv.tv_usec = (long)((ms % 1000) * 1000);
    int rc1;
    int rc2;
    unsafe {
        rc1 = setsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_RCVTIMEO, (const void*)&tv, 16U);
        rc2 = setsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_SNDTIMEO, (const void*)&tv, 16U);
    }
    return (rc1 == 0 && rc2 == 0) ? 0 : -1;
}

} // namespace std
