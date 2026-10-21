// SafeC Standard Library — Non-blocking file/socket helpers: BSD backend
// (macOS / iOS / FreeBSD — see io_nb.h for the portable API and the full
// backend file list)
#pragma once
#include <std/sched/io_nb.h>
#include <std/errno.h>
#include <std/time.h>

#define SCHED_O_CREAT     0x0200
#define SCHED_O_NONBLOCK  0x0004
#define SCHED_F_GETFL     3
#define SCHED_F_SETFL     4

namespace std {

// BSD/macOS 'struct sockaddr_in' (16 bytes): sin_len(1) sin_family(1)
// sin_port(2) sin_addr(4) sin_zero(8) — note macOS has the extra 1-byte
// sin_len field Linux's/Windows' sockaddr_in doesn't, ahead of sin_family
// (which is also only 1 byte here, not 2 as on those platforms).
struct SockAddrIn {
    unsigned char  sin_len;
    unsigned char  sin_family;
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
// Real libc fcntl is 'int fcntl(int fd, int cmd, ...)' — genuinely
// variadic, not (fd, cmd, arg). Declaring it with a fixed third 'int arg'
// parameter instead of '...' is a real ABI mismatch on arm64/AAPCS64
// (Apple Silicon passes variadic-position arguments differently than
// fixed ones): verified empirically that fd_set_nonblocking's
// fcntl(fd, F_SETFL, flags|O_NONBLOCK) call silently failed to actually
// set O_NONBLOCK with the fixed-arg declaration — a freshly-listened
// socket read back as still blocking, so accept() on it blocked forever
// instead of returning EAGAIN, hanging the reactor's acceptor task. Read
// back with the correct variadic declaration below, the exact same call
// works. x86_64 (Linux, and this same declaration in io_nb_linux.sc)
// happens not to hit this — its calling convention doesn't special-case
// variadic-position arguments the way AAPCS64 does — but this was never
// something to rely on either way.
extern int fcntl(int fd, int cmd, ...);
extern int setsockopt(int fd, int level, int optname, const void* optval, unsigned int optlen);

#define SCHED_SOL_SOCKET    0xffff
#define SCHED_SO_REUSEADDR  0x0004
// A write to a socket whose peer has closed raises SIGPIPE, which kills the
// process by default; BSD sockets opt out per socket (Linux does it per
// send with MSG_NOSIGNAL). Accepted sockets don't reliably inherit it.
#if defined(__APPLE__)
#define SCHED_SO_NOSIGPIPE  0x1022
#else
#define SCHED_SO_NOSIGPIPE  0x0800   // FreeBSD
#endif

static void sched_no_sigpipe_(int fd) {
    unsafe {
        int on = 1;
        setsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_NOSIGPIPE, (const void*)&on, 4U);
    }
}
#define SCHED_IPPROTO_TCP   6
#define SCHED_TCP_NODELAY   1
#define SCHED_SO_ERROR      0x1007
#define SCHED_SO_RCVTIMEO   0x1006
#define SCHED_SO_SNDTIMEO   0x1005
#define SCHED_POLLOUT       0x004
#define SCHED_EINTR         4
#define SCHED_EINPROGRESS   36

extern int poll(void* fds, unsigned int nfds, int timeout);
extern int getsockopt(int fd, int level, int optname, void* optval, void* optlen);
extern int close(int fd);

struct SchedPollFd {
    int   fd;
    short events;
    short revents;
};

// 'struct timeval' for SO_RCVTIMEO/SO_SNDTIMEO (16 bytes). FreeBSD's
// tv_usec is a long; macOS's is a 4-byte int followed by 4 bytes of
// padding, which on these little-endian targets has the same memory image
// as a long holding a value below 1e6.
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
    // Without SO_REUSEADDR, rebinding this exact port fails (EADDRINUSE)
    // for as long as any connection this process just served still has a
    // socket in TIME_WAIT on this port -- which after any real traffic is
    // the normal case, not an edge case. Verified: kill a server that just
    // handled a real load test, then immediately try to start a new one on
    // the same port -- it silently fails to bind (the old code didn't
    // check tcp_listen_nb's return value against this failure mode either,
    // so the process would print its own "listening" banner and then just
    // exit). Setting this is what lets "stop the server, start it again"
    // actually work like every other language's server here does.
    unsafe {
        int reuseOpt = 1;
        setsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_REUSEADDR,
                   (const void*)&reuseOpt, 4U);
    }

    struct SockAddrIn addr;
    addr.sin_len    = (unsigned char)16;
    addr.sin_family = (unsigned char)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = 0U; // INADDR_ANY
    addr.sin_zero   = 0UL;

    int rc;
    unsafe { rc = bind(fd, (const void*)&addr, 16U); }
    if (rc != 0) {
        return -1;
    }
    // 512, not the socket API's traditional tiny default (historically
    // 5, sometimes seen as low as 16 in example code) -- verified this
    // actually matters: at 16, a burst of ~50 concurrent client
    // connections (a completely ordinary load-test concurrency, see
    // safec-docs's Benchmarks page) overflows the kernel's pending-accept
    // queue, and the connections that don't fit hit real TCP SYN-retransmit
    // delays (observed: 500ms+ tail latency on connect(), not on request
    // processing) waiting for a slot to free up -- not a "the OS is slow"
    // effect, a "we told it to only queue 16" one.
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
    // Nagle's algorithm batches small writes to wait for either a full
    // segment or the peer's ACK before sending — request/response
    // protocols like HTTP, where each side writes once and waits, get no
    // benefit from that batching and instead just pay its latency
    // (interacting with the peer's delayed-ACK timer, tens of ms per
    // hop). Disabling it on the accepted socket is the standard fix for
    // any server whose protocol already frames its own messages.
    unsafe {
        int nodelayOpt = 1;
        setsockopt(fd, SCHED_IPPROTO_TCP, SCHED_TCP_NODELAY,
                   (const void*)&nodelayOpt, 4U);
    }
    sched_no_sigpipe_(fd);
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
    sched_no_sigpipe_(fd);
    if (fd_set_nonblocking(fd) != 0) {
        unsafe { close(fd); }
        return -1;
    }

    struct SockAddrIn addr;
    addr.sin_len    = (unsigned char)16;
    addr.sin_family = (unsigned char)SCHED_AF_INET;
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

// Waits for an in-flight non-blocking connect on 'fd' to finish, for at
// most 'timeout_ms' (<= 0: no limit). Returns 0 if it connected,
// otherwise the error (ETIMEDOUT on timeout, else SO_ERROR's value).
static int sched_wait_connected_(int fd, int timeout_ms) {
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
        unsafe { n = poll((void*)&pfd, 1U, wait_ms); }
        if (n > 0) {
            int soerr = 0;
            unsigned int len = 4U;
            int grc;
            unsafe { grc = getsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_ERROR,
                                      (void*)&soerr, (void*)&len); }
            return (grc != 0) ? errno_get() : soerr;
        }
        if (n == 0) {
            return ERRNO_ETIMEDOUT();
        }
        int perr = errno_get();
        if (perr != SCHED_EINTR) {
            return perr;
        }
    }
    return 0;
}

inline int tcp_wait_connected(int fd) {
    int err = sched_wait_connected_(fd, 0);
    if (err != 0) {
        errno_set(err);
        return -1;
    }
    return 0;
}

inline int tcp_connect_timeout(unsigned int addr_network_order, unsigned short port,
                               int timeout_ms) {
    int err = 0;
    int fd = sched_connect_begin_(addr_network_order, port, &err);
    if (fd < 0) {
        return -1;
    }
    if (err == SCHED_EINPROGRESS) {
        err = sched_wait_connected_(fd, timeout_ms);
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
