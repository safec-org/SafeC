// SafeC Standard Library — Non-blocking file/socket helpers: Win32 backend
// (see io_nb.h for the portable API and the full backend file list)
//
// Sockets and files are different worlds on Windows, more so than on
// POSIX where both share one fd namespace and one non-blocking-mode
// mechanism (fcntl+O_NONBLOCK):
//
//   - Sockets: real non-blocking support, via ioctlsocket()+FIONBIO
//     instead of fcntl(). socket()/bind()/listen()/accept()/connect() are
//     the same function *names* as BSD sockets (Winsock deliberately
//     mirrors the BSD API for source compatibility) but live in ws2_32.dll
//     and require WSAStartup() before any of them work — tcp_listen_nb/
//     tcp_connect_nb call it defensively (WSAStartup is safe to call more
//     than once; std::Reactor::init() also calls it for the reactor's own
//     WSAPoll use, so this is redundant but harmless if both run).
//
//   - Files: fd_open_nb here is *best-effort only* — it calls the CRT's
//     _open() (which does accept O_RDONLY/O_WRONLY/O_RDWR-compatible flag
//     values) but Windows has no non-blocking-file-open equivalent to
//     O_NONBLOCK at all; true asynchronous file I/O on Windows means
//     CreateFile with FILE_FLAG_OVERLAPPED plus a completion mechanism
//     entirely different from this reactor's readiness-based model, which
//     is out of scope here. A fd_open_nb'd file handle on this backend is
//     a perfectly normal (blocking) file descriptor — don't
//     await_fd()/register it with the reactor expecting non-blocking
//     semantics the way a socket from tcp_listen_nb/tcp_connect_nb has.
// safeguard: link windows -lws2_32
#pragma once
#include <std/sched/io_nb.h>
#include <std/time.h>

#define SCHED_O_CREAT  0x0100
#define SAFEC_FIONBIO  0x8004667EU
#define SAFEC_WSA_VERSION 0x0202  // MAKEWORD(2, 2) — request Winsock 2.2
// Winsock deliberately mirrors the original 4.3BSD values for these —
// verified empirically against a real <winsock2.h>/<ws2tcpip.h>, not
// assumed: SOL_SOCKET=0xffff and SO_REUSEADDR=0x0004 match io_nb_bsd.sc's
// values exactly; IPPROTO_TCP=6/TCP_NODELAY=1 match every backend here.
#define SCHED_SOL_SOCKET    0xffff
#define SCHED_SO_REUSEADDR  0x0004
#define SCHED_IPPROTO_TCP   6
#define SCHED_TCP_NODELAY   1
#define SCHED_SO_ERROR      0x1007
#define SCHED_SO_RCVTIMEO   0x1006
#define SCHED_SO_SNDTIMEO   0x1005

namespace std {

// Windows 'struct sockaddr_in' (16 bytes): sin_family(2) sin_port(2)
// sin_addr(4) sin_zero(8) — same shape as Linux's (no leading sin_len
// byte the way BSD/macOS has).
struct SockAddrIn {
    unsigned short sin_family;
    unsigned short sin_port;   // network byte order — see sched_htons
    unsigned int   sin_addr;   // network byte order — see sched_htonl
    unsigned long  sin_zero;   // 8 bytes, always zero
};

// WSADATA, which WSAStartup fills in — see reactor_win32.sc's identical
// definition for why it has to be the real, full-size layout.
// reactor_win32.sc declares this same struct (same purpose, same shape) —
// guarded so a caller that includes both this file
// and reactor_win32.sc (e.g. std::http_serve_reactor, which needs both)
// doesn't hit a duplicate-definition error. #pragma once only guards a
// single file against itself, not two different files defining the same
// struct name.
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

extern int WSAStartup(unsigned short versionRequested, WSAData_* wsaData);
extern int _open(const char* path, int flags, int mode); // MSVCRT — plain (blocking) file open
extern unsigned long long socket(int domain, int type, int protocol);
extern int bind(unsigned long long fd, const SockAddrIn* addr, int addrlen);
extern int listen(unsigned long long fd, int backlog);
extern unsigned long long accept(unsigned long long fd, SockAddrIn* addr, int* addrlen);
extern int connect(unsigned long long fd, const SockAddrIn* addr, int addrlen);
extern int ioctlsocket(unsigned long long fd, int cmd, unsigned int* argp);   // u_long: 32 bits on Windows
extern int setsockopt(unsigned long long fd, int level, int optname, const unsigned char* optval, int optlen);
extern int WSAGetLastError();
extern void WSASetLastError(int err);
extern int getsockopt(unsigned long long fd, int level, int optname, unsigned char* optval, int* optlen);
extern int closesocket(unsigned long long fd);

#define SAFEC_WSAEWOULDBLOCK 10035
#define SAFEC_WSAETIMEDOUT   10060

// Winsock 'fd_set' holding a single socket: u_int fd_count, then the
// SOCKET array (8-byte aligned, so fd_array starts at offset 8 exactly as
// in the real 64-entry struct; select() only reads fd_count entries).
struct SchedFdSet1 {
    unsigned int       fd_count;
    unsigned long long fd_array[1];
};

// Winsock 'struct timeval': two 4-byte longs (Windows 'long' is 32-bit).
struct SchedWinTimeval {
    int tv_sec;
    int tv_usec;
};
extern int select(int nfds, SchedFdSet1* readfds, SchedFdSet1* writefds, SchedFdSet1* exceptfds, const SchedWinTimeval* timeout);

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

static void io_nb_wsa_init_() {
    struct WSAData_ wsaData;
    unsafe { WSAStartup((unsigned short)SAFEC_WSA_VERSION, &wsaData); }
}

inline int sock_would_block() {
    int e;
    unsafe { e = WSAGetLastError(); }
    return e == SAFEC_WSAEWOULDBLOCK;
}

inline int fd_set_nonblocking(int fd) {
    unsigned int mode = 1u; // non-zero = enable non-blocking mode
    int rc;
    unsafe { rc = ioctlsocket((unsigned long long)fd, (int)SAFEC_FIONBIO, &mode); }
    return rc;
}

inline int fd_set_blocking(int fd) {
    unsigned int mode = 0u; // zero = disable non-blocking mode
    int rc;
    unsafe { rc = ioctlsocket((unsigned long long)fd, (int)SAFEC_FIONBIO, &mode); }
    return rc;
}

// See file header comment — this is a plain (blocking) file open on this
// backend; the O_NONBLOCK-style flag argument callers may pass has no
// effect here.
inline int fd_open_nb(const &char path, int flags, int mode) {
    int fd;
    unsafe { fd = _open(path, flags, mode); }
    return fd;
}

inline int tcp_listen_nb(unsigned short port) {
    io_nb_wsa_init_();
    unsigned long long fd;
    unsafe { fd = socket(SCHED_AF_INET, SCHED_SOCK_STREAM, 0); }
    // socket() failure returns INVALID_SOCKET (~0, every bit set) — must
    // check this *before* truncating to int and calling fd_set_nonblocking
    // on it. Real bug found via this exact gap in tcp_accept_nb below:
    // calling any further Winsock function on an invalid handle fails
    // with its own error (WSAENOTSOCK) that overwrites whatever
    // WSAGetLastError() the real failure had set, so a caller checking
    // sock_would_block() afterward sees the wrong code entirely.
    if (fd == 0xFFFFFFFFFFFFFFFFULL) {
        return -1;
    }
    int ifd = (int)fd;
    if (fd_set_nonblocking(ifd) != 0) {
        return -1;
    }
    // See io_nb_bsd.sc's tcp_listen_nb for why: without this, rebinding
    // the port fails for as long as a just-served connection's socket
    // sits in TIME_WAIT on it. Missing here before (unlike the BSD/Linux
    // backends) — real gap, found while restarting this server rapidly
    // during benchmarking (see benchmarks.md's web service section).
    unsafe {
        int reuseOpt = 1;
        setsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_REUSEADDR,
                   (const unsigned char*)&reuseOpt, 4);
    }

    struct SockAddrIn addr;
    addr.sin_family = (unsigned short)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = 0U; // INADDR_ANY
    addr.sin_zero   = 0UL;

    int rc;
    unsafe { rc = bind(fd, &addr, 16); }
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
    return ifd;
}

inline int tcp_accept_nb(int listenfd) {
    unsigned long long fd;
    unsafe { fd = accept((unsigned long long)listenfd, (SockAddrIn*)0, (int*)0); }
    // See tcp_listen_nb's identical check for why this must come before
    // touching 'fd' any further — this is the specific call site where
    // the missing check was actually found: an empty accept queue makes
    // accept() correctly fail with WSAEWOULDBLOCK, but without this
    // early return, the very next line (fd_set_nonblocking on the
    // resulting INVALID_SOCKET) fails too and clobbers that error to
    // WSAENOTSOCK before std::sock_would_block() ever gets to check it —
    // observed as std::http_serve_reactor's Windows acceptor task
    // silently taking the "real error" branch on every single
    // connection attempt and hanging forever instead of registering
    // with the reactor and waiting correctly.
    if (fd == 0xFFFFFFFFFFFFFFFFULL) {
        return -1;
    }
    int ifd = (int)fd;
    if (fd_set_nonblocking(ifd) != 0) {
        return -1;
    }
    // See io_nb_bsd.sc's tcp_accept_nb for why: request/response protocols
    // like HTTP get nothing from Nagle's write-batching and only pay its
    // latency against the peer's delayed-ACK timer.
    unsafe {
        int nodelayOpt = 1;
        setsockopt(fd, SCHED_IPPROTO_TCP, SCHED_TCP_NODELAY,
                   (const unsigned char*)&nodelayOpt, 4);
    }
    return ifd;
}

// socket()+non-blocking connect() shared by tcp_connect_nb and
// tcp_connect_timeout. Returns the socket (or -1, with nothing left open)
// and stores 0 in 'err_out' if connect() succeeded immediately, otherwise
// WSAGetLastError() (normally WSAEWOULDBLOCK).
static int sched_connect_begin_(unsigned int addr_network_order, unsigned short port,
                                &int err_out) {
    io_nb_wsa_init_();
    unsigned long long fd;
    unsafe { fd = socket(SCHED_AF_INET, SCHED_SOCK_STREAM, 0); }
    // See tcp_listen_nb's identical check.
    if (fd == 0xFFFFFFFFFFFFFFFFULL) {
        return -1;
    }
    int ifd = (int)fd;
    if (fd_set_nonblocking(ifd) != 0) {
        unsafe { closesocket(fd); }
        return -1;
    }

    struct SockAddrIn addr;
    addr.sin_family = (unsigned short)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = addr_network_order;
    addr.sin_zero   = 0UL;

    int rc;
    unsafe { rc = connect(fd, &addr, 16); }
    int e = 0;
    if (rc != 0) {
        unsafe { e = WSAGetLastError(); }
    }
    unsafe { *err_out = e; }
    return ifd;
}

inline int tcp_connect_nb(unsigned int addr_network_order, unsigned short port) {
    // Non-blocking connect(): a non-zero return here almost always just
    // means WSAEWOULDBLOCK (the handshake hasn't finished yet), not
    // failure — the caller awaits SCHED_WRITE on 'fd' and treats
    // writability as "connect attempt finished" per the header comment,
    // so the return value here isn't itself meaningful the way it is for
    // a blocking connect().
    int err = 0;
    return sched_connect_begin_(addr_network_order, port, &err);
}

// Waits for an in-flight non-blocking connect on 'fd' to finish, for at
// most 'timeout_ms' (<= 0: no limit). Returns 0 if it connected,
// otherwise the Winsock error (WSAETIMEDOUT on timeout). select(), not
// WSAPoll: WSAPoll never reports a failed non-blocking connect on Windows
// releases before 10 2004 (with no timeout it would wait forever), while
// select() reports it through exceptfds.
static int sched_wait_connected_(unsigned long long fd, int timeout_ms) {
    struct SchedFdSet1 wset;
    struct SchedFdSet1 eset;
    wset.fd_count = 1U; wset.fd_array[0] = fd;
    eset.fd_count = 1U; eset.fd_array[0] = fd;
    struct SchedWinTimeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int n;
    unsafe {
        n = select(0, (SchedFdSet1*)0, &wset, &eset,
                   (timeout_ms > 0) ? (const SchedWinTimeval*)&tv : (const SchedWinTimeval*)0);
    }
    if (n == 0) {
        return SAFEC_WSAETIMEDOUT;
    }
    int err = 0;
    if (n < 0) {
        unsafe { err = WSAGetLastError(); }
        return err;
    }
    int soerr = 0;
    int len = 4;
    int grc;
    unsafe { grc = getsockopt(fd, SCHED_SOL_SOCKET, SCHED_SO_ERROR,
                              (unsigned char*)&soerr, &len); }
    if (grc != 0) {
        unsafe { err = WSAGetLastError(); }
    } else {
        err = soerr;
    }
    // A failed connect is reported only through exceptfds; make sure it
    // never reads as success even if SO_ERROR came back 0.
    if (err == 0 && eset.fd_count != 0U) {
        err = SAFEC_WSAETIMEDOUT;
    }
    return err;
}

inline int tcp_wait_connected(int fd) {
    int err = sched_wait_connected_((unsigned long long)(unsigned int)fd, 0);
    if (err != 0) {
        unsafe { WSASetLastError(err); }
        return -1;
    }
    return 0;
}

inline int tcp_connect_timeout(unsigned int addr_network_order, unsigned short port,
                               int timeout_ms) {
    int err = 0;
    int ifd = sched_connect_begin_(addr_network_order, port, &err);
    if (ifd < 0) {
        return -1;
    }
    unsigned long long fd = (unsigned long long)(unsigned int)ifd;
    if (err == SAFEC_WSAEWOULDBLOCK) {
        err = sched_wait_connected_(fd, timeout_ms);
    }
    if (err == 0 && fd_set_blocking(ifd) != 0) {
        unsafe { err = WSAGetLastError(); }
    }
    if (err != 0) {
        unsafe { closesocket(fd); WSASetLastError(err); }
        return -1;
    }
    return ifd;
}

// Winsock takes SO_RCVTIMEO/SO_SNDTIMEO as a DWORD of milliseconds, not a
// struct timeval.
inline int sock_set_io_timeout(int fd, int timeout_ms) {
    unsigned int ms = (timeout_ms > 0) ? (unsigned int)timeout_ms : 0U;
    unsigned long long s = (unsigned long long)(unsigned int)fd;
    int rc1;
    int rc2;
    unsafe {
        rc1 = setsockopt(s, SCHED_SOL_SOCKET, SCHED_SO_RCVTIMEO, (const unsigned char*)&ms, 4);
        rc2 = setsockopt(s, SCHED_SOL_SOCKET, SCHED_SO_SNDTIMEO, (const unsigned char*)&ms, 4);
    }
    return (rc1 == 0 && rc2 == 0) ? 0 : -1;
}

} // namespace std
