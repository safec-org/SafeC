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
#pragma once
#include <std/sched/io_nb.h>

#define SCHED_O_CREAT  0x0100
#define SAFEC_FIONBIO  0x8004667EU
#define SAFEC_WSA_VERSION 0x0202  // MAKEWORD(2, 2) — request Winsock 2.2

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

struct WSAData_ {
    unsigned char reserved[32]; // oversized on purpose — see reactor_win32.sc's WSAData_
};

extern int WSAStartup(unsigned short versionRequested, void* wsaData);
extern int _open(const char* path, int flags, int mode); // MSVCRT — plain (blocking) file open
extern unsigned long long socket(int domain, int type, int protocol);
extern int bind(unsigned long long fd, const void* addr, int addrlen);
extern int listen(unsigned long long fd, int backlog);
extern unsigned long long accept(unsigned long long fd, void* addr, void* addrlen);
extern int connect(unsigned long long fd, const void* addr, int addrlen);
extern int ioctlsocket(unsigned long long fd, int cmd, unsigned long* argp);

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
    unsafe { WSAStartup((unsigned short)SAFEC_WSA_VERSION, (void*)&wsaData); }
}

inline int fd_set_nonblocking(int fd) {
    unsigned long mode = 1UL; // non-zero = enable non-blocking mode
    int rc;
    unsafe { rc = ioctlsocket((unsigned long long)fd, (int)SAFEC_FIONBIO, &mode); }
    return rc;
}

inline int fd_set_blocking(int fd) {
    unsigned long mode = 0UL; // zero = disable non-blocking mode
    int rc;
    unsafe { rc = ioctlsocket((unsigned long long)fd, (int)SAFEC_FIONBIO, &mode); }
    return rc;
}

// See file header comment — this is a plain (blocking) file open on this
// backend; the O_NONBLOCK-style flag argument callers may pass has no
// effect here.
inline int fd_open_nb(const char* path, int flags, int mode) {
    int fd;
    unsafe { fd = _open(path, flags, mode); }
    return fd;
}

inline int tcp_listen_nb(unsigned short port) {
    io_nb_wsa_init_();
    unsigned long long fd;
    unsafe { fd = socket(SCHED_AF_INET, SCHED_SOCK_STREAM, 0); }
    int ifd = (int)fd;
    if (fd_set_nonblocking(ifd) != 0) {
        return -1;
    }

    struct SockAddrIn addr;
    addr.sin_family = (unsigned short)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = 0U; // INADDR_ANY
    addr.sin_zero   = 0UL;

    int rc;
    unsafe { rc = bind(fd, (const void*)&addr, 16); }
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
    unsafe { fd = accept((unsigned long long)listenfd, (void*)0, (void*)0); }
    int ifd = (int)fd;
    if (fd_set_nonblocking(ifd) != 0) {
        return -1;
    }
    return ifd;
}

inline int tcp_connect_nb(unsigned int addr_network_order, unsigned short port) {
    io_nb_wsa_init_();
    unsigned long long fd;
    unsafe { fd = socket(SCHED_AF_INET, SCHED_SOCK_STREAM, 0); }
    int ifd = (int)fd;
    if (fd_set_nonblocking(ifd) != 0) {
        return -1;
    }

    struct SockAddrIn addr;
    addr.sin_family = (unsigned short)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = addr_network_order;
    addr.sin_zero   = 0UL;

    unsafe { connect(fd, (const void*)&addr, 16); }
    // Non-blocking connect(): a non-zero return here almost always just
    // means WSAEWOULDBLOCK (the handshake hasn't finished yet), not
    // failure — the caller awaits SCHED_WRITE on 'fd' and treats
    // writability as "connect attempt finished" per the header comment,
    // so the return value here isn't itself meaningful the way it is for
    // a blocking connect().
    return ifd;
}

} // namespace std
