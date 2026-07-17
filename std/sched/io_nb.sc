// SafeC Standard Library — Non-blocking file/socket helpers (see io_nb.h)
#pragma once
#include <std/sched/io_nb.h>

#define SCHED_F_GETFL 3
#define SCHED_F_SETFL 4

namespace std {

extern int open(const char* path, int flags, int mode);
extern int socket(int domain, int type, int protocol);
extern int bind(int fd, const void* addr, unsigned int addrlen);
extern int listen(int fd, int backlog);
extern int accept(int fd, void* addr, void* addrlen);
extern int connect(int fd, const void* addr, unsigned int addrlen);
extern int fcntl(int fd, int cmd, int arg);

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

inline int fd_open_nb(const char* path, int flags, int mode) {
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
    unsafe { rc = listen(fd, 16); }
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
    return fd;
}

inline int tcp_connect_nb(unsigned int addr_network_order, unsigned short port) {
    int fd;
    unsafe { fd = socket(SCHED_AF_INET, SCHED_SOCK_STREAM, 0); }
    if (fd < 0) {
        return -1;
    }
    if (fd_set_nonblocking(fd) != 0) {
        return -1;
    }

    struct SockAddrIn addr;
    addr.sin_len    = (unsigned char)16;
    addr.sin_family = (unsigned char)SCHED_AF_INET;
    addr.sin_port   = sched_htons(port);
    addr.sin_addr   = addr_network_order;
    addr.sin_zero   = 0UL;

    unsafe { connect(fd, (const void*)&addr, 16U); }
    // Non-blocking connect(): a non-zero return here almost always just
    // means EINPROGRESS (the handshake hasn't finished yet), not failure —
    // the caller awaits SCHED_WRITE on 'fd' and treats writability as
    // "connect attempt finished" per the header comment, so the return
    // value here isn't itself meaningful the way it is for a blocking
    // connect().
    return fd;
}

} // namespace std
