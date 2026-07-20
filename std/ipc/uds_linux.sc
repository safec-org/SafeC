// SafeC Standard Library — Unix Domain Sockets: Linux backend
// (Linux / Android — see uds.h for the portable API and the full backend
// file list)
//
// struct sockaddr_un's real Linux layout (<sys/un.h>): sun_family(2) +
// sun_path[108] = 110 bytes — no leading sun_len byte, and sun_family is
// a plain 2-byte field, matching sockaddr_in's Linux shape (see
// std/sched/io_nb_linux.sc's SockAddrIn comment) rather than BSD's.
#pragma once
#include <std/ipc/uds.h>
#include <std/str.sc>

#define SAFEC_AF_UNIX     1
#define SAFEC_SOCK_STREAM 1
#define SAFEC_SUN_PATH_MAX 108UL
#define SAFEC_UDS_O_NONBLOCK 0x800
#define SAFEC_UDS_F_GETFL    3
#define SAFEC_UDS_F_SETFL    4

namespace std {

struct SockAddrUn {
    unsigned short sun_family;
    char           sun_path[108];
};

extern int socket(int domain, int type, int protocol);
extern int bind(int fd, const void* addr, unsigned int addrlen);
extern int listen(int fd, int backlog);
extern int accept(int fd, void* addr, void* addrlen);
extern int connect(int fd, const void* addr, unsigned int addrlen);
extern int fcntl(int fd, int cmd, int arg);
extern int unlink(const char* path);

static int uds_fill_addr_(struct SockAddrUn* addr, const char* path) {
    unsafe { addr->sun_family = (unsigned short)SAFEC_AF_UNIX; }
    unsigned long len = str_len(path);
    if (len > SAFEC_SUN_PATH_MAX - 1UL) {
        return -1; // path too long to fit sun_path + NUL
    }
    unsigned long i = 0UL;
    while (i < len) {
        unsafe { addr->sun_path[i] = path[i]; }
        i = i + 1UL;
    }
    unsafe { addr->sun_path[i] = (char)0; }
    return 0;
}

static int uds_set_nonblocking_(int fd) {
    int flags;
    unsafe { flags = fcntl(fd, SAFEC_UDS_F_GETFL, 0); }
    if (flags < 0) {
        return -1;
    }
    int rc;
    unsafe { rc = fcntl(fd, SAFEC_UDS_F_SETFL, flags | SAFEC_UDS_O_NONBLOCK); }
    return rc;
}

inline int uds_listen_nb(const char* path) {
    int fd;
    unsafe { fd = socket(SAFEC_AF_UNIX, SAFEC_SOCK_STREAM, 0); }
    if (fd < 0) {
        return -1;
    }
    if (uds_set_nonblocking_(fd) != 0) {
        return -1;
    }
    struct SockAddrUn addr;
    if (uds_fill_addr_(&addr, path) != 0) {
        return -1;
    }
    int rc;
    unsafe { rc = bind(fd, (const void*)&addr, 110U); }
    if (rc != 0) {
        return -1;
    }
    // See std/sched/io_nb_bsd.sc's tcp_listen_nb for why 512, not a small
    // literal like 16 -- same backlog-overflow tail-latency issue applies
    // to any listening socket, not just TCP ones.
    unsafe { rc = listen(fd, 512); }
    if (rc != 0) {
        return -1;
    }
    return fd;
}

inline int uds_accept_nb(int listenfd) {
    int fd;
    unsafe { fd = accept(listenfd, (void*)0, (void*)0); }
    if (fd < 0) {
        return -1;
    }
    if (uds_set_nonblocking_(fd) != 0) {
        return -1;
    }
    return fd;
}

inline int uds_connect_nb(const char* path) {
    int fd;
    unsafe { fd = socket(SAFEC_AF_UNIX, SAFEC_SOCK_STREAM, 0); }
    if (fd < 0) {
        return -1;
    }
    if (uds_set_nonblocking_(fd) != 0) {
        return -1;
    }
    struct SockAddrUn addr;
    if (uds_fill_addr_(&addr, path) != 0) {
        return -1;
    }
    unsafe { connect(fd, (const void*)&addr, 110U); }
    // Non-blocking connect(): a non-zero return here almost always just
    // means EINPROGRESS, not failure — see uds.h's comment.
    return fd;
}

inline int uds_unlink(const char* path) {
    int rc;
    unsafe { rc = unlink(path); }
    return rc;
}

} // namespace std
