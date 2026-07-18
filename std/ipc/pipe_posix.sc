// SafeC Standard Library — Anonymous Pipes: POSIX backend
// (macOS / iOS / Linux / Android / FreeBSD — see pipe.h for the portable
// API and the full backend file list)
#pragma once
#include <std/ipc/pipe.h>

#define SAFEC_PIPE_F_GETFL 3
#define SAFEC_PIPE_F_SETFL 4
#define SAFEC_PIPE_O_NONBLOCK_BSD   0x0004
#define SAFEC_PIPE_O_NONBLOCK_LINUX 0x0800

namespace std {

extern int   pipe(void* pipefd);       // int pipefd[2]
extern long  read(int fd, void* buf, unsigned long count);
extern long  write(int fd, const void* buf, unsigned long count);
extern int   close(int fd);
extern int   fcntl(int fd, int cmd, int arg);

inline int pipe_create(struct Pipe* out) {
    int fds[2];
    int rc;
    unsafe { rc = pipe((void*)&fds[0]); }
    if (rc != 0) {
        return -1;
    }
    unsafe {
        out->read_fd  = (long long)fds[0];
        out->write_fd = (long long)fds[1];
    }
    return 0;
}

inline long long pipe_read(struct Pipe* p, void* buf, unsigned long count) {
    long n;
    unsafe { n = read((int)p->read_fd, buf, count); }
    return (long long)n;
}

inline long long pipe_write(struct Pipe* p, const void* buf, unsigned long count) {
    long n;
    unsafe { n = write((int)p->write_fd, buf, count); }
    return (long long)n;
}

// macOS/BSD and Linux disagree on O_NONBLOCK's numeric value (see
// std/sched/io_nb_bsd.sc vs io_nb_linux.sc for the same split) — since
// this one file covers both, try the BSD value first and fall back to
// the Linux value if fcntl rejects it outright, rather than requiring a
// third pipe_posix_bsd.sc/pipe_posix_linux.sc split for a single flag
// bit. (F_GETFL/F_SETFL themselves are the same value, 3/4, on both.)
static int pipe_set_nonblocking_(int fd) {
    int flags;
    unsafe { flags = fcntl(fd, SAFEC_PIPE_F_GETFL, 0); }
    if (flags < 0) {
        return -1;
    }
    int rc;
    unsafe { rc = fcntl(fd, SAFEC_PIPE_F_SETFL, flags | SAFEC_PIPE_O_NONBLOCK_BSD); }
    if (rc == 0) {
        return 0;
    }
    unsafe { rc = fcntl(fd, SAFEC_PIPE_F_SETFL, flags | SAFEC_PIPE_O_NONBLOCK_LINUX); }
    return rc;
}

inline int pipe_set_read_nonblocking(struct Pipe* p) {
    int fd;
    unsafe { fd = (int)p->read_fd; }
    return pipe_set_nonblocking_(fd);
}

inline int pipe_set_write_nonblocking(struct Pipe* p) {
    int fd;
    unsafe { fd = (int)p->write_fd; }
    return pipe_set_nonblocking_(fd);
}

inline int pipe_close_read(struct Pipe* p) {
    int fd;
    unsafe { fd = (int)p->read_fd; }
    int rc;
    unsafe { rc = close(fd); }
    return rc;
}

inline int pipe_close_write(struct Pipe* p) {
    int fd;
    unsafe { fd = (int)p->write_fd; }
    int rc;
    unsafe { rc = close(fd); }
    return rc;
}

} // namespace std
