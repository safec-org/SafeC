// SafeC Standard Library — Anonymous Pipes: POSIX backend
// (macOS / iOS / Linux / Android / FreeBSD — see pipe.h for the portable
// API and the full backend file list)
#pragma once
#include <std/stddef.h>
#include <std/ipc/pipe.h>

#define SAFEC_PIPE_F_GETFL 3
#define SAFEC_PIPE_F_SETFL 4
#define SAFEC_PIPE_O_NONBLOCK_BSD   0x0004
#define SAFEC_PIPE_O_NONBLOCK_LINUX 0x0800

namespace std {

extern int   pipe(void* pipefd);       // int pipefd[2]
extern ssize_t  read(int fd, void* buf, size_t count);
extern ssize_t  write(int fd, const void* buf, size_t count);
extern int   close(int fd);
extern int   fcntl(int fd, int cmd, ...);   // variadic: arm64 Apple passes the 3rd arg on the stack

inline int Pipe::init() {
    int fds[2];
    int rc;
    unsafe { rc = pipe((void*)&fds[0]); }
    if (rc != 0) {
        return -1;
    }
    unsafe {
        self.read_fd  = (long long)fds[0];
        self.write_fd = (long long)fds[1];
    }
    return 0;
}

inline long long Pipe::read([]unsigned char buf_s) {
    unsigned long count = buf_s.len;
    &unsigned char buf;
    unsafe { buf = (&unsigned char)buf_s.ptr; }
    long n;
    unsafe { n = read((int)self.read_fd, (void*)buf, count); }
    return (long long)n;
}

inline long long Pipe::write([]const unsigned char buf_s) {
    unsigned long count = buf_s.len;
    const &unsigned char buf;
    unsafe { buf = (const &unsigned char)buf_s.ptr; }
    long n;
    unsafe { n = write((int)self.write_fd, (const void*)buf, count); }
    return (long long)n;
}

// macOS/BSD and Linux disagree on O_NONBLOCK's numeric value (see
// std/sched/io_nb_bsd.sc vs io_nb_linux.sc for the same split), so pick it
// by platform. (Trying one value and falling back on failure doesn't work:
// Linux's F_SETFL silently ignores unknown bits and returns 0.)
// F_GETFL/F_SETFL themselves are the same value, 3/4, on both.
#if defined(__linux__)
#define SAFEC_PIPE_O_NONBLOCK_ SAFEC_PIPE_O_NONBLOCK_LINUX
#else
#define SAFEC_PIPE_O_NONBLOCK_ SAFEC_PIPE_O_NONBLOCK_BSD
#endif
static int pipe_set_nonblocking_(int fd) {
    int flags;
    unsafe { flags = fcntl(fd, SAFEC_PIPE_F_GETFL, 0); }
    if (flags < 0) {
        return -1;
    }
    int rc;
    unsafe { rc = fcntl(fd, SAFEC_PIPE_F_SETFL, flags | SAFEC_PIPE_O_NONBLOCK_); }
    return rc;
}

inline int Pipe::set_read_nonblocking() {
    int fd;
    unsafe { fd = (int)self.read_fd; }
    return pipe_set_nonblocking_(fd);
}

inline int Pipe::set_write_nonblocking() {
    int fd;
    unsafe { fd = (int)self.write_fd; }
    return pipe_set_nonblocking_(fd);
}

inline int Pipe::close_read() {
    int fd;
    unsafe { fd = (int)self.read_fd; }
    int rc;
    unsafe { rc = close(fd); }
    return rc;
}

inline int Pipe::close_write() {
    int fd;
    unsafe { fd = (int)self.write_fd; }
    int rc;
    unsafe { rc = close(fd); }
    return rc;
}

} // namespace std
