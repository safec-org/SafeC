// SafeC Standard Library — Anonymous Pipes
//
// The simplest form of hosted inter-process communication: a one-way byte
// stream with a read end and a write end, most commonly used for parent-
// child communication (a spawned child process inherits the pipe fds/
// handles, or they're wired to its stdin/stdout — that wiring itself is
// out of scope here, this module only creates and operates the pipe).
//
// Handles are 'long long' uniformly across platforms — a POSIX pipe fd
// fits directly; a Windows pipe HANDLE (a pointer-sized value) is stored
// via the same pointer-to-integer convention std::thread.h's OS-thread
// handles already use.
//
// Backends (picked the same way as std/sched/reactor.h's):
//   pipe_posix.sc  — macOS, iOS, Linux, Android, FreeBSD (pipe() is
//                    identical across all of these — no per-OS split
//                    needed the way sockaddr_in required one)
//   pipe_win32.sc  — Windows (CreatePipe/ReadFile/WriteFile/CloseHandle)
#pragma once

namespace std {

struct Pipe {
    long long read_fd;
    long long write_fd;
};

// Creates a new pipe. Returns 0 on success, -1 on failure.
int pipe_create(struct Pipe* out);

// Reads up to 'count' bytes from the pipe's read end into 'buf'. Returns
// the number of bytes read (0 means the write end was closed and the
// pipe is drained — EOF), or -1 on error.
long long pipe_read(struct Pipe* p, void* buf, unsigned long count);

// Writes up to 'count' bytes from 'buf' to the pipe's write end. Returns
// the number of bytes written, or -1 on error.
long long pipe_write(struct Pipe* p, const void* buf, unsigned long count);

// Puts the read end into non-blocking mode (pairs with std::Reactor —
// await_fd(p->read_fd, SCHED_READ) once this returns 0). Returns 0 on
// success, -1 on failure.
int pipe_set_read_nonblocking(struct Pipe* p);

// Puts the write end into non-blocking mode. Returns 0 on success, -1 on
// failure.
int pipe_set_write_nonblocking(struct Pipe* p);

// Closes the read end. Safe to call once; closing twice is undefined the
// same way double-close(2) is on any platform.
int pipe_close_read(struct Pipe* p);

// Closes the write end. A reader blocked in pipe_read (or awaiting
// SCHED_READ on read_fd) sees EOF (pipe_read returns 0) once this runs
// and the pipe is drained.
int pipe_close_write(struct Pipe* p);

} // namespace std
