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
// Pipe conforms to io_traits.h's Reader and Writer (read from the read end,
// write to the write end), so read_exact / write_all / copy_stream work on
// it.
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

    // Creates a new pipe in place (fills read_fd/write_fd). Returns 0 on
    // success, -1 on failure.
    int init();

    // Reads up to 'count' bytes from the pipe's read end into 'buf'.
    // Returns the number of bytes read (0 means the write end was closed
    // and the pipe is drained — EOF), or -1 on error.
    long long read(&unsigned char buf, unsigned long count);

    // Writes up to 'count' bytes from 'buf' to the pipe's write end.
    // Returns the number of bytes written, or -1 on error.
    long long write(const &unsigned char buf, unsigned long count);

    // Puts the read end into non-blocking mode (pairs with std::Reactor —
    // await_fd(p.read_fd, SCHED_READ) once this returns 0). Returns 0 on
    // success, -1 on failure.
    int set_read_nonblocking();

    // Puts the write end into non-blocking mode. Returns 0 on success, -1
    // on failure.
    int set_write_nonblocking();

    // Closes the read end. Safe to call once; closing twice is undefined
    // the same way double-close(2) is on any platform.
    int close_read();

    // Closes the write end. A reader blocked in read() (or awaiting
    // SCHED_READ on read_fd) sees EOF (read() returns 0) once this runs
    // and the pipe is drained.
    int close_write();
};

} // namespace std
