// SafeC Standard Library — Anonymous Pipes: Win32 backend
// (see pipe.h for the portable API and the full backend file list)
//
// CreatePipe's ends are HANDLEs (pointer-sized), stored via the same
// pointer-to-'long long' convention std::thread.h's OS-thread handles
// already use. Non-blocking anonymous pipes have no direct Win32
// equivalent to O_NONBLOCK — overlapped I/O (FILE_FLAG_OVERLAPPED) needs
// the pipe created with CreateNamedPipe instead of CreatePipe, so
// pipe_set_read_nonblocking/pipe_set_write_nonblocking are honest no-ops
// here (return -1) rather than silently pretending to succeed; a program
// needing non-blocking pipe I/O on Windows should read/write from a
// dedicated thread instead of the reactor's readiness-polling model, the
// same real-world workaround most libraries use for this exact Win32 gap.
#pragma once
#include <std/ipc/pipe.h>

namespace std {

extern int  CreatePipe(void* hReadPipe, void* hWritePipe, void* lpPipeAttributes, unsigned int nSize);
extern int  ReadFile(unsigned long long hFile, void* lpBuffer, unsigned int nNumberOfBytesToRead, void* lpNumberOfBytesRead, void* lpOverlapped);
extern int  WriteFile(unsigned long long hFile, const void* lpBuffer, unsigned int nNumberOfBytesToWrite, void* lpNumberOfBytesWritten, void* lpOverlapped);
extern int  CloseHandle(unsigned long long hObject);

inline int pipe_create(&Pipe out) {
    unsigned long long readH = 0ULL;
    unsigned long long writeH = 0ULL;
    int ok;
    unsafe { ok = CreatePipe((void*)&readH, (void*)&writeH, (void*)0, 0U); }
    if (ok == 0) {
        return -1;
    }
    unsafe {
        out->read_fd  = (long long)readH;
        out->write_fd = (long long)writeH;
    }
    return 0;
}

inline long long pipe_read(&Pipe p, void* buf, unsigned long count) {
    unsigned long long h;
    unsafe { h = (unsigned long long)p->read_fd; }
    unsigned int got = 0U;
    int ok;
    unsafe { ok = ReadFile(h, buf, (unsigned int)count, (void*)&got, (void*)0); }
    if (ok == 0) {
        return -1LL;
    }
    return (long long)got;
}

inline long long pipe_write(&Pipe p, const void* buf, unsigned long count) {
    unsigned long long h;
    unsafe { h = (unsigned long long)p->write_fd; }
    unsigned int sent = 0U;
    int ok;
    unsafe { ok = WriteFile(h, buf, (unsigned int)count, (void*)&sent, (void*)0); }
    if (ok == 0) {
        return -1LL;
    }
    return (long long)sent;
}

// See file header comment — anonymous CreatePipe handles have no
// non-blocking mode on this backend.
inline int pipe_set_read_nonblocking(&Pipe p) {
    return -1;
}

inline int pipe_set_write_nonblocking(&Pipe p) {
    return -1;
}

inline int pipe_close_read(&Pipe p) {
    unsigned long long h;
    unsafe { h = (unsigned long long)p->read_fd; }
    int ok;
    unsafe { ok = CloseHandle(h); }
    return ok != 0 ? 0 : -1;
}

inline int pipe_close_write(&Pipe p) {
    unsigned long long h;
    unsafe { h = (unsigned long long)p->write_fd; }
    int ok;
    unsafe { ok = CloseHandle(h); }
    return ok != 0 ? 0 : -1;
}

} // namespace std
