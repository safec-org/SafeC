#pragma once
// SafeC Standard Library — errno (C11 <errno.h>)
// Access the thread-local errno value and error descriptions.

// Read the current thread-local errno value.
namespace std {

int  errno_get();
// Set errno explicitly (rarely needed).
void errno_set(int code);
// Return a static string describing the error code (wraps strerror).
const char* errno_str(int code);
// Print "prefix: <error description>" to stderr.
void errno_print(const char* prefix);

// Common POSIX error codes (subset; actual values are platform-defined).
// Use errno_get() to retrieve the actual numeric code from the OS.
const int ERRNO_OK()     { return 0; }
const int ERRNO_EPERM()  { return 1; }
const int ERRNO_ENOENT() { return 2; }
const int ERRNO_EIO()    { return 5; }
const int ERRNO_EBADF()  { return 9; }
const int ERRNO_ENOMEM() { return 12; }
const int ERRNO_EACCES() { return 13; }
const int ERRNO_EBUSY()  { return 16; }
const int ERRNO_EEXIST() { return 17; }
const int ERRNO_EINVAL() { return 22; }
const int ERRNO_ERANGE() { return 34; }
// EAGAIN/EWOULDBLOCK/ETIMEDOUT are the one place this "subset" stops being
// platform-stable: every other code above happens to share the same
// number on Linux/macOS/BSD (a historical Unix errno.h accident), but
// these three genuinely differ — verified empirically (a small C program
// against each platform's real <errno.h>), not copied from memory:
//   Linux (glibc):  EAGAIN=11  EWOULDBLOCK=11   ETIMEDOUT=110
//   macOS/BSD:      EAGAIN=35  EWOULDBLOCK=35   ETIMEDOUT=60
// Windows sockets don't use errno for this at all — recv/send/etc. report
// failure via WSAGetLastError() on a completely separate per-thread error
// slot, not the CRT errno _errno() wraps. Code that needs "did this
// non-blocking socket call fail because it would've blocked" should call
// std::sock_would_block() (std/sched/io_nb.h) instead of comparing
// errno_get() to these — that helper is the one that's actually correct
// on all three platforms, including Windows.
#ifdef __APPLE__
const int ERRNO_EAGAIN()      { return 35; }
const int ERRNO_EWOULDBLOCK() { return 35; }
const int ERRNO_ETIMEDOUT()   { return 60; }
#else
const int ERRNO_EAGAIN()      { return 11; }
const int ERRNO_EWOULDBLOCK() { return 11; }
const int ERRNO_ETIMEDOUT()   { return 110; }
#endif

} // namespace std
