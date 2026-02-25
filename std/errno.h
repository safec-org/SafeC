#pragma once
// SafeC Standard Library — errno (C11 <errno.h>)
// Access the thread-local errno value and error descriptions.

// Read the current thread-local errno value.
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
const int ERRNO_EAGAIN() { return 11; }
const int ERRNO_EWOULDBLOCK() { return 11; }
const int ERRNO_ETIMEDOUT()   { return 110; }
