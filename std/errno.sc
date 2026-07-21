// SafeC Standard Library — errno implementation
#pragma once
#include <std/errno.h>
#include <std/errno_compat.h>
#include <std/stderr_compat.h>

// Access errno via a C helper to handle its macro nature
namespace std {

extern int  strerror_r(int errnum, char* buf, unsigned long buflen); // POSIX
extern char* strerror(int errnum);
extern void  fputs(const char* s, void* stream);

inline int errno_get() {
    unsafe { return errno; }
}

inline void errno_set(int code) {
    unsafe { errno = code; }
}

inline const char* errno_str(int code) {
    unsafe { return strerror(code); }
}

inline void errno_print(const char* prefix) {
    unsafe {
        fputs(prefix, SAFEC_STDERR_);
        fputs(": ", SAFEC_STDERR_);
        fputs(strerror(errno_get()), SAFEC_STDERR_);
        fputs("\n", SAFEC_STDERR_);
    }
}

} // namespace std
