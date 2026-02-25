// SafeC Standard Library — errno implementation
#include "errno.h"

// Access errno via a C helper to handle its macro nature
extern int* __errno_location();   // Linux glibc
extern int* __error();            // macOS/BSD
extern int  strerror_r(int errnum, char* buf, unsigned long buflen); // POSIX
extern char* strerror(int errnum);
extern void  fputs(const char* s, void* stream);
extern void* stderr;

int errno_get() {
    unsafe {
#ifdef __linux__
        return *(__errno_location());
#else
        return *(__error());
#endif
    }
}

void errno_set(int code) {
    unsafe {
#ifdef __linux__
        *(__errno_location()) = code;
#else
        *(__error()) = code;
#endif
    }
}

const char* errno_str(int code) {
    unsafe { return strerror(code); }
}

void errno_print(const char* prefix) {
    unsafe {
        fputs(prefix, stderr);
        fputs(": ", stderr);
        fputs(strerror(errno_get()), stderr);
        fputs("\n", stderr);
    }
}
