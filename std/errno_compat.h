// SafeC Standard Library — internal errno-accessor compatibility shim.
//
// Not part of the public std:: API — included by the handful of std/ files
// that need 'errno' as a plain lvalue (convert.sc, errno.sc). Every libc
// here exposes 'errno' as a thread-local value reached through a per-libc
// accessor function, never a plain extern global:
//   - macOS/BSD: '__error()' — the documented macOS <errno.h> expansion.
//   - glibc:     '__errno_location()'.
//   - MSVC/UCRT: '_errno()' (returns 'errno_t*', itself just 'int*').
//
// This header hides that split behind one 'errno' object-like macro (fine
// in safe mode — it isn't function-like, just an object-like macro whose
// replacement text happens to contain a call), usable as a plain lvalue
// exactly like C's own <errno.h> macro: 'errno = 0;', 'if (errno == ...)'.
#pragma once

namespace std {

#if defined(__APPLE__)
extern int* __error(void);
#define errno (*__error())
#elif defined(_WIN32)
extern int* _errno(void);
#define errno (*_errno())
#else
extern int* __errno_location(void);
#define errno (*__errno_location())
#endif

}
