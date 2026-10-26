// SafeC Standard Library — internal stderr-handle compatibility shim.
//
// Not part of the public std:: API — included only by the handful of std/
// files that need to print a fatal diagnostic before abort()ing
// (mem.sc, io.sc, assert.sc, crypto/rng.sc, alloc/{pool,slab,tlsf}.sc) and
// have no other reason to pull in the full <std/io.h>.
//
// There is no single libc symbol for "the stderr FILE*" across platforms:
//   - macOS/BSD:  'stderr' is a macro expanding to '__stderrp', an actual
//                 exported global of that name.
//   - glibc:      'stderr' is a plain exported global itself — no '__stderrp'.
//   - MSVC/UCRT:  no exported 'stderr' symbol at all; the macro expands to
//                 a call, '__acrt_iob_func(2)' (index 2 = stderr in the
//                 standard fd 0/1/2 ordering), whose result is not a fixed
//                 address and can't be taken as a plain extern global.
//
// SAFEC_STDERR_ hides that split behind one token, usable everywhere the
// old direct '__stderrp' reference was (a 'CFile_*'), on whichever of __APPLE__ /
// __linux__ / _WIN32 'safec' auto-defined for this host/target (see
// main.cpp's platform-detection block).
#pragma once

namespace std {

// The C library's FILE, opaque: std only ever holds a 'CFile_*' the C
// library handed out and passes it back (fprintf, fputs, fclose, ...). The
// playground compiler's runtime (selfhost/runtime/rt.sc, compiled into the
// same translation unit) is a C library with a FILE of its own: there, a
// CFile_ is that one.
#if defined(__SCC__)
typedef struct __FILE CFile_;
#else
struct CFile_ { char opaque_; };
#endif

#if defined(__APPLE__)
extern CFile_* __stderrp;
#define SAFEC_STDERR_ __stderrp
#elif defined(_WIN32)
extern CFile_* __acrt_iob_func(unsigned int n);
#define SAFEC_STDERR_ __acrt_iob_func(2)
#else
// glibc and most other hosted libcs: a plain exported 'stderr' global.
extern CFile_* stderr;
#define SAFEC_STDERR_ stderr
#endif

}
