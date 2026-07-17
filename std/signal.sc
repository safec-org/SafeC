// SafeC Standard Library — signal implementation
#pragma once
#include <std/signal.h>

namespace std {

extern void* signal(int sig, void* handler);
extern int   raise(int sig);
extern int   kill(int pid, int sig);
extern int   pause();

// SIG_DFL = 0, SIG_IGN = 1 on POSIX
inline void* signal_handle(int sig, void* handler) {
    unsafe { return signal(sig, handler); }
}

inline void signal_default(int sig) {
    unsafe { signal(sig, (void*)0); }
}

inline void signal_ignore(int sig) {
    unsafe { signal(sig, (void*)1); }
}

int signal_raise(int sig)        { unsafe { return raise(sig); } }
int signal_kill(int pid, int sig) { unsafe { return kill(pid, sig); } }
void signal_pause()               { unsafe { pause(); } }

} // namespace std
