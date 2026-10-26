// SafeC Standard Library — signal implementation
#pragma once
#include <std/signal.h>

namespace std {

// sighandler_t; SIG_DFL, SIG_IGN and SIG_ERR are the integers 0, 1 and -1
// as handlers.
typedef fn void(int) SigHandler_;
extern SigHandler_ signal(int sig, SigHandler_ handler);
extern int   raise(int sig);
extern int   kill(int pid, int sig);
extern int   pause();

// SIG_DFL = 0, SIG_IGN = 1 on POSIX
inline int signal_handle(int sig, fn void(int) handler) {
    unsafe {
        SigHandler_ prev = signal(sig, handler);
        return prev == (SigHandler_)~0UL ? -1 : 0;   // SIG_ERR
    }
}

inline void signal_default(int sig) {
    unsafe { signal(sig, (SigHandler_)0UL); }   // SIG_DFL
}

inline void signal_ignore(int sig) {
    unsafe { signal(sig, (SigHandler_)1UL); }   // SIG_IGN
}

int signal_raise(int sig)        { unsafe { return raise(sig); } }
int signal_kill(int pid, int sig) { unsafe { return kill(pid, sig); } }
void signal_pause()               { unsafe { pause(); } }

} // namespace std
