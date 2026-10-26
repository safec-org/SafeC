// SafeC Standard Library — signal implementation
#pragma once
#include <std/signal.h>

namespace std {

extern void* signal(int sig, void* handler);
extern int   raise(int sig);
extern int   kill(int pid, int sig);
extern int   pause();

// SIG_DFL = 0, SIG_IGN = 1 on POSIX
inline int signal_handle(int sig, fn void(int) handler) {
    unsafe {
        void* prev = signal(sig, (void*)handler);
        return prev == (void*)-1 ? -1 : 0;   // SIG_ERR
    }
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
