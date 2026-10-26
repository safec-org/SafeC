// SafeC Standard Library — signal implementation
#pragma once
#include <std/signal.h>

namespace std {

extern unsigned char* signal(int sig, unsigned char* handler);
extern int   raise(int sig);
extern int   kill(int pid, int sig);
extern int   pause();

// SIG_DFL = 0, SIG_IGN = 1 on POSIX
inline int signal_handle(int sig, fn void(int) handler) {
    unsafe {
        unsigned char* prev = signal(sig, (unsigned char*)handler);
        return prev == (unsigned char*)-1 ? -1 : 0;   // SIG_ERR
    }
}

inline void signal_default(int sig) {
    unsafe { signal(sig, (unsigned char*)0); }
}

inline void signal_ignore(int sig) {
    unsafe { signal(sig, (unsigned char*)1); }
}

int signal_raise(int sig)        { unsafe { return raise(sig); } }
int signal_kill(int pid, int sig) { unsafe { return kill(pid, sig); } }
void signal_pause()               { unsafe { pause(); } }

} // namespace std
