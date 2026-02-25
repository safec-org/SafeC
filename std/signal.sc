// SafeC Standard Library — signal implementation
#include "signal.h"

extern void* signal(int sig, void* handler);
extern int   raise(int sig);
extern int   kill(int pid, int sig);
extern int   pause();

// SIG_DFL = 0, SIG_IGN = 1 on POSIX
void* signal_handle(int sig, void* handler) {
    unsafe { return signal(sig, handler); }
}

void signal_default(int sig) {
    unsafe { signal(sig, (void*)0); }
}

void signal_ignore(int sig) {
    unsafe { signal(sig, (void*)1); }
}

int signal_raise(int sig)        { unsafe { return raise(sig); } }
int signal_kill(int pid, int sig) { unsafe { return kill(pid, sig); } }
void signal_pause()               { unsafe { pause(); } }
