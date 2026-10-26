#pragma once
// SafeC Standard Library — Signal (C11 <signal.h>)
// Install signal handlers and send signals.

// Standard signal numbers (POSIX values; matches Linux/macOS).
namespace std {

static const int SIG_HUP()  { return 1; }   // Hangup
static const int SIG_INT()  { return 2; }   // Interrupt (Ctrl+C)
static const int SIG_QUIT() { return 3; }   // Quit
static const int SIG_ILL()  { return 4; }   // Illegal instruction
static const int SIG_ABRT() { return 6; }   // Abort
static const int SIG_FPE()  { return 8; }   // Floating-point exception
static const int SIG_KILL() { return 9; }   // Kill (cannot be caught)
static const int SIG_SEGV() { return 11; }  // Segmentation fault
static const int SIG_PIPE() { return 13; }  // Broken pipe
static const int SIG_ALRM() { return 14; }  // Alarm clock
static const int SIG_TERM() { return 15; }  // Termination
// The numbers above are the same everywhere; these differ between Linux and
// macOS/BSD.
#if defined(__APPLE__) || defined(__FreeBSD__)
static const int SIG_USR1() { return 30; }  // User-defined signal 1
static const int SIG_USR2() { return 31; }  // User-defined signal 2
static const int SIG_CHLD() { return 20; }  // Child stopped/terminated
static const int SIG_CONT() { return 19; }  // Continue if stopped
static const int SIG_STOP() { return 17; }  // Stop process (cannot be caught)
static const int SIG_TSTP() { return 18; }  // Stop typed at terminal
#else
static const int SIG_USR1() { return 10; }  // User-defined signal 1
static const int SIG_USR2() { return 12; }  // User-defined signal 2
static const int SIG_CHLD() { return 17; }  // Child stopped/terminated
static const int SIG_CONT() { return 18; }  // Continue if stopped
static const int SIG_STOP() { return 19; }  // Stop process (cannot be caught)
static const int SIG_TSTP() { return 20; }  // Stop typed at terminal
#endif
static const int SIG_WINCH(){ return 28; }  // Window size change

// Install a handler for signal sig. 0 on success, -1 on error (an
// invalid or uncatchable signal).
int signal_handle(int sig, fn void(int) handler);

// Restore default handling for signal sig.
void signal_default(int sig);

// Ignore signal sig.
void signal_ignore(int sig);

// Send signal sig to the calling process.
int signal_raise(int sig);

// Send signal sig to process with the given PID (requires <sys/types.h>).
int signal_kill(int pid, int sig);

// Block until any signal arrives (pause).
void signal_pause();

} // namespace std
