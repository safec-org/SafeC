#pragma once
// SafeC Standard Library — Signal (C11 <signal.h>)
// Install signal handlers and send signals.

// Standard signal numbers (POSIX values; matches Linux/macOS).
const int SIG_HUP()  { return 1; }   // Hangup
const int SIG_INT()  { return 2; }   // Interrupt (Ctrl+C)
const int SIG_QUIT() { return 3; }   // Quit
const int SIG_ILL()  { return 4; }   // Illegal instruction
const int SIG_ABRT() { return 6; }   // Abort
const int SIG_FPE()  { return 8; }   // Floating-point exception
const int SIG_KILL() { return 9; }   // Kill (cannot be caught)
const int SIG_SEGV() { return 11; }  // Segmentation fault
const int SIG_PIPE() { return 13; }  // Broken pipe
const int SIG_ALRM() { return 14; }  // Alarm clock
const int SIG_TERM() { return 15; }  // Termination
const int SIG_USR1() { return 10; }  // User-defined signal 1
const int SIG_USR2() { return 12; }  // User-defined signal 2
const int SIG_CHLD() { return 17; }  // Child stopped/terminated
const int SIG_CONT() { return 18; }  // Continue if stopped
const int SIG_STOP() { return 19; }  // Stop process (cannot be caught)
const int SIG_TSTP() { return 20; }  // Stop typed at terminal
const int SIG_WINCH(){ return 28; }  // Window size change

// Install a handler for signal sig.
// handler: void (*handler)(int) — pass as void*
// Returns previous handler (as void*), or (void*)-1 on error.
void* signal_handle(int sig, void* handler);

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
