// SafeC Standard Library — Thread (cross-platform implementation)
// safeguard: link linux -lpthread
// safeguard: runtime-symbol pthread_create  (what `spawn` compiles to on a POSIX target)
#pragma once
#include <std/thread.h>
#include <std/mem.h>

namespace std {

// ════════════════════════════════ WIN32 ══════════════════════════════════════
#ifdef _WIN32

// kernel32.dll — always linked on Windows

// The kernel objects std hands to kernel32, opaque: a HANDLE, a
// CRITICAL_SECTION, a CONDITION_VARIABLE and an SRWLOCK (the last three live
// in heap blocks of their size, see mutex_init & co.), and the
// SECURITY_ATTRIBUTES std never passes (always null).
struct Win32Handle_ { char opaque_; };
struct Win32CritSec_ { char opaque_; };
struct Win32CondVar_ { char opaque_; };
struct Win32SrwLock_ { char opaque_; };
struct Win32SecAttrs_ { char opaque_; };

// The thread's start routine: DWORD WINAPI f(LPVOID) — std's entry returns
// a pointer, whose low 32 bits are the (unused) exit code.
extern Win32Handle_* CreateThread(Win32SecAttrs_* sec, unsigned long stack_sz, fn ?&Erased(?&Erased) func, ?&Erased arg,
                                  unsigned int flags, unsigned int* tid_out);
extern unsigned int WaitForSingleObject(Win32Handle_* handle, unsigned int ms);
extern int CloseHandle(Win32Handle_* handle);
extern void Sleep(unsigned int ms);
extern int SwitchToThread();
extern unsigned int GetCurrentThreadId();

// CRITICAL_SECTION (40 bytes on x64; we alloc 64 to be safe)
extern void InitializeCriticalSection(Win32CritSec_* cs);
extern void DeleteCriticalSection(Win32CritSec_* cs);
extern void EnterCriticalSection(Win32CritSec_* cs);
extern void LeaveCriticalSection(Win32CritSec_* cs);
extern int  TryEnterCriticalSection(Win32CritSec_* cs);

// CONDITION_VARIABLE (8 bytes on Vista+; we alloc 16)
extern void InitializeConditionVariable(Win32CondVar_* cv);
extern int  SleepConditionVariableCS(Win32CondVar_* cv, Win32CritSec_* cs, unsigned int ms);
extern void WakeConditionVariable(Win32CondVar_* cv);
extern void WakeAllConditionVariable(Win32CondVar_* cv);
extern unsigned int GetLastError();

// SRWLOCK (8 bytes; we alloc 16)
extern void InitializeSRWLock(Win32SrwLock_* rw);
extern void AcquireSRWLockShared(Win32SrwLock_* rw);
extern void AcquireSRWLockExclusive(Win32SrwLock_* rw);
extern void ReleaseSRWLockShared(Win32SrwLock_* rw);
extern void ReleaseSRWLockExclusive(Win32SrwLock_* rw);
extern int  TryAcquireSRWLockShared(Win32SrwLock_* rw);
extern int  TryAcquireSRWLockExclusive(Win32SrwLock_* rw);

inline int thread_create_raw_(&unsigned long long tid, fn ?&Erased(?&Erased) func, ?&Erased arg) {
    unsafe {
        Win32Handle_* h = CreateThread((Win32SecAttrs_*)0, 0UL, func, arg, 0u, (unsigned int*)0);
        if (h == (Win32Handle_*)0) return -1;
        *tid = (unsigned long long)h;
        return 0;
    }
}

inline int thread_join(unsigned long long tid) {
    unsafe {
        Win32Handle_* h = (Win32Handle_*)tid;
        unsigned int r = WaitForSingleObject(h, 0xFFFFFFFFu); // INFINITE
        CloseHandle(h);
        return (int)r;
    }
}

inline int thread_detach(unsigned long long tid) {
    unsafe { return CloseHandle((Win32Handle_*)tid) ? 0 : -1; }
}

void thread_yield()                  { unsafe { SwitchToThread(); } }
void thread_sleep_ms(unsigned long ms) { unsafe { Sleep(ms > 0xFFFFFFFEUL ? 0xFFFFFFFEu : (unsigned int)ms); } }

inline unsigned long long thread_self() {
    unsafe { return (unsigned long long)GetCurrentThreadId(); }
}

inline int mutex_init(&unsigned long long m) {
    unsafe {
        Win32CritSec_* cs = (Win32CritSec_*)alloc_raw_(64UL);
        if (cs == (Win32CritSec_*)0) return -1;
        InitializeCriticalSection(cs);
        *m = (unsigned long long)cs;
        return 0;
    }
}

inline int mutex_destroy(&unsigned long long m) {
    unsafe {
        Win32CritSec_* cs = (Win32CritSec_*)(*m);
        DeleteCriticalSection(cs);
        dealloc_raw_((unsigned char*)cs);
        *m = 0ULL;
        return 0;
    }
}

int mutex_lock(&unsigned long long m)    { unsafe { EnterCriticalSection((Win32CritSec_*)(*m)); return 0; } }
int mutex_trylock(&unsigned long long m) { unsafe { return TryEnterCriticalSection((Win32CritSec_*)(*m)) ? 0 : 1; } }
int mutex_unlock(&unsigned long long m)  { unsafe { LeaveCriticalSection((Win32CritSec_*)(*m)); return 0; } }

inline int cond_init(&unsigned long long cv) {
    unsafe {
        Win32CondVar_* c = (Win32CondVar_*)alloc_raw_(16UL);
        if (c == (Win32CondVar_*)0) return -1;
        InitializeConditionVariable(c);
        *cv = (unsigned long long)c;
        return 0;
    }
}

inline int cond_destroy(&unsigned long long cv) {
    unsafe { dealloc_raw_((unsigned char*)(*cv)); *cv = 0ULL; return 0; }
}

inline int cond_wait(&unsigned long long cv, &unsigned long long m) {
    unsafe {
        return SleepConditionVariableCS((Win32CondVar_*)(*cv), (Win32CritSec_*)(*m), 0xFFFFFFFFu) ? 0 : -1;
    }
}

inline int cond_timedwait_ms(&unsigned long long cv, &unsigned long long m, unsigned long ms) {
    unsafe {
        if (SleepConditionVariableCS((Win32CondVar_*)(*cv), (Win32CritSec_*)(*m), ms > 0xFFFFFFFEUL ? 0xFFFFFFFEu : (unsigned int)ms)) return 0;
        return GetLastError() == 1460u ? THREAD_TIMEDOUT : -1;   // ERROR_TIMEOUT
    }
}

int cond_signal(&unsigned long long cv)    { unsafe { WakeConditionVariable((Win32CondVar_*)(*cv)); return 0; } }
int cond_broadcast(&unsigned long long cv) { unsafe { WakeAllConditionVariable((Win32CondVar_*)(*cv)); return 0; } }

inline int rwlock_init(&unsigned long long rw) {
    unsafe {
        Win32SrwLock_* r = (Win32SrwLock_*)alloc_raw_(16UL);
        if (r == (Win32SrwLock_*)0) return -1;
        InitializeSRWLock(r);
        *rw = (unsigned long long)r;
        return 0;
    }
}

inline int rwlock_destroy(&unsigned long long rw) {
    unsafe { dealloc_raw_((unsigned char*)(*rw)); *rw = 0ULL; return 0; }
}

int rwlock_rdlock(&unsigned long long rw)    { unsafe { AcquireSRWLockShared((Win32SrwLock_*)(*rw)); return 0; } }
int rwlock_wrlock(&unsigned long long rw)    { unsafe { AcquireSRWLockExclusive((Win32SrwLock_*)(*rw)); return 0; } }
int rwlock_tryrdlock(&unsigned long long rw) { unsafe { return TryAcquireSRWLockShared((Win32SrwLock_*)(*rw)) ? 0 : 1; } }
int rwlock_trywrlock(&unsigned long long rw) { unsafe { return TryAcquireSRWLockExclusive((Win32SrwLock_*)(*rw)) ? 0 : 1; } }
int rwlock_rdunlock(&unsigned long long rw)  { unsafe { ReleaseSRWLockShared((Win32SrwLock_*)(*rw)); return 0; } }
int rwlock_wrunlock(&unsigned long long rw)  { unsafe { ReleaseSRWLockExclusive((Win32SrwLock_*)(*rw)); return 0; } }

// ════════════════════════════════ POSIX ══════════════════════════════════════
#else

// pthread_t is an integer handle (unsigned long long holds it on every
// target); the mutex/cond/rwlock objects live in heap blocks of their size
// (see mutex_init & co.), opaque here; std never passes attributes (null).
struct PthreadMutex_ { char opaque_; };
struct PthreadCond_ { char opaque_; };
struct PthreadRwlock_ { char opaque_; };
struct PthreadAttr_ { char opaque_; };

extern int pthread_create(unsigned long long* tid, PthreadAttr_* attr, fn ?&Erased(?&Erased) func, ?&Erased arg);
extern int pthread_join(unsigned long long tid, ?&Erased* retval);
extern int pthread_detach(unsigned long long tid);
extern unsigned long long pthread_self();
extern int sched_yield();
// pthread_cond_timedwait's timeout code differs per OS.
#if defined(__APPLE__) || defined(__FreeBSD__)
#define SAFEC_ETIMEDOUT_ 60
#else
#define SAFEC_ETIMEDOUT_ 110
#endif

// struct timespec { time_t tv_sec; long tv_nsec; }: two 4-byte fields on a
// 32-bit C ABI, two 8-byte fields on a 64-bit one. (SafeC's own 'long' is 64
// bits everywhere, so it can't stand in for C's here.)
#if __SIZEOF_POINTER__ == 4
#define SAFEC_TS_FIELD_ int
#else
#define SAFEC_TS_FIELD_ long long
#endif
struct SafecTimespec_ {
    SAFEC_TS_FIELD_ tv_sec;
    SAFEC_TS_FIELD_ tv_nsec;
};
extern int nanosleep(const SafecTimespec_* req, SafecTimespec_* rem);
extern int clock_gettime(int clk, SafecTimespec_* ts);

extern int pthread_mutex_init(PthreadMutex_* m, PthreadAttr_* attr);
extern int pthread_mutex_destroy(PthreadMutex_* m);
extern int pthread_mutex_lock(PthreadMutex_* m);
extern int pthread_mutex_trylock(PthreadMutex_* m);
extern int pthread_mutex_unlock(PthreadMutex_* m);

extern int pthread_cond_init(PthreadCond_* cv, PthreadAttr_* attr);
extern int pthread_cond_destroy(PthreadCond_* cv);
extern int pthread_cond_wait(PthreadCond_* cv, PthreadMutex_* m);
extern int pthread_cond_timedwait(PthreadCond_* cv, PthreadMutex_* m, const SafecTimespec_* ts);
extern int pthread_cond_signal(PthreadCond_* cv);
extern int pthread_cond_broadcast(PthreadCond_* cv);

extern int pthread_rwlock_init(PthreadRwlock_* rw, PthreadAttr_* attr);
extern int pthread_rwlock_destroy(PthreadRwlock_* rw);
extern int pthread_rwlock_rdlock(PthreadRwlock_* rw);
extern int pthread_rwlock_wrlock(PthreadRwlock_* rw);
extern int pthread_rwlock_tryrdlock(PthreadRwlock_* rw);
extern int pthread_rwlock_trywrlock(PthreadRwlock_* rw);
extern int pthread_rwlock_unlock(PthreadRwlock_* rw);

inline int thread_create_raw_(&unsigned long long tid, fn ?&Erased(?&Erased) func, ?&Erased arg) {
    unsafe { return pthread_create((unsigned long long*)tid, (PthreadAttr_*)0, func, arg); }
}
int thread_join(unsigned long long tid)    { unsafe { return pthread_join(tid, (?&Erased*)0); } }
int thread_detach(unsigned long long tid)  { unsafe { return pthread_detach(tid); } }
void thread_yield()                        { unsafe { sched_yield(); } }
unsigned long long thread_self()           { unsafe { return pthread_self(); } }

inline void thread_sleep_ms(unsigned long ms) {
    unsafe {
        struct SafecTimespec_ ts;
        ts.tv_sec = (SAFEC_TS_FIELD_)(ms / 1000UL);
        ts.tv_nsec = (SAFEC_TS_FIELD_)((ms % 1000UL) * 1000000UL);
        nanosleep(&ts, (SafecTimespec_*)0);
    }
}

inline int mutex_init(&unsigned long long m) {
    unsafe {
        PthreadMutex_* pm = (PthreadMutex_*)alloc_raw_(128UL); // pthread_mutex_t <= 128 bytes (64 on macOS, 40 on Linux)
        if (pm == (PthreadMutex_*)0) return -1;
        int r = pthread_mutex_init(pm, (PthreadAttr_*)0);
        if (r != 0) { dealloc_raw_((unsigned char*)pm); return r; }
        *m = (unsigned long long)pm;
        return 0;
    }
}

inline int mutex_destroy(&unsigned long long m) {
    unsafe {
        PthreadMutex_* pm = (PthreadMutex_*)(*m);
        int r = pthread_mutex_destroy(pm);
        dealloc_raw_((unsigned char*)pm);
        *m = 0ULL;
        return r;
    }
}

int mutex_lock(&unsigned long long m)    { unsafe { return pthread_mutex_lock((PthreadMutex_*)(*m)); } }
int mutex_trylock(&unsigned long long m) { unsafe { return pthread_mutex_trylock((PthreadMutex_*)(*m)); } }
int mutex_unlock(&unsigned long long m)  { unsafe { return pthread_mutex_unlock((PthreadMutex_*)(*m)); } }

inline int cond_init(&unsigned long long cv) {
    unsafe {
        PthreadCond_* pc = (PthreadCond_*)alloc_raw_(128UL); // pthread_cond_t <= 128 bytes
        if (pc == (PthreadCond_*)0) return -1;
        int r = pthread_cond_init(pc, (PthreadAttr_*)0);
        if (r != 0) { dealloc_raw_((unsigned char*)pc); return r; }
        *cv = (unsigned long long)pc;
        return 0;
    }
}

inline int cond_destroy(&unsigned long long cv) {
    unsafe {
        PthreadCond_* pc = (PthreadCond_*)(*cv);
        int r = pthread_cond_destroy(pc);
        dealloc_raw_((unsigned char*)pc);
        *cv = 0ULL;
        return r;
    }
}

inline int cond_wait(&unsigned long long cv, &unsigned long long m) {
    unsafe { return pthread_cond_wait((PthreadCond_*)(*cv), (PthreadMutex_*)(*m)); }
}

inline int cond_timedwait_ms(&unsigned long long cv, &unsigned long long m, unsigned long ms) {
    unsafe {
        struct SafecTimespec_ ts;
        clock_gettime(0, &ts); // CLOCK_REALTIME = 0
        long long ns_add = (long long)ms * 1000000LL;
        long long nsec = (long long)ts.tv_nsec + ns_add;
        ts.tv_sec = ts.tv_sec + (SAFEC_TS_FIELD_)(nsec / 1000000000LL);
        ts.tv_nsec = (SAFEC_TS_FIELD_)(nsec % 1000000000LL);
        int r = pthread_cond_timedwait((PthreadCond_*)(*cv), (PthreadMutex_*)(*m), &ts);
        if (r == 0) return 0;
        return r == SAFEC_ETIMEDOUT_ ? THREAD_TIMEDOUT : -1;
    }
}

int cond_signal(&unsigned long long cv)    { unsafe { return pthread_cond_signal((PthreadCond_*)(*cv)); } }
int cond_broadcast(&unsigned long long cv) { unsafe { return pthread_cond_broadcast((PthreadCond_*)(*cv)); } }

inline int rwlock_init(&unsigned long long rw) {
    unsafe {
        PthreadRwlock_* pr = (PthreadRwlock_*)alloc_raw_(256UL); // pthread_rwlock_t <= 256 bytes (200 on macOS)
        if (pr == (PthreadRwlock_*)0) return -1;
        int r = pthread_rwlock_init(pr, (PthreadAttr_*)0);
        if (r != 0) { dealloc_raw_((unsigned char*)pr); return r; }
        *rw = (unsigned long long)pr;
        return 0;
    }
}

inline int rwlock_destroy(&unsigned long long rw) {
    unsafe {
        PthreadRwlock_* pr = (PthreadRwlock_*)(*rw);
        int r = pthread_rwlock_destroy(pr);
        dealloc_raw_((unsigned char*)pr);
        *rw = 0ULL;
        return r;
    }
}

int rwlock_rdlock(&unsigned long long rw)    { unsafe { return pthread_rwlock_rdlock((PthreadRwlock_*)(*rw)); } }
int rwlock_wrlock(&unsigned long long rw)    { unsafe { return pthread_rwlock_wrlock((PthreadRwlock_*)(*rw)); } }
int rwlock_tryrdlock(&unsigned long long rw) { unsafe { return pthread_rwlock_tryrdlock((PthreadRwlock_*)(*rw)); } }
int rwlock_trywrlock(&unsigned long long rw) { unsafe { return pthread_rwlock_trywrlock((PthreadRwlock_*)(*rw)); } }
int rwlock_rdunlock(&unsigned long long rw)  { unsafe { return pthread_rwlock_unlock((PthreadRwlock_*)(*rw)); } }
int rwlock_wrunlock(&unsigned long long rw)  { unsafe { return pthread_rwlock_unlock((PthreadRwlock_*)(*rw)); } }

#endif

} // namespace std
