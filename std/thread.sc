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

extern unsigned char* CreateThread(unsigned char* sec, unsigned long stack_sz, unsigned char* func, unsigned char* arg,
                           unsigned int flags, unsigned int* tid_out);
extern unsigned int WaitForSingleObject(unsigned char* handle, unsigned int ms);
extern int CloseHandle(unsigned char* handle);
extern void Sleep(unsigned int ms);
extern int SwitchToThread();
extern unsigned int GetCurrentThreadId();

// CRITICAL_SECTION (40 bytes on x64; we alloc 64 to be safe)
extern void InitializeCriticalSection(unsigned char* cs);
extern void DeleteCriticalSection(unsigned char* cs);
extern void EnterCriticalSection(unsigned char* cs);
extern void LeaveCriticalSection(unsigned char* cs);
extern int  TryEnterCriticalSection(unsigned char* cs);

// CONDITION_VARIABLE (8 bytes on Vista+; we alloc 16)
extern void InitializeConditionVariable(unsigned char* cv);
extern int  SleepConditionVariableCS(unsigned char* cv, unsigned char* cs, unsigned int ms);
extern void WakeConditionVariable(unsigned char* cv);
extern void WakeAllConditionVariable(unsigned char* cv);
extern unsigned int GetLastError();

// SRWLOCK (8 bytes; we alloc 16)
extern void InitializeSRWLock(unsigned char* rw);
extern void AcquireSRWLockShared(unsigned char* rw);
extern void AcquireSRWLockExclusive(unsigned char* rw);
extern void ReleaseSRWLockShared(unsigned char* rw);
extern void ReleaseSRWLockExclusive(unsigned char* rw);
extern int  TryAcquireSRWLockShared(unsigned char* rw);
extern int  TryAcquireSRWLockExclusive(unsigned char* rw);

inline int thread_create_raw_(&unsigned long long tid, fn ?&Erased(?&Erased) func, ?&Erased arg) {
    unsafe {
        unsigned char* h = CreateThread((unsigned char*)0, 0UL, (unsigned char*)func, (unsigned char*)arg, 0u, (unsigned int*)0);
        if (h == (unsigned char*)0) return -1;
        *tid = (unsigned long long)h;
        return 0;
    }
}

inline int thread_join(unsigned long long tid) {
    unsafe {
        unsigned char* h = (unsigned char*)tid;
        unsigned int r = WaitForSingleObject(h, 0xFFFFFFFFu); // INFINITE
        CloseHandle(h);
        return (int)r;
    }
}

inline int thread_detach(unsigned long long tid) {
    unsafe { return CloseHandle((unsigned char*)tid) ? 0 : -1; }
}

void thread_yield()                  { unsafe { SwitchToThread(); } }
void thread_sleep_ms(unsigned long ms) { unsafe { Sleep(ms > 0xFFFFFFFEUL ? 0xFFFFFFFEu : (unsigned int)ms); } }

inline unsigned long long thread_self() {
    unsafe { return (unsigned long long)GetCurrentThreadId(); }
}

inline int mutex_init(&unsigned long long m) {
    unsafe {
        unsigned char* cs = alloc_raw_(64UL);
        if (cs == (unsigned char*)0) return -1;
        InitializeCriticalSection(cs);
        *m = (unsigned long long)cs;
        return 0;
    }
}

inline int mutex_destroy(&unsigned long long m) {
    unsafe {
        unsigned char* cs = (unsigned char*)(*m);
        DeleteCriticalSection(cs);
        dealloc_raw_(cs);
        *m = 0ULL;
        return 0;
    }
}

int mutex_lock(&unsigned long long m)    { unsafe { EnterCriticalSection((unsigned char*)(*m)); return 0; } }
int mutex_trylock(&unsigned long long m) { unsafe { return TryEnterCriticalSection((unsigned char*)(*m)) ? 0 : 1; } }
int mutex_unlock(&unsigned long long m)  { unsafe { LeaveCriticalSection((unsigned char*)(*m)); return 0; } }

inline int cond_init(&unsigned long long cv) {
    unsafe {
        unsigned char* c = alloc_raw_(16UL);
        if (c == (unsigned char*)0) return -1;
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
        return SleepConditionVariableCS((unsigned char*)(*cv), (unsigned char*)(*m), 0xFFFFFFFFu) ? 0 : -1;
    }
}

inline int cond_timedwait_ms(&unsigned long long cv, &unsigned long long m, unsigned long ms) {
    unsafe {
        if (SleepConditionVariableCS((unsigned char*)(*cv), (unsigned char*)(*m), ms > 0xFFFFFFFEUL ? 0xFFFFFFFEu : (unsigned int)ms)) return 0;
        return GetLastError() == 1460u ? THREAD_TIMEDOUT : -1;   // ERROR_TIMEOUT
    }
}

int cond_signal(&unsigned long long cv)    { unsafe { WakeConditionVariable((unsigned char*)(*cv)); return 0; } }
int cond_broadcast(&unsigned long long cv) { unsafe { WakeAllConditionVariable((unsigned char*)(*cv)); return 0; } }

inline int rwlock_init(&unsigned long long rw) {
    unsafe {
        unsigned char* r = alloc_raw_(16UL);
        if (r == (unsigned char*)0) return -1;
        InitializeSRWLock(r);
        *rw = (unsigned long long)r;
        return 0;
    }
}

inline int rwlock_destroy(&unsigned long long rw) {
    unsafe { dealloc_raw_((unsigned char*)(*rw)); *rw = 0ULL; return 0; }
}

int rwlock_rdlock(&unsigned long long rw)    { unsafe { AcquireSRWLockShared((unsigned char*)(*rw)); return 0; } }
int rwlock_wrlock(&unsigned long long rw)    { unsafe { AcquireSRWLockExclusive((unsigned char*)(*rw)); return 0; } }
int rwlock_tryrdlock(&unsigned long long rw) { unsafe { return TryAcquireSRWLockShared((unsigned char*)(*rw)) ? 0 : 1; } }
int rwlock_trywrlock(&unsigned long long rw) { unsafe { return TryAcquireSRWLockExclusive((unsigned char*)(*rw)) ? 0 : 1; } }
int rwlock_rdunlock(&unsigned long long rw)  { unsafe { ReleaseSRWLockShared((unsigned char*)(*rw)); return 0; } }
int rwlock_wrunlock(&unsigned long long rw)  { unsafe { ReleaseSRWLockExclusive((unsigned char*)(*rw)); return 0; } }

// ════════════════════════════════ POSIX ══════════════════════════════════════
#else

extern int pthread_create(unsigned long long* tid, unsigned char* attr, unsigned char* func, unsigned char* arg);
extern int pthread_join(unsigned long long tid, unsigned char** retval);
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
extern int nanosleep(unsigned char* req, unsigned char* rem);
extern int clock_gettime(int clk, unsigned char* ts);

extern int pthread_mutex_init(unsigned char* m, unsigned char* attr);
extern int pthread_mutex_destroy(unsigned char* m);
extern int pthread_mutex_lock(unsigned char* m);
extern int pthread_mutex_trylock(unsigned char* m);
extern int pthread_mutex_unlock(unsigned char* m);

extern int pthread_cond_init(unsigned char* cv, unsigned char* attr);
extern int pthread_cond_destroy(unsigned char* cv);
extern int pthread_cond_wait(unsigned char* cv, unsigned char* m);
extern int pthread_cond_timedwait(unsigned char* cv, unsigned char* m, unsigned char* ts);
extern int pthread_cond_signal(unsigned char* cv);
extern int pthread_cond_broadcast(unsigned char* cv);

extern int pthread_rwlock_init(unsigned char* rw, unsigned char* attr);
extern int pthread_rwlock_destroy(unsigned char* rw);
extern int pthread_rwlock_rdlock(unsigned char* rw);
extern int pthread_rwlock_wrlock(unsigned char* rw);
extern int pthread_rwlock_tryrdlock(unsigned char* rw);
extern int pthread_rwlock_trywrlock(unsigned char* rw);
extern int pthread_rwlock_unlock(unsigned char* rw);

inline int thread_create_raw_(&unsigned long long tid, fn ?&Erased(?&Erased) func, ?&Erased arg) {
    unsafe { return pthread_create((unsigned long long*)tid, (unsigned char*)0, (unsigned char*)func, (unsigned char*)arg); }
}
int thread_join(unsigned long long tid)    { unsafe { return pthread_join(tid, (unsigned char**)0); } }
int thread_detach(unsigned long long tid)  { unsafe { return pthread_detach(tid); } }
void thread_yield()                        { unsafe { sched_yield(); } }
unsigned long long thread_self()           { unsafe { return pthread_self(); } }

inline void thread_sleep_ms(unsigned long ms) {
    unsafe {
        struct SafecTimespec_ ts;
        ts.tv_sec = (SAFEC_TS_FIELD_)(ms / 1000UL);
        ts.tv_nsec = (SAFEC_TS_FIELD_)((ms % 1000UL) * 1000000UL);
        nanosleep((unsigned char*)&ts, (unsigned char*)0);
    }
}

inline int mutex_init(&unsigned long long m) {
    unsafe {
        unsigned char* pm = alloc_raw_(128UL); // pthread_mutex_t <= 128 bytes (64 on macOS, 40 on Linux)
        if (pm == (unsigned char*)0) return -1;
        int r = pthread_mutex_init(pm, (unsigned char*)0);
        if (r != 0) { dealloc_raw_(pm); return r; }
        *m = (unsigned long long)pm;
        return 0;
    }
}

inline int mutex_destroy(&unsigned long long m) {
    unsafe {
        unsigned char* pm = (unsigned char*)(*m);
        int r = pthread_mutex_destroy(pm);
        dealloc_raw_(pm);
        *m = 0ULL;
        return r;
    }
}

int mutex_lock(&unsigned long long m)    { unsafe { return pthread_mutex_lock((unsigned char*)(*m)); } }
int mutex_trylock(&unsigned long long m) { unsafe { return pthread_mutex_trylock((unsigned char*)(*m)); } }
int mutex_unlock(&unsigned long long m)  { unsafe { return pthread_mutex_unlock((unsigned char*)(*m)); } }

inline int cond_init(&unsigned long long cv) {
    unsafe {
        unsigned char* pc = alloc_raw_(128UL); // pthread_cond_t <= 128 bytes
        if (pc == (unsigned char*)0) return -1;
        int r = pthread_cond_init(pc, (unsigned char*)0);
        if (r != 0) { dealloc_raw_(pc); return r; }
        *cv = (unsigned long long)pc;
        return 0;
    }
}

inline int cond_destroy(&unsigned long long cv) {
    unsafe {
        unsigned char* pc = (unsigned char*)(*cv);
        int r = pthread_cond_destroy(pc);
        dealloc_raw_(pc);
        *cv = 0ULL;
        return r;
    }
}

inline int cond_wait(&unsigned long long cv, &unsigned long long m) {
    unsafe { return pthread_cond_wait((unsigned char*)(*cv), (unsigned char*)(*m)); }
}

inline int cond_timedwait_ms(&unsigned long long cv, &unsigned long long m, unsigned long ms) {
    unsafe {
        struct SafecTimespec_ ts;
        clock_gettime(0, (unsigned char*)&ts); // CLOCK_REALTIME = 0
        long long ns_add = (long long)ms * 1000000LL;
        long long nsec = (long long)ts.tv_nsec + ns_add;
        ts.tv_sec = ts.tv_sec + (SAFEC_TS_FIELD_)(nsec / 1000000000LL);
        ts.tv_nsec = (SAFEC_TS_FIELD_)(nsec % 1000000000LL);
        int r = pthread_cond_timedwait((unsigned char*)(*cv), (unsigned char*)(*m), (unsigned char*)&ts);
        if (r == 0) return 0;
        return r == SAFEC_ETIMEDOUT_ ? THREAD_TIMEDOUT : -1;
    }
}

int cond_signal(&unsigned long long cv)    { unsafe { return pthread_cond_signal((unsigned char*)(*cv)); } }
int cond_broadcast(&unsigned long long cv) { unsafe { return pthread_cond_broadcast((unsigned char*)(*cv)); } }

inline int rwlock_init(&unsigned long long rw) {
    unsafe {
        unsigned char* pr = alloc_raw_(256UL); // pthread_rwlock_t <= 256 bytes (200 on macOS)
        if (pr == (unsigned char*)0) return -1;
        int r = pthread_rwlock_init(pr, (unsigned char*)0);
        if (r != 0) { dealloc_raw_(pr); return r; }
        *rw = (unsigned long long)pr;
        return 0;
    }
}

inline int rwlock_destroy(&unsigned long long rw) {
    unsafe {
        unsigned char* pr = (unsigned char*)(*rw);
        int r = pthread_rwlock_destroy(pr);
        dealloc_raw_(pr);
        *rw = 0ULL;
        return r;
    }
}

int rwlock_rdlock(&unsigned long long rw)    { unsafe { return pthread_rwlock_rdlock((unsigned char*)(*rw)); } }
int rwlock_wrlock(&unsigned long long rw)    { unsafe { return pthread_rwlock_wrlock((unsigned char*)(*rw)); } }
int rwlock_tryrdlock(&unsigned long long rw) { unsafe { return pthread_rwlock_tryrdlock((unsigned char*)(*rw)); } }
int rwlock_trywrlock(&unsigned long long rw) { unsafe { return pthread_rwlock_trywrlock((unsigned char*)(*rw)); } }
int rwlock_rdunlock(&unsigned long long rw)  { unsafe { return pthread_rwlock_unlock((unsigned char*)(*rw)); } }
int rwlock_wrunlock(&unsigned long long rw)  { unsafe { return pthread_rwlock_unlock((unsigned char*)(*rw)); } }

#endif

} // namespace std
