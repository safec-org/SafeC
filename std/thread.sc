// SafeC Standard Library — Thread (cross-platform implementation)
#include "thread.h"
#include "mem.h"

// ════════════════════════════════ WIN32 ══════════════════════════════════════
#ifdef __WINDOWS__

// kernel32.dll — always linked on Windows
extern void* CreateThread(void* sec, unsigned long stack_sz, void* fn, void* arg,
                           unsigned long flags, unsigned long* tid_out);
extern unsigned long WaitForSingleObject(void* handle, unsigned long ms);
extern int CloseHandle(void* handle);
extern void Sleep(unsigned long ms);
extern int SwitchToThread();
extern unsigned long GetCurrentThreadId();

// CRITICAL_SECTION (40 bytes on x64; we alloc 64 to be safe)
extern void InitializeCriticalSection(void* cs);
extern void DeleteCriticalSection(void* cs);
extern void EnterCriticalSection(void* cs);
extern void LeaveCriticalSection(void* cs);
extern int  TryEnterCriticalSection(void* cs);

// CONDITION_VARIABLE (8 bytes on Vista+; we alloc 16)
extern void InitializeConditionVariable(void* cv);
extern int  SleepConditionVariableCS(void* cv, void* cs, unsigned long ms);
extern void WakeConditionVariable(void* cv);
extern void WakeAllConditionVariable(void* cv);

// SRWLOCK (8 bytes; we alloc 16)
extern void InitializeSRWLock(void* rw);
extern void AcquireSRWLockShared(void* rw);
extern void AcquireSRWLockExclusive(void* rw);
extern void ReleaseSRWLockShared(void* rw);
extern void ReleaseSRWLockExclusive(void* rw);
extern int  TryAcquireSRWLockShared(void* rw);
extern int  TryAcquireSRWLockExclusive(void* rw);

int thread_create(unsigned long long* tid, void* fn, void* arg) {
    unsafe {
        void* h = CreateThread((void*)0, 0UL, fn, arg, 0UL, (unsigned long*)0);
        if (h == (void*)0) return -1;
        *tid = (unsigned long long)h;
        return 0;
    }
}

int thread_join(unsigned long long tid) {
    unsafe {
        void* h = (void*)tid;
        unsigned long r = WaitForSingleObject(h, 4294967295UL); // INFINITE
        CloseHandle(h);
        return (int)r;
    }
}

int thread_detach(unsigned long long tid) {
    unsafe { return CloseHandle((void*)tid) ? 0 : -1; }
}

void thread_yield()                  { unsafe { SwitchToThread(); } }
void thread_sleep_ms(unsigned long ms) { unsafe { Sleep(ms); } }

unsigned long long thread_self() {
    unsafe { return (unsigned long long)GetCurrentThreadId(); }
}

int mutex_init(unsigned long long* m) {
    unsafe {
        void* cs = alloc(64UL);
        if (cs == (void*)0) return -1;
        InitializeCriticalSection(cs);
        *m = (unsigned long long)cs;
        return 0;
    }
}

int mutex_destroy(unsigned long long* m) {
    unsafe {
        void* cs = (void*)(*m);
        DeleteCriticalSection(cs);
        dealloc(cs);
        *m = 0ULL;
        return 0;
    }
}

int mutex_lock(unsigned long long* m)    { unsafe { EnterCriticalSection((void*)(*m)); return 0; } }
int mutex_trylock(unsigned long long* m) { unsafe { return TryEnterCriticalSection((void*)(*m)) ? 0 : 1; } }
int mutex_unlock(unsigned long long* m)  { unsafe { LeaveCriticalSection((void*)(*m)); return 0; } }

int cond_init(unsigned long long* cv) {
    unsafe {
        void* c = alloc(16UL);
        if (c == (void*)0) return -1;
        InitializeConditionVariable(c);
        *cv = (unsigned long long)c;
        return 0;
    }
}

int cond_destroy(unsigned long long* cv) {
    unsafe { dealloc((void*)(*cv)); *cv = 0ULL; return 0; }
}

int cond_wait(unsigned long long* cv, unsigned long long* m) {
    unsafe {
        return SleepConditionVariableCS((void*)(*cv), (void*)(*m), 4294967295UL) ? 0 : -1;
    }
}

int cond_timedwait_ms(unsigned long long* cv, unsigned long long* m, unsigned long ms) {
    unsafe {
        return SleepConditionVariableCS((void*)(*cv), (void*)(*m), (unsigned long)ms) ? 0 : -1;
    }
}

int cond_signal(unsigned long long* cv)    { unsafe { WakeConditionVariable((void*)(*cv)); return 0; } }
int cond_broadcast(unsigned long long* cv) { unsafe { WakeAllConditionVariable((void*)(*cv)); return 0; } }

int rwlock_init(unsigned long long* rw) {
    unsafe {
        void* r = alloc(16UL);
        if (r == (void*)0) return -1;
        InitializeSRWLock(r);
        *rw = (unsigned long long)r;
        return 0;
    }
}

int rwlock_destroy(unsigned long long* rw) {
    unsafe { dealloc((void*)(*rw)); *rw = 0ULL; return 0; }
}

int rwlock_rdlock(unsigned long long* rw)    { unsafe { AcquireSRWLockShared((void*)(*rw)); return 0; } }
int rwlock_wrlock(unsigned long long* rw)    { unsafe { AcquireSRWLockExclusive((void*)(*rw)); return 0; } }
int rwlock_tryrdlock(unsigned long long* rw) { unsafe { return TryAcquireSRWLockShared((void*)(*rw)) ? 0 : 1; } }
int rwlock_trywrlock(unsigned long long* rw) { unsafe { return TryAcquireSRWLockExclusive((void*)(*rw)) ? 0 : 1; } }
int rwlock_rdunlock(unsigned long long* rw)  { unsafe { ReleaseSRWLockShared((void*)(*rw)); return 0; } }
int rwlock_wrunlock(unsigned long long* rw)  { unsafe { ReleaseSRWLockExclusive((void*)(*rw)); return 0; } }

// ════════════════════════════════ POSIX ══════════════════════════════════════
#else

extern int pthread_create(unsigned long long* tid, void* attr, void* fn, void* arg);
extern int pthread_join(unsigned long long tid, void** retval);
extern int pthread_detach(unsigned long long tid);
extern unsigned long long pthread_self();
extern int sched_yield();
extern int nanosleep(void* req, void* rem);
extern int clock_gettime(int clk, void* ts);

extern int pthread_mutex_init(void* m, void* attr);
extern int pthread_mutex_destroy(void* m);
extern int pthread_mutex_lock(void* m);
extern int pthread_mutex_trylock(void* m);
extern int pthread_mutex_unlock(void* m);

extern int pthread_cond_init(void* cv, void* attr);
extern int pthread_cond_destroy(void* cv);
extern int pthread_cond_wait(void* cv, void* m);
extern int pthread_cond_timedwait(void* cv, void* m, void* ts);
extern int pthread_cond_signal(void* cv);
extern int pthread_cond_broadcast(void* cv);

extern int pthread_rwlock_init(void* rw, void* attr);
extern int pthread_rwlock_destroy(void* rw);
extern int pthread_rwlock_rdlock(void* rw);
extern int pthread_rwlock_wrlock(void* rw);
extern int pthread_rwlock_tryrdlock(void* rw);
extern int pthread_rwlock_trywrlock(void* rw);
extern int pthread_rwlock_unlock(void* rw);

int thread_create(unsigned long long* tid, void* fn, void* arg) {
    unsafe { return pthread_create(tid, (void*)0, fn, arg); }
}
int thread_join(unsigned long long tid)    { unsafe { return pthread_join(tid, (void**)0); } }
int thread_detach(unsigned long long tid)  { unsafe { return pthread_detach(tid); } }
void thread_yield()                        { unsafe { sched_yield(); } }
unsigned long long thread_self()           { unsafe { return pthread_self(); } }

void thread_sleep_ms(unsigned long ms) {
    unsafe {
        // struct timespec { time_t tv_sec; long tv_nsec; } = 16 bytes on 64-bit
        long long ts[2];
        ts[0] = (long long)(ms / 1000UL);
        ts[1] = (long long)((ms % 1000UL) * 1000000UL);
        nanosleep((void*)ts, (void*)0);
    }
}

int mutex_init(unsigned long long* m) {
    unsafe {
        void* pm = alloc(128UL); // pthread_mutex_t <= 128 bytes (64 on macOS, 40 on Linux)
        if (pm == (void*)0) return -1;
        int r = pthread_mutex_init(pm, (void*)0);
        if (r != 0) { dealloc(pm); return r; }
        *m = (unsigned long long)pm;
        return 0;
    }
}

int mutex_destroy(unsigned long long* m) {
    unsafe {
        void* pm = (void*)(*m);
        int r = pthread_mutex_destroy(pm);
        dealloc(pm);
        *m = 0ULL;
        return r;
    }
}

int mutex_lock(unsigned long long* m)    { unsafe { return pthread_mutex_lock((void*)(*m)); } }
int mutex_trylock(unsigned long long* m) { unsafe { return pthread_mutex_trylock((void*)(*m)); } }
int mutex_unlock(unsigned long long* m)  { unsafe { return pthread_mutex_unlock((void*)(*m)); } }

int cond_init(unsigned long long* cv) {
    unsafe {
        void* pc = alloc(128UL); // pthread_cond_t <= 128 bytes
        if (pc == (void*)0) return -1;
        int r = pthread_cond_init(pc, (void*)0);
        if (r != 0) { dealloc(pc); return r; }
        *cv = (unsigned long long)pc;
        return 0;
    }
}

int cond_destroy(unsigned long long* cv) {
    unsafe {
        void* pc = (void*)(*cv);
        int r = pthread_cond_destroy(pc);
        dealloc(pc);
        *cv = 0ULL;
        return r;
    }
}

int cond_wait(unsigned long long* cv, unsigned long long* m) {
    unsafe { return pthread_cond_wait((void*)(*cv), (void*)(*m)); }
}

int cond_timedwait_ms(unsigned long long* cv, unsigned long long* m, unsigned long ms) {
    unsafe {
        long long ts[2];
        clock_gettime(0, (void*)ts); // CLOCK_REALTIME = 0
        long long ns_add = (long long)ms * 1000000LL;
        ts[1] = ts[1] + ns_add;
        if (ts[1] >= 1000000000LL) {
            ts[0] = ts[0] + ts[1] / 1000000000LL;
            ts[1] = ts[1] % 1000000000LL;
        }
        return pthread_cond_timedwait((void*)(*cv), (void*)(*m), (void*)ts);
    }
}

int cond_signal(unsigned long long* cv)    { unsafe { return pthread_cond_signal((void*)(*cv)); } }
int cond_broadcast(unsigned long long* cv) { unsafe { return pthread_cond_broadcast((void*)(*cv)); } }

int rwlock_init(unsigned long long* rw) {
    unsafe {
        void* pr = alloc(256UL); // pthread_rwlock_t <= 256 bytes (200 on macOS)
        if (pr == (void*)0) return -1;
        int r = pthread_rwlock_init(pr, (void*)0);
        if (r != 0) { dealloc(pr); return r; }
        *rw = (unsigned long long)pr;
        return 0;
    }
}

int rwlock_destroy(unsigned long long* rw) {
    unsafe {
        void* pr = (void*)(*rw);
        int r = pthread_rwlock_destroy(pr);
        dealloc(pr);
        *rw = 0ULL;
        return r;
    }
}

int rwlock_rdlock(unsigned long long* rw)    { unsafe { return pthread_rwlock_rdlock((void*)(*rw)); } }
int rwlock_wrlock(unsigned long long* rw)    { unsafe { return pthread_rwlock_wrlock((void*)(*rw)); } }
int rwlock_tryrdlock(unsigned long long* rw) { unsafe { return pthread_rwlock_tryrdlock((void*)(*rw)); } }
int rwlock_trywrlock(unsigned long long* rw) { unsafe { return pthread_rwlock_trywrlock((void*)(*rw)); } }
int rwlock_rdunlock(unsigned long long* rw)  { unsafe { return pthread_rwlock_unlock((void*)(*rw)); } }
int rwlock_wrunlock(unsigned long long* rw)  { unsafe { return pthread_rwlock_unlock((void*)(*rw)); } }

#endif
