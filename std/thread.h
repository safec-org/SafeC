#pragma once
// SafeC Standard Library — Thread
// Cross-platform: POSIX pthreads (Linux/macOS) or Win32 (Windows).
// Backend is selected automatically from '_WIN32', which safec auto-defines
// for a Windows host/target (see main.cpp's platform-detection block) —
// same convention as std/stderr_compat.h.
//
// Thread handles: unsigned long long
//   POSIX → pthread_t value (8 bytes on 64-bit)
//   Win32 → HANDLE (kernel object pointer, 8 bytes on x64)
//
// Mutex / condvar / rwlock handles: unsigned long long
//   POSIX → pointer to heap-allocated pthread object
//   Win32 → pointer to heap-allocated Windows primitive

// ── Thread ───────────────────────────────────────────────────────────────────
namespace std {

// OS-boundary entry point (POSIX/Win32 take an untyped start routine);
// call thread_create below instead.
int  thread_create_raw_(&unsigned long long tid, void* func, void* arg);

// Starts func(arg) on a new thread and stores its handle in *tid.
// 0 on success. func's return value is discarded (thread_join only waits).
//   static ?&Job work(?&Job j) { ... return null; }
//   unsigned long long tid = 0ULL;
//   std::thread_create(&tid, work, (?&Job)&job);
generic<T>
int thread_create(&unsigned long long tid, fn ?&T(?&T) func, ?&T arg) {
    unsafe { return thread_create_raw_(tid, (void*)func, (void*)arg); }
}
int  thread_join(unsigned long long tid);
int  thread_detach(unsigned long long tid);
void thread_yield();
void thread_sleep_ms(unsigned long ms);
unsigned long long thread_self();

// ── Mutex ─────────────────────────────────────────────────────────────────────
int mutex_init(&unsigned long long m);
int mutex_destroy(&unsigned long long m);
int mutex_lock(&unsigned long long m);
int mutex_trylock(&unsigned long long m);  // 0=locked, non-zero=busy
// Don't re-lock a mutex the calling thread already holds: a Windows
// CRITICAL_SECTION allows it (recursive), a default POSIX mutex deadlocks
// (mutex_lock) or reports busy (mutex_trylock).
int mutex_unlock(&unsigned long long m);

// ── Condition variable ────────────────────────────────────────────────────────
// cond_wait atomically releases mutex and sleeps; re-acquires on wake.
int cond_init(&unsigned long long cv);
int cond_destroy(&unsigned long long cv);
int cond_wait(&unsigned long long cv, &unsigned long long m);
// Waits at most 'ms' milliseconds: 0 if woken, THREAD_TIMEDOUT if the time
// ran out, -1 on another error (the same codes on every platform).
#define THREAD_TIMEDOUT 1
int cond_timedwait_ms(&unsigned long long cv, &unsigned long long m, unsigned long ms);
int cond_signal(&unsigned long long cv);
int cond_broadcast(&unsigned long long cv);

// ── Read-write lock ───────────────────────────────────────────────────────────
// Multiple simultaneous readers; exclusive writers.
int rwlock_init(&unsigned long long rw);
int rwlock_destroy(&unsigned long long rw);
int rwlock_rdlock(&unsigned long long rw);
int rwlock_wrlock(&unsigned long long rw);
int rwlock_tryrdlock(&unsigned long long rw);
int rwlock_trywrlock(&unsigned long long rw);
int rwlock_rdunlock(&unsigned long long rw);
int rwlock_wrunlock(&unsigned long long rw);

} // namespace std
