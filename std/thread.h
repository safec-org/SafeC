#pragma once
// SafeC Standard Library — Thread
// Cross-platform: POSIX pthreads (Linux/macOS) or Win32 (Windows).
// Compile with -D__WINDOWS__ to select the Win32 backend.
//
// Thread handles: unsigned long long
//   POSIX → pthread_t value (8 bytes on 64-bit)
//   Win32 → HANDLE (kernel object pointer, 8 bytes on x64)
//
// Mutex / condvar / rwlock handles: unsigned long long
//   POSIX → pointer to heap-allocated pthread object
//   Win32 → pointer to heap-allocated Windows primitive

// ── Thread ───────────────────────────────────────────────────────────────────
// fn signature: void* fn(void*) on POSIX; DWORD WINAPI fn(LPVOID) on Win32.
int  thread_create(unsigned long long* tid, void* fn, void* arg);
int  thread_join(unsigned long long tid);
int  thread_detach(unsigned long long tid);
void thread_yield();
void thread_sleep_ms(unsigned long ms);
unsigned long long thread_self();

// ── Mutex ─────────────────────────────────────────────────────────────────────
int mutex_init(unsigned long long* m);
int mutex_destroy(unsigned long long* m);
int mutex_lock(unsigned long long* m);
int mutex_trylock(unsigned long long* m);  // 0=locked, non-zero=busy
int mutex_unlock(unsigned long long* m);

// ── Condition variable ────────────────────────────────────────────────────────
// cond_wait atomically releases mutex and sleeps; re-acquires on wake.
int cond_init(unsigned long long* cv);
int cond_destroy(unsigned long long* cv);
int cond_wait(unsigned long long* cv, unsigned long long* m);
int cond_timedwait_ms(unsigned long long* cv, unsigned long long* m, unsigned long ms);
int cond_signal(unsigned long long* cv);
int cond_broadcast(unsigned long long* cv);

// ── Read-write lock ───────────────────────────────────────────────────────────
// Multiple simultaneous readers; exclusive writers.
int rwlock_init(unsigned long long* rw);
int rwlock_destroy(unsigned long long* rw);
int rwlock_rdlock(unsigned long long* rw);
int rwlock_wrlock(unsigned long long* rw);
int rwlock_tryrdlock(unsigned long long* rw);
int rwlock_trywrlock(unsigned long long* rw);
int rwlock_rdunlock(unsigned long long* rw);
int rwlock_wrunlock(unsigned long long* rw);
