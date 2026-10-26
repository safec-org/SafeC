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

#include <std/mem.h>
#include <std/panic.h>
#include <std/ownership.h>

// ── Thread ───────────────────────────────────────────────────────────────────
#include <std/erased.h>

namespace std {

// OS-boundary entry point (POSIX/Win32 take an untyped start routine; see
// std/erased.h); call thread_create below instead.
int  thread_create_raw_(&unsigned long long tid, fn ?&Erased(?&Erased) func, ?&Erased arg);

// Starts func(arg) on a new thread and stores its handle in *tid.
// 0 on success. func's return value is discarded (thread_join only waits).
//   static ?&Job work(?&Job j) { ... return null; }
//   unsigned long long tid = 0ULL;
//   std::thread_create(&tid, work, (?&Job)&job);
generic<T>
int thread_create(&unsigned long long tid, fn ?&T(?&T) func, ?&T arg) {
    unsafe { return thread_create_raw_(tid, (fn ?&Erased(?&Erased))func, (?&Erased)arg); }
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


// ══ Typed, checked layer ═════════════════════════════════════════════════════
// The functions above are the OS primitives: they move untyped handles and
// nothing checks what the threads share. These build on them so the
// compiler can (see the language reference's "Thread safety" page):
//
//   * thread_spawn(f, arg) moves 'arg' (a Send value) into a new thread;
//     the returned SpawnedThread is a Drop value: join() it for f's result (drop()
//     joins and drops it). Nothing is shared, so nothing can race.
//   * Mutex<T> owns a T that is reachable only through the MutexGuard
//     lock() returns; the guard unlocks in drop(), and a reference from
//     guard.get() can't be used after the guard is dropped. A Mutex is Sync,
//     so threads may share one (a static Mutex, or a local lent to
//     spawn_scoped threads).
//   * RwLock<T>: many readers (read() guards give 'const &T') or one writer.

// ── thread_spawn / Thread ────────────────────────────────────────────────────

generic<T, R>
struct ThreadStart_ {
    fn R(T) f;
    T       arg;    // moved in by thread_spawn, moved out by the thread
    R       out;    // the thread's result, moved out by join()
};

generic<T, R>
?&ThreadStart_<T, R> thread_start_main_(?&ThreadStart_<T, R> sp) {
    match (sp) {
        case some(s): { unsafe { T a = s.arg; s.out = s.f(a); } }
        case null: { }
    }
    return null;
}

// A running thread started by thread_spawn, and its result. A Drop value:
// join() it (or drop() it, which joins and drops the result).
generic<T, R>
struct SpawnedThread {
    unsigned long long        tid_;
    ?&heap ThreadStart_<T, R> start_;   // null if the thread couldn't start
    impl !Send;                         // joined by the thread that started it

    int  started() const;               // 0 if the thread couldn't be created
    R    join() consume;                // waits; f's result (panics if it never started)
    void drop();                        // join() and drop the result
};

// Runs f(arg) on a new thread. 'arg' moves into it (using it afterwards is
// an error); T and R must be Send.
generic<T: Send, R: Send>
struct SpawnedThread<T, R> thread_spawn(fn R(T) f, T arg) {
    struct SpawnedThread<T, R> t;
    t.tid_ = 0ULL;
    t.start_ = null;
    ?&heap ThreadStart_<T, R> sp = null;
    unsafe { sp = (?&heap ThreadStart_<T, R>)alloc_raw_(sizeof(struct ThreadStart_<T, R>)); }
    match (sp) {
        case null: { drop(arg); return t; }
        case some(s): {
            unsafe { s.f = f; s.arg = arg; }   // arg moves into the start block
            fn ?&ThreadStart_<T, R>(?&ThreadStart_<T, R>) entry = thread_start_main_;
            if (thread_create(&t.tid_, entry, (?&ThreadStart_<T, R>)s) != 0) {
                unsafe { drop(s.arg); dealloc(s); }
                return t;
            }
            t.start_ = s;
        }
    }
    return t;
}

generic<T, R>
int SpawnedThread::started() const { return self.start_.is_null() ? 0 : 1; }

generic<T, R>
R SpawnedThread::join() {
    match (self.start_) {
        case some(s): {
            thread_join(self.tid_);
            self.start_ = null;
            unsafe {
                R r = s.out;   // moves out of the start block, which goes away
                dealloc(s);
                return r;
            }
        }
        case null: { }
    }
    panic_at("SpawnedThread::join: the thread never started", "thread.h", 0);
    unsafe { return *(R*)0; }   // unreachable
}

generic<T, R>
void SpawnedThread::drop() {
    if (self.start_.is_null()) return;
    R r = self.join();
    drop(r);
}

// ── Mutex<T> / MutexGuard<T> ─────────────────────────────────────────────────

// A T shared between threads, reachable only while locked. Moving a Mutex
// after locking it is not allowed (guards point at it).
generic<T: Send>
struct Mutex {
    unsigned long long raw_;     // the OS lock
    T*                 value_;   // heap box; only a guard reaches it (raw: unsafe to touch)
    unsafe impl Sync;            // every access to *value_ is under raw_

    static struct Mutex<T> new(T value);
    // Blocks until this thread holds the lock. The guard is the only way to
    // the value; dropping it unlocks. Don't lock twice on one thread.
    struct MutexGuard<T> lock() const;
    // The value back; the Mutex is gone (no guard may be alive).
    T    into_inner() consume;
    void drop();                 // drops the value, destroys the lock
};

// Proof of holding a Mutex's lock: get() is the protected value, valid
// until the guard is dropped (which unlocks).
generic<T>
struct MutexGuard {
    const &Mutex<T> m_;
    &T              value_;
    impl !Send;                  // must unlock on the locking thread

    &T   get();                  // the value: don't keep it past drop()
    void set(T value);           // replace it (dropping the old one)
    void drop();                 // unlocks
};

generic<T>
struct Mutex<T> Mutex::new(T value) {
    struct Mutex<T> m;
    m.raw_ = 0ULL;
    mutex_init(&m.raw_);
    unsafe {
        m.value_ = (T*)alloc_raw_(sizeof(T));
        if (m.value_ == (T*)0) panic_at("Mutex::new: out of memory", "thread.h", 0);
        *m.value_ = value;   // moves in
    }
    return m;
}

generic<T>
struct MutexGuard<T> Mutex::lock() const {
    unsafe { mutex_lock((&unsigned long long)&self.raw_); }
    struct MutexGuard<T> g;
    g.m_ = self;
    unsafe { g.value_ = (&T)self.value_; }
    return g;
}

generic<T>
T Mutex::into_inner() {
    unsafe {
        T v = *self.value_;   // moves out of the box
        dealloc(self.value_);
        mutex_destroy(&self.raw_);
        return v;
    }
}

generic<T>
void Mutex::drop() {
    unsafe {
        drop(*self.value_);
        dealloc(self.value_);
        self.value_ = (T*)0;
    }
    mutex_destroy(&self.raw_);
}

generic<T>
&T MutexGuard::get() { return self.value_; }

generic<T>
void MutexGuard::set(T value) { drop(std::replace(self.value_, value)); }

generic<T>
void MutexGuard::drop() { unsafe { mutex_unlock((&unsigned long long)&self.m_.raw_); } }

// ── RwLock<T> ────────────────────────────────────────────────────────────────

// Like Mutex<T>, with many readers at once (const access) or one writer.
// Readers share the value, so T must be Sync as well as Send.
generic<T: Send + Sync>
struct RwLock {
    unsigned long long raw_;
    T*                 value_;
    unsafe impl Sync;

    static struct RwLock<T> new(T value);
    struct RwReadGuard<T>  read() const;    // shared: get() is const
    struct RwWriteGuard<T> write() const;   // exclusive
    void drop();
};

generic<T>
struct RwReadGuard {
    const &RwLock<T> l_;
    const &T         value_;
    impl !Send;
    const &T get() const;
    void     drop();                // read-unlocks
};

generic<T>
struct RwWriteGuard {
    const &RwLock<T> l_;
    &T               value_;
    impl !Send;
    &T   get();
    void drop();                    // write-unlocks
};

generic<T>
struct RwLock<T> RwLock::new(T value) {
    struct RwLock<T> l;
    l.raw_ = 0ULL;
    rwlock_init(&l.raw_);
    unsafe {
        l.value_ = (T*)alloc_raw_(sizeof(T));
        if (l.value_ == (T*)0) panic_at("RwLock::new: out of memory", "thread.h", 0);
        *l.value_ = value;
    }
    return l;
}

generic<T>
struct RwReadGuard<T> RwLock::read() const {
    unsafe { rwlock_rdlock((&unsigned long long)&self.raw_); }
    struct RwReadGuard<T> g;
    g.l_ = self;
    unsafe { g.value_ = (const &T)self.value_; }
    return g;
}

generic<T>
struct RwWriteGuard<T> RwLock::write() const {
    unsafe { rwlock_wrlock((&unsigned long long)&self.raw_); }
    struct RwWriteGuard<T> g;
    g.l_ = self;
    unsafe { g.value_ = (&T)self.value_; }
    return g;
}

generic<T>
void RwLock::drop() {
    unsafe {
        drop(*self.value_);
        dealloc(self.value_);
        self.value_ = (T*)0;
    }
    rwlock_destroy(&self.raw_);
}

generic<T>
const &T RwReadGuard::get() const { return self.value_; }

generic<T>
void RwReadGuard::drop() { unsafe { rwlock_rdunlock((&unsigned long long)&self.l_.raw_); } }

generic<T>
&T RwWriteGuard::get() { return self.value_; }

generic<T>
void RwWriteGuard::drop() { unsafe { rwlock_wrunlock((&unsigned long long)&self.l_.raw_); } }

} // namespace std
