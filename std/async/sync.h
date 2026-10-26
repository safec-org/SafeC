// SafeC Standard Library — async: Mutex, Semaphore, Notify, cancellation
//
// Synchronization whose waiting is a future (tokio::sync's Mutex,
// Semaphore, Notify, and tokio_util's CancellationToken). Waiting tasks
// are Pending with their waker registered, never a blocked thread; every
// operation takes the primitive's spinlock, so they work across tasks,
// executors and threads. Nothing allocates. Like channels, a primitive
// must not move while futures from it exist.
//
// SafeC has no destructors, so there are no lock guards: whoever completes
// a lock()/acquire() future owns the lock/permits and calls unlock() /
// release() itself (holding them across awaits is what these are for).
//
//   self.lk = mtx.lock();          AWAIT(1, self.lk);   ... mtx.unlock();
//   self.acq = sem.acquire(2);     AWAIT(2, self.acq);  ... sem.release(2);
//   self.n = notify.notified();    AWAIT(3, self.n);    // notify.notify_one() elsewhere
//   self.c = token.cancelled();    AWAIT(4, self.c);    // token.cancel() elsewhere
//
//   struct Cancellable<struct Fetch, int> w = with_cancel(Fetch::new(), &token);
//   AWAIT_INTO(5, self.res, self.w);   // ?int: null if the token was cancelled first
#pragma once
#include <std/async/future.h>
#include <std/sync/spinlock.h>

namespace std {

// ── AsyncMutex ─────────────────────────────────────────────────────────────

struct AsyncMutex {
    unsafe impl Sync;   // state changes under its spinlock
    int             locked;
    struct WaitList waiters;
    struct Spinlock lk;

    static struct AsyncMutex new();
    // Future: Ready once this task holds the lock.
    struct MutexLock lock();
    int  try_lock();     // 1 if acquired
    void unlock();       // and wake a waiter
    int  is_locked() const;
};

struct MutexLock {
    ?&AsyncMutex m;
    int          wait_id;
    int          done;
    struct Poll<Unit> poll(&Context cx);
    void drop();
};

// ── AsyncSemaphore ─────────────────────────────────────────────────────────

struct AsyncSemaphore {
    unsafe impl Sync;   // state changes under its spinlock
    long long       permits;
    struct WaitList waiters;
    struct Spinlock lk;

    static struct AsyncSemaphore new(long long permits);
    // Future: Ready once 'n' permits are taken (all at once).
    struct SemAcquire acquire(long long n);
    int  try_acquire(long long n);    // 1 if taken
    void release(long long n);        // return permits, wake waiters
    long long available() const;
};

struct SemAcquire {
    ?&AsyncSemaphore s;
    long long        n;
    int              wait_id;
    int              done;
    struct Poll<Unit> poll(&Context cx);
    void drop();
};

// ── AsyncNotify ────────────────────────────────────────────────────────────
// notify_one() wakes one waiting notified() future, or — if none is
// waiting — leaves a single permit that the next notified() takes at once.
// notify_waiters() wakes every future waiting right now and leaves no
// permit (tokio::sync::Notify).

struct AsyncNotify {
    unsafe impl Sync;   // state changes under its spinlock
    int             permit;
    int             generation;   // bumped by notify_waiters
    int             pending_one;  // notify_one()s handed to woken waiters
    struct WaitList waiters;
    struct Spinlock lk;

    static struct AsyncNotify new();
    struct Notified notified();
    void notify_one();
    void notify_waiters();
};

struct Notified {
    ?&AsyncNotify n;
    int           generation;   // at the first poll
    int           started;
    int           wait_id;
    int           done;
    struct Poll<Unit> poll(&Context cx);
    void drop();
};

// ── CancelToken ────────────────────────────────────────────────────────────
// A flag that, once set, stays set, and wakes everything waiting for it.
// A child token is cancelled when it or any ancestor is (up to
// CANCEL_DEPTH levels); cancelling a child doesn't touch its parent.

#define CANCEL_DEPTH 4

struct CancelToken {
    unsafe impl Sync;   // state changes under its spinlock
    int             cancelled;    // atomic
    ?&CancelToken   parent;
    struct WaitList waiters;
    struct Spinlock lk;

    static struct CancelToken new();
    static struct CancelToken child_of(&CancelToken parent);
    void cancel();
    int  is_cancelled() const;
    // Future: Ready once cancelled.
    struct Cancelled cancelled_fut();
};

struct Cancelled {
    ?&CancelToken t;
    int           ids[CANCEL_DEPTH];   // registration in each token of the chain
    int           done;
    struct Poll<Unit> poll(&Context cx);
    void drop();
};

// 'inner' until it finishes or 'token' is cancelled, whichever is first;
// on cancellation 'inner' is dropped. Output: ?T, null if cancelled.
generic<F: Future, T>
struct Cancellable {
    F                inner;
    struct Cancelled c;
    int              finished;
    struct Poll<?T> poll(&Context cx);
    void drop();
};

generic<F, T>
struct Cancellable<F, T> with_cancel(F inner, &CancelToken token) {
    struct Cancellable<F, T> w;
    w.inner = inner;
    w.c = token.cancelled_fut();
    w.finished = 0;
    return w;
}

generic<F: Future, T>
struct Poll<?T> Cancellable::poll(&Context cx) {
    struct Poll<Unit> pc = self.c.poll(cx);
    if (pc.ready_ != 0) {
        self.inner.drop();
        self.finished = 1;
        ?T none = null;
        return Poll::ready(none);
    }
    struct Poll<T> p = self.inner.poll(cx);
    if (p.ready_ != 0) {
        self.c.drop();
        self.finished = 1;
        ?T v = p.value;
        return Poll::ready(v);
    }
    return Poll::pending();
}

generic<F: Future, T>
void Cancellable::drop() {
    if (self.finished == 0) { self.inner.drop(); self.c.drop(); self.finished = 1; }
}

} // namespace std
