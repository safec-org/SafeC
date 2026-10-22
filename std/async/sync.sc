// SafeC Standard Library — async: AsyncMutex, AsyncSemaphore, AsyncNotify, CancelToken
#pragma once
#include <std/async/sync.h>

namespace std {

// ── AsyncMutex ─────────────────────────────────────────────────────────────

struct AsyncMutex AsyncMutex::new() {
    struct AsyncMutex m;
    m.locked = 0;
    m.waiters = WaitList::new();
    m.lk = Spinlock::new();
    return m;
}

struct MutexLock AsyncMutex::lock() {
    struct MutexLock l;
    l.m = self;
    l.wait_id = 0;
    l.done = 0;
    return l;
}

int AsyncMutex::try_lock() {
    self.lk.lock();
    int ok = self.locked == 0 ? 1 : 0;
    if (ok != 0) self.locked = 1;
    self.lk.unlock();
    return ok;
}

void AsyncMutex::unlock() {
    self.lk.lock();
    self.locked = 0;
    // Wake one: it re-polls and takes the lock (or, if someone else got it
    // first, waits again). A waiter that was woken but is dropped before
    // re-polling passes the wake on (see MutexLock::drop).
    self.waiters.wake_one();
    self.lk.unlock();
}

int AsyncMutex::is_locked() const { return self.locked; }

struct Poll<Unit> MutexLock::poll(&Context cx) {
    if (self.done != 0) return Poll::ready(UNIT);
    match (self.m) {
        case null: { self.done = 1; return Poll::ready(UNIT); }
        case some(m): {
            m.lk.lock();
            if (m.locked == 0) {
                m.locked = 1;
                m.waiters.remove(self.wait_id);
                self.wait_id = 0;
                self.done = 1;
                m.lk.unlock();
                return Poll::ready(UNIT);
            }
            self.wait_id = m.waiters.add(self.wait_id, &cx.waker);
            m.lk.unlock();
            if (self.wait_id == 0) cx.wake_by_ref();
            return Poll::pending();
        }
    }
    return Poll::ready(UNIT);
}

void MutexLock::drop() {
    match (self.m) {
        case some(m): {
            m.lk.lock();
            if (self.done == 0 && self.wait_id != 0 && m.waiters.contains(self.wait_id) == 0 && m.locked == 0) {
                // We were woken to take the lock but are going away: pass it on.
                m.waiters.wake_one();
            }
            m.waiters.remove(self.wait_id);
            m.lk.unlock();
        }
        case null: { }
    }
    self.wait_id = 0;
    self.done = 1;
}

// ── AsyncSemaphore ─────────────────────────────────────────────────────────

struct AsyncSemaphore AsyncSemaphore::new(long long permits) {
    struct AsyncSemaphore s;
    s.permits = permits;
    s.waiters = WaitList::new();
    s.lk = Spinlock::new();
    return s;
}

struct SemAcquire AsyncSemaphore::acquire(long long n) {
    struct SemAcquire a;
    a.s = self;
    a.n = n;
    a.wait_id = 0;
    a.done = 0;
    return a;
}

int AsyncSemaphore::try_acquire(long long n) {
    self.lk.lock();
    int ok = self.permits >= n ? 1 : 0;
    if (ok != 0) self.permits = self.permits - n;
    self.lk.unlock();
    return ok;
}

void AsyncSemaphore::release(long long n) {
    self.lk.lock();
    self.permits = self.permits + n;
    // Waiters may want different counts: let all of them look.
    self.waiters.wake_all();
    self.lk.unlock();
}

long long AsyncSemaphore::available() const { return self.permits; }

struct Poll<Unit> SemAcquire::poll(&Context cx) {
    if (self.done != 0) return Poll::ready(UNIT);
    match (self.s) {
        case null: { self.done = 1; return Poll::ready(UNIT); }
        case some(s): {
            s.lk.lock();
            if (s.permits >= self.n) {
                s.permits = s.permits - self.n;
                s.waiters.remove(self.wait_id);
                self.wait_id = 0;
                self.done = 1;
                s.lk.unlock();
                return Poll::ready(UNIT);
            }
            self.wait_id = s.waiters.add(self.wait_id, &cx.waker);
            s.lk.unlock();
            if (self.wait_id == 0) cx.wake_by_ref();
            return Poll::pending();
        }
    }
    return Poll::ready(UNIT);
}

void SemAcquire::drop() {
    match (self.s) {
        case some(s): { s.lk.lock(); s.waiters.remove(self.wait_id); s.lk.unlock(); }
        case null: { }
    }
    self.wait_id = 0;
    self.done = 1;
}

// ── AsyncNotify ────────────────────────────────────────────────────────────

struct AsyncNotify AsyncNotify::new() {
    struct AsyncNotify n;
    n.permit = 0;
    n.generation = 0;
    n.pending_one = 0;
    n.waiters = WaitList::new();
    n.lk = Spinlock::new();
    return n;
}

struct Notified AsyncNotify::notified() {
    struct Notified f;
    f.n = self;
    f.generation = 0;
    f.started = 0;
    f.wait_id = 0;
    f.done = 0;
    return f;
}

void AsyncNotify::notify_one() {
    self.lk.lock();
    if (self.waiters.len() > self.pending_one) {
        self.pending_one = self.pending_one + 1;
        self.waiters.wake_one();
    } else {
        self.permit = 1;
    }
    self.lk.unlock();
}

void AsyncNotify::notify_waiters() {
    self.lk.lock();
    self.generation = self.generation + 1;
    self.waiters.wake_all();
    self.lk.unlock();
}

struct Poll<Unit> Notified::poll(&Context cx) {
    if (self.done != 0) return Poll::ready(UNIT);
    match (self.n) {
        case null: { self.done = 1; return Poll::ready(UNIT); }
        case some(n): {
            n.lk.lock();
            if (self.started == 0) {
                self.started = 1;
                self.generation = n.generation;
                if (n.permit != 0) {
                    n.permit = 0;
                    self.done = 1;
                    n.lk.unlock();
                    return Poll::ready(UNIT);
                }
            }
            int woken = self.wait_id != 0 && n.waiters.contains(self.wait_id) == 0 ? 1 : 0;
            if (n.generation != self.generation) {
                self.done = 1;
            } else if (woken != 0 && n.pending_one > 0) {
                n.pending_one = n.pending_one - 1;
                self.done = 1;
            }
            if (self.done != 0) {
                n.waiters.remove(self.wait_id);
                self.wait_id = 0;
                n.lk.unlock();
                return Poll::ready(UNIT);
            }
            self.wait_id = n.waiters.add(woken != 0 ? 0 : self.wait_id, &cx.waker);
            n.lk.unlock();
            if (self.wait_id == 0) cx.wake_by_ref();
            return Poll::pending();
        }
    }
    return Poll::ready(UNIT);
}

void Notified::drop() {
    match (self.n) {
        case some(n): {
            n.lk.lock();
            if (self.done == 0 && self.wait_id != 0 && n.waiters.contains(self.wait_id) == 0 &&
                n.pending_one > 0) {
                // Woken by notify_one but going away: hand it to another waiter.
                n.pending_one = n.pending_one - 1;
                if (n.waiters.len() > n.pending_one) {
                    n.pending_one = n.pending_one + 1;
                    n.waiters.wake_one();
                } else {
                    n.permit = 1;
                }
            }
            n.waiters.remove(self.wait_id);
            n.lk.unlock();
        }
        case null: { }
    }
    self.wait_id = 0;
    self.done = 1;
}

// ── CancelToken ────────────────────────────────────────────────────────────

struct CancelToken CancelToken::new() {
    struct CancelToken t;
    t.cancelled = 0;
    t.parent = null;
    t.waiters = WaitList::new();
    t.lk = Spinlock::new();
    return t;
}

struct CancelToken CancelToken::child_of(&CancelToken parent) {
    struct CancelToken t = CancelToken::new();
    t.parent = parent;
    return t;
}

void CancelToken::cancel() {
    self.lk.lock();
    atomic_store_int(&self.cancelled, 1);
    self.waiters.wake_all();
    self.lk.unlock();
}

int CancelToken::is_cancelled() const {
    if (atomic_load_int(&self.cancelled) != 0) return 1;
    ?&CancelToken p = self.parent;
    int depth = 1;
    while (depth < CANCEL_DEPTH) {
        match (p) {
            case null: return 0;
            case some(t): {
                if (atomic_load_int(&t.cancelled) != 0) return 1;
                p = t.parent;
            }
        }
        depth = depth + 1;
    }
    return 0;
}

struct Cancelled CancelToken::cancelled_fut() {
    struct Cancelled c;
    c.t = self;
    int i = 0;
    while (i < CANCEL_DEPTH) { c.ids[i] = 0; i = i + 1; }
    c.done = 0;
    return c;
}

// Applies f to each token of the chain starting at 't' (index = depth).
static void cancelled_unregister_(&Cancelled c) {
    ?&CancelToken p = c.t;
    int i = 0;
    while (i < CANCEL_DEPTH) {
        match (p) {
            case null: return;
            case some(t): {
                t.lk.lock();
                t.waiters.remove(c.ids[i]);
                t.lk.unlock();
                c.ids[i] = 0;
                p = t.parent;
            }
        }
        i = i + 1;
    }
}

struct Poll<Unit> Cancelled::poll(&Context cx) {
    if (self.done != 0) return Poll::ready(UNIT);
    match (self.t) {
        case null: { self.done = 1; return Poll::ready(UNIT); }
        case some(t0): {
            if (t0.is_cancelled() != 0) {
                cancelled_unregister_(self);
                self.done = 1;
                return Poll::ready(UNIT);
            }
            // Register with every token in the chain, then look again (a
            // cancel between the check and the registration still wakes us
            // or is seen here).
            ?&CancelToken p = self.t;
            int busy = 0;
            int i = 0;
            while (i < CANCEL_DEPTH) {
                match (p) {
                    case null: i = CANCEL_DEPTH;
                    case some(t): {
                        t.lk.lock();
                        self.ids[i] = t.waiters.add(self.ids[i], &cx.waker);
                        if (self.ids[i] == 0) busy = 1;
                        t.lk.unlock();
                        p = t.parent;
                        i = i + 1;
                    }
                }
            }
            if (t0.is_cancelled() != 0) {
                cancelled_unregister_(self);
                self.done = 1;
                return Poll::ready(UNIT);
            }
            if (busy != 0) cx.wake_by_ref();
            return Poll::pending();
        }
    }
    return Poll::ready(UNIT);
}

void Cancelled::drop() {
    cancelled_unregister_(self);
    self.done = 1;
}

} // namespace std
