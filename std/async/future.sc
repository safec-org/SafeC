// SafeC Standard Library — async: Waker, Context, WakerSlot, WaitList, YieldNow
#pragma once
#include <std/async/future.h>

namespace std {

// ── atomics ────────────────────────────────────────────────────────────────

int async_load_(const &int p) { unsafe { return atomic_load((const _Atomic int*)p); } }
void async_store_(&int p, int v) { unsafe { atomic_store((_Atomic int*)p, v); } }
int async_xchg_(&int p, int v) { unsafe { return atomic_exchange((_Atomic int*)p, v); } }
int async_fetch_or_(&int p, int bits) { unsafe { return atomic_fetch_or((_Atomic int*)p, bits); } }
int async_fetch_and_(&int p, int bits) { unsafe { return atomic_fetch_and((_Atomic int*)p, bits); } }

// ── Waker ──────────────────────────────────────────────────────────────────

static void waker_noop_fn_(?&int data, unsigned long tag) { }

struct Waker Waker::empty_() {
    struct Waker w;
    w.wake_  = null;
    w.clone_ = null;
    w.drop_  = null;
    w.data_  = null;
    w.tag_   = 0UL;
    return w;
}

struct Waker Waker::noop() {
    return Waker::new(waker_noop_fn_, (?&int)null, 0UL);
}

void Waker::wake_by_ref() const {
    match (self.wake_) {
        case some(f): f(self.data_, self.tag_);
        case null: { }
    }
}

void Waker::wake() {
    self.wake_by_ref();
    self.drop();
}

struct Waker Waker::clone() const {
    match (self.clone_) {
        case some(f): f(self.data_, self.tag_);
        case null: { }
    }
    unsafe { return *self; }   // a bitwise copy: clone_ took the reference it holds
}

void Waker::drop() {
    match (self.drop_) {
        case some(f): f(self.data_, self.tag_);
        case null: { }
    }
    // A dropped waker wakes nothing (and drops nothing) again.
    self.wake_ = null;
    self.drop_ = null;
    self.clone_ = null;
}

int Waker::will_wake(const &Waker other) const {
    unsafe {
        return (self.data_ == other.data_ && self.wake_ == other.wake_ &&
                self.tag_ == other.tag_) ? 1 : 0;
    }
}

int Waker::is_noop() const {
    fn void(?&int, unsigned long) n = waker_noop_fn_;
    unsafe { return (self.wake_ == (?fn void(?&Erased, unsigned long))n || self.wake_.is_null()) ? 1 : 0; }
}

// ── Context ────────────────────────────────────────────────────────────────

struct Context Context::from_waker(struct Waker w) {
    struct Context cx;
    cx.waker  = w;
    cx.timers = null;
    cx.io_    = null;
    cx.exec_  = null;
    cx.spawn_ = null;
    return cx;
}

struct Context Context::noop() {
    return Context::from_waker(Waker::noop());
}

void Context::drop() { self.waker.drop(); }

void Context::wake_by_ref() const {
    self.waker.wake_by_ref();
}

// ── WakerSlot ──────────────────────────────────────────────────────────────

struct WakerSlot WakerSlot::new() {
    struct WakerSlot s;
    s.w = Waker::empty_();
    s.set = 0;
    return s;
}

void WakerSlot::register_(const &Waker w) {
    if (self.set != 0) {
        if (self.w.will_wake(w) != 0) return;
    }
    std::replace(&self.w, w.clone()).drop();   // the old waker, or the empty one
    self.set = 1;
}

int WakerSlot::wake() {
    if (self.set == 0) return 0;
    self.set = 0;
    struct Waker w = std::replace(&self.w, Waker::empty_());
    w.wake();
    return 1;
}

void WakerSlot::drop() {
    if (self.set != 0) { self.set = 0; self.w.drop(); }
}

void WakerSlot::clear() {
    if (self.set == 0) return;
    self.set = 0;
    std::replace(&self.w, Waker::noop()).drop();
}

// ── WaitList ───────────────────────────────────────────────────────────────
// id = serial * 64 + slot + 1 (ASYNC_WAITERS <= 63).

static int waitlist_slot_(int id) { return (id & 63) - 1; }

struct WaitList WaitList::new() {
    struct WaitList l;
    int i = 0;
    while (i < ASYNC_WAITERS) {
        l.w[i] = Waker::empty_();
        l.serial[i] = 0;
        i = i + 1;
    }
    l.count = 0;
    l.next_serial = 1;
    l.next_ = 0U;
    return l;
}

int WaitList::contains(int id) const {
    if (id <= 0) return 0;
    int slot = waitlist_slot_(id);
    if (slot < 0 || slot >= ASYNC_WAITERS) return 0;
    return (self.serial[slot] != 0 && self.serial[slot] == id / 64) ? 1 : 0;
}

int WaitList::add(int id, const &Waker w) {
    if (self.contains(id) != 0) {
        int slot = waitlist_slot_(id);
        if (self.w[slot].will_wake(w) == 0) std::replace(&self.w[slot], w.clone()).drop();
        return id;
    }
    int i = 0;
    while (i < ASYNC_WAITERS) {
        if (self.serial[i] == 0) {
            int ser = self.next_serial;
            self.next_serial = self.next_serial >= 16777215 ? 1 : self.next_serial + 1;
            self.serial[i] = ser;
            self.w[i] = w.clone();
            self.count = self.count + 1;
            return ser * 64 + i + 1;
        }
        i = i + 1;
    }
    return 0;
}

void WaitList::remove(int id) {
    if (self.contains(id) == 0) return;
    int slot = waitlist_slot_(id);
    self.serial[slot] = 0;
    std::replace(&self.w[slot], Waker::noop()).drop();
    self.count = self.count - 1;
}

int WaitList::wake_one() {
    if (self.count == 0) return 0;
    int k = 0;
    while (k < ASYNC_WAITERS) {
        int i = (int)((self.next_ + (unsigned int)k) % (unsigned int)ASYNC_WAITERS);
        if (self.serial[i] != 0) {
            self.serial[i] = 0;
            self.count = self.count - 1;
            self.next_ = (unsigned int)(i + 1);
            struct Waker w = std::replace(&self.w[i], Waker::noop());
            w.wake();
            return 1;
        }
        k = k + 1;
    }
    return 0;
}

int WaitList::wake_all() {
    int n = 0;
    int i = 0;
    while (i < ASYNC_WAITERS && self.count > 0) {
        if (self.serial[i] != 0) {
            self.serial[i] = 0;
            self.count = self.count - 1;
            struct Waker w = std::replace(&self.w[i], Waker::noop());
            w.wake();
            n = n + 1;
        }
        i = i + 1;
    }
    return n;
}

int WaitList::len() const { return self.count; }

void WaitList::drop() {
    int i = 0;
    while (i < ASYNC_WAITERS) {
        if (self.serial[i] != 0) { self.serial[i] = 0; self.w[i].drop(); }
        i = i + 1;
    }
    self.count = 0;
}

// ── YieldNow ───────────────────────────────────────────────────────────────

struct YieldNow YieldNow::new() {
    struct YieldNow y;
    y.yielded = 0;
    return y;
}

struct Poll<Unit> YieldNow::poll(&Context cx) {
    if (self.yielded != 0) return Poll::ready(UNIT);
    self.yielded = 1;
    cx.wake_by_ref();
    return Poll::pending();
}

void YieldNow::drop() { self.yielded = 1; }

} // namespace std
