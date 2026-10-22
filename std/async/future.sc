// SafeC Standard Library — async: Waker, Context, WakerSlot, WaitList, YieldNow
#pragma once
#include <std/async/future.h>

namespace std {

// ── Waker ──────────────────────────────────────────────────────────────────

static void waker_noop_fn_(?&int data, unsigned long tag) { }

struct Waker Waker::raw_(void* wake, void* clone, void* drop, void* data, unsigned long tag) {
    struct Waker w;
    w.wake_  = wake;
    w.clone_ = clone;
    w.drop_  = drop;
    w.data_  = data;
    w.tag_   = tag;
    return w;
}

struct Waker Waker::noop() {
    return Waker::new(waker_noop_fn_, (?&int)null, 0UL);
}

void Waker::wake_by_ref() const {
    unsafe {
        if (self.wake_ == (void*)0) return;
        fn void(void*, unsigned long) f = (fn void(void*, unsigned long))self.wake_;
        f(self.data_, self.tag_);
    }
}

void Waker::wake() {
    self.wake_by_ref();
    self.drop();
}

struct Waker Waker::clone() const {
    unsafe {
        if (self.clone_ != (void*)0) {
            fn void(void*, unsigned long) f = (fn void(void*, unsigned long))self.clone_;
            f(self.data_, self.tag_);
        }
    }
    return Waker::raw_(self.wake_, self.clone_, self.drop_, self.data_, self.tag_);
}

void Waker::drop() {
    unsafe {
        if (self.drop_ != (void*)0) {
            fn void(void*, unsigned long) f = (fn void(void*, unsigned long))self.drop_;
            f(self.data_, self.tag_);
        }
        // A dropped waker wakes nothing (and drops nothing) again.
        self.wake_ = (void*)0;
        self.drop_ = (void*)0;
        self.clone_ = (void*)0;
    }
}

int Waker::will_wake(const &Waker other) const {
    unsafe {
        return (self.data_ == other.data_ && self.wake_ == other.wake_ &&
                self.tag_ == other.tag_) ? 1 : 0;
    }
}

int Waker::is_noop() const {
    fn void(?&int, unsigned long) n = waker_noop_fn_;
    unsafe { return (self.wake_ == (void*)n || self.wake_ == (void*)0) ? 1 : 0; }
}

// ── Context ────────────────────────────────────────────────────────────────

struct Context Context::from_waker(struct Waker w) {
    struct Context cx;
    cx.waker  = w;
    cx.timers = null;
    unsafe {
        cx.io_    = (void*)0;
        cx.exec_  = (void*)0;
        cx.spawn_ = (void*)0;
    }
    return cx;
}

struct Context Context::noop() {
    return Context::from_waker(Waker::noop());
}

void Context::wake_by_ref() const {
    self.waker.wake_by_ref();
}

// ── WakerSlot ──────────────────────────────────────────────────────────────

struct WakerSlot WakerSlot::new() {
    struct WakerSlot s;
    s.w = Waker::raw_((void*)0, (void*)0, (void*)0, (void*)0, 0UL);
    s.set = 0;
    return s;
}

void WakerSlot::register_(const &Waker w) {
    if (self.set != 0) {
        if (self.w.will_wake(w) != 0) return;
        self.w.drop();
    }
    self.w = w.clone();
    self.set = 1;
}

int WakerSlot::wake() {
    if (self.set == 0) return 0;
    self.set = 0;
    struct Waker w = self.w;
    self.w = Waker::raw_((void*)0, (void*)0, (void*)0, (void*)0, 0UL);
    w.wake();
    return 1;
}

void WakerSlot::clear() {
    if (self.set == 0) return;
    self.set = 0;
    self.w.drop();
}

// ── WaitList ───────────────────────────────────────────────────────────────
// id = serial * 64 + slot + 1 (ASYNC_WAITERS <= 63).

static int waitlist_slot_(int id) { return (id & 63) - 1; }

struct WaitList WaitList::new() {
    struct WaitList l;
    int i = 0;
    while (i < ASYNC_WAITERS) {
        l.w[i] = Waker::raw_((void*)0, (void*)0, (void*)0, (void*)0, 0UL);
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
        if (self.w[slot].will_wake(w) == 0) {
            self.w[slot].drop();
            self.w[slot] = w.clone();
        }
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
    self.w[slot].drop();
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
            struct Waker w = self.w[i];
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
            struct Waker w = self.w[i];
            w.wake();
            n = n + 1;
        }
        i = i + 1;
    }
    return n;
}

int WaitList::len() const { return self.count; }

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
