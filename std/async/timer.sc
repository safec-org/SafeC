// SafeC Standard Library — async: TimerQueue, Sleep, Interval
#pragma once
#include <std/async/timer.h>
#include <std/panic.h>

namespace std {

// ── TimerQueue ─────────────────────────────────────────────────────────────

struct TimerQueue TimerQueue::new() {
    struct TimerQueue q;
    q.wheel.init();
    q.origin_ms = 0LL;
    q.now_ms_ = 0LL;
    q.started = 0;
    q.lock = Spinlock::new();
    return q;
}

// Earliest 'expires' among live timers after the current tick; 0 if none.
static unsigned long timerq_next_tick_(const &TimerWheel w) {
    unsigned long best = 0UL;
    int i = 0;
    while (i < WHEEL_TIMERS) {
        if (w.timers[i].used == 1 && w.timers[i].expires > w.current_tick) {
            if (best == 0UL || w.timers[i].expires < best) best = w.timers[i].expires;
        }
        i = i + 1;
    }
    return best;
}

int TimerQueue::advance(long long now_ms) {
    self.lock.lock();
    if (self.started == 0) {
        self.started = 1;
        self.origin_ms = now_ms;
        self.now_ms_ = now_ms;
        self.lock.unlock();
        return 0;
    }
    if (now_ms <= self.now_ms_) {
        self.lock.unlock();
        return 0;
    }
    unsigned long target = (unsigned long)(now_ms - self.origin_ms);
    int fired = 0;
    // Jump from one due timer to the next instead of ticking every ms:
    // TimerWheel::tick() fires the timers whose deadline is exactly the new
    // tick, so put the wheel one tick before each.
    while (self.wheel.current_tick < target) {
        unsigned long next = timerq_next_tick_(&self.wheel);
        if (next == 0UL || next > target) {
            self.wheel.current_tick = target;
            break;
        }
        self.wheel.current_tick = next - 1UL;
        fired = fired + self.wheel.tick();
    }
    self.now_ms_ = now_ms;
    self.lock.unlock();
    return fired;
}

long long TimerQueue::now_ms() const { return self.now_ms_; }

long long TimerQueue::next_deadline() const {
    unsigned long next = timerq_next_tick_(&self.wheel);
    if (next == 0UL) return -1LL;
    return self.origin_ms + (long long)next;
}

long long TimerQueue::park_timeout() const {
    long long d = self.next_deadline();
    if (d < 0LL) return -1LL;
    if (d <= self.now_ms_) return 0LL;
    return d - self.now_ms_;
}

int TimerQueue::pending() const {
    int n = 0;
    int i = 0;
    while (i < WHEEL_TIMERS) {
        if (self.wheel.timers[i].used == 1 && self.wheel.timers[i].expires > self.wheel.current_tick) n = n + 1;
        i = i + 1;
    }
    return n;
}

int TimerQueue::add_locked_(long long deadline_ms, void* cb, void* ctx) {
    if (self.started == 0) {
        self.started = 1;
        self.origin_ms = self.now_ms_;
    }
    if (deadline_ms <= self.now_ms_) return 0;
    unsigned long tick = (unsigned long)(deadline_ms - self.origin_ms);
    if (tick <= self.wheel.current_tick) return 0;
    return self.wheel.add_raw_(tick, cb, ctx);
}

void TimerQueue::cancel_locked_(int id) {
    self.wheel.cancel(id);
}

// ── Sleep ──────────────────────────────────────────────────────────────────

// The TimerQueue's callback for a Sleep: runs inside advance(), queue locked.
static void sleep_fire_(?&Sleep sp) {
    match (sp) {
        case some(s): {
            s.fired = 1;
            s.id = 0;
            if (s.has_waker != 0) {
                s.has_waker = 0;
                struct Waker w = s.waker;
                w.wake();
            }
        }
        case null: { }
    }
}

struct Sleep Sleep::new(long long ms) {
    struct Sleep s;
    s.duration_ms = ms < 0LL ? 0LL : ms;
    s.deadline = 0LL;
    s.armed = 0;
    s.fired = 0;
    s.id = 0;
    s.waker = Waker::noop();
    s.has_waker = 0;
    s.tq = null;
    return s;
}

struct Sleep Sleep::until(long long deadline_ms) {
    struct Sleep s = Sleep::new(0LL);
    s.deadline = deadline_ms;
    s.armed = 1;
    return s;
}

struct Poll<Unit> Sleep::poll(&Context cx) {
    if (self.fired != 0) return Poll::ready(UNIT);
    match (cx.timers) {
        case null: {
            panic_at("std/async: Sleep polled by an executor without a timer queue (Context::timers is null)",
                     "std/async/timer.sc", 0);
        }
        case some(tq): {
            self.tq = tq;
            tq.lock.lock();
            if (self.fired != 0) { tq.lock.unlock(); return Poll::ready(UNIT); }
            if (self.armed == 0) {
                self.deadline = tq.now_ms_ + self.duration_ms;
                self.armed = 1;
            }
            if (self.deadline <= tq.now_ms_) {
                if (self.id != 0) { tq.cancel_locked_(self.id); self.id = 0; }
                if (self.has_waker != 0) { self.has_waker = 0; self.waker.drop(); }
                self.fired = 1;
                tq.lock.unlock();
                return Poll::ready(UNIT);
            }
            if (self.has_waker == 0 || self.waker.will_wake(&cx.waker) == 0) {
                if (self.has_waker != 0) self.waker.drop();
                self.waker = cx.waker.clone();
                self.has_waker = 1;
            }
            int busy = 0;
            if (self.id == 0) {
                fn void(?&Sleep) cb = sleep_fire_;
                unsafe { self.id = tq.add_locked_(self.deadline, (void*)cb, (void*)self); }
                if (self.id == 0) busy = 1;   // wheel full: re-poll until the deadline
            }
            tq.lock.unlock();
            if (busy != 0) cx.wake_by_ref();
            return Poll::pending();
        }
    }
    return Poll::pending();
}

void Sleep::drop() {
    match (self.tq) {
        case some(tq): {
            tq.lock.lock();
            if (self.id != 0 && self.fired == 0) tq.cancel_locked_(self.id);
            self.id = 0;
            if (self.has_waker != 0) { self.has_waker = 0; self.waker.drop(); }
            self.fired = 1;
            tq.lock.unlock();
        }
        case null: { self.fired = 1; }
    }
}

void Sleep::reset(long long ms) {
    self.drop();
    self.duration_ms = ms < 0LL ? 0LL : ms;
    self.armed = 0;
    self.fired = 0;
}

int Sleep::is_elapsed() const { return self.fired; }

// ── Interval ───────────────────────────────────────────────────────────────

struct Interval Interval::new(long long period_ms) {
    struct Interval iv;
    iv.period_ms = period_ms < 1LL ? 1LL : period_ms;
    iv.next_ms = 0LL;
    iv.ticks = 0LL;
    iv.sleep = Sleep::new(iv.period_ms);
    iv.started = 0;
    return iv;
}

struct Poll<long long> Interval::poll(&Context cx) {
    struct Poll<Unit> p = self.sleep.poll(cx);
    if (p.is_pending() != 0) return Poll::pending();
    // The tick that just elapsed was due at sleep.deadline; the next one
    // is a period after it (not after now), so the schedule doesn't drift.
    self.next_ms = self.sleep.deadline + self.period_ms;
    self.ticks = self.ticks + 1LL;
    self.sleep = Sleep::until(self.next_ms);
    return Poll::ready(self.ticks);
}

void Interval::drop() { self.sleep.drop(); }

} // namespace std
