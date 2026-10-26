// SafeC Standard Library — Timer Wheel Scheduler
// Efficient O(1) timer management using a hashed timing wheel.
#pragma once
#include <std/erased.h>

#define WHEEL_SLOTS   256    // power-of-two number of buckets
#define WHEEL_TIMERS  64     // max timers (indices 0..WHEEL_TIMERS-1)

namespace std {

struct WheelTimer {
    unsigned long expires;   // absolute tick when the timer fires
    // One TimerWheel is a multi-tenant scheduler: its slots host timers
    // with unrelated context types at once, so each slot stores its
    // callback and context type-erased. They are only ever written by
    // TimerWheel::add/add_periodic below, which take a matching
    // fn void(?&T) / ?&T pair, so the erased call is always well-typed
    // (see std/erased.h).
    ?fn void(?&Erased) callback;  // the fn void(?&T); null when free
    ?&Erased           ctx;       // the ?&T
    int           used;      // 1 if slot occupied, 0 if free
    int           periodic;  // 1 → auto-reschedule every `period` ticks
    unsigned long period;    // interval for periodic timers
};

struct TimerWheel {
    struct WheelTimer timers[WHEEL_TIMERS];
    // slots[s] stores timer index+1 for the head of the list in bucket s,
    // 0 means the bucket is empty.  Chaining uses timers[id].ctx as next
    // pointer only conceptually; this implementation uses a flat scan for
    // simplicity and correctness within the WHEEL_TIMERS limit.
    unsigned long     current_tick;

    // Initialise the wheel in place to a clean state (all slots free,
    // tick = 0). Must be called before any other operation.
    void init();

    // One-shot timer calling cb(ctx) at absolute tick `tick`.
    // Returns the timer ID (1-based) or 0 if the wheel is full.
    generic<T> int add(unsigned long tick, fn void(?&T) cb, ?&T ctx);

    // Periodic timer calling cb(ctx) every `period` ticks, first at
    // current_tick + period. Returns the timer ID (1-based) or 0 if full.
    generic<T> int add_periodic(unsigned long period, fn void(?&T) cb, ?&T ctx);

    // Type-erased registration behind add/add_periodic.
    int  add_raw_(unsigned long tick, fn void(?&Erased) callback, ?&Erased ctx);
    int  add_periodic_raw_(unsigned long period, fn void(?&Erased) callback, ?&Erased ctx);

    // Cancel a timer by its ID.  Silently ignores invalid IDs.
    void cancel(int id);

    // Advance the wheel by one tick; fires all timers expiring at the new tick.
    // Periodic timers are automatically rescheduled.
    // Returns the number of timers fired.
    int  tick();

    // Return the current tick counter.
    unsigned long now() const;
};

generic<T>
int TimerWheel::add(unsigned long tick, fn void(?&T) cb, ?&T ctx) {
    unsafe { return self.add_raw_(tick, (fn void(?&Erased))cb, (?&Erased)ctx); }
}

generic<T>
int TimerWheel::add_periodic(unsigned long period, fn void(?&T) cb, ?&T ctx) {
    unsafe { return self.add_periodic_raw_(period, (fn void(?&Erased))cb, (?&Erased)ctx); }
}

} // namespace std
