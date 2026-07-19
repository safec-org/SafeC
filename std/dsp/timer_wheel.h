// SafeC Standard Library — Timer Wheel Scheduler
// Efficient O(1) timer management using a hashed timing wheel.
#pragma once

#define WHEEL_SLOTS   256    // power-of-two number of buckets
#define WHEEL_TIMERS  64     // max timers (indices 0..WHEEL_TIMERS-1)

namespace std {

struct WheelTimer {
    unsigned long expires;   // absolute tick when the timer fires
    void*         callback;  // function pointer: void(*)(void* ctx)
    // 'ctx' stays 'void*' rather than a typed '?&T': one TimerWheel is
    // meant to be a single, process-wide, multi-tenant scheduler — its
    // WHEEL_TIMERS slots routinely host unrelated subsystems' timers
    // (a network retransmit timer next to a UI blink-cursor timer) with
    // different context types at once. Genericizing TimerWheel<T> would
    // force every timer in a given wheel to share one T instead — see
    // std/gui/gui_widget.h's Widget.userData for the fuller writeup of
    // this tradeoff, and std/dma.h's DmaChannel<T> for a case where a
    // concrete T *does* fit (one driver owns a whole channel, no
    // multi-tenancy to lose).
    void*         ctx;       // user context passed to callback
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

    // Add a one-shot timer firing at absolute tick `tick`.
    // Returns timer ID (1-based) or 0 if the wheel is full.
    int  add(unsigned long tick, void* callback, void* ctx);

    // Add a periodic timer firing every `period` ticks starting at
    // current_tick + period.
    // Returns timer ID (1-based) or 0 if the wheel is full.
    int  add_periodic(unsigned long period, void* callback, void* ctx);

    // Cancel a timer by its ID.  Silently ignores invalid IDs.
    void cancel(int id);

    // Advance the wheel by one tick; fires all timers expiring at the new tick.
    // Periodic timers are automatically rescheduled.
    // Returns the number of timers fired.
    int  tick();

    // Return the current tick counter.
    unsigned long now() const;
};

// Initialise a TimerWheel to a clean state (all slots free, tick = 0).
void timer_wheel_init(&stack TimerWheel tw);

} // namespace std
