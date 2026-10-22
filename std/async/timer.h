// SafeC Standard Library — async: timers (Sleep, Interval, a timer queue)
//
// A TimerQueue is the clock-and-deadlines half of an executor: futures
// register "wake me at time T" there, and the executor advances it with
// the time its Park source reports (std/async/executor.h), which fires the
// due wakers. Every executor in std/async owns one and hands it to the
// futures it polls through Context::timers, so
//
//   struct Sleep s = Sleep::new(250LL);    // ms; the deadline is fixed at first poll
//   AWAIT(1, self.s);                      // inside a coroutine (coro.h)
//
// needs no handle to anything. Times are milliseconds on the executor's
// clock (a monotonic clock on a hosted target, whatever the Park source
// counts on bare metal). The queue only knows the time it was last
// advanced to, which keeps it deterministic: a test can drive it with a
// fake clock.
//
// The queue is std/dsp/timer_wheel.h's TimerWheel with a 1 ms tick
// (WHEEL_TIMERS = 64 concurrent timers), advanced in jumps from one due
// timer to the next rather than tick by tick. A Sleep that finds the wheel
// full waits in the queue's overflow list (ASYNC_WAITERS more) and retries
// when a slot frees up or its deadline comes; past that it re-polls until
// its deadline (correct, but busy). All operations take the queue's spinlock, so a multi-threaded
// executor's workers share one queue.
#pragma once
#include <std/async/future.h>

namespace std {

// struct TimerQueue is declared in future.h (Context points at one).


// Ready (with UNIT) once 'ms' milliseconds have passed since its first
// poll (Rust's tokio::time::sleep). Needs a context with timers.
struct Sleep {
    long long      duration_ms;
    long long      deadline;    // set at the first poll (or by until())
    int            armed;       // deadline set
    int            fired;       // the queue woke it (deadline passed)
    int            id;          // TimerQueue registration, 0 = none
    int            ovf_id;      // waiting for a free wheel slot (TimerQueue::overflow), 0 = no
    struct Waker   waker;       // what the timer wakes
    int            has_waker;
    ?&TimerQueue   tq;

    static struct Sleep new(long long ms);
    // Ready once the executor's clock reaches 'deadline_ms'.
    static struct Sleep until(long long deadline_ms);
    struct Poll<Unit> poll(&Context cx);
    void drop();
    // Re-arm for 'ms' from the next poll (it may be pending: the old
    // registration is dropped first).
    void reset(long long ms);
    int  is_elapsed() const;
};

// A periodic timer (tokio::time::interval): every poll that returns Ready
// gives the number of the tick (1, 2, 3, ...), one per 'period_ms', the
// first one 'period_ms' after the first poll. Unlike a plain future it is
// polled again after Ready — it is a stream of ticks. Missed ticks (the
// task was busy) are delivered back to back, then the schedule continues
// (tokio's MissedTickBehavior::Burst).
struct Interval {
    long long    period_ms;
    long long    next_ms;     // deadline of the next tick
    long long    ticks;
    struct Sleep sleep;
    int          started;

    static struct Interval new(long long period_ms);
    struct Poll<long long> poll(&Context cx);
    void drop();
};

} // namespace std
