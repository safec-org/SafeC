// SafeC Standard Library — async: futures, Poll, Waker, Context
//
// Rust's async model without language-level coroutines. A *future* is a
// struct with two methods:
//
//   struct Poll<T> poll(&Context cx);   // make progress; Ready(value) or Pending
//   void           drop();              // abandon it (release timers, waiters)
//
// poll() never blocks. When it can't finish yet it returns
// Poll::pending() — but only after arranging for cx's Waker to be woken
// once progress is possible again (a timer firing, a channel getting a
// value, a socket becoming readable). The executor then polls it again.
// Nothing runs a future except polling it: a future that is never polled
// does nothing, exactly like Rust's.
//
// The pieces map one to one onto Rust's core::task / core::future:
//
//   Rust                         SafeC (std/async)
//   Poll<T>::Ready / Pending     struct Poll<T>: Poll::ready(v) / Poll::pending()
//   Future::poll(Pin<&mut Self>, &mut Context) -> Poll<Output>
//                                struct Poll<T> F::poll(&Context cx)
//   Drop for a future            F::drop() (SafeC has no destructors)
//   Context, cx.waker()          struct Context, cx.waker / cx.wake_by_ref()
//   Waker (RawWaker + vtable)    struct Waker (type-erased behind Waker::new<T>)
//   async fn / .await            ASYNC_BEGIN / AWAIT / ASYNC_RETURN (std/async/coro.h)
//   Pin                          the rule below: a polled future doesn't move
//
// Two rules follow from futures being plain structs:
//
//   * Pinning. Once polled, a future must stay where it is until it
//     completes or is dropped — timers, channels and the I/O driver keep
//     its address. Move it (into a combinator, an AsyncTask, ...) only
//     before the first poll.
//   * Dropping. A future abandoned before it completes must be drop()ped:
//     that is what unregisters it. Combinators (select, timeout) and
//     executors (JoinHandle::abort) do this for the futures they own.
//     drop() is idempotent and also fine after completion.
//
// The Output type is the T of the Poll<T> that poll() returns. Futures
// that produce nothing return struct Poll<Unit> (Unit is an int; pass
// UNIT).
#pragma once
#include <std/sync/spinlock.h>
#include <std/ownership.h>
#include <std/dsp/timer_wheel.h>

typedef int Unit;     // the output of a future that produces nothing: Rust's ()
#define UNIT 0

namespace std {

// Sequentially consistent atomics on an int, for the task and executor
// state shared with wakers (which may run on other threads or in an
// interrupt handler). The compiler's atomic built-ins, without std/atomic.h
// (whose 64-bit operations need libcalls some bare-metal cores lack).
int  async_load_(const &int p);
void async_store_(&int p, int v);
int  async_xchg_(&int p, int v);
int  async_fetch_or_(&int p, int bits);
int  async_fetch_and_(&int p, int bits);

// What a future is. The structural check is by name: a type with
// methods poll and drop fits, whatever its Output. Executors and
// combinators bound their future parameters with it, so a struct missing
// drop() is rejected where it is spawned or combined.
trait Future {
    void poll(&Context cx);
    void drop();
}

// The result of one poll: Ready with a value, or Pending (value unset).
generic<T>
struct Poll {
    int ready_;   // 1 = Ready, 0 = Pending
    T   value;    // set when ready_ == 1

    static struct Poll<T> ready(T v);
    static struct Poll<T> pending();
    int is_ready() const;
    int is_pending() const;
    // The value if Ready, null if Pending.
    ?T  take() const;
};

// ── Waker ──────────────────────────────────────────────────────────────────
// A handle that tells an executor "poll the task this belongs to again".
// It is a type-erased (data, functions) pair, like Rust's RawWaker and its
// vtable, made only through the generic constructors below, which take a
// matching fn(?&T, ...) and ?&T so every erased call is well-typed:
//
//   wake      (data, tag): schedule the task. Called by wake/wake_by_ref.
//   clone, drop (data, tag): a copy of this waker was made / let go of —
//             for executors whose wakers own a reference count. null
//             (Waker::new) when copies are free, as for every executor in
//             std/async: their wakers point at task storage that outlives
//             every registration (a future unregisters in drop()).
//
// Wakers are small values; copy them with clone() (which runs the clone
// hook) and release them with drop(). wake() consumes the waker,
// wake_by_ref() doesn't.
struct Waker {
    void*         data_;    // ?&T
    void*         wake_;    // fn void(?&T, unsigned long)
    void*         clone_;   // fn void(?&T, unsigned long), or null
    void*         drop_;    // fn void(?&T, unsigned long), or null
    unsigned long tag_;     // passed back to the functions (a slot, a generation, ...)

    // A waker calling wake(data, tag); copies need no bookkeeping.
    generic<T> static struct Waker new(fn void(?&T, unsigned long) wake, ?&T data, unsigned long tag);
    // A waker whose copies are counted: clone(data, tag) runs for every
    // clone(), drop(data, tag) for every drop() (and after wake()).
    generic<T> static struct Waker counted(fn void(?&T, unsigned long) wake,
                                           fn void(?&T, unsigned long) clone,
                                           fn void(?&T, unsigned long) drop,
                                           ?&T data, unsigned long tag);
    // A waker that does nothing (Rust's Waker::noop): for polling a future
    // by hand, in tests, or where nobody needs to be told.
    static struct Waker noop();
    // Type-erased constructor behind new/counted.
    static struct Waker raw_(void* wake, void* clone, void* drop, void* data, unsigned long tag);

    void wake() consume;                 // wake_by_ref(), then drop()
    void wake_by_ref() const;
    struct Waker clone() const;
    void drop();
    // True if both wake the same task (same data, functions and tag):
    // lets a future skip re-registering an unchanged waker.
    int  will_wake(const &Waker other) const;
    int  is_noop() const;
};

generic<T>
struct Waker Waker::new(fn void(?&T, unsigned long) wake, ?&T data, unsigned long tag) {
    unsafe { return Waker::raw_((void*)wake, (void*)0, (void*)0, (void*)data, tag); }
}

generic<T>
struct Waker Waker::counted(fn void(?&T, unsigned long) wake,
                            fn void(?&T, unsigned long) clone,
                            fn void(?&T, unsigned long) drop,
                            ?&T data, unsigned long tag) {
    unsafe { return Waker::raw_((void*)wake, (void*)clone, (void*)drop, (void*)data, tag); }
}

// ── WakerSlot ──────────────────────────────────────────────────────────────
// One registered waker (Rust's AtomicWaker): where a single waiting future
// leaves its waker for whoever completes it. Not locked by itself; the
// structure that holds it (a channel, a task) guards it.
struct WakerSlot {
    struct Waker w;
    int          set;

    static struct WakerSlot new();
    // Keep a clone of 'w' (dropping a different one already held).
    void register_(const &Waker w);
    // Wake and release the held waker. Returns 1 if there was one.
    int  wake();
    // Release it without waking.
    void clear();
};

// ── WaitList ───────────────────────────────────────────────────────────────
// A fixed set of registered wakers, for primitives several futures wait on
// at once (a channel's senders, a mutex's lockers). No heap: ASYNC_WAITERS
// slots. add() returns an id the future keeps and passes back to refresh
// or remove its registration; an id stays tied to its registration (a
// slot reused after a wake gets a new id), so a stale one is harmless. A
// waiter that finds the list full gets id 0 and must re-poll on its own
// (cx.wake_by_ref()) — slower, never wrong.
#define ASYNC_WAITERS 32

struct WaitList {
    struct Waker w[ASYNC_WAITERS];
    int          serial[ASYNC_WAITERS];   // 0 = free, else the registration's serial
    int          count;
    int          next_serial;
    unsigned int next_;                   // where wake_one starts looking (fairness)

    static struct WaitList new();
    // Register (or, with the id of a live registration, refresh) a waiter.
    // Returns its id, 0 if full.
    int  add(int id, const &Waker w);
    // Remove a registration (a stale or 0 id is ignored).
    void remove(int id);
    // Whether 'id' is still registered (not woken or removed yet).
    int  contains(int id) const;
    // Wake (and remove) one waiter. Returns 1 if there was one.
    int  wake_one();
    // Wake (and remove) every waiter. Returns how many.
    int  wake_all();
    int  len() const;
};

// ── TimerQueue ─────────────────────────────────────────────────────────────
// The executor's timer queue (implemented in timer.sc; Sleep, Interval and
// the queue's role are described in std/async/timer.h). Declared here
// because Context points at one.
struct TimerQueue {
    struct TimerWheel wheel;     // tick = 1 ms since origin_ms
    long long         origin_ms; // clock value of tick 0
    long long         now_ms_;   // the time the queue was last advanced to
    int               started;   // origin set (by the first advance)
    struct Spinlock   lock;
    // Sleeps that found the wheel full wait here, and are woken to retry
    // when a wheel slot frees up or the earliest of their deadlines comes.
    struct WaitList   overflow;
    long long         ovf_min;   // earliest deadline waiting in 'overflow', -1 if none

    static struct TimerQueue new();
    // Moves the queue's clock to 'now_ms' (never backwards) and wakes every
    // timer due by then. The first call sets the origin. Returns how many
    // fired.
    int       advance(long long now_ms);
    // The current time: what the last advance() was given.
    long long now_ms() const;
    // The earliest pending deadline, or -1 if no timer is pending.
    long long next_deadline() const;
    // Milliseconds from now to the earliest deadline (0 if already due),
    // or -1 if no timer is pending: how long an executor may park.
    long long park_timeout() const;
    int       pending() const;

    // Internal, with 'lock' held: register cb(ctx) for 'deadline_ms'
    // (0 = the wheel is full, or the deadline has passed), and cancel one.
    int       add_locked_(long long deadline_ms, void* cb, void* ctx);
    void      cancel_locked_(int id);
    // Internal, with 'lock' held: wait in 'overflow' for a free slot (0 =
    // that is full too), and leave it.
    int       overflow_locked_(long long deadline_ms, const &Waker w, int id);
};

// ── Context ────────────────────────────────────────────────────────────────
// What poll() gets: the waker of the task being polled, and the services of
// the executor polling it. Executors fill it in for every poll; a future
// only reads it (and clones the waker if it needs to keep it).
//
//   timers  the executor's timer queue (std/async/timer.h): Sleep,
//           Interval and Timeout register there. null when the executor
//           has none (Context::from_waker), and then sleeping is an error.
//   io_     the executor's I/O driver (std/async/net.h's IoDriver), or
//           null: the futures there read it through io_driver(cx).
//   exec_, spawn_  the executor, and its type-erased spawn: what
//           Context::spawn uses to start another task from inside a task.
struct Context {
    struct Waker      waker;
    ?&TimerQueue      timers;
    void*             io_;
    void*             exec_;
    void*             spawn_;    // fn int(void* exec, ?&TaskHeader task)

    // A context with only a waker: no timers, no I/O, no spawning.
    static struct Context from_waker(struct Waker w);
    // A context with Waker::noop(): poll by hand and look at the result.
    static struct Context noop();

    void wake_by_ref() const;    // waker.wake_by_ref(): "poll me again soon"
    // Releases the waker. A Context owns its waker (a Drop value), so the
    // executor that made one drops it after the poll.
    void drop();
};

// ── Leaf futures ───────────────────────────────────────────────────────────

// Ready with a value on the first poll (Rust's future::ready).
generic<T>
struct Ready {
    T   value;
    int done;
    static struct Ready<T> new(T v);
    struct Poll<T> poll(&Context cx);
    void drop();
};

// Never ready (Rust's future::pending): useful in select/timeouts, tests.
generic<T>
struct Pending {
    int unused_;
    static struct Pending<T> new();
    struct Poll<T> poll(&Context cx);
    void drop();
};

// Pending once (after waking itself), then Ready: gives the executor a
// chance to run other tasks (Rust's yield_now / tokio::task::yield_now).
struct YieldNow {
    int yielded;
    static struct YieldNow new();
    struct Poll<Unit> poll(&Context cx);
    void drop();
};

generic<T> struct Poll<T> Poll::ready(T v) { struct Poll<T> p; p.ready_ = 1; p.value = v; return p; }
generic<T> struct Poll<T> Poll::pending() { struct Poll<T> p; p.ready_ = 0; return p; }
generic<T> int Poll::is_ready() const { return self.ready_; }
generic<T> int Poll::is_pending() const { return self.ready_ == 0 ? 1 : 0; }
generic<T> ?T Poll::take() const { if (self.ready_ == 0) return null; return self.value; }

generic<T> struct Ready<T> Ready::new(T v) { struct Ready<T> r; r.value = v; r.done = 0; return r; }
generic<T> struct Poll<T> Ready::poll(&Context cx) { self.done = 1; return Poll::ready(self.value); }
generic<T> void Ready::drop() { self.done = 1; }

generic<T> struct Pending<T> Pending::new() { struct Pending<T> p; p.unused_ = 0; return p; }
generic<T> struct Poll<T> Pending::poll(&Context cx) { return Poll::pending(); }
generic<T> void Pending::drop() { }

} // namespace std

#include <std/async/timer.h>
