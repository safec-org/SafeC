// SafeC Standard Library — async: tasks, JoinHandle, the local executor
//
// An executor polls futures to completion. Each top-level future it runs
// is a *task*: an AsyncTask<F, T> holding the future, its output slot and
// the bookkeeping the executor and the task's wakers share. Tasks live in
// memory the caller provides (a static, a local of main, a field of a
// longer-lived struct, or heap memory you manage) — std/async never
// allocates — and must stay put from spawn() until they finish (or are
// aborted and their executor has seen it).
//
//   static struct AsyncTask<struct Fetch, int> t1;    // any storage that outlives the run
//   struct LocalExecutor ex = LocalExecutor::new();
//   t1 = AsyncTask::new(Fetch::new());
//   struct JoinHandle<int> h = ex.spawn(&t1);
//   struct VirtualClock clk = VirtualClock::new(0LL);
//   ex.run(&clk);                                      // until every task is done
//   match (t1.take()) { case some(v): ...; case none: ... }
//
// LocalExecutor is single-threaded and deterministic: tasks are polled in
// spawn order, each one once per wake-up, and it has no heap, no locks
// and no OS dependency — it builds for bare metal and wasm32 as well as
// hosted targets. Its wakers are safe to call from another thread or an
// interrupt handler: a wake is two atomic stores (the task's NOTIFIED bit
// and the executor's 'notified' flag), plus Park::unpark() to cut a park
// short. What it does while nothing is runnable is up to its Park source:
//
//   trait Park { now_ms(); park(timeout_ms); unpark(); }
//
//   VirtualClock   a clock that jumps straight to the next timer deadline
//                  instead of waiting (tokio's paused time): deterministic
//                  tests and simulations, any target.
//   FnPark         your own clock and idle functions (a SysTick counter and
//                  WFI on bare metal, say).
//   HostPark       the OS monotonic clock, parking on a condition variable
//                  (std/async/host.h; hosted).
//   IoDriver       epoll/kqueue/WSAPoll readiness plus the OS clock
//                  (std/async/net.h; hosted).
//
// For hosted programs that just want a result, std/async/host.h's
// block_on(&fut) wraps all of this. The multi-threaded executor is
// std/async/pool.h's ThreadPool; its tasks and JoinHandles are the same
// types.
#pragma once
#include <std/async/future.h>
#include <std/atomic.h>
#include <std/sync/spinlock.h>

// TaskHeader::state bits
#define ASYNC_TASK_NOTIFIED 1    // woken since it was last polled
#define ASYNC_TASK_RUNNING  2    // being polled right now
#define ASYNC_TASK_DONE     4    // finished (output stored) or aborted (dropped)
#define ASYNC_TASK_ABORTED  8    // JoinHandle::abort() was called
#define ASYNC_TASK_QUEUED   16   // in a ThreadPool's run queue

namespace std {

// A source of time and of waiting, for executors (see above).
trait Park {
    long long now_ms();               // monotonic milliseconds
    void park(long long timeout_ms);  // wait up to timeout (-1: until unpark), or less
    void unpark();                    // end the current (or the next) park early
}

// The part of an AsyncTask an executor sees. First field of every
// AsyncTask, so a ?&TaskHeader and the task have the same address.
struct TaskHeader {
    void*            poll_;     // fn int(?&AsyncTask<F, T>, &Context): 1 = finished
    void*            drop_;     // fn void(?&AsyncTask<F, T>): drop the future
    void*            exec_;     // the executor it was spawned on, null before
    void*            wake_;     // that executor's wake: fn void(?&TaskHeader, unsigned long)
    int              state;     // ASYNC_TASK_* bits, changed atomically
    ?&TaskHeader     next;      // the executor's list link
    struct WakerSlot joiner;    // a JoinHandle waiting for it
    struct Spinlock  lock;      // guards 'joiner' and the DONE transition
};

// A future and the room for its output: the unit an executor runs.
generic<F: Future, T>
struct AsyncTask {
    struct TaskHeader hdr;      // first: executors see &task.hdr
    F                 fut;
    T                 out[1];   // the output, once finished (not aborted)

    // A task for 'fut', not yet spawned. Move it to its final place before
    // spawning it; it must not move after.
    static struct AsyncTask<F, T> new(F fut);
    int is_finished() const;    // done, either way
    int is_aborted() const;
    // The output, if it finished (not aborted).
    ?T  take() const;
};

// What spawn() returns: await it (it is a future) or ask it about the task.
// Ready(some(v)) once the task finished with v, Ready(null) if it was
// aborted. Copies refer to the same task; only one may be awaited at a time.
generic<T>
struct JoinHandle {
    ?&TaskHeader task;
    ?&T          out_;

    struct Poll<?T> poll(&Context cx);
    // Stop waiting for it (the task keeps running: dropping a JoinHandle
    // detaches, as in Rust).
    void drop();
    // Cancel the task: its executor drops the future (instead of polling it
    // again) and the task finishes as aborted. Cancelling a finished task
    // does nothing.
    void abort();
    int  is_finished() const;
};

// The single-threaded, no-heap executor (see the top of this file).
struct LocalExecutor {
    ?&TaskHeader      head;        // live tasks, in spawn order
    ?&TaskHeader      tail;
    int               live;        // spawned and not finished
    int               notified;    // atomic: some task (or the block_on root) was woken
    int               root_woken;  // atomic: block_on's own future was woken
    struct TimerQueue timers;
    void*             io_;         // an attached IoDriver (std/async/net.h), or null
    void*             unpark_;     // fn void(?&P): the running Park's unpark, or null
    void*             park_;       // that Park
    long long         polls;       // statistics: task polls so far

    static struct LocalExecutor new();

    // Start running 'task' (returns its handle). The task joins the end of
    // the poll order and is polled on the next round.
    generic<F, T> struct JoinHandle<T> spawn(&AsyncTask<F, T> task);
    // Type-erased spawn behind spawn() and Context's: 1 on success.
    int  spawn_raw_(?&TaskHeader t);

    // One round: poll every task woken when its turn in the spawn order
    // comes (a task woken by one later in the order waits for the next
    // round). Doesn't advance the timers or wait. Returns the polls made.
    int  run_ready();
    // Advance the timer queue to now_ms (firing what is due), then one
    // round. Returns the polls made.
    int  turn(long long now_ms);
    // Run until every spawned task has finished: rounds, with the timers
    // advanced by the Park's clock before each, parking on 'park' whenever
    // nothing is runnable. A task that keeps waking itself doesn't stop
    // timers from firing.
    generic<P: Park> void run(&P park);
    // Run 'fut' (and every spawned task, meanwhile) until 'fut' is Ready;
    // returns its value. Spawned tasks that are still running stay spawned.
    generic<P: Park, F: Future, T> T block_on(&P park, &F fut);

    int  live_count() const;
    int  has_ready() const;       // some task is woken
    // Internal: one round of run() with the Park hooks installed.
    void install_park_(void* unpark, void* park);
};

// Spawn 'task' on the executor that is polling 'cx' — from inside a task
// (tokio::spawn). Panics if that context has no executor.
generic<F, T>
struct JoinHandle<T> async_spawn(&Context cx, &AsyncTask<F, T> task);

// ── Park sources (see the top of the file) ─────────────────────────────────

// Simulated time: park(t) moves the clock forward by t instead of waiting,
// so a run takes no real time and is fully deterministic. park(-1) — every
// task waits and no timer is pending — is a deadlock in a program that has
// no other threads or interrupts, and panics.
struct VirtualClock {
    long long now;
    long long parked;   // how many times it parked
    static struct VirtualClock new(long long start_ms);
    long long now_ms();
    void park(long long timeout_ms);
    void unpark();
};

// A Park from functions: now() for the clock, idle(timeout_ms) to wait (WFI
// on bare metal; timeout -1 = until an interrupt), wake() (or null) from
// another context to end idle() early.
struct FnPark {
    fn long long()     now_fn;
    fn void(long long) idle_fn;
    ?&int              flag;    // unused by FnPark itself; for the caller's idle_fn
    static struct FnPark new(fn long long() now, fn void(long long) idle);
    long long now_ms();
    void park(long long timeout_ms);
    void unpark();
};

// ── internal: the generic halves of AsyncTask/JoinHandle/LocalExecutor ─────

generic<F, T>
int async_task_poll_(?&AsyncTask<F, T> tp, &Context cx) {
    match (tp) {
        case some(t): {
            struct Poll<T> p = t.fut.poll(cx);
            if (p.ready_ == 0) return 0;
            t.out[0] = p.value;
            return 1;
        }
        case null: return 1;
    }
    return 1;
}

generic<F, T>
void async_task_drop_(?&AsyncTask<F, T> tp) {
    match (tp) {
        case some(t): t.fut.drop();
        case null: { }
    }
}

// Make 'hdr' a fresh, unspawned header.
void task_header_init_(&TaskHeader hdr, void* poll, void* drop);
// Mark 'hdr' finished (aborted or not) and wake its JoinHandle.
void task_header_finish_(&TaskHeader hdr, int aborted);

generic<F: Future, T>
struct AsyncTask<F, T> AsyncTask::new(F fut) {
    struct AsyncTask<F, T> t;
    fn int(?&AsyncTask<F, T>, &Context) pf = async_task_poll_;
    fn void(?&AsyncTask<F, T>) df = async_task_drop_;
    unsafe { task_header_init_(&t.hdr, (void*)pf, (void*)df); }
    t.fut = fut;
    return t;
}

generic<F: Future, T>
int AsyncTask::is_finished() const {
    return (atomic_load_int(&self.hdr.state) & ASYNC_TASK_DONE) != 0 ? 1 : 0;
}

generic<F: Future, T>
int AsyncTask::is_aborted() const {
    return (atomic_load_int(&self.hdr.state) & ASYNC_TASK_ABORTED) != 0 ? 1 : 0;
}

generic<F: Future, T>
?T AsyncTask::take() const {
    int st = atomic_load_int(&self.hdr.state);
    if ((st & ASYNC_TASK_DONE) == 0 || (st & ASYNC_TASK_ABORTED) != 0) return null;
    return self.out[0];
}

// Whether 'hdr' is done, registering 'w' to be woken when it is if not.
int task_header_poll_join_(&TaskHeader hdr, const &Waker w);
void task_header_abort_(&TaskHeader hdr);

generic<T>
struct Poll<?T> JoinHandle::poll(&Context cx) {
    match (self.task) {
        case null: { ?T none = null; return Poll::ready(none); }
        case some(t): {
            if (task_header_poll_join_(t, &cx.waker) == 0) return Poll::pending();
            int st = atomic_load_int(&t.state);
            if ((st & ASYNC_TASK_ABORTED) != 0) { ?T none = null; return Poll::ready(none); }
            match (self.out_) {
                case some(o): { ?T v = *o; return Poll::ready(v); }
                case null: { ?T none = null; return Poll::ready(none); }
            }
        }
    }
    ?T none = null;
    return Poll::ready(none);
}

generic<T>
void JoinHandle::drop() {
    match (self.task) {
        case some(t): { t.lock.lock(); t.joiner.clear(); t.lock.unlock(); }
        case null: { }
    }
    self.task = null;
}

generic<T>
void JoinHandle::abort() {
    match (self.task) {
        case some(t): task_header_abort_(t);
        case null: { }
    }
}

generic<T>
int JoinHandle::is_finished() const {
    match (self.task) {
        case some(t): return (atomic_load_int(&t.state) & ASYNC_TASK_DONE) != 0 ? 1 : 0;
        case null: return 1;
    }
    return 1;
}

generic<F, T>
struct JoinHandle<T> join_handle_for_(&AsyncTask<F, T> task) {
    struct JoinHandle<T> h;
    h.task = &task.hdr;
    h.out_ = &task.out[0];
    return h;
}

generic<F, T>
struct JoinHandle<T> LocalExecutor::spawn(&AsyncTask<F, T> task) {
    self.spawn_raw_(&task.hdr);
    return join_handle_for_(task);
}

// Calls the spawn function an executor left in a Context.
int context_spawn_raw_(&Context cx, ?&TaskHeader t);

generic<F, T>
struct JoinHandle<T> async_spawn(&Context cx, &AsyncTask<F, T> task) {
    context_spawn_raw_(cx, &task.hdr);
    return join_handle_for_(task);
}

generic<P>
void park_unpark_thunk_(?&P p) {
    match (p) {
        case some(x): x.unpark();
        case null: { }
    }
}

generic<P: Park>
void LocalExecutor::run(&P park) {
    fn void(?&P) up = park_unpark_thunk_;
    unsafe { self.install_park_((void*)up, (void*)park); }
    while (self.live > 0) {
        self.turn(park.now_ms());
        if (self.live == 0 || self.has_ready() != 0) continue;
        park.park(self.timers.park_timeout());
    }
    unsafe { self.install_park_((void*)0, (void*)0); }
}

// The block_on root future's waker (data = the executor).
void local_root_wake_(?&LocalExecutor ex, unsigned long tag);
// A Context for polling on 'ex' with waker 'w'.
struct Context local_context_(&LocalExecutor ex, struct Waker w);

generic<P: Park, F: Future, T>
T LocalExecutor::block_on(&P park, &F fut) {
    fn void(?&P) up = park_unpark_thunk_;
    unsafe { self.install_park_((void*)up, (void*)park); }
    struct Context cx = local_context_(self, Waker::new(local_root_wake_, (?&LocalExecutor)self, 0UL));
    atomic_store_int(&self.root_woken, 1);
    self.timers.advance(park.now_ms());
    while (1) {
        if (atomic_exchange_int(&self.root_woken, 0) != 0) {
            struct Poll<T> p = fut.poll(&cx);
            if (p.ready_ != 0) {
                unsafe { self.install_park_((void*)0, (void*)0); }
                return p.value;
            }
        }
        self.turn(park.now_ms());
        if (atomic_load_int(&self.root_woken) != 0 || self.has_ready() != 0) continue;
        park.park(self.timers.park_timeout());
    }
    struct Poll<T> never = fut.poll(&cx);   // unreachable
    return never.value;
}

} // namespace std
