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
#include <std/sync/spinlock.h>

// TaskHeader::state bits
#define ASYNC_TASK_NOTIFIED 1    // woken since it was last polled
#define ASYNC_TASK_RUNNING  2    // being polled right now
#define ASYNC_TASK_DONE     4    // finished (output stored) or aborted (dropped)
#define ASYNC_TASK_ABORTED  8    // JoinHandle::abort() was called
#define ASYNC_TASK_QUEUED   16   // in a ThreadPool's run queue
#define ASYNC_TASK_TAKEN    32   // the output was taken (take() or the JoinHandle)

namespace std {

// A source of time and of waiting, for executors (see above).
trait Park {
    long long now_ms();               // monotonic milliseconds
    void park(long long timeout_ms);  // wait up to timeout (-1: until unpark), or less
    void unpark();                    // end the current (or the next) park early
}

// The part of an AsyncTask an executor sees. First field of every
// AsyncTask, so a ?&TaskHeader and the task have the same address: poll_
// and drop_ are AsyncTask<F, T>'s own functions, stored by AsyncTask::new
// as functions of the header (see std/erased.h).
struct TaskHeader {
    fn int(?&TaskHeader, &Context)          poll_;  // 1 = finished
    fn void(?&TaskHeader)                   drop_;  // drop the future
    ?&Erased                                exec_;  // the executor it was spawned on, null before
    ?fn void(?&TaskHeader, unsigned long)   wake_;  // that executor's wake
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
    // The output, if it finished (not aborted) and wasn't taken yet: it
    // moves out of the task (once).
    ?T  take();
    // The task owns its future and, until taken, its output. Drop it once
    // it finished (or was aborted) and nothing polls it any more.
    void drop();
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
    ?&IoDriver        io_;         // an attached IoDriver (std/async/net.h), or null
    ?fn void(?&Erased) unpark_;    // the running Park's unpark, or null
    ?&Erased          park_;       // that Park
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
    // Releases the timer queue. Spawned tasks are owned by their AsyncTask
    // storage, not by the executor: finish or abort them first.
    void drop();
    // Internal: install (or, with nulls, remove) the running Park's unpark.
    generic<P> void install_park_(?fn void(?&P) unpark, ?&P park);
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
            if (p.ready_ == 0) { p.drop(); return 0; }
            t.out[0] = p.unwrap();
            return 1;
        }
        case null: return 1;
    }
    return 1;
}

generic<F, T>
void async_task_drop_(?&AsyncTask<F, T> tp) {
    match (tp) {
        case some(t): { unsafe { t.fut.drop(); } }   // the executor's abort: the task drops nothing again
        case null: { }
    }
}

// Make 'hdr' a fresh, unspawned header.
void task_header_init_(&TaskHeader hdr, fn int(?&TaskHeader, &Context) poll, fn void(?&TaskHeader) drop);
// Mark 'hdr' finished (aborted or not) and wake its JoinHandle.
void task_header_finish_(&TaskHeader hdr, int aborted);

generic<F: Future, T>
struct AsyncTask<F, T> AsyncTask::new(F fut) {
    struct AsyncTask<F, T> t;
    fn int(?&AsyncTask<F, T>, &Context) pf = async_task_poll_;
    fn void(?&AsyncTask<F, T>) df = async_task_drop_;
    unsafe { task_header_init_(&t.hdr, (fn int(?&TaskHeader, &Context))pf, (fn void(?&TaskHeader))df); }
    t.fut = fut;
    return t;
}

generic<F: Future, T>
int AsyncTask::is_finished() const {
    return (async_load_(&self.hdr.state) & ASYNC_TASK_DONE) != 0 ? 1 : 0;
}

generic<F: Future, T>
int AsyncTask::is_aborted() const {
    return (async_load_(&self.hdr.state) & ASYNC_TASK_ABORTED) != 0 ? 1 : 0;
}

generic<F: Future, T>
?T AsyncTask::take() {
    int st = async_fetch_or_(&self.hdr.state, ASYNC_TASK_TAKEN);
    if ((st & ASYNC_TASK_DONE) == 0 || (st & ASYNC_TASK_ABORTED) != 0 || (st & ASYNC_TASK_TAKEN) != 0) {
        if ((st & ASYNC_TASK_TAKEN) == 0) async_fetch_and_(&self.hdr.state, ~ASYNC_TASK_TAKEN);
        return null;
    }
    unsafe { return self.out[0]; }   // moves out: TAKEN keeps it from moving twice
}

generic<F: Future, T>
void AsyncTask::drop() {
    // A spawned task that hasn't finished is still in its executor's list:
    // dropping it (its future may borrow locals that are going away) would
    // leave the executor polling freed memory. Finish, or abort() and run
    // the executor until it is done, first.
    if (self.hdr.exec_ != (unsigned char*)0 && (async_load_(&self.hdr.state) & ASYNC_TASK_DONE) == 0)
        panic_at("AsyncTask dropped while its executor may still poll it", "std/async/executor.h", 0);
    self.fut.drop();                 // idempotent: a finished future drops nothing
    self.hdr.joiner.drop();
    int st = async_load_(&self.hdr.state);
    if ((st & ASYNC_TASK_DONE) != 0 && (st & (ASYNC_TASK_ABORTED | ASYNC_TASK_TAKEN)) == 0) {
        unsafe { drop(self.out[0]); }   // a finished task's output nobody took
    }
    unsafe { forget(self.out); }        // otherwise the slot holds nothing
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
            int st = async_load_(&t.state);
            if ((st & ASYNC_TASK_ABORTED) != 0) { ?T none = null; return Poll::ready(none); }
            if ((async_fetch_or_(&t.state, ASYNC_TASK_TAKEN) & ASYNC_TASK_TAKEN) != 0) {
                ?T taken = null;   // the output already moved out (take())
                return Poll::ready(taken);
            }
            match (self.out_) {
                case some(o): { ?T v = null; unsafe { v = *o; } return Poll::ready(v); }   // moves out once
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
        case some(t): return (async_load_(&t.state) & ASYNC_TASK_DONE) != 0 ? 1 : 0;
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

// Executors: make async_spawn on 'cx' call sp(exec, task).
generic<E>
void context_set_spawn_(&Context cx, fn int(?&E, ?&TaskHeader) sp, ?&E exec) {
    unsafe {
        cx.spawn_ = (?fn int(?&Erased, ?&TaskHeader))sp;
        cx.exec_  = (?&Erased)exec;
    }
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

generic<P>
void LocalExecutor::install_park_(?fn void(?&P) unpark, ?&P park) {
    unsafe {
        self.unpark_ = (?fn void(?&Erased))unpark;
        self.park_ = (?&Erased)park;
    }
}

generic<P: Park>
void LocalExecutor::run(&P park) {
    fn void(?&P) up = park_unpark_thunk_;
    self.install_park_(up, (?&P)park);
    while (self.live > 0) {
        self.turn(park.now_ms());
        if (self.live == 0 || self.has_ready() != 0) continue;
        park.park(self.timers.park_timeout());
    }
    self.unpark_ = null;
    self.park_ = null;
}

// The block_on root future's waker (data = the executor).
void local_root_wake_(?&LocalExecutor ex, unsigned long tag);
// A Context for polling on 'ex' with waker 'w'.
struct Context local_context_(&LocalExecutor ex, struct Waker w);

generic<P: Park, F: Future, T>
T LocalExecutor::block_on(&P park, &F fut) {
    fn void(?&P) up = park_unpark_thunk_;
    self.install_park_(up, (?&P)park);
    struct Context cx = local_context_(self, Waker::new(local_root_wake_, (?&LocalExecutor)self, 0UL));
    async_store_(&self.root_woken, 1);
    self.timers.advance(park.now_ms());
    while (1) {
        if (async_xchg_(&self.root_woken, 0) != 0) {
            struct Poll<T> p = fut.poll(&cx);
            if (p.ready_ != 0) {
                self.unpark_ = null;
                self.park_ = null;
                cx.drop();
                return p.unwrap();
            }
            p.drop();
        }
        self.turn(park.now_ms());
        if (async_load_(&self.root_woken) != 0 || self.has_ready() != 0) continue;
        park.park(self.timers.park_timeout());
    }
    struct Poll<T> never = fut.poll(&cx);   // unreachable
    return never.value;
}

} // namespace std
