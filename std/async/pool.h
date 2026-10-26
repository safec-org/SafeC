// SafeC Standard Library — async: ThreadPool, the multi-threaded executor
//
// N worker threads (std/thread.h) share one FIFO run queue: a woken task
// is queued once, and whichever worker is free polls it next (a shared-
// queue executor, like tokio's multi-thread runtime minus work stealing).
// The task types are the executor-independent ones from executor.h —
// AsyncTask, JoinHandle — and so are timers (one TimerQueue shared by the
// workers), channels and the sync primitives, which all lock internally.
// A task never runs on two threads at once: the RUNNING/NOTIFIED bits in
// its header make a wake during a poll re-queue it after the poll.
//
//   static struct ThreadPool pool;               // must not move once started
//   pool = ThreadPool::new();
//   pool.start(4);
//   struct JoinHandle<int> h = pool.spawn(&task);   // task storage as for LocalExecutor
//   int v = pool.block_on(Fetch::new());          // spawn + wait, from a non-worker thread
//   pool.wait_idle();                              // every spawned task finished
//   pool.shutdown();                               // stop and join the workers
//
// The futures it runs may be polled on any worker, one after another, so
// whatever they share must be thread-safe (std/async's primitives are;
// plain globals need atomics or a lock). I/O futures (std/async/net.h)
// need a Runtime and don't run here. Hosted targets only.
#pragma once
#include <std/async/executor.h>
#include <std/thread.h>

#define ASYNC_POOL_MAX_THREADS 64

namespace std {

struct ThreadPool {
    unsafe impl Sync;   // shared with its workers; state changes under 'mu'
    unsigned long long mu;         // guards the queue, live, the flags
    unsigned long long work_cv;    // signalled when a task is queued (or on shutdown)
    unsigned long long done_cv;    // broadcast when a task finishes
    ?&TaskHeader       qhead;      // run queue, linked through TaskHeader::next
    ?&TaskHeader       qtail;
    int                live;       // spawned and not finished
    int                stopping;
    int                nthreads;
    int                idle;       // workers waiting for work
    unsigned long long tids[ASYNC_POOL_MAX_THREADS];
    struct TimerQueue  timers;
    long long          polls;      // statistics

    // A pool with no threads yet. Put it where it will stay, then start().
    static struct ThreadPool new();
    // Start 'n' worker threads (1..ASYNC_POOL_MAX_THREADS). 0 on success.
    int  start(int n);
    // F and T must be Send: the task runs (and finishes) on a worker thread.
    generic<F: Send, T: Send> struct JoinHandle<T> spawn(&AsyncTask<F, T> task);
    int  spawn_raw_(?&TaskHeader t);
    // Run 'fut' as a task on the pool and wait (this thread blocks) for its
    // value. Call from outside the pool's workers.
    generic<F: Future + Send, T: Send> T block_on(F fut);
    // Block until the task behind 'hdr' is done.
    void wait_task_(&TaskHeader hdr);
    // Block until every spawned task has finished.
    void wait_idle();
    // Stop the workers (after their current poll) and join them. Tasks
    // still queued or waiting are abandoned, not dropped.
    void drop();
    void shutdown() consume;   // = drop()
    int  live_count();
};

generic<F: Send, T: Send>
struct JoinHandle<T> ThreadPool::spawn(&AsyncTask<F, T> task) {
    self.spawn_raw_(&task.hdr);
    return join_handle_for_(task);
}

generic<F: Future + Send, T: Send>
T ThreadPool::block_on(F fut) {
    struct AsyncTask<F, T> t = AsyncTask::new(fut);
    self.spawn_raw_(&t.hdr);
    self.wait_task_(&t.hdr);
    ?T out = t.take();
    t.drop();
    match (out) {
        case some(v): return v;
        case none: { }
    }
    panic_at("ThreadPool::block_on: the task was aborted", "pool.h", 0);
    unsafe { return *(T*)(unsigned char*)&t.out[0]; }   // unreachable: panic_at doesn't return
}

} // namespace std
