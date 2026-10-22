// SafeC Standard Library — async: ThreadPool
#pragma once
#include <std/async/pool.h>
#include <std/async/host.h>
#include <std/panic.h>

namespace std {

// With mu held: append 't' to the run queue.
static void pool_push_(&ThreadPool p, &TaskHeader t) {
    t.next = null;
    match (p.qtail) {
        case some(last): last.next = t;
        case null: p.qhead = t;
    }
    p.qtail = t;
}

// With mu held: take the oldest queued task.
static ?&TaskHeader pool_pop_(&ThreadPool p) {
    ?&TaskHeader h = p.qhead;
    match (h) {
        case some(t): {
            p.qhead = t.next;
            match (p.qhead) {
                case null: p.qtail = null;
                case some(x): { }
            }
            t.next = null;
        }
        case null: { }
    }
    return h;
}

// A pool task's waker: queue it unless it is queued, running (then it is
// re-queued after the poll) or done.
static void pool_task_wake_(?&TaskHeader tp, unsigned long tag) {
    match (tp) {
        case some(t): {
            ?&ThreadPool pq;
            unsafe { pq = (?&ThreadPool)t.exec_; }
            match (pq) {
                case some(p): {
                    mutex_lock(&p.mu);
                    int st = async_fetch_or_(&t.state, ASYNC_TASK_NOTIFIED);
                    if ((st & (ASYNC_TASK_DONE | ASYNC_TASK_RUNNING | ASYNC_TASK_QUEUED)) == 0) {
                        async_fetch_or_(&t.state, ASYNC_TASK_QUEUED);
                        pool_push_(p, t);
                        cond_signal(&p.work_cv);
                    }
                    mutex_unlock(&p.mu);
                }
                case null: { }
            }
        }
        case null: { }
    }
}

static int pool_spawn_thunk_(?&ThreadPool pq, ?&TaskHeader t) {
    match (pq) {
        case some(p): return p.spawn_raw_(t);
        case null: return 0;
    }
    return 0;
}

struct ThreadPool ThreadPool::new() {
    struct ThreadPool p;
    p.mu = 0ULL;
    p.work_cv = 0ULL;
    p.done_cv = 0ULL;
    p.qhead = null;
    p.qtail = null;
    p.live = 0;
    p.stopping = 0;
    p.nthreads = 0;
    p.idle = 0;
    int i = 0;
    while (i < ASYNC_POOL_MAX_THREADS) { p.tids[i] = 0ULL; i = i + 1; }
    p.timers = TimerQueue::new();
    p.polls = 0LL;
    return p;
}

// Polls (or drops) one task taken off the queue; mu is not held.
static void pool_run_one_(&ThreadPool p, &TaskHeader t) {
    int st = async_load_(&t.state);
    int done = 0;
    int aborted = 0;
    if ((st & ASYNC_TASK_ABORTED) != 0) {
        unsafe {
            fn void(?&TaskHeader) df = (fn void(?&TaskHeader))t.drop_;
            df(t);
        }
        done = 1;
        aborted = 1;
    } else {
        struct Context cx = Context::from_waker(Waker::new(pool_task_wake_, (?&TaskHeader)t, 0UL));
        cx.timers = &p.timers;
        fn int(?&ThreadPool, ?&TaskHeader) sp = pool_spawn_thunk_;
        unsafe {
            cx.exec_ = (void*)p;
            cx.spawn_ = (void*)sp;
            fn int(?&TaskHeader, &Context) pf = (fn int(?&TaskHeader, &Context))t.poll_;
            done = pf(t, &cx);
        }
    }
    if (done != 0) {
        task_header_finish_(t, aborted);
        mutex_lock(&p.mu);
        p.live = p.live - 1;
        p.polls = p.polls + 1LL;
        cond_broadcast(&p.done_cv);
        mutex_unlock(&p.mu);
        return;
    }
    mutex_lock(&p.mu);
    p.polls = p.polls + 1LL;
    int now = async_fetch_and_(&t.state, ~ASYNC_TASK_RUNNING);
    if ((now & (ASYNC_TASK_NOTIFIED | ASYNC_TASK_ABORTED)) != 0) {
        // Woken (or aborted) while it ran: run it again.
        async_fetch_or_(&t.state, ASYNC_TASK_QUEUED);
        pool_push_(p, t);
        cond_signal(&p.work_cv);
    }
    mutex_unlock(&p.mu);
}

static ?&ThreadPool pool_worker_(?&ThreadPool pq) {
    match (pq) {
        case some(p): {
            mutex_lock(&p.mu);
            while (p.stopping == 0) {
                ?&TaskHeader next = pool_pop_(p);
                match (next) {
                    case some(t): {
                        // Clear NOTIFIED before polling: a wake from now on
                        // re-queues it after this poll.
                        async_fetch_and_(&t.state, ~(ASYNC_TASK_QUEUED | ASYNC_TASK_NOTIFIED));
                        async_fetch_or_(&t.state, ASYNC_TASK_RUNNING);
                        mutex_unlock(&p.mu);
                        pool_run_one_(p, t);
                        // Keep the timers going while there is always work:
                        // otherwise a busy pool would never fire them.
                        p.timers.advance(async_mono_ms());
                        mutex_lock(&p.mu);
                    }
                    case null: {
                        // Nothing to run: sleep until work arrives or the next
                        // timer is due, then fire the timers (without mu:
                        // firing wakes tasks, which takes mu).
                        long long to = p.timers.park_timeout();
                        p.idle = p.idle + 1;
                        if (to < 0LL) {
                            cond_wait(&p.work_cv, &p.mu);
                        } else if (to > 0LL) {
                            cond_timedwait_ms(&p.work_cv, &p.mu, (unsigned long)to);
                        }
                        p.idle = p.idle - 1;
                        mutex_unlock(&p.mu);
                        p.timers.advance(async_mono_ms());
                        mutex_lock(&p.mu);
                    }
                }
            }
            mutex_unlock(&p.mu);
        }
        case null: { }
    }
    return pq;
}

int ThreadPool::start(int n) {
    if (n < 1) n = 1;
    if (n > ASYNC_POOL_MAX_THREADS) n = ASYNC_POOL_MAX_THREADS;
    if (mutex_init(&self.mu) != 0) return -1;
    if (cond_init(&self.work_cv) != 0) return -1;
    if (cond_init(&self.done_cv) != 0) return -1;
    self.timers.advance(async_mono_ms());
    int i = 0;
    while (i < n) {
        if (thread_create(&self.tids[i], pool_worker_, (?&ThreadPool)self) != 0) break;
        i = i + 1;
    }
    self.nthreads = i;
    return i == n ? 0 : -1;
}

int ThreadPool::spawn_raw_(?&TaskHeader tq) {
    match (tq) {
        case some(t): {
            unsafe {
                if (t.exec_ != (void*)0) {
                    panic_at("std/async: task spawned twice", "std/async/pool.sc", 0);
                    return 0;
                }
            }
            // The timers count from the pool's start; keep the clock current
            // for the new task's first Sleep.
            self.timers.advance(async_mono_ms());
            mutex_lock(&self.mu);
            fn void(?&TaskHeader, unsigned long) wk = pool_task_wake_;
            unsafe {
                t.exec_ = (void*)self;
                t.wake_ = (void*)wk;
            }
            self.live = self.live + 1;
            async_fetch_or_(&t.state, ASYNC_TASK_QUEUED);
            pool_push_(self, t);
            cond_signal(&self.work_cv);
            mutex_unlock(&self.mu);
            return 1;
        }
        case null: return 0;
    }
    return 0;
}

void ThreadPool::wait_task_(&TaskHeader hdr) {
    mutex_lock(&self.mu);
    while ((async_load_(&hdr.state) & ASYNC_TASK_DONE) == 0) cond_wait(&self.done_cv, &self.mu);
    mutex_unlock(&self.mu);
}

void ThreadPool::wait_idle() {
    mutex_lock(&self.mu);
    while (self.live > 0) cond_wait(&self.done_cv, &self.mu);
    mutex_unlock(&self.mu);
}

void ThreadPool::shutdown() {
    if (self.nthreads == 0) return;
    mutex_lock(&self.mu);
    self.stopping = 1;
    cond_broadcast(&self.work_cv);
    mutex_unlock(&self.mu);
    int i = 0;
    while (i < self.nthreads) { thread_join(self.tids[i]); i = i + 1; }
    self.nthreads = 0;
    cond_destroy(&self.work_cv);
    cond_destroy(&self.done_cv);
    mutex_destroy(&self.mu);
}

int ThreadPool::live_count() {
    mutex_lock(&self.mu);
    int n = self.live;
    mutex_unlock(&self.mu);
    return n;
}

} // namespace std
