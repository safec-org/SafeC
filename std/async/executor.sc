// SafeC Standard Library — async: task headers, JoinHandle, LocalExecutor, Park sources
#pragma once
#include <std/async/executor.h>
#include <std/panic.h>

namespace std {

// ── TaskHeader ─────────────────────────────────────────────────────────────

void task_header_init_(&TaskHeader hdr, fn int(?&TaskHeader, &Context) poll, fn void(?&TaskHeader) drop) {
    hdr.poll_ = poll;
    hdr.drop_ = drop;
    hdr.exec_ = null;
    hdr.wake_ = null;
    hdr.state = 0;
    hdr.next = null;
    hdr.joiner = WakerSlot::new();
    hdr.lock = Spinlock::new();
}

void task_header_finish_(&TaskHeader hdr, int aborted) {
    hdr.lock.lock();
    int bits = ASYNC_TASK_DONE;
    if (aborted != 0) bits = bits | ASYNC_TASK_ABORTED;
    async_fetch_or_(&hdr.state, bits);
    async_fetch_and_(&hdr.state, ~(ASYNC_TASK_RUNNING | ASYNC_TASK_NOTIFIED | ASYNC_TASK_QUEUED));
    hdr.joiner.wake();
    hdr.lock.unlock();
}

int task_header_poll_join_(&TaskHeader hdr, const &Waker w) {
    hdr.lock.lock();
    if ((async_load_(&hdr.state) & ASYNC_TASK_DONE) != 0) {
        hdr.lock.unlock();
        return 1;
    }
    hdr.joiner.register_(w);
    hdr.lock.unlock();
    return 0;
}

// Every executor's wake starts by setting NOTIFIED and makes sure the task
// is polled; abort sets ABORTED first and wakes the task the usual way,
// and the executor then drops the future instead of polling it.
void task_header_abort_(&TaskHeader hdr) {
    int st = async_fetch_or_(&hdr.state, ASYNC_TASK_ABORTED);
    if ((st & ASYNC_TASK_DONE) != 0) {
        // Too late: it finished. Leave the result as it is.
        if ((st & ASYNC_TASK_ABORTED) == 0) async_fetch_and_(&hdr.state, ~ASYNC_TASK_ABORTED);
        return;
    }
    match (hdr.wake_) {
        case some(f): f(hdr, 0UL);
        case null: { }
    }
}

int context_spawn_raw_(&Context cx, ?&TaskHeader t) {
    match (cx.spawn_) {
        case some(f): if (!cx.exec_.is_null()) return f(cx.exec_, t);
        case null: { }
    }
    panic_at("std/async: async_spawn from a context without an executor", "std/async/executor.sc", 0);
    return 0;
}

// ── LocalExecutor ──────────────────────────────────────────────────────────

static void local_unpark_(&LocalExecutor ex) {
    match (ex.unpark_) {
        case some(f): f(ex.park_);
        case null: { }
    }
}

// A task's waker on a LocalExecutor: two atomic stores, then unpark —
// callable from any thread or an interrupt handler.
static void local_task_wake_(?&TaskHeader tp, unsigned long tag) {
    match (tp) {
        case some(t): {
            int st = async_fetch_or_(&t.state, ASYNC_TASK_NOTIFIED);
            if ((st & ASYNC_TASK_DONE) != 0) return;
            ?&LocalExecutor exq;
            unsafe { exq = (?&LocalExecutor)t.exec_; }
            match (exq) {
                case some(ex): {
                    async_store_(&ex.notified, 1);
                    local_unpark_(ex);
                }
                case null: { }
            }
        }
        case null: { }
    }
}

void local_root_wake_(?&LocalExecutor exq, unsigned long tag) {
    match (exq) {
        case some(ex): {
            async_store_(&ex.root_woken, 1);
            async_store_(&ex.notified, 1);
            local_unpark_(ex);
        }
        case null: { }
    }
}

static int local_spawn_thunk_(?&LocalExecutor exq, ?&TaskHeader t) {
    match (exq) {
        case some(ex): return ex.spawn_raw_(t);
        case null: return 0;
    }
    return 0;
}

struct Context local_context_(&LocalExecutor ex, struct Waker w) {
    struct Context cx = Context::from_waker(w);
    cx.timers = &ex.timers;
    cx.io_ = ex.io_;
    context_set_spawn_(&cx, local_spawn_thunk_, (?&LocalExecutor)ex);
    return cx;
}

void LocalExecutor::drop() { self.timers.drop(); }

struct LocalExecutor LocalExecutor::new() {
    struct LocalExecutor ex;
    ex.head = null;
    ex.tail = null;
    ex.live = 0;
    ex.notified = 0;
    ex.root_woken = 0;
    ex.timers = TimerQueue::new();
    ex.io_ = null;
    ex.unpark_ = null;
    ex.park_ = null;
    ex.polls = 0LL;
    return ex;
}

int LocalExecutor::spawn_raw_(?&TaskHeader tq) {
    match (tq) {
        case some(t): {
            if (!t.exec_.is_null()) {
                panic_at("std/async: task spawned twice", "std/async/executor.sc", 0);
                return 0;
            }
            unsafe { t.exec_ = (?&Erased)self; }
            t.wake_ = local_task_wake_;
            t.next = null;
            match (self.tail) {
                case some(last): last.next = t;
                case null: self.head = t;
            }
            self.tail = t;
            self.live = self.live + 1;
            // Polled on the next round.
            async_fetch_or_(&t.state, ASYNC_TASK_NOTIFIED);
            async_store_(&self.notified, 1);
            return 1;
        }
        case null: return 0;
    }
    return 0;
}

// Polls (or drops, if aborted) one woken task. Returns 1 if it finished.
static int local_step_(&LocalExecutor ex, &TaskHeader t) {
    int st = async_fetch_and_(&t.state, ~ASYNC_TASK_NOTIFIED);
    if ((st & ASYNC_TASK_DONE) != 0) return 1;
    if ((st & ASYNC_TASK_ABORTED) != 0) {
        t.drop_(t);
        task_header_finish_(t, 1);
        return 1;
    }
    async_fetch_or_(&t.state, ASYNC_TASK_RUNNING);
    fn void(?&TaskHeader, unsigned long) wk = local_task_wake_;
    struct Context cx = local_context_(ex, Waker::new(wk, (?&TaskHeader)t, 0UL));
    int done = t.poll_(t, &cx);
    cx.drop();
    ex.polls = ex.polls + 1LL;
    async_fetch_and_(&t.state, ~ASYNC_TASK_RUNNING);
    if (done != 0) {
        task_header_finish_(t, 0);
        return 1;
    }
    return 0;
}

int LocalExecutor::run_ready() {
    if (async_xchg_(&self.notified, 0) == 0) return 0;
    int n = 0;
    ?&TaskHeader prev = null;
    ?&TaskHeader cur = self.head;
    int going = 1;
    while (going != 0) {
        match (cur) {
            case null: going = 0;
            case some(t): {
                int finished = 0;
                int st = async_load_(&t.state);
                if ((st & ASYNC_TASK_DONE) != 0) {
                    finished = 1;
                } else if ((st & ASYNC_TASK_NOTIFIED) != 0) {
                    finished = local_step_(self, t);
                    n = n + 1;
                }
                ?&TaskHeader nxt = t.next;
                if (finished != 0) {
                    // Unlink it: the executor forgets finished tasks.
                    match (prev) {
                        case some(p): p.next = nxt;
                        case null: self.head = nxt;
                    }
                    match (nxt) {
                        case null: self.tail = prev;
                        case some(x): { }
                    }
                    t.next = null;
                    self.live = self.live - 1;
                } else {
                    prev = t;
                }
                cur = nxt;
            }
        }
    }
    return n;
}

int LocalExecutor::turn(long long now_ms) {
    self.timers.advance(now_ms);
    return self.run_ready();
}

int LocalExecutor::live_count() const { return self.live; }

int LocalExecutor::has_ready() const {
    return async_load_(&self.notified) != 0 ? 1 : 0;
}

// ── Park sources ───────────────────────────────────────────────────────────

struct VirtualClock VirtualClock::new(long long start_ms) {
    struct VirtualClock c;
    c.now = start_ms;
    c.parked = 0LL;
    return c;
}

long long VirtualClock::now_ms() { return self.now; }

void VirtualClock::park(long long timeout_ms) {
    self.parked = self.parked + 1LL;
    if (timeout_ms < 0LL) {
        panic_at("std/async: deadlock -- every task is waiting and no timer is pending (VirtualClock)",
                 "std/async/executor.sc", 0);
        return;
    }
    self.now = self.now + timeout_ms;
}

void VirtualClock::unpark() { }

struct FnPark FnPark::new(fn long long() now, fn void(long long) idle) {
    struct FnPark p;
    p.now_fn = now;
    p.idle_fn = idle;
    p.flag = null;
    return p;
}

long long FnPark::now_ms() { return self.now_fn(); }
void FnPark::park(long long timeout_ms) { self.idle_fn(timeout_ms); }
void FnPark::unpark() { }

} // namespace std
