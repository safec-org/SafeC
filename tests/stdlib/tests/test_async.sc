// std::async — Poll/Waker/Context, goto coroutines, LocalExecutor (spawn,
// JoinHandle, abort, async_spawn, block_on, wakes from other threads),
// timers (Sleep, Interval, a full wheel), combinators (join, select,
// timeout, map, and_then, join_all, select_any), channels (Oneshot,
// AsyncChannel), AsyncMutex/Semaphore/Notify/CancelToken, the ThreadPool,
// host block_on with real time, and the Runtime's async TCP over loopback
// (echo with concurrent clients, refused connects, timeouts, a small HTTP
// exchange). Virtual-clock runs check exact times: they are deterministic.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/thread.h>
#include <std/atomic.h>
#include <std/async/async.h>
#include <std/async/host.h>
#include <std/async/pool.h>
#include <std/async/net.h>

// ── helpers ────────────────────────────────────────────────────────────────

static int g_log[256];
static int g_nlog = 0;
static int g_drops = 0;

static void log_(int v) {
    if (g_nlog < 256) { g_log[g_nlog] = v; g_nlog = g_nlog + 1; }
}

static void log_reset_() { g_nlog = 0; g_drops = 0; }

static int log_is_(const &int want, int n) {
    if (g_nlog != n) return 0;
    int i = 0;
    while (i < n) {
        unsafe { if (g_log[i] != want[i]) return 0; }
        i = i + 1;
    }
    return 1;
}

// Counts wakes into an int.
static void count_wake_(?&int n, unsigned long tag) {
    match (n) {
        case some(p): *p = *p + (int)tag;
        case null: { }
    }
}

static int g_clones = 0;
static int g_wdrops = 0;
static void count_clone_(?&int n, unsigned long tag) { g_clones = g_clones + 1; }
static void count_drop_(?&int n, unsigned long tag) { g_wdrops = g_wdrops + 1; }

// Ready with 'v' after sleeping 'ms'; counts drops.
struct Val {
    int          st;
    int          v;
    struct Sleep d;
    static struct Val new(int v, long long ms);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Val Val::new(int v, long long ms) {
    struct Val x;
    x.st = ASYNC_INIT;
    x.v = v;
    x.d = Sleep::new(ms);
    return x;
}

struct Poll<int> Val::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    AWAIT(1, self.d);
    log_(self.v);
    ASYNC_RETURN(self.v);
    ASYNC_END;
}

void Val::drop() {
    if (self.st == 1) self.d.drop();
    if (self.st != ASYNC_DONE) g_drops = g_drops + 1;
    self.st = ASYNC_DONE;
}

// Logs id*100+i for i in 0..n, sleeping 'ms' (0: yielding) between.
struct Ticker {
    int          st;
    int          id;
    int          n;
    int          i;
    long long    ms;
    struct Sleep d;
    static struct Ticker new(int id, int n, long long ms);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Ticker Ticker::new(int id, int n, long long ms) {
    struct Ticker t;
    t.st = ASYNC_INIT;
    t.id = id;
    t.n = n;
    t.i = 0;
    t.ms = ms;
    t.d = Sleep::new(0LL);
    return t;
}

struct Poll<int> Ticker::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    for (self.i = 0; self.i < self.n; self.i = self.i + 1) {
        log_(self.id * 100 + self.i);
        if (self.ms > 0LL) {
            self.d = Sleep::new(self.ms);
            AWAIT(1, self.d);
        } else {
            ASYNC_YIELD(2);
        }
    }
    ASYNC_RETURN(self.id);
    ASYNC_END;
}

void Ticker::drop() {
    if (self.st == 1) self.d.drop();
    if (self.st != ASYNC_DONE) g_drops = g_drops + 1;
    self.st = ASYNC_DONE;
}

// ── Poll, Waker, WakerSlot, WaitList ───────────────────────────────────────

void t_poll_waker() {
    struct Poll<int> r = Poll::ready(7);
    struct Poll<int> p = Poll::pending();
    ASSERT_TRUE(r.is_ready());
    ASSERT_TRUE(p.is_pending());
    ASSERT_EQ(r.take().default(0), 7);
    ASSERT_TRUE(p.take().is_none());

    int hits = 0;
    struct Waker w = Waker::new(count_wake_, (?&int)&hits, 1UL); defer drop(w);
    w.wake_by_ref();
    w.wake_by_ref();
    ASSERT_EQ(hits, 2);
    struct Waker w2 = w.clone();
    ASSERT_TRUE(w.will_wake(&w2));
    struct Waker w3 = Waker::new(count_wake_, (?&int)&hits, 5UL); defer drop(w3);
    ASSERT_FALSE(w.will_wake(&w3));
    w2.wake();                 // consumes (using w2 after this is rejected)
    ASSERT_EQ(hits, 3);
    struct Waker nw = Waker::noop();
    ASSERT_TRUE(nw.is_noop());
    nw.drop();
    ASSERT_FALSE(w.is_noop());

    // Counted wakers: clone/drop hooks run once per copy.
    g_clones = 0;
    g_wdrops = 0;
    struct Waker c = Waker::counted(count_wake_, count_clone_, count_drop_, (?&int)&hits, 1UL);
    struct Waker c2 = c.clone();
    struct Waker c3 = c2.clone();
    c2.wake();
    c3.drop();
    c.drop();
    ASSERT_EQ(g_clones, 2);
    ASSERT_EQ(g_wdrops, 3);
    ASSERT_EQ(hits, 4);

    // WakerSlot keeps one; re-registering the same waker doesn't clone again.
    g_clones = 0;
    struct WakerSlot slot = WakerSlot::new(); defer drop(slot);
    struct Waker cw = Waker::counted(count_wake_, count_clone_, count_drop_, (?&int)&hits, 10UL); defer drop(cw);
    slot.register_(&cw);
    slot.register_(&cw);
    ASSERT_EQ(g_clones, 1);
    ASSERT_EQ(slot.wake(), 1);
    ASSERT_EQ(hits, 14);
    ASSERT_EQ(slot.wake(), 0);

    // WaitList: ids, stale ids, wake_one fairness, full list.
    struct WaitList wl = WaitList::new(); defer drop(wl);
    int a = wl.add(0, &w);
    int b = wl.add(0, &w3);
    ASSERT_TRUE(a > 0 && b > 0 && a != b);
    ASSERT_EQ(wl.add(a, &w), a);       // refresh keeps the id
    ASSERT_EQ(wl.len(), 2);
    hits = 0;
    ASSERT_EQ(wl.wake_one(), 1);
    ASSERT_EQ(hits, 1);                // 'a' (tag 1) went first
    ASSERT_FALSE(wl.contains(a));
    int a2 = wl.add(0, &w);            // may reuse a's slot: a new id
    ASSERT_TRUE(a2 != a);
    wl.remove(a);                      // stale: ignored
    ASSERT_TRUE(wl.contains(a2));
    ASSERT_EQ(wl.wake_all(), 2);
    ASSERT_EQ(hits, 1 + 1 + 5);
    ASSERT_EQ(wl.len(), 0);
    int i = 0;
    int last = 1;
    while (i < ASYNC_WAITERS) { last = wl.add(0, &w); i = i + 1; }
    ASSERT_TRUE(last > 0);
    ASSERT_EQ(wl.add(0, &w), 0);       // full
    ASSERT_EQ(wl.wake_all(), ASYNC_WAITERS);

    // Leaf futures.
    struct Context cx = Context::noop(); defer drop(cx);
    struct Ready<int> rd = Ready::new(3); defer drop(rd);
    ASSERT_EQ(rd.poll(&cx).take().default(0), 3);
    struct Pending<int> pn = Pending::new(); defer drop(pn);
    ASSERT_TRUE(pn.poll(&cx).is_pending());
    hits = 0;
    struct Context cx2 = Context::from_waker(Waker::new(count_wake_, (?&int)&hits, 1UL)); defer drop(cx2);
    struct YieldNow y = YieldNow::new(); defer drop(y);
    ASSERT_TRUE(y.poll(&cx2).is_pending());
    ASSERT_EQ(hits, 1);                // woke itself
    ASSERT_TRUE(y.poll(&cx2).is_ready());
}

// ── coroutines polled by hand ──────────────────────────────────────────────

// Exercises every macro: two awaits in a loop, AWAIT_INTO, a yield, and
// ASYNC_RETURN as a match arm.
struct Coro {
    int              st;
    int              i;
    int              got;
    int              total;
    struct Ready<int> r;
    struct YieldNow  y;
    ?int             maybe;
    static struct Coro new();
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Coro Coro::new() {
    struct Coro c;
    c.st = ASYNC_INIT;
    c.i = 0;
    c.got = 0;
    c.total = 0;
    c.maybe = null;
    return c;
}

struct Poll<int> Coro::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 3);
    for (self.i = 1; self.i <= 4; self.i = self.i + 1) {
        self.y = YieldNow::new();
        AWAIT(1, self.y);
        self.r = Ready::new(self.i * 10);
        AWAIT_INTO(2, self.got, self.r);
        { int scratch = self.got + 1; self.total = self.total + scratch; }   // block ends before the next await
    }
    ASYNC_YIELD(3);
    self.maybe = self.total;
    match (self.maybe) {
        case some(v): ASYNC_RETURN(v);
        case none: ASYNC_RETURN(-1);
    }
    ASYNC_END;
}

void Coro::drop() {
    if (self.st == 1) self.y.drop();
    if (self.st == 2) self.r.drop();
    self.st = ASYNC_DONE;
}

void t_coroutine() {
    int hits = 0;
    struct Context cx = Context::from_waker(Waker::new(count_wake_, (?&int)&hits, 1UL)); defer drop(cx);
    struct Coro c = Coro::new(); defer drop(c);
    int polls = 0;
    struct Poll<int> p = c.poll(&cx);
    polls = 1;
    while (p.is_pending()) { p = c.poll(&cx); polls = polls + 1; }
    ASSERT_EQ(p.value, 11 + 21 + 31 + 41);
    ASSERT_EQ(polls, 6);               // 4 yields, one ASYNC_YIELD, then done
    ASSERT_EQ(hits, 5);                // every Pending woke itself first
    ASSERT_EQ(c.st, ASYNC_DONE);
}

// ── LocalExecutor ──────────────────────────────────────────────────────────

static struct AsyncTask<struct Ticker, int> g_tt[4];

void t_executor_order() {
    log_reset_();
    struct LocalExecutor ex = LocalExecutor::new(); defer drop(ex);
    struct VirtualClock clk = VirtualClock::new(0LL);
    // Yielding tickers interleave round-robin in spawn order.
    g_tt[0] = AsyncTask::new(Ticker::new(1, 3, 0LL));
    g_tt[1] = AsyncTask::new(Ticker::new(2, 2, 0LL));
    g_tt[2] = AsyncTask::new(Ticker::new(3, 3, 0LL));
    struct JoinHandle<int> h0 = ex.spawn(&g_tt[0]); defer drop(h0);
    drop(ex.spawn(&g_tt[1]));
    drop(ex.spawn(&g_tt[2]));
    ASSERT_EQ(ex.live_count(), 3);
    ex.run(&clk);
    int want[8] = {100, 200, 300, 101, 201, 301, 102, 302};
    ASSERT_TRUE(log_is_(&want[0], 8));
    ASSERT_EQ(ex.live_count(), 0);
    ASSERT_TRUE(h0.is_finished());
    ASSERT_EQ(g_tt[2].take().default(0), 3);
    ASSERT_EQ(clk.now, 0LL);           // nothing slept
    ASSERT_EQ(clk.parked, 0LL);

    // Sleeping tickers: wake-ups ordered by virtual time, exact.
    log_reset_();
    g_tt[0] = AsyncTask::new(Ticker::new(1, 3, 30LL));   // logs at 0, 30, 60
    g_tt[1] = AsyncTask::new(Ticker::new(2, 4, 20LL));   // 0, 20, 40, 60
    drop(ex.spawn(&g_tt[0]));
    drop(ex.spawn(&g_tt[1]));
    ex.run(&clk);
    int want2[7] = {100, 200, 201, 101, 202, 102, 203};
    ASSERT_TRUE(log_is_(&want2[0], 7));
    ASSERT_EQ(clk.now, 90LL);          // ticker 1's last sleep ends at 90
}

// Awaits another task's JoinHandle, then spawns a child from inside.
struct Parent {
    int                       st;
    struct JoinHandle<int>    h;
    ?int                      got;
    struct AsyncTask<struct Val, int> child;
    struct JoinHandle<int>    ch;
    ?int                      child_got;
    int                       spawned;   // child/ch hold a task
    static struct Parent new(struct JoinHandle<int> h);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Parent Parent::new(struct JoinHandle<int> h) {
    struct Parent p;
    p.st = ASYNC_INIT;
    p.h = h;
    p.got = null;
    p.child_got = null;
    p.spawned = 0;
    return p;
}

struct Poll<int> Parent::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    AWAIT_INTO(1, self.got, self.h);
    self.child = AsyncTask::new(Val::new(5, 10LL));
    self.ch = async_spawn(cx, &self.child);
    self.spawned = 1;
    AWAIT_INTO(2, self.child_got, self.ch);
    ASYNC_RETURN(self.got.default(-100) + self.child_got.default(-1000));
    ASYNC_END;
}

void Parent::drop() {
    self.h.drop();
    if (self.spawned != 0) { self.ch.drop(); self.child.drop(); self.spawned = 0; }
    self.st = ASYNC_DONE;
}

static struct AsyncTask<struct Val, int> g_vt[4];
static struct AsyncTask<struct Parent, int> g_pt;

void t_executor_join_abort() {
    log_reset_();
    struct LocalExecutor ex = LocalExecutor::new(); defer drop(ex);
    struct VirtualClock clk = VirtualClock::new(1000LL);
    g_vt[0] = AsyncTask::new(Val::new(40, 25LL));
    g_vt[1] = AsyncTask::new(Val::new(41, 500LL));
    struct JoinHandle<int> h0 = ex.spawn(&g_vt[0]);
    struct JoinHandle<int> h1 = ex.spawn(&g_vt[1]); defer drop(h1);
    g_pt = AsyncTask::new(Parent::new(h0));
    drop(ex.spawn(&g_pt));
    // Run to t=1100, then abort the 500 ms one: its timer goes away with it.
    while (clk.now < 1100LL && ex.live_count() > 0) {
        ex.turn(clk.now);
        if (ex.has_ready()) continue;
        long long to = ex.timers.park_timeout();
        if (to < 0LL || clk.now + to > 1100LL) to = 1100LL - clk.now;
        clk.park(to);
    }
    ASSERT_EQ(g_pt.take().default(0), 45);      // 40 from h0 + 5 from the child it spawned
    ASSERT_EQ(ex.timers.pending(), 1);
    h1.abort();
    ex.run(&clk);
    ASSERT_TRUE(g_vt[1].is_finished());
    ASSERT_TRUE(g_vt[1].is_aborted());
    ASSERT_TRUE(g_vt[1].take().is_none());
    ASSERT_EQ(g_drops, 1);
    ASSERT_EQ(ex.timers.pending(), 0);
    ASSERT_EQ(clk.now, 1100LL);                 // no park waited for the aborted timer
    // Aborting a finished task changes nothing (h0 itself moved into the
    // Parent: abort through another handle to the same task).
    struct JoinHandle<int> h0b = std::join_handle_for_(&g_vt[0]);
    h0b.abort();
    h0b.drop();
    ASSERT_FALSE(g_vt[0].is_aborted());
    ASSERT_TRUE(g_vt[0].take().is_none());      // its output (40) already moved to the Parent through h0
    // An aborted task's handle reports null.
    struct Context cx = Context::noop(); defer drop(cx);
    struct Poll<?int> hp = h1.poll(&cx);
    ASSERT_TRUE(hp.is_ready());
    ASSERT_TRUE(hp.value.is_none());

    // block_on runs the root and the spawned tasks together.
    log_reset_();
    g_vt[2] = AsyncTask::new(Val::new(7, 5LL));
    drop(ex.spawn(&g_vt[2]));
    struct Val root = Val::new(9, 20LL); defer drop(root);
    int r = ex.block_on(&clk, &root);
    ASSERT_EQ(r, 9);
    int want[2] = {7, 9};
    ASSERT_TRUE(log_is_(&want[0], 2));
    ASSERT_EQ(clk.now, 1120LL);
}

// A task waits on a oneshot; an OS thread sends after 30 ms. The executor
// parks (HostPark / IoDriver) and must be woken from the other thread.
static struct Oneshot<int> g_os;

struct WaitOneshot {
    int                    st;
    struct OneshotRecv<int> rx;
    ?int                   got;
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> WaitOneshot::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    self.rx = g_os.recv();
    AWAIT_INTO(1, self.got, self.rx);
    ASYNC_RETURN(self.got.default(-1));
    ASYNC_END;
}

void WaitOneshot::drop() { if (self.st == 1) self.rx.drop(); self.st = ASYNC_DONE; }

static ?&int send_later_(?&int ms) {
    match (ms) {
        case some(p): thread_sleep_ms((unsigned long)*p);
        case null: { }
    }
    g_os.send(77);
    return ms;
}

void t_remote_wake() {
    // HostPark (condition variable).
    g_os = Oneshot::new();
    int delay = 30;
    unsigned long long tid = 0ULL;
    long long t0 = async_mono_ms();
    ASSERT_EQ(thread_create(&tid, send_later_, (?&int)&delay), 0);
    struct WaitOneshot w; defer drop(w);
    w.st = ASYNC_INIT;
    w.got = null;
    int v = block_on(&w);
    long long dt = async_mono_ms() - t0;
    thread_join(tid);
    ASSERT_EQ(v, 77);
    ASSERT_TRUE(dt >= 25LL);
    ASSERT_TRUE(dt < 2000LL);

    // IoDriver (the wake pipe interrupts the reactor's wait).
    struct Runtime rt = Runtime::new(); defer drop(rt);
    ASSERT_EQ(rt.init(), 0);
    g_os = Oneshot::new();
    t0 = async_mono_ms();
    ASSERT_EQ(thread_create(&tid, send_later_, (?&int)&delay), 0);
    w.st = ASYNC_INIT;
    w.got = null;
    v = rt.block_on(&w);
    dt = async_mono_ms() - t0;
    thread_join(tid);
    ASSERT_EQ(v, 77);
    ASSERT_TRUE(dt >= 25LL);
    ASSERT_TRUE(dt < 2000LL);
    rt.close_();
}

// ── timers ─────────────────────────────────────────────────────────────────

static long long now_of_(&Context cx) {
    match (cx.timers) {
        case some(q): return q.now_ms();
        case null: return -1LL;
    }
    return -1LL;
}

struct Ticks {
    int             st;
    struct Interval iv;
    long long       n;
    long long       stamps[8];
    int             k;
    struct Sleep    busy;
    struct Poll<int> poll(&Context cx);
    void drop();
};

// Five ticks of a 10 ms interval; between ticks 2 and 3 it is "busy" for
// 35 ms, so ticks 3..5 come in a burst and the schedule keeps its phase.
struct Poll<int> Ticks::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    self.iv = Interval::new(10LL);
    for (self.k = 0; self.k < 5; self.k = self.k + 1) {
        AWAIT_INTO(1, self.n, self.iv);
        self.stamps[self.k] = now_of_(cx);
        if (self.k == 1) {
            self.busy = Sleep::new(35LL);
            AWAIT(2, self.busy);
        }
    }
    ASYNC_RETURN((int)self.n);
    ASYNC_END;
}

void Ticks::drop() {
    if (self.st == 2) self.busy.drop();
    self.iv.drop();
    self.st = ASYNC_DONE;
}

static struct AsyncTask<struct Val, int> g_many[80];

void t_timers() {
    struct LocalExecutor ex = LocalExecutor::new(); defer drop(ex);
    struct VirtualClock clk = VirtualClock::new(0LL);
    struct Ticks tk; defer drop(tk);
    tk.st = ASYNC_INIT;
    int n = ex.block_on(&clk, &tk);
    ASSERT_EQ(n, 5);
    ASSERT_EQ(tk.stamps[0], 10LL);
    ASSERT_EQ(tk.stamps[1], 20LL);
    ASSERT_EQ(tk.stamps[2], 55LL);     // busy until 55: ticks due at 30, 40, 50 burst
    ASSERT_EQ(tk.stamps[3], 55LL);
    ASSERT_EQ(tk.stamps[4], 55LL);

    // TimerQueue by hand.
    struct TimerQueue q = TimerQueue::new(); defer drop(q);
    q.advance(500LL);
    ASSERT_EQ(q.now_ms(), 500LL);
    ASSERT_EQ(q.next_deadline(), -1LL);
    ASSERT_EQ(q.park_timeout(), -1LL);
    int hits = 0;
    struct Context cx = Context::from_waker(Waker::new(count_wake_, (?&int)&hits, 1UL)); defer drop(cx);
    cx.timers = &q;
    struct Sleep s1 = Sleep::new(100LL);
    struct Sleep s2 = Sleep::new(40LL); defer drop(s2);
    ASSERT_TRUE(s1.poll(&cx).is_pending());
    ASSERT_TRUE(s2.poll(&cx).is_pending());
    ASSERT_EQ(q.pending(), 2);
    ASSERT_EQ(q.next_deadline(), 540LL);
    ASSERT_EQ(q.park_timeout(), 40LL);
    ASSERT_EQ(q.advance(539LL), 0);
    ASSERT_EQ(q.advance(560LL), 1);
    ASSERT_EQ(hits, 1);
    ASSERT_TRUE(s2.poll(&cx).is_ready());
    s1.reset(10LL);                    // re-armed from the next poll
    ASSERT_EQ(q.pending(), 0);
    ASSERT_TRUE(s1.poll(&cx).is_pending());
    ASSERT_EQ(q.next_deadline(), 570LL);
    s1.drop();
    ASSERT_EQ(q.pending(), 0);
    struct Sleep past = Sleep::until(100LL); defer drop(past);
    ASSERT_TRUE(past.poll(&cx).is_ready());   // deadline already passed

    // More concurrent sleeps than the wheel holds (64): the rest re-poll
    // until their deadline, and everything still finishes in order.
    log_reset_();
    int i = 0;
    while (i < 80) {
        g_many[i] = AsyncTask::new(Val::new(i, (long long)(100 + i)));
        drop(ex.spawn(&g_many[i]));
        i = i + 1;
    }
    long long start = clk.now;
    ex.run(&clk);
    ASSERT_EQ(g_nlog, 80);
    int sorted = 1;
    i = 1;
    while (i < 80) { if (g_log[i] < g_log[i - 1]) sorted = 0; i = i + 1; }
    ASSERT_TRUE(sorted);
    ASSERT_EQ(clk.now - start, 179LL);
}

// ── combinators ────────────────────────────────────────────────────────────

static int twice_(int x) { return x * 2; }
static struct Val after_(int x) { return Val::new(x + 1, 5LL); }

void t_combinators() {
    struct LocalExecutor ex = LocalExecutor::new(); defer drop(ex);
    struct VirtualClock clk = VirtualClock::new(0LL);
    log_reset_();

    struct Join2<struct Val, struct Val, int, int> j = join2(Val::new(1, 30LL), Val::new(2, 10LL)); defer drop(j);
    struct Both<int, int> b = ex.block_on(&clk, &j);
    ASSERT_EQ(b.first, 1);
    ASSERT_EQ(b.second, 2);
    ASSERT_EQ(clk.now, 30LL);          // concurrently, not 40

    struct Join3<struct Val, struct Val, struct Ready<int>, int, int, int> j3 =
        join3(Val::new(3, 5LL), Val::new(4, 15LL), Ready::new(5));
    struct Both<int, struct Both<int, int>> b3 = ex.block_on(&clk, &j3);
    j3.drop();
    ASSERT_EQ(b3.first + b3.second.first * 10 + b3.second.second * 100, 543);
    ASSERT_EQ(clk.now, 45LL);

    g_drops = 0;
    struct Select2<struct Val, struct Val, int, int> s = select2(Val::new(6, 30LL), Val::new(7, 10LL)); defer drop(s);
    struct Either<int, int> e = ex.block_on(&clk, &s);
    ASSERT_EQ(e.which, 2);
    ASSERT_FALSE(e.is_left());
    ASSERT_EQ(e.right, 7);
    ASSERT_EQ(g_drops, 1);             // the loser was dropped ...
    ASSERT_EQ(ex.timers.pending(), 0); // ... with its timer
    ASSERT_EQ(clk.now, 55LL);

    struct Timeout<struct Val, int> t = timeout(Val::new(8, 100LL), 20LL); defer drop(t);
    ?int r = ex.block_on(&clk, &t);
    ASSERT_TRUE(r.is_none());
    ASSERT_EQ(clk.now, 75LL);
    ASSERT_EQ(ex.timers.pending(), 0);
    struct Timeout<struct Val, int> t2 = timeout(Val::new(9, 10LL), 20LL); defer drop(t2);
    ?int r2 = ex.block_on(&clk, &t2);
    ASSERT_EQ(r2.default(-1), 9);
    ASSERT_EQ(clk.now, 85LL);
    ASSERT_EQ(ex.timers.pending(), 0);

    struct MapFut<struct Val, int, int> m = map_fut(Val::new(21, 1LL), twice_); defer drop(m);
    int mv = ex.block_on(&clk, &m);
    ASSERT_EQ(mv, 42);
    struct AndThen<struct Val, int, struct Val, int> at = and_then(Val::new(10, 5LL), after_); defer drop(at);
    int av = ex.block_on(&clk, &at);
    ASSERT_EQ(av, 11);
    ASSERT_EQ(clk.now, 96LL);

    struct Val vs[4];
    vs[0] = Val::new(30, 15LL);
    vs[1] = Val::new(31, 5LL);
    vs[2] = Val::new(32, 25LL);
    vs[3] = Val::new(33, 1LL);
    int outs[4];
    struct JoinAll<struct Val, int> ja = join_all(vs[0..4], outs[0..4]);
    int jn = ex.block_on(&clk, &ja);
    ASSERT_EQ(jn, 4);
    ASSERT_EQ(outs[0] + outs[1] + outs[2] + outs[3], 30 + 31 + 32 + 33);
    ASSERT_EQ(outs[2], 32);
    ASSERT_EQ(clk.now, 121LL);

    ja.drop();
    unsafe { forget(vs); }   // join_all owned the futures behind the slice and dropped them
    g_drops = 0;
    vs[0] = Val::new(40, 15LL);
    vs[1] = Val::new(41, 5LL);
    vs[2] = Val::new(42, 25LL);
    struct SelectAny<struct Val, int> sa = select_any(vs[0..3]);
    struct Picked<int> pk = ex.block_on(&clk, &sa);
    ASSERT_EQ(pk.index, 1);
    ASSERT_EQ(pk.value, 41);
    ASSERT_EQ(g_drops, 2);
    ASSERT_EQ(ex.timers.pending(), 0);
    sa.drop();
    unsafe { forget(vs); }   // select_any owned (and dropped) vs[0..3]; vs[3] was join_all's

    // A combinator dropped half-way drops its children.
    g_drops = 0;
    struct Join2<struct Val, struct Val, int, int> jd = join2(Val::new(1, 10LL), Val::new(2, 50LL));
    struct Context cx = local_context_(&ex, Waker::noop()); defer drop(cx);
    ASSERT_TRUE(jd.poll(&cx).is_pending());
    ASSERT_EQ(ex.timers.pending(), 2);
    jd.drop();
    ASSERT_EQ(g_drops, 2);
    ASSERT_EQ(ex.timers.pending(), 0);
}

// ── channels ───────────────────────────────────────────────────────────────

static int g_slots[3];
static struct AsyncChannel<int> g_ch;

struct Producer {
    int                 st;
    int                 base;
    int                 i;
    int                 ok;
    struct ChanSend<int> tx;
    static struct Producer new(int base);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Producer Producer::new(int base) {
    struct Producer p;
    p.st = ASYNC_INIT;
    p.base = base;
    p.i = 0;
    p.ok = 0;
    return p;
}

struct Poll<int> Producer::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    for (self.i = 0; self.i < 10; self.i = self.i + 1) {
        self.tx = g_ch.send(self.base + self.i);
        AWAIT_INTO(1, self.ok, self.tx);
        if (self.ok == 0) ASYNC_RETURN(-1);
    }
    ASYNC_RETURN(10);
    ASYNC_END;
}

void Producer::drop() { if (self.st == 1) self.tx.drop(); self.st = ASYNC_DONE; }

struct Consumer {
    int                 st;
    int                 n;
    int                 sum;
    int                 max_len;
    ?int                got;
    struct ChanRecv<int> rx;
    struct Sleep        d;
    static struct Consumer new();
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Consumer Consumer::new() {
    struct Consumer c;
    c.st = ASYNC_INIT;
    c.n = 0;
    c.sum = 0;
    c.max_len = 0;
    c.got = null;
    return c;
}

struct Poll<int> Consumer::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    while (1) {
        if ((int)g_ch.len() > self.max_len) self.max_len = (int)g_ch.len();
        self.rx = g_ch.recv();
        AWAIT_INTO(1, self.got, self.rx);
        if (self.got.is_none()) break;
        self.sum = self.sum + self.got.default(0);
        self.n = self.n + 1;
        self.d = Sleep::new(2LL);      // slow consumer: producers block on a full channel
        AWAIT(2, self.d);
    }
    ASYNC_RETURN(self.n);
    ASYNC_END;
}

void Consumer::drop() {
    if (self.st == 1) self.rx.drop();
    if (self.st == 2) self.d.drop();
    self.st = ASYNC_DONE;
}

static struct AsyncTask<struct Producer, int> g_prod[2];
static struct AsyncTask<struct Consumer, int> g_cons[2];

void t_channels() {
    // Oneshot.
    struct Context cx = Context::noop(); defer drop(cx);
    struct Oneshot<int> o = Oneshot::new(); defer drop(o);
    struct OneshotRecv<int> orx = o.recv(); defer drop(orx);
    ASSERT_TRUE(orx.poll(&cx).is_pending());
    ASSERT_EQ(o.send(5), 1);
    ASSERT_EQ(o.send(6), 0);           // only once
    struct Poll<?int> op = orx.poll(&cx);
    ASSERT_TRUE(op.is_ready());
    ASSERT_EQ(op.value.default(0), 5);
    struct Oneshot<int> o2 = Oneshot::new(); defer drop(o2);
    o2.close();
    struct OneshotRecv<int> orx2 = o2.recv(); defer drop(orx2);
    struct Poll<?int> op2 = orx2.poll(&cx);
    ASSERT_TRUE(op2.is_ready());
    ASSERT_TRUE(op2.value.is_none());
    ASSERT_TRUE(o2.is_closed());
    struct Oneshot<int> o3 = Oneshot::new(); defer drop(o3);
    o3.send(9);
    ASSERT_EQ(o3.try_recv().default(0), 9);
    ASSERT_TRUE(o3.try_recv().is_none());

    // try_send/try_recv.
    g_ch = AsyncChannel::new(g_slots[0..3]);
    ASSERT_EQ(g_ch.capacity(), 3UL);
    ASSERT_EQ(g_ch.try_send(1), 1);
    ASSERT_EQ(g_ch.try_send(2), 1);
    ASSERT_EQ(g_ch.try_send(3), 1);
    ASSERT_EQ(g_ch.try_send(4), 0);    // full
    ASSERT_EQ(g_ch.try_recv().default(0), 1);
    ASSERT_EQ(g_ch.try_send(4), 1);    // wraps around
    ASSERT_EQ(g_ch.len(), 3UL);
    ASSERT_EQ(g_ch.try_recv().default(0), 2);
    ASSERT_EQ(g_ch.try_recv().default(0), 3);
    ASSERT_EQ(g_ch.try_recv().default(0), 4);
    ASSERT_TRUE(g_ch.try_recv().is_none());

    // Two producers, two consumers over a 3-slot channel; close when the
    // producers are done: consumers drain, then see null.
    g_ch = AsyncChannel::new(g_slots[0..3]);
    struct LocalExecutor ex = LocalExecutor::new(); defer drop(ex);
    struct VirtualClock clk = VirtualClock::new(0LL);
    g_prod[0] = AsyncTask::new(Producer::new(0));
    g_prod[1] = AsyncTask::new(Producer::new(100));
    g_cons[0] = AsyncTask::new(Consumer::new());
    g_cons[1] = AsyncTask::new(Consumer::new());
    drop(ex.spawn(&g_prod[0]));
    drop(ex.spawn(&g_prod[1]));
    drop(ex.spawn(&g_cons[0]));
    drop(ex.spawn(&g_cons[1]));
    while (g_prod[0].is_finished() == 0 || g_prod[1].is_finished() == 0) {
        ex.turn(clk.now);
        if (ex.has_ready() == 0) clk.park(ex.timers.park_timeout());
    }
    g_ch.close();
    ASSERT_EQ(g_ch.try_send(1), -1);
    ex.run(&clk);
    ASSERT_EQ(g_prod[0].take().default(0), 10);
    ASSERT_EQ(g_prod[1].take().default(0), 10);
    int n0 = g_cons[0].take().default(0);
    int n1 = g_cons[1].take().default(0);
    ASSERT_EQ(n0 + n1, 20);
    ASSERT_TRUE(n0 > 0 && n1 > 0);     // both consumers got work
    unsafe {
        &AsyncTask<struct Consumer, int> c0 = &g_cons[0];
        &AsyncTask<struct Consumer, int> c1 = &g_cons[1];
        ASSERT_EQ(c0.fut.sum + c1.fut.sum, 45 + 1045);
        ASSERT_TRUE(c0.fut.max_len <= 3 && c1.fut.max_len <= 3);
    }
    ASSERT_EQ(clk.now, 20LL);          // 20 values, 2 consumers, 2 ms each

    // A send on a closed channel completes with 0; a pending one is woken by close.
    struct AsyncChannel<int> c2 = AsyncChannel::new(g_slots[0..1]); defer drop(c2);
    struct ChanSend<int> s1 = c2.send(1); defer drop(s1);
    struct ChanSend<int> s2 = c2.send(2); defer drop(s2);
    ASSERT_EQ(s1.poll(&cx).take().default(-1), 1);
    ASSERT_TRUE(s2.poll(&cx).is_pending());
    c2.close();
    ASSERT_EQ(s2.poll(&cx).take().default(-1), 0);
    struct ChanRecv<int> r1 = c2.recv(); defer drop(r1);
    ASSERT_EQ(r1.poll(&cx).value.default(-1), 1);   // buffered values still drain
    struct ChanRecv<int> r2 = c2.recv(); defer drop(r2);
    struct Poll<?int> rp = r2.poll(&cx);
    ASSERT_TRUE(rp.is_ready() && rp.value.is_none());
}

// ── sync primitives ────────────────────────────────────────────────────────

static struct AsyncMutex g_mx;
static struct AsyncSemaphore g_sem;
static struct AsyncNotify g_ntf;
static int g_inside = 0;
static int g_max_inside = 0;

struct Locker {
    int              st;
    int              id;
    struct MutexLock l;
    struct Sleep     d;
    static struct Locker new(int id);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Locker Locker::new(int id) {
    struct Locker k;
    k.st = ASYNC_INIT;
    k.id = id;
    return k;
}

struct Poll<int> Locker::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    self.l = g_mx.lock();
    AWAIT(1, self.l);
    g_inside = g_inside + 1;
    if (g_inside > g_max_inside) g_max_inside = g_inside;
    log_(self.id);
    self.d = Sleep::new(10LL);         // hold it across an await
    AWAIT(2, self.d);
    g_inside = g_inside - 1;
    g_mx.unlock();
    ASYNC_RETURN(self.id);
    ASYNC_END;
}

void Locker::drop() {
    if (self.st == 1) self.l.drop();
    if (self.st == 2) { self.d.drop(); g_inside = g_inside - 1; g_mx.unlock(); }
    self.st = ASYNC_DONE;
}

struct Permits {
    int               st;
    int               id;
    struct SemAcquire a;
    struct Sleep      d;
    static struct Permits new(int id);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Permits Permits::new(int id) {
    struct Permits p;
    p.st = ASYNC_INIT;
    p.id = id;
    return p;
}

struct Poll<int> Permits::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    self.a = g_sem.acquire(1LL);
    AWAIT(1, self.a);
    g_inside = g_inside + 1;
    if (g_inside > g_max_inside) g_max_inside = g_inside;
    self.d = Sleep::new(10LL);
    AWAIT(2, self.d);
    g_inside = g_inside - 1;
    g_sem.release(1LL);
    ASYNC_RETURN(self.id);
    ASYNC_END;
}

void Permits::drop() {
    if (self.st == 1) self.a.drop();
    if (self.st == 2) self.d.drop();
    self.st = ASYNC_DONE;
}

struct Waiter {
    int             st;
    int             id;
    struct Notified n;
    static struct Waiter new(int id);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Waiter Waiter::new(int id) {
    struct Waiter w;
    w.st = ASYNC_INIT;
    w.id = id;
    return w;
}

struct Poll<int> Waiter::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    self.n = g_ntf.notified();
    AWAIT(1, self.n);
    log_(self.id);
    ASYNC_RETURN(self.id);
    ASYNC_END;
}

void Waiter::drop() { if (self.st == 1) self.n.drop(); self.st = ASYNC_DONE; }

static struct AsyncTask<struct Locker, int> g_lk[4];
static struct AsyncTask<struct Permits, int> g_pm[6];
static struct AsyncTask<struct Waiter, int> g_wt[3];

void t_sync() {
    struct LocalExecutor ex = LocalExecutor::new(); defer drop(ex);
    struct VirtualClock clk = VirtualClock::new(0LL);

    // Mutex: one holder at a time, FIFO-ish hand-off, held across awaits.
    log_reset_();
    g_mx = AsyncMutex::new();
    g_inside = 0;
    g_max_inside = 0;
    int i = 0;
    while (i < 4) { g_lk[i] = AsyncTask::new(Locker::new(i + 1)); drop(ex.spawn(&g_lk[i])); i = i + 1; }
    ex.run(&clk);
    ASSERT_EQ(g_max_inside, 1);
    ASSERT_EQ(g_nlog, 4);
    ASSERT_EQ(clk.now, 40LL);
    ASSERT_FALSE(g_mx.is_locked());
    ASSERT_EQ(g_mx.try_lock(), 1);
    ASSERT_EQ(g_mx.try_lock(), 0);
    g_mx.unlock();

    // Semaphore(2): at most two inside, six tasks take 30 ms.
    g_sem = AsyncSemaphore::new(2LL);
    g_inside = 0;
    g_max_inside = 0;
    long long t0 = clk.now;
    i = 0;
    while (i < 6) { g_pm[i] = AsyncTask::new(Permits::new(i)); drop(ex.spawn(&g_pm[i])); i = i + 1; }
    ex.run(&clk);
    ASSERT_EQ(g_max_inside, 2);
    ASSERT_EQ(clk.now - t0, 30LL);
    ASSERT_EQ(g_sem.available(), 2LL);
    ASSERT_EQ(g_sem.try_acquire(3LL), 0);
    ASSERT_EQ(g_sem.try_acquire(2LL), 1);
    g_sem.release(2LL);

    // Notify: a stored permit, notify_one with waiters, notify_waiters.
    log_reset_();
    g_ntf = AsyncNotify::new();
    g_ntf.notify_one();                // no waiter: a permit for the next one
    g_wt[0] = AsyncTask::new(Waiter::new(1));
    drop(ex.spawn(&g_wt[0]));
    ex.turn(clk.now);
    ASSERT_TRUE(g_wt[0].is_finished());
    g_wt[1] = AsyncTask::new(Waiter::new(2));
    g_wt[2] = AsyncTask::new(Waiter::new(3));
    drop(ex.spawn(&g_wt[1]));
    drop(ex.spawn(&g_wt[2]));
    ex.turn(clk.now);
    ASSERT_EQ(ex.live_count(), 2);
    g_ntf.notify_one();
    ex.turn(clk.now);
    ASSERT_EQ(ex.live_count(), 1);
    g_ntf.notify_waiters();
    ex.turn(clk.now);
    ASSERT_EQ(ex.live_count(), 0);
    ASSERT_EQ(g_nlog, 3);
    // notify_waiters leaves no permit behind.
    struct Notified late = g_ntf.notified();
    struct Context cx = Context::noop(); defer drop(cx);
    ASSERT_TRUE(late.poll(&cx).is_pending());
    late.drop();

    // CancelToken, a child, and with_cancel.
    struct CancelToken root = CancelToken::new(); defer drop(root);
    struct CancelToken child = CancelToken::child_of(&root); defer drop(child);
    struct CancelToken other = CancelToken::new(); defer drop(other);
    ASSERT_FALSE(child.is_cancelled());
    struct Cancelled cf = child.cancelled_fut(); defer drop(cf);
    int hits = 0;
    struct Context cxw = Context::from_waker(Waker::new(count_wake_, (?&int)&hits, 1UL)); defer drop(cxw);
    ASSERT_TRUE(cf.poll(&cxw).is_pending());
    root.cancel();                     // cancelling the parent cancels the child
    ASSERT_EQ(hits, 1);
    ASSERT_TRUE(child.is_cancelled());
    ASSERT_TRUE(cf.poll(&cxw).is_ready());
    ASSERT_FALSE(other.is_cancelled());

    g_drops = 0;
    struct CancelToken tok = CancelToken::new(); defer drop(tok);
    struct Cancellable<struct Val, int> wc = with_cancel(Val::new(5, 100LL), &tok); defer drop(wc);
    g_vt[3] = AsyncTask::new(Val::new(0, 30LL));   // something to cancel at t+30
    struct Context cxe = local_context_(&ex, Waker::noop()); defer drop(cxe);
    ASSERT_TRUE(wc.poll(&cxe).is_pending());
    tok.cancel();
    struct Poll<?int> wp = wc.poll(&cxe);
    ASSERT_TRUE(wp.is_ready());
    ASSERT_TRUE(wp.value.is_none());
    ASSERT_EQ(g_drops, 1);
    ASSERT_EQ(ex.timers.pending(), 0);
    struct CancelToken tok2 = CancelToken::new(); defer drop(tok2);
    struct Cancellable<struct Val, int> wc2 = with_cancel(Val::new(6, 10LL), &tok2); defer drop(wc2);
    ?int got = ex.block_on(&clk, &wc2);
    ASSERT_EQ(got.default(-1), 6);
}

// ── ThreadPool ─────────────────────────────────────────────────────────────

static int g_pool_count = 0;

struct PoolJob {
    int             st;
    int             id;
    int             i;
    struct Sleep    d;
    struct YieldNow y;
    static struct PoolJob new(int id);
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct PoolJob PoolJob::new(int id) {
    struct PoolJob j;
    j.st = ASYNC_INIT;
    j.id = id;
    j.i = 0;
    return j;
}

struct Poll<int> PoolJob::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    for (self.i = 0; self.i < 5; self.i = self.i + 1) {
        self.d = Sleep::new(1LL + (long long)(self.id % 4));
        AWAIT(1, self.d);
        atomic_fetch_add_int(&g_pool_count, 1);
        self.y = YieldNow::new();
        AWAIT(2, self.y);
    }
    ASYNC_RETURN(self.id);
    ASYNC_END;
}

void PoolJob::drop() {
    if (self.st == 1) self.d.drop();
    if (self.st == 2) self.y.drop();
    self.st = ASYNC_DONE;
}

// Pool tasks talking through a channel.
static int g_pslots[4];
static struct AsyncChannel<int> g_pch;
struct PoolSender {
    int st; int i; int ok; struct ChanSend<int> tx;
    struct Poll<int> poll(&Context cx);
    void drop();
};
struct Poll<int> PoolSender::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    for (self.i = 1; self.i <= 100; self.i = self.i + 1) {
        self.tx = g_pch.send(self.i);
        AWAIT_INTO(1, self.ok, self.tx);
    }
    g_pch.close();
    ASYNC_RETURN(100);
    ASYNC_END;
}
void PoolSender::drop() { if (self.st == 1) self.tx.drop(); self.st = ASYNC_DONE; }
struct PoolSummer {
    int st; int sum; ?int got; struct ChanRecv<int> rx;
    struct Poll<int> poll(&Context cx);
    void drop();
};
struct Poll<int> PoolSummer::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    while (1) {
        self.rx = g_pch.recv();
        AWAIT_INTO(1, self.got, self.rx);
        if (self.got.is_none()) break;
        self.sum = self.sum + self.got.default(0);
    }
    ASYNC_RETURN(self.sum);
    ASYNC_END;
}
void PoolSummer::drop() { if (self.st == 1) self.rx.drop(); self.st = ASYNC_DONE; }

static struct ThreadPool g_pool;
static struct AsyncTask<struct PoolJob, int> g_pj[32];
static struct AsyncTask<struct PoolSender, int> g_ps;
static struct AsyncTask<struct PoolSummer, int> g_psum;
static struct AsyncTask<struct Val, int> g_pv;

void t_thread_pool() {
    g_pool = ThreadPool::new();
    ASSERT_EQ(g_pool.start(4), 0);
    g_pool_count = 0;
    int i = 0;
    while (i < 32) { g_pj[i] = AsyncTask::new(PoolJob::new(i)); drop(g_pool.spawn(&g_pj[i])); i = i + 1; }
    g_pool.wait_idle();
    ASSERT_EQ(atomic_load_int(&g_pool_count), 160);
    int sum = 0;
    i = 0;
    while (i < 32) { sum = sum + g_pj[i].take().default(-1000); i = i + 1; }
    ASSERT_EQ(sum, 31 * 32 / 2);

    g_pch = AsyncChannel::new(g_pslots[0..4]);
    struct PoolSender ps;
    ps.st = ASYNC_INIT;
    g_ps = AsyncTask::new(ps);
    struct PoolSummer pm;
    pm.st = ASYNC_INIT;
    pm.sum = 0;
    pm.got = null;
    g_psum = AsyncTask::new(pm);
    drop(g_pool.spawn(&g_psum));
    drop(g_pool.spawn(&g_ps));
    g_pool.wait_idle();
    ASSERT_EQ(g_psum.take().default(0), 5050);

    // block_on from the main thread, and abort of a sleeping pool task.
    int pb = g_pool.block_on(PoolJob::new(7));
    ASSERT_EQ(pb, 7);
    g_drops = 0;
    g_pv = AsyncTask::new(Val::new(1, 10000LL));
    struct JoinHandle<int> h = g_pool.spawn(&g_pv); defer drop(h);
    thread_sleep_ms(20UL);
    h.abort();
    g_pool.wait_idle();
    ASSERT_TRUE(g_pv.is_aborted());
    ASSERT_EQ(g_drops, 1);
    ASSERT_EQ(g_pool.timers.pending(), 0);
    g_pool.shutdown();
}

// A task that keeps yielding must not keep timers from firing.
static int g_flag = 0;

struct Spinner {
    int             st;
    int             spins;
    struct YieldNow y;
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> Spinner::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    self.spins = 0;
    while (atomic_load_int(&g_flag) == 0 && self.spins < 50000000) {
        self.spins = self.spins + 1;
        self.y = YieldNow::new();
        AWAIT(1, self.y);
    }
    ASYNC_RETURN(self.spins);
    ASYNC_END;
}

void Spinner::drop() { if (self.st == 1) self.y.drop(); self.st = ASYNC_DONE; }

struct FlagAfter {
    int          st;
    struct Sleep d;
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> FlagAfter::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    self.d = Sleep::new(20LL);
    AWAIT(1, self.d);
    atomic_store_int(&g_flag, 1);
    ASYNC_RETURN(1);
    ASYNC_END;
}

void FlagAfter::drop() { if (self.st == 1) self.d.drop(); self.st = ASYNC_DONE; }

static struct AsyncTask<struct FlagAfter, int> g_fa;
static struct AsyncTask<struct Spinner, int> g_sp;

void t_no_starvation() {
    // LocalExecutor with the real clock: the spinner is the block_on root.
    atomic_store_int(&g_flag, 0);
    struct LocalExecutor ex = LocalExecutor::new(); defer drop(ex);
    struct HostPark park = HostPark::new();
    struct FlagAfter fa;
    fa.st = ASYNC_INIT;
    g_fa = AsyncTask::new(fa);
    drop(ex.spawn(&g_fa));
    struct Spinner sp;
    sp.st = ASYNC_INIT;
    int spins = ex.block_on(&park, &sp);
    sp.drop();
    ASSERT_TRUE(spins < 50000000);
    ASSERT_TRUE(g_fa.is_finished());
    park.destroy();

    // A one-thread pool kept busy by the spinner.
    atomic_store_int(&g_flag, 0);
    ?&heap ThreadPool pq = alloc(sizeof(struct ThreadPool));
    ASSERT_FALSE(pq.is_null());
    match (pq) {
        case null: return;
        case some(p): {
            *p = ThreadPool::new();
            ASSERT_EQ(p.start(1), 0);
            struct Spinner sp2;
            sp2.st = ASYNC_INIT;
            g_sp = AsyncTask::new(sp2);
            struct FlagAfter fa2;
            fa2.st = ASYNC_INIT;
            g_fa = AsyncTask::new(fa2);
            drop(p.spawn(&g_sp));
            drop(p.spawn(&g_fa));
            p.wait_idle();
            ASSERT_TRUE(g_sp.take().default(50000000) < 50000000);
            p.shutdown();
        }
    }
    dealloc(pq);
}

void t_host_block_on() {
    long long t0 = async_mono_ms();
    struct Ticker tk = Ticker::new(9, 3, 15LL); defer drop(tk);   // three real 15 ms sleeps
    log_reset_();
    int bv = block_on(&tk);
    ASSERT_EQ(bv, 9);
    long long dt = async_mono_ms() - t0;
    ASSERT_TRUE(dt >= 40LL);
    ASSERT_TRUE(dt < 3000LL);
    ASSERT_EQ(g_nlog, 3);
}

// ── async TCP over loopback ────────────────────────────────────────────────

// Accepts 'expect' connections, one echo handler task per connection,
// spawned from the accept loop.
struct EchoConn {
    int                   st;
    struct AsyncTcpStream c;
    struct ReadFut        rd;
    struct WriteAllFut    wr;
    long long             n;
    long long             bytes;
    int                   ok;
    unsigned char         buf[512];
    struct Poll<long long> poll(&Context cx);
    void drop();
};

struct Poll<long long> EchoConn::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    self.bytes = 0LL;
    while (1) {
        self.rd = self.c.read(self.buf[0..512]);
        AWAIT_INTO(1, self.n, self.rd);
        if (self.n <= 0LL) break;
        self.wr = self.c.write_all(self.buf[0..(unsigned long)self.n]);
        AWAIT_INTO(2, self.ok, self.wr);
        if (self.ok == 0) break;
        self.bytes = self.bytes + self.n;
    }
    unsafe { self.c.close(); }   // early, in place: drop() closes nothing again
    ASYNC_RETURN(self.bytes);
    ASYNC_END;
}

void EchoConn::drop() {
    if (self.st == 1) self.rd.drop();
    if (self.st == 2) self.wr.drop();
    self.c.close();
    self.st = ASYNC_DONE;
}

#define ECHO_CLIENTS 8

struct EchoServer {
    int                     st;
    int                     expect;
    int                     accepted;
    struct AsyncTcpListener l;
    struct AcceptFut        acc;
    struct AsyncTcpStream   c;
    struct AsyncTask<struct EchoConn, long long> conns[ECHO_CLIENTS];
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> EchoServer::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    self.accepted = 0;
    while (self.accepted < self.expect) {
        self.acc = self.l.accept();
        AWAIT_INTO(1, self.c, self.acc);
        if (self.c.fd < 0) ASYNC_RETURN(-1);
        {
            struct EchoConn ec;
            ec.st = ASYNC_INIT;
            ec.c = std::replace(&self.c, AsyncTcpStream::from_fd(-1));
            self.conns[self.accepted] = AsyncTask::new(ec);
            drop(async_spawn(cx, &self.conns[self.accepted]));
        }
        self.accepted = self.accepted + 1;
    }
    unsafe { self.l.close(); }   // early, in place: drop() closes nothing again
    ASYNC_RETURN(self.accepted);
    ASYNC_END;
}

void EchoServer::drop() {
    if (self.st == 1) self.acc.drop();
    self.c.drop();
    int i = 0;
    while (i < self.accepted) { self.conns[i].drop(); i = i + 1; }   // finished echo tasks
    self.l.close();
    self.st = ASYNC_DONE;
}

struct EchoClient {
    int                   st;
    int                   id;
    unsigned short        port;
    struct ConnectFut     cn;
    struct AsyncTcpStream s;
    struct WriteAllFut    wr;
    struct ReadFut        rd;
    struct Sleep          d;
    int                   ok;
    long long             n;
    unsigned long         got;
    unsigned char         out[300];
    unsigned char         in[300];
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> EchoClient::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 4);
    self.cn = AsyncTcpStream::connect(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1), self.port);
    AWAIT_INTO(1, self.s, self.cn);
    if (self.s.fd < 0) ASYNC_RETURN(-1);
    for (int i = 0; i < 300; i = i + 1) self.out[i] = (unsigned char)((i * 7 + self.id) & 255);
    self.d = Sleep::new((long long)(self.id % 3));     // stagger the clients
    AWAIT(2, self.d);
    self.wr = self.s.write_all(self.out[0..300]);
    AWAIT_INTO(3, self.ok, self.wr);
    if (self.ok == 0) ASYNC_RETURN(-2);
    self.got = 0UL;
    while (self.got < 300UL) {
        self.rd = self.s.read(self.in[self.got..300]);
        AWAIT_INTO(4, self.n, self.rd);
        if (self.n <= 0LL) break;
        self.got = self.got + (unsigned long)self.n;
    }
    unsafe { self.s.close(); }   // early, in place: drop() closes nothing again
    if (self.got != 300UL) ASYNC_RETURN(-3);
    for (int i = 0; i < 300; i = i + 1) { if (self.in[i] != self.out[i]) ASYNC_RETURN(-4); }
    ASYNC_RETURN(self.id);
    ASYNC_END;
}

void EchoClient::drop() {
    if (self.st == 1) self.cn.drop();
    if (self.st == 2) self.d.drop();
    if (self.st == 3) self.wr.drop();
    if (self.st == 4) self.rd.drop();
    self.s.close();
    self.st = ASYNC_DONE;
}

// A minimal HTTP/1.0 server task (one request) and client.
#define HTTP_REPLY "HTTP/1.0 200 OK\r\nContent-Length: 5\r\n\r\nhello"

struct HttpOnce {
    int                     st;
    struct AsyncTcpListener l;
    struct AcceptFut        acc;
    struct AsyncTcpStream   c;
    struct ReadFut          rd;
    struct WriteAllFut      wr;
    long long               n;
    unsigned long           have;
    int                     ok;
    unsigned char           req[512];
    unsigned char           resp[128];
    struct Poll<int> poll(&Context cx);
    void drop();
};

// Copies the C string 's' into 'dst'; returns its length.
static unsigned long put_str_([]unsigned char dst, const char* s) {
    unsigned long n = 0UL;
    unsafe {
        while (n < dst.len && s[n] != (char)0) { dst[n] = (unsigned char)s[n]; n = n + 1UL; }
    }
    return n;
}

static int has_blank_line_(const &unsigned char b, unsigned long n) {
    unsigned long i = 3UL;
    while (i < n) {
        unsafe {
            if (b[i - 3UL] == (unsigned char)13 && b[i - 2UL] == (unsigned char)10 &&
                b[i - 1UL] == (unsigned char)13 && b[i] == (unsigned char)10) return 1;
        }
        i = i + 1UL;
    }
    return 0;
}

struct Poll<int> HttpOnce::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 3);
    self.acc = self.l.accept();
    AWAIT_INTO(1, self.c, self.acc);
    self.have = 0UL;
    while (self.have < 512UL && has_blank_line_(&self.req[0], self.have) == 0) {
        self.rd = self.c.read(self.req[self.have..512]);
        AWAIT_INTO(2, self.n, self.rd);
        if (self.n <= 0LL) ASYNC_RETURN(-1);
        self.have = self.have + (unsigned long)self.n;
    }
    self.n = (long long)put_str_(self.resp[0..128], HTTP_REPLY);
    self.wr = self.c.write_all(self.resp[0..(unsigned long)self.n]);
    AWAIT_INTO(3, self.ok, self.wr);
    unsafe { self.c.close(); }   // early, in place: drop() closes nothing again
    unsafe { self.l.close(); }   // early, in place: drop() closes nothing again
    // 1 if the request line was a GET of /hello.
    if (self.req[0] == (unsigned char)71 && self.req[4] == (unsigned char)47 && self.req[5] == (unsigned char)104) ASYNC_RETURN(1);
    ASYNC_RETURN(0);
    ASYNC_END;
}

void HttpOnce::drop() {
    if (self.st == 1) self.acc.drop();
    if (self.st == 2) self.rd.drop();
    if (self.st == 3) self.wr.drop();
    self.c.close();
    self.l.close();
    self.st = ASYNC_DONE;
}

struct HttpGet {
    int                   st;
    unsigned short        port;
    struct ConnectFut     cn;
    struct AsyncTcpStream s;
    struct WriteAllFut    wr;
    struct ReadFut        rd;
    int                   ok;
    long long             n;
    unsigned long         have;
    unsigned char         req[64];
    unsigned char         resp[256];
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> HttpGet::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 3);
    self.cn = AsyncTcpStream::connect(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1), self.port);
    AWAIT_INTO(1, self.s, self.cn);
    if (self.s.fd < 0) ASYNC_RETURN(-1);
    self.n = (long long)put_str_(self.req[0..64], "GET /hello HTTP/1.0\r\nHost: x\r\n\r\n");
    self.wr = self.s.write_all(self.req[0..(unsigned long)self.n]);
    AWAIT_INTO(2, self.ok, self.wr);
    self.have = 0UL;
    while (self.have < 255UL) {
        self.rd = self.s.read(self.resp[self.have..255]);
        AWAIT_INTO(3, self.n, self.rd);
        if (self.n <= 0LL) break;           // HTTP/1.0: the server closes at the end
        self.have = self.have + (unsigned long)self.n;
    }
    self.resp[self.have] = (unsigned char)0;
    unsafe { self.s.close(); }   // early, in place: drop() closes nothing again
    ASYNC_RETURN((int)self.have);
    ASYNC_END;
}

void HttpGet::drop() {
    if (self.st == 1) self.cn.drop();
    if (self.st == 2) self.wr.drop();
    if (self.st == 3) self.rd.drop();
    self.s.close();
    self.st = ASYNC_DONE;
}

static struct Runtime g_rt;
static struct AsyncTask<struct EchoServer, int> g_srv;
static struct AsyncTask<struct EchoClient, int> g_cli[ECHO_CLIENTS];
static struct AsyncTask<struct HttpOnce, int> g_http;

void t_net_loopback() {
    g_rt = Runtime::new();
    ASSERT_EQ(g_rt.init(), 0);

    // Echo: ECHO_CLIENTS concurrent clients, a handler task per connection.
    struct EchoServer srv;
    srv.st = ASYNC_INIT;
    srv.accepted = 0;
    srv.c = AsyncTcpStream::from_fd(-1);
    srv.expect = ECHO_CLIENTS;
    srv.l = AsyncTcpListener::bind((unsigned short)0);
    ASSERT_TRUE(srv.l.fd >= 0);
    unsigned short port = srv.l.local_port();
    ASSERT_TRUE(port != (unsigned short)0);
    g_srv = AsyncTask::new(srv);
    drop(g_rt.spawn(&g_srv));
    int i = 0;
    while (i < ECHO_CLIENTS) {
        struct EchoClient c;
        c.st = ASYNC_INIT;
        c.id = i + 1;
        c.port = port;
        g_cli[i] = AsyncTask::new(c);
        drop(g_rt.spawn(&g_cli[i]));
        i = i + 1;
    }
    g_rt.run();
    ASSERT_EQ(g_srv.take().default(0), ECHO_CLIENTS);
    int ok = 0;
    i = 0;
    while (i < ECHO_CLIENTS) { if (g_cli[i].take().default(-9) == i + 1) ok = ok + 1; i = i + 1; }
    ASSERT_EQ(ok, ECHO_CLIENTS);
    long long echoed = 0LL;
    unsafe {
        &AsyncTask<struct EchoServer, int> s = &g_srv;
        i = 0;
        while (i < ECHO_CLIENTS) { echoed = echoed + s.fut.conns[i].take().default(0LL); i = i + 1; }
    }
    ASSERT_EQ(echoed, 300LL * (long long)ECHO_CLIENTS);
    ASSERT_EQ(g_rt.io.waiting(), 0);
    ASSERT_TRUE(g_rt.io.events > 0LL);

    // A refused connect (the listener above is closed now).
    struct EchoClient refused; defer drop(refused);
    refused.st = ASYNC_INIT;
    refused.id = 1;
    refused.port = port;
    int rv = g_rt.block_on(&refused);
    ASSERT_EQ(rv, -1);

    // Timeouts on I/O: an accept nobody connects to, dropped by timeout();
    // its registration goes away with it.
    struct AsyncTcpListener idle = AsyncTcpListener::bind((unsigned short)0);
    struct Timeout<struct AcceptFut, struct AsyncTcpStream> ta = timeout(idle.accept(), 30LL);
    long long t0 = async_mono_ms();
    ?struct AsyncTcpStream none = g_rt.block_on(&ta); defer drop(none);
    ASSERT_TRUE(none.is_none());
    ASSERT_TRUE(async_mono_ms() - t0 >= 25LL);
    ASSERT_EQ(g_rt.io.waiting(), 0);
    ta.drop();   // before the listener its accept future refers to
    idle.close();

    // A tiny HTTP exchange.
    struct HttpOnce h;
    h.st = ASYNC_INIT;
    h.c = AsyncTcpStream::from_fd(-1);
    h.l = AsyncTcpListener::bind((unsigned short)0);
    ASSERT_TRUE(h.l.fd >= 0);
    struct HttpGet g; defer drop(g);
    g.st = ASYNC_INIT;
    g.port = h.l.local_port();
    g_http = AsyncTask::new(h);
    drop(g_rt.spawn(&g_http));
    int got = g_rt.block_on(&g);
    g_rt.run();
    ASSERT_EQ(g_http.take().default(-5), 1);
    unsigned char tmp[128];
    ASSERT_EQ(got, (int)put_str_(tmp[0..128], HTTP_REPLY));
    unsafe { ASSERT_STR_EQ((const char*)&g.resp[0], HTTP_REPLY); }

    g_rt.close_();
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("Poll/Waker/WakerSlot/WaitList", t_poll_waker);
        t.add("coroutine macros polled by hand", t_coroutine);
        t.add("LocalExecutor order + virtual time", t_executor_order);
        t.add("JoinHandle, async_spawn, abort, block_on", t_executor_join_abort);
        t.add("wakes from another thread (HostPark, IoDriver)", t_remote_wake);
        t.add("timers: Sleep, Interval, full wheel", t_timers);
        t.add("combinators", t_combinators);
        t.add("channels: Oneshot, AsyncChannel", t_channels);
        t.add("Mutex/Semaphore/Notify/CancelToken", t_sync);
        t.add("ThreadPool", t_thread_pool);
        t.add("host block_on (real time)", t_host_block_on);
        t.add("busy tasks don't starve timers", t_no_starvation);
        t.add("async TCP over loopback", t_net_loopback);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
