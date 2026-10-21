// std::alloc (bump/slab/pool/tlsf), heap, result, thread/atomic, std::sync
// (spinlock/mpsc/lockfree/channel/task/thread_bare), time, io_file, io_traits,
// test/bench+fuzz, simd (+ vmath, sgemm, cpu), ipc (pipe, uds)
#include <std/test/test.h>
#include <std/test/bench.h>
#include <std/test/fuzz.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/str.h>
#include <std/heap.h>
#include <std/result.h>
#include <std/alloc/bump.h>
#include <std/alloc/slab.h>
#include <std/alloc/pool.h>
#include <std/alloc/tlsf.h>
#include <std/thread.h>
#include <std/atomic.h>
#include <std/time.h>
#include <std/io_file.h>
#include <std/io_traits.h>
#include <std/io.h>
#include <std/sync/spinlock.h>
#include <std/sync/mpsc.h>
#include <std/sync/lockfree.h>
#include <std/sync/channel.h>
#include <std/sync/task.h>
#include <std/sync/thread_bare.h>
#include <std/sync/parallel.h>
#include <std/simd/simd.h>
#include <std/simd/vmath.h>
#include <std/simd/gemm.h>
#include <std/simd/cpu.h>
#include <std/math.h>
#include <std/ipc/pipe.h>
#include <std/ipc/uds.h>
#include <std/sched/io_nb.h>
#include <std/reactive/signal.h>

struct Pkt { long long id; int len; };

void t_typed_alloc() {
    struct BumpAllocator b = BumpAllocator::new(256UL);
    ?&heap Pkt p = alloc_t(&b);
    match (p) { case some(x): { x.id = 7LL; x.len = 3; } case null: ASSERT_TRUE(0); }
    ?&heap long long arr = alloc_array_t(&b, 8UL);
    ASSERT_FALSE(arr.is_null());
    unsafe { ASSERT_EQ((unsigned long)arr % 8UL, 0); }
    ?&heap long long tooMany = alloc_array_t(&b, 1000UL);
    ASSERT_TRUE(tooMany.is_null());
    b.destroy();

    struct SlabAllocator s = SlabAllocator::new(sizeof(struct Pkt), 2UL);
    ?&heap Pkt s1 = alloc_t(&s);
    ?&heap Pkt s2 = alloc_t(&s);
    ?&heap Pkt s3 = alloc_t(&s);
    ASSERT_TRUE(s3.is_null());
    match (s1) { case some(x): dealloc_t(&s, x); case null: ASSERT_TRUE(0); }
    match (s2) { case some(x): dealloc_t(&s, x); case null: ASSERT_TRUE(0); }
    ASSERT_EQ(s.available(), 2);
    s.destroy();

    struct PoolAllocator pl = PoolAllocator::new(4096UL);
    unsigned long before = pl.available();
    ?&heap Pkt pp = alloc_t(&pl);
    match (pp) {
        case some(x): dealloc_t(&pl, x);
        case null: ASSERT_TRUE(0);
    }
    ASSERT_EQ(pl.available(), before);
    pl.destroy();

    struct TlsfAllocator t = TlsfAllocator::new(65536UL);
    ?&heap int ints = alloc_array_t(&t, 100UL);
    match (ints) {
        case some(x): { unsafe { x[99] = 5; ASSERT_EQ(x[99], 5); } dealloc_t(&t, x); }
        case null: ASSERT_TRUE(0);
    }
    t.destroy();
}

void t_bump() {
    unsafe {
        struct BumpAllocator b = BumpAllocator::new(1024UL);
        ?&heap char p1 = (?&heap char)b.alloc(10UL, 1UL);
        ?&heap char p2 = (?&heap char)b.alloc(8UL, 8UL);
        ASSERT_FALSE(p1.is_null());
        ASSERT_FALSE(p2.is_null());
        unsigned long a1 = (unsigned long)p1;
        unsigned long a2 = (unsigned long)p2;
        ASSERT_EQ(a2 % 8UL, 0);
        ASSERT_TRUE(a2 >= a1 + 10UL);
        ?&heap char tooBig = (?&heap char)b.alloc(4096UL, 1UL);
        ASSERT_TRUE(tooBig.is_null());
        // The address is aligned, past the 16 malloc gives the buffer.
        ?&heap char p3 = (?&heap char)b.alloc(8UL, 256UL);
        ASSERT_FALSE(p3.is_null());
        ASSERT_EQ((unsigned long)p3 % 256UL, 0);
        ASSERT_TRUE(b.alloc(8UL, 24UL).is_null());   // not a power of two
        unsigned long left = b.remaining();
        b.dealloc((?&heap void)p3);                  // a no-op on an arena
        ASSERT_EQ(b.remaining(), left);
        b.reset();
        ASSERT_EQ(b.remaining(), 1024);
        b.destroy();
    }
}

void t_slab() {
    unsafe {
        struct SlabAllocator s = SlabAllocator::new(32UL, 4UL);
        ?&heap void a = s.alloc(32UL, 8UL);
        ?&heap void b = s.alloc(32UL, 8UL);
        ?&heap void c = s.alloc(32UL, 8UL);
        ?&heap void d = s.alloc(32UL, 8UL);
        ASSERT_FALSE(d.is_null());
        ?&heap void none = s.alloc(32UL, 8UL);
        ASSERT_TRUE(none.is_null());
        ASSERT_EQ(s.available(), 0);
        s.dealloc(b);
        ASSERT_EQ(s.available(), 1);
        ASSERT_TRUE(s.alloc(33UL, 1UL).is_null());   // bigger than one object
        ASSERT_TRUE(s.alloc(8UL, 16UL).is_null());   // beyond max_align() (8)
        ASSERT_EQ(s.available(), 1);
        ?&heap void e = s.alloc(32UL, 8UL);
        ASSERT_TRUE((unsigned long)e == (unsigned long)b);
        s.dealloc(a); s.dealloc(c); s.dealloc(d); s.dealloc(e);
        ASSERT_EQ(s.available(), 4);
        s.destroy();
    }
}

void t_pool() {
    unsafe {
        struct PoolAllocator p = PoolAllocator::new(4096UL);
        unsigned long before = p.available();
        ?&heap char aq = (?&heap char)p.alloc(100UL, 1UL);
        ?&heap char bq = (?&heap char)p.alloc(200UL, 8UL);
        ASSERT_FALSE(aq.is_null());
        ASSERT_FALSE(bq.is_null());
        if (aq.is_null() || bq.is_null()) return;
        &char a = std::expect_ref(aq, "aq is null");
        &char b = std::expect_ref(bq, "bq is null");
        safe_memset(a, 1, 100UL);
        safe_memset(b, 2, 200UL);
        ASSERT_EQ(a[99], 1);
        p.dealloc((?&heap void)aq);
        p.dealloc((?&heap void)bq);
        ASSERT_EQ(p.available(), before);   // coalesced back
        ?&heap void huge = p.alloc(1000000UL, 1UL);
        ASSERT_TRUE(huge.is_null());
        ASSERT_TRUE(p.alloc(~0UL - 3UL, 1UL).is_null());   // size rounding mustn't wrap
        ASSERT_TRUE(p.alloc(16UL, 16UL).is_null());       // beyond the 8 it guarantees
        ASSERT_EQ(p.available(), before);
        p.destroy();
    }
}

void t_tlsf() {
    unsafe {
        struct TlsfAllocator t = TlsfAllocator::new(65536UL);
        ?&heap unsigned char ptrs[64];
        int i = 0;
        while (i < 64) {
            ptrs[i] = (?&heap unsigned char)t.alloc((unsigned long)(16 + i * 13), 1UL);
            if (ptrs[i].is_null()) break;
            safe_memset(std::expect_ref(ptrs[i], "tlsf"), i, (unsigned long)(16 + i * 13));
            i = i + 1;
        }
        ASSERT_EQ(i, 64);
        if (i != 64) return;
        int ok = 1;
        i = 0;
        while (i < 64) {
            match (ptrs[i]) {
                case null: { ok = 0; }
                case some(blk): { if (blk[15] != (unsigned char)i) ok = 0; }
            }
            i = i + 2;
        }
        ASSERT_TRUE(ok);
        i = 0;
        while (i < 64) {
            match (ptrs[i]) {
                case null: { }
                case some(blk): { t.dealloc((?&heap void)blk); }
            }
            i = i + 1;
        }
        // After freeing everything a large block must fit again (coalescing).
        ?&heap void bigq = t.alloc(50000UL, 8UL);
        match (bigq) {
            case null: { ASSERT_TRUE(0); }
            case some(big): { t.dealloc(big); }
        }
        ASSERT_TRUE(t.alloc(~0UL - 3UL, 1UL).is_null());   // size rounding mustn't wrap
        ASSERT_TRUE(t.alloc(64UL, 32UL).is_null());       // beyond the 8 it guarantees
        t.destroy();
    }
}

void t_heap_result() {
    unsafe {
        heap_init();
        ?&heap char pq = (?&heap char)heap_alloc(100UL);
        ASSERT_FALSE(pq.is_null());
        if (pq.is_null()) return;
        &char p = std::expect_ref(pq, "pq is null");
        str_copy(p, "heap", 100UL);
        ?&heap char q = (?&heap char)heap_realloc((&heap void)p, 1000UL);
        ASSERT_FALSE(q.is_null());
        if (q.is_null()) return;
        ASSERT_STR_EQ(q, "heap");
        match (q) {
            case null: { }
            case some(qq): { heap_free((&heap void)qq); }
        }
        ASSERT_EQ(heap_capacity(), 0);   // hosted: unbounded (malloc-backed)

        struct Result<int, long long> r = Result::new_ok(42);
        ASSERT_TRUE(r.ok());
        ASSERT_EQ(r.get_ok().default(0), 42);
        ASSERT_TRUE(r.get_err().is_null());
        struct Result<int, long long> e = Result::new_err(7LL);
        ASSERT_TRUE(e.err());
        ASSERT_EQ(e.get_err().default(0LL), 7);
        ASSERT_EQ(e.unwrap_or(-1), -1);
    }
}

// ── threads ──────────────────────────────────────────────────────────────────
static int g_counter = 0;
static long long g_atomic = 0LL;
static unsigned long long g_mtx = 0ULL;
static struct Spinlock g_spin;
static int g_spin_counter = 0;

// Thread entry (pthread ABI: one pointer in, one pointer out); the context
// is unused, so it is typed as a nullable reference to nothing in particular.
static ?&int worker(?&int arg) {
    unsafe {
        int i = 0;
        while (i < 10000) {
            mutex_lock(&g_mtx);
            g_counter = g_counter + 1;
            mutex_unlock(&g_mtx);
            atomic_fetch_add_ll(&g_atomic, 1LL);
            g_spin.lock();
            g_spin_counter = g_spin_counter + 1;
            g_spin.unlock();
            i = i + 1;
        }
    }
    return null;
}

void t_threads() {
    unsafe {
        mutex_init(&g_mtx);
        g_spin = Spinlock::new();
        unsigned long long tids[4];
        int i = 0;
        while (i < 4) { ASSERT_EQ(thread_create(&tids[i], worker, null), 0); i = i + 1; }
        i = 0;
        while (i < 4) { thread_join(tids[i]); i = i + 1; }
        ASSERT_EQ(g_counter, 40000);
        ASSERT_EQ(atomic_load_ll(&g_atomic), 40000);
        ASSERT_EQ(g_spin_counter, 40000);
        mutex_destroy(&g_mtx);

        int x = 5;
        int exp = 5;
        ASSERT_TRUE(atomic_cas_int(&x, &exp, 9));
        ASSERT_EQ(x, 9);
        exp = 1;
        ASSERT_FALSE(atomic_cas_int(&x, &exp, 3));
        ASSERT_EQ(exp, 9);
        ASSERT_EQ(atomic_exchange_int(&x, 1), 9);

        unsigned long long rw = 0ULL;
        rwlock_init(&rw);
        rwlock_rdlock(&rw);
        ASSERT_NE(rwlock_trywrlock(&rw), 0);
        rwlock_rdunlock(&rw);
        ASSERT_EQ(rwlock_trywrlock(&rw), 0);
        rwlock_wrunlock(&rw);
        rwlock_destroy(&rw);

        unsigned long long m = 0ULL;
        unsigned long long cv = 0ULL;
        mutex_init(&m);
        cond_init(&cv);
        mutex_lock(&m);
        long long t0 = time_mono_ns();
        cond_timedwait_ms(&cv, &m, 20UL);   // nobody signals → times out
        long long dt = time_mono_ns() - t0;
        mutex_unlock(&m);
        ASSERT_TRUE(dt >= 15000000LL);
        cond_destroy(&cv);
        mutex_destroy(&m);
    }
}

static struct MpscQueue<int> g_mq;
static int g_bases[2];
static ?&int mpsc_producer(?&int arg) {
    int base = arg.default(0);
    int i = 0;
    while (i < 1000) {
        while (!g_mq.enqueue(base + i)) { thread_yield(); }
        i = i + 1;
    }
    return null;
}

static struct Chan<int> g_ch;
static ?&int chan_producer(?&int arg) {
    int i = 1;
    while (i <= 100) { g_ch.send(i); i = i + 1; }
    g_ch.close();
    return null;
}

void t_queues() {
    g_mq = MpscQueue::new(64UL);
    unsigned long long t1 = 0ULL;
    unsigned long long t2 = 0ULL;
    unsafe { g_bases[0] = 0; g_bases[1] = 100000; }
    unsafe {
        thread_create(&t1, mpsc_producer, (?&int)&g_bases[0]);
        thread_create(&t2, mpsc_producer, (?&int)&g_bases[1]);
    }
    long long sum = 0LL;
    int got = 0;
    while (got < 2000) {
        match (g_mq.dequeue()) {
            case some(v): { sum = sum + (long long)v; got = got + 1; }
            case none: thread_yield();
        }
    }
    thread_join(t1);
    thread_join(t2);
    // 2 * (0+..+999) + 1000*100000
    ASSERT_EQ(sum, 999000LL + 100000000LL);
    ASSERT_TRUE(g_mq.dequeue().is_none());
    g_mq.destroy();

    struct LFQueue<long long> q = LFQueue::new(8UL);
    long long i = 0LL;
    while (i < 8LL) { ASSERT_TRUE(q.enqueue(i)); i = i + 1LL; }
    ASSERT_TRUE(q.is_full());
    ASSERT_FALSE(q.enqueue(i));
    ASSERT_EQ(q.dequeue().default(-1LL), 0);
    ASSERT_EQ(q.len(), 7);
    q.destroy();

    struct Chan<int> ch = Chan::new(4);
    ch.send(11);
    ch.send(22);
    ASSERT_EQ(ch.recv().default(0), 11);
    ASSERT_EQ(ch.recv().default(0), 22);
    ch.close();
    ASSERT_TRUE(ch.recv().is_none());
    ASSERT_FALSE(ch.send(33));

    // blocking hand-off across threads, closed by the producer
    g_ch = Chan::new(2);
    unsigned long long tp = 0ULL;
    thread_create(&tp, chan_producer, null);
    long long total = 0LL;
    int more = 1;
    while (more) {
        match (g_ch.recv()) {
            case some(v): total = total + (long long)v;
            case none: more = 0;
        }
    }
    thread_join(tp);
    ASSERT_EQ(total, 5050);
}

// Cooperative task: counts to 3 across three resumptions.
static int g_task_hits = 0;
static int counting_task(?&int arg, int resume) {
    unsafe { g_task_hits = g_task_hits + 1; }
    if (resume >= 3) return 0;
    return resume + 1;
}

// ── parallel_for ─────────────────────────────────────────────────────────────
static int g_par_hits[100000];
struct ParJob { &int hits; long long sum; int calls; };

static void par_mark(&ParJob j, unsigned long lo, unsigned long hi) {
    unsigned long i = lo;
    long long s = 0LL;
    while (i < hi) {
        unsafe { j.hits[i] = j.hits[i] + 1; }
        s = s + (long long)i;
        i = i + 1UL;
    }
    atomic_fetch_add_ll(&j.sum, s);
    atomic_fetch_add_int(&j.calls, 1);
}

// A body that itself calls parallel_for: the inner call runs inline.
static void par_nested(&ParJob j, unsigned long lo, unsigned long hi) {
    struct ParJob inner;
    inner.hits = j.hits;
    inner.sum = 0LL;
    inner.calls = 0;
    unsigned long i = lo;
    while (i < hi) {
        unsafe { parallel_for(1UL, 1UL, par_mark, &inner); }
        i = i + 1UL;
    }
    atomic_fetch_add_int(&j.calls, inner.calls == (int)(hi - lo) ? 1 : 1000);
}

void t_parallel() {
    ASSERT_TRUE(parallel_threads() >= 1UL);
    unsigned long n = 100000UL;
    &int hits = &g_par_hits[0];
    int round = 0;
    while (round < 50) {           // repeated jobs reuse the parked workers
        unsigned long i = 0UL;
        while (i < n) { unsafe { hits[i] = 0; } i = i + 1UL; }
        struct ParJob j;
        j.hits = hits;
        j.sum = 0LL;
        j.calls = 0;
        parallel_for(n, 1000UL, par_mark, &j);
        ASSERT_TRUE(j.sum == (long long)(n * (n - 1UL) / 2UL));
        int bad = 0;
        i = 0UL;
        while (i < n) { unsafe { if (hits[i] != 1) bad = bad + 1; } i = i + 1UL; }
        ASSERT_EQ(bad, 0);
        ASSERT_TRUE(j.calls >= 1);
        round = round + 1;
    }
    struct ParJob small;
    small.hits = hits;
    small.sum = 0LL;
    small.calls = 0;
    parallel_for(10UL, 100UL, par_mark, &small);       // n <= grain: one inline call
    ASSERT_EQ(small.calls, 1);
    parallel_for(0UL, 1UL, par_mark, &small);          // empty range: no call
    ASSERT_EQ(small.calls, 1);

    struct ParJob nest;
    nest.hits = hits;
    nest.sum = 0LL;
    nest.calls = 0;
    parallel_for(64UL, 4UL, par_nested, &nest);
    ASSERT_TRUE(nest.calls >= 1 && nest.calls < 1000);

    unsigned long saved = parallel_threads();
    parallel_set_threads(1UL);
    ASSERT_EQ(parallel_threads(), 1UL);
    struct ParJob one;
    one.hits = hits;
    one.sum = 0LL;
    one.calls = 0;
    parallel_for(n, 10UL, par_mark, &one);
    ASSERT_EQ(one.calls, 1);
    parallel_set_threads(saved);
}

void t_tasks() {
    unsafe {
        ?&heap TaskScheduler sq = alloc(sizeof(struct TaskScheduler));
        ?&heap ThreadSched tsq = alloc(sizeof(struct ThreadSched));
        ASSERT_FALSE(sq.is_null());
        ASSERT_FALSE(tsq.is_null());
        if (sq.is_null() || tsq.is_null()) return;
        &TaskScheduler s = std::expect_ref(sq, "sq is null");
        *s = TaskScheduler::new();
        ASSERT_TRUE(task_spawn(s, counting_task, null) >= 0);
        task_spawn(s, counting_task, null);
        s.run_all();
        ASSERT_EQ(g_task_hits, 8);
        ASSERT_EQ(s.active_count(), 0);
        dealloc(sq);

        g_task_hits = 0;
        &ThreadSched ts = std::expect_ref(tsq, "tsq is null");
        *ts = ThreadSched::new();
        Thread a = thread_spawn(ts, counting_task, null, 1);
        thread_spawn(ts, counting_task, null, 5);
        ts.join_thread(a);
        ASSERT_FALSE(ts.is_active(a));
        ts.run_all();
        ASSERT_EQ(g_task_hits, 8);
        dealloc(tsq);
    }
}

static int g_sig_seen = 0;
static void on_sig(?&int seen) {
    match (seen) {
        case some(p): *p = *p + 1;
        case null: { }
    }
}

void t_signal() {
    struct Signal<int> c = Signal::new(5);
    unsafe { signal_subscribe(&c, on_sig, (?&int)&g_sig_seen); }
    unsigned long last = c.get_version();
    ASSERT_FALSE(c.changed(&last));
    c.set(c.get() + 1);
    ASSERT_EQ(c.get(), 6);
    ASSERT_EQ(g_sig_seen, 1);
    ASSERT_TRUE(c.changed(&last));
    ASSERT_FALSE(c.changed(&last));
    *c.get_ref() = 40;
    c.notify();
    ASSERT_EQ(g_sig_seen, 2);
    c.unsubscribe_all();
    c.set(1);
    ASSERT_EQ(g_sig_seen, 2);
    c.free();
}

void t_time() {
    unsafe {
        long long a = time_mono_ns();
        time_sleep_ns(2000000LL);
        long long b = time_mono_ns();
        ASSERT_TRUE(b - a >= 1500000LL);
        int tm[9];
        time_gmtime(0LL, (int*)tm);
        char buf[64];
        time_format((char*)buf, 64UL, "%Y-%m-%d", (int*)tm);
        ASSERT_STR_EQ((const char*)buf, "1970-01-01");
        time_gmtime(951782400LL, (int*)tm);   // 2000-02-29
        time_format((char*)buf, 64UL, "%Y-%m-%d", (int*)tm);
        ASSERT_STR_EQ((const char*)buf, "2000-02-29");
        ASSERT_TRUE(time_is_leap(2000));
        ASSERT_FALSE(time_is_leap(1900));
        ASSERT_EQ(time_days_in_month(1, 2024), 29);   // month is 0-based
        ASSERT_TRUE(time_now() > 1600000000LL);
    }
}

void t_file() {
    unsafe {
        const &char path = "/tmp/safec_stdlib_test_file.txt";
        ?&File wq = File::open(path, "wb");
        ASSERT_FALSE(wq.is_null());
        if (wq.is_null()) return;
        &File w = std::expect_ref(wq, "write handle");
        w.puts("line1\n");
        w.write((const &unsigned char)"abc", 3UL);
        w.close();
        &File f = std::expect_ref(File::open(path, "rb"), "read handle");
        ASSERT_EQ(f.size(), 9);
        char buf[32];
        f.gets(buf, 32);
        ASSERT_STR_EQ(buf, "line1\n");
        ASSERT_EQ(f.getc(), 'a');
        f.seek(0LL, 0);
        ASSERT_EQ(f.tell(), 0);
        f.close();
        ASSERT_EQ(io_remove(path), 0);
        ASSERT_TRUE(File::open("/nonexistent_dir_safec/x", "rb").is_null());
    }
}

// io_traits.h's Reader/Writer helpers over File and Pipe: write_all,
// read_exact (and its early-EOF result), copy_stream in both directions.
void t_io_traits() {
    unsafe {
        const &char path = "/tmp/safec_stdlib_test_traits.bin";
        unsigned char data[10000];
        unsigned long i = 0UL;
        while (i < 10000UL) { data[i] = (unsigned char)(i * 7UL + 3UL); i = i + 1UL; }

        &File w = std::expect_ref(File::open(path, "wb"), "write handle");
        ASSERT_EQ(write_all(w, &data[0], 10000UL), 1);
        w.close();

        &File r = std::expect_ref(File::open(path, "rb"), "read handle");
        unsigned char back[10000];
        ASSERT_EQ(read_exact(r, &back[0], 6000UL), 1);
        ASSERT_EQ(back[5999], data[5999]);
        ASSERT_EQ(read_exact(r, &back[6000], 5000UL), 0);    // only 4000 left
        ASSERT_EQ(back[9999], data[9999]);
        r.close();

        // File -> Pipe -> File, the pipe drained by a second copy after
        // its write end closes. 10000 bytes fit the pipe's buffer.
        struct Pipe p;
        ASSERT_EQ(p.init(), 0);
        &File src = std::expect_ref(File::open(path, "rb"), "source");
        ASSERT_EQ(copy_stream(src, &p), 10000);
        src.close();
        ASSERT_EQ(p.close_write(), 0);
        const &char path2 = "/tmp/safec_stdlib_test_traits2.bin";
        &File dst = std::expect_ref(File::open(path2, "wb"), "dest");
        ASSERT_EQ(copy_stream(&p, dst), 10000);
        dst.close();
        ASSERT_EQ(p.close_read(), 0);

        &File chk = std::expect_ref(File::open(path2, "rb"), "check");
        ASSERT_EQ(chk.size(), 10000);
        unsigned char again[10000];
        ASSERT_EQ(read_exact(chk, &again[0], 10000UL), 1);
        ASSERT_EQ(again[0], data[0]);
        ASSERT_EQ(again[4321], data[4321]);
        ASSERT_EQ(again[9999], data[9999]);
        ASSERT_EQ(chk.read(&again[0], 16UL), 0);              // EOF
        chk.close();

        // Pipe read_exact / write_all directly.
        struct Pipe q;
        ASSERT_EQ(q.init(), 0);
        ASSERT_EQ(write_all(&q, (const &unsigned char)"hello", 5UL), 1);
        ASSERT_EQ(q.close_write(), 0);
        unsigned char hb[8];
        ASSERT_EQ(read_exact(&q, &hb[0], 3UL), 1);
        ASSERT_EQ(hb[2], 'l');
        ASSERT_EQ(read_exact(&q, &hb[0], 3UL), 0);            // 2 left, then EOF
        ASSERT_EQ(q.close_read(), 0);

        ASSERT_EQ(io_remove(path), 0);
        ASSERT_EQ(io_remove(path2), 0);
    }
}

static unsigned long g_bench_calls = 0UL;
static void bench_fn(?&int arg) { unsafe { g_bench_calls = g_bench_calls + 1UL; } }
static unsigned long g_fuzz_calls = 0UL;
static void fuzz_fn(const &unsigned char d, unsigned long n) { unsafe { g_fuzz_calls = g_fuzz_calls + 1UL; } }

void t_bench_fuzz() {
    unsafe {
        ?&heap BenchSuite bq = alloc(sizeof(struct BenchSuite));
        ASSERT_FALSE(bq.is_null());
        if (bq.is_null()) return;
        &BenchSuite b = std::expect_ref(bq, "bq is null");
        *b = BenchSuite::new();
        bench_add(b, "noop", bench_fn, null, 1000UL);
        b.run();
        ASSERT_EQ(g_bench_calls, 1000);
        ASSERT_TRUE(b.cases[0].elapsed_s >= 0.0);
        dealloc(bq);
        struct FuzzTarget ft = FuzzTarget::new(fuzz_fn, 50UL);
        unsigned char corpus[4] = { (unsigned char)115, (unsigned char)101, (unsigned char)101, (unsigned char)100 };   // "seed"
        ft.run(&corpus[0], 4UL);
        ASSERT_EQ(g_fuzz_calls, 51);
    }
}

void t_simd() {
    unsafe {
        float a[8];
        float b[8];
        float o[8];
        int i = 0;
        while (i < 8) { a[i] = (float)i; b[i] = (float)(2 * i); i = i + 1; }
        f32x8 va = simd_load_f32x8((const float*)a);
        f32x8 vb = simd_load_f32x8((const float*)b);
        f32x8 vc = simd_fma_f32x8(va, vb, simd_splat_f32x8((float)1.0));
        simd_store_f32x8(vc, (float*)o);
        ASSERT_TRUE(o[3] == (float)19.0);   // 3*6+1
        ASSERT_TRUE(simd_hsum_f32x8(va) == (float)28.0);
        int ia[4];
        ia[0] = 4; ia[1] = -2; ia[2] = 9; ia[3] = 1;
        i32x4 vi = simd_load_i32x4((const int*)ia);
        ASSERT_EQ(simd_hsum_i32x4(vi), 12);
        i32x4 mx = simd_max_i32x4(vi, simd_splat_i32x4(3));
        ASSERT_EQ(simd_hsum_i32x4(mx), 4 + 3 + 9 + 3);
        f64x4 d = simd_splat_f64x4(0.5);
        ASSERT_TRUE(simd_hsum_f64x4(d) == 2.0);
        ASSERT_TRUE(simd_hmax_f32x4(simd_load_f32x4((const float*)b)) == (float)6.0);
    }
}

static int near_f(float a, float b, float eps) {
    float d = a - b;
    if (d < (float)0.0) d = -d;
    return d <= eps;
}

void t_simd_more() {
    unsafe {
        double d2[2];
        d2[0] = 1.5; d2[1] = -2.0;
        f64x2 vd = simd_load_f64x2((const double*)d2);
        f64x2 fd = simd_fma_f64x2(vd, simd_splat_f64x2(2.0), simd_splat_f64x2(1.0));
        simd_store_f64x2(fd, (double*)d2);
        ASSERT_TRUE(d2[0] == 4.0 && d2[1] == -3.0);
        ASSERT_TRUE(simd_hsum_f64x2(fd) == 1.0);
        double d4[4];
        d4[0] = 1.0; d4[1] = 2.0; d4[2] = 3.0; d4[3] = 4.0;
        f64x4 v4 = simd_fma_f64x4(simd_load_f64x4((const double*)d4), simd_splat_f64x4(2.0), simd_splat_f64x4(0.5));
        ASSERT_TRUE(simd_hsum_f64x4(v4) == 22.0);
        float f4[4];
        f4[0] = (float)3.0; f4[1] = (float)-1.0; f4[2] = (float)7.0; f4[3] = (float)2.0;
        f32x4 a = simd_load_f32x4((const float*)f4);
        ASSERT_TRUE(simd_hmin_f32x4(a) == (float)-1.0);
        f32x4 mx = simd_max_f32x4(a, simd_splat_f32x4((float)2.5));
        simd_store_f32x4(mx, (float*)f4);
        ASSERT_TRUE(f4[1] == (float)2.5 && f4[2] == (float)7.0);
        ASSERT_TRUE(simd_hsum_f32x4(simd_min_f32x4(a, simd_splat_f32x4((float)0.0))) == (float)-1.0);
        float f8[8];
        int i = 0;
        while (i < 8) { f8[i] = (float)(i - 4); i = i + 1; }
        f32x8 v8 = simd_max_f32x8(simd_load_f32x8((const float*)f8), simd_splat_f32x8((float)0.0));
        ASSERT_TRUE(simd_hsum_f32x8(v8) == (float)6.0);
        int i8[8];
        i = 0;
        while (i < 8) { i8[i] = i * 3; i = i + 1; }
        i32x8 vi8 = simd_load_i32x8((const int*)i8);
        ASSERT_EQ(simd_hsum_i32x8(vi8), 84);
        int i4[4];
        simd_store_i32x4(simd_min_i32x4(simd_splat_i32x4(5), simd_load_i32x4((const int*)i8)), (int*)i4);
        ASSERT_EQ(i4[3], 5);
        unsigned int u4[4];
        u4[0] = 1u; u4[1] = 2u; u4[2] = 3u; u4[3] = 4000000000u;
        u32x4 vu = simd_load_u32x4((const unsigned int*)u4);
        simd_store_u32x4(vu, (unsigned int*)u4);
        test_assert_eq_u(simd_hsum_u32x4(simd_splat_u32x4(1000000000u)), 4000000000ULL, "hsum u32", __FILE__, __LINE__);
        signed char sb[16];
        i = 0;
        while (i < 16) { sb[i] = (signed char)(i - 8); i = i + 1; }
        signed char sb2[16];
        simd_store_i8x16(simd_load_i8x16((const signed char*)sb), (signed char*)sb2);
        ASSERT_EQ(sb2[0], -8);
        ASSERT_EQ(sb2[15], 7);

        // vmath against libm
        float x[9];
        float y[9];
        i = 0;
        while (i < 9) { x[i] = (float)(i - 4) * (float)1.5; i = i + 1; }
        int ok = 1;
        vm_exp_f32((const float*)x, (float*)y, 9UL);
        i = 0;
        while (i < 9) { if (!near_f(y[i], exp_f(x[i]), exp_f(x[i]) * (float)1e-5)) ok = 0; i = i + 1; }
        ASSERT_TRUE(ok);
        vm_tanh_f32((const float*)x, (float*)y, 9UL);
        i = 0;
        while (i < 9) { if (!near_f(y[i], tanh_f(x[i]), (float)1e-5)) ok = 0; i = i + 1; }
        ASSERT_TRUE(ok);
        vm_sigmoid_f32((const float*)x, (float*)y, 9UL);
        i = 0;
        while (i < 9) {
            float ref = (float)1.0 / ((float)1.0 + exp_f(-x[i]));
            if (!near_f(y[i], ref, (float)1e-5)) ok = 0;
            i = i + 1;
        }
        ASSERT_TRUE(ok);

        // sgemm against a naive product (odd sizes hit the edge paths)
        unsigned long m = 7UL;
        unsigned long n = 5UL;
        unsigned long k = 9UL;
        float A[63];
        float B[45];
        float C[35];
        float R[35];
        i = 0;
        while (i < 63) { A[i] = (float)((i * 7) % 11) - (float)5.0; i = i + 1; }
        i = 0;
        while (i < 45) { B[i] = (float)((i * 5) % 13) * (float)0.25; i = i + 1; }
        unsigned long r = 0UL;
        while (r < m) {
            unsigned long cc = 0UL;
            while (cc < n) {
                float acc = (float)0.0;
                unsigned long p = 0UL;
                while (p < k) { acc = acc + A[r * k + p] * B[p * n + cc]; p = p + 1UL; }
                R[r * n + cc] = (float)2.0 * acc + (float)1.0;
                C[r * n + cc] = (float)1.0;
                cc = cc + 1UL;
            }
            r = r + 1UL;
        }
        sgemm(SGEMM_NO_TRANS, SGEMM_NO_TRANS, m, n, k, (float)2.0, (const float*)A, k,
              (const float*)B, n, (float)1.0, (float*)C, n);
        i = 0;
        while (i < 35) { if (!near_f(C[i], R[i], (float)1e-3)) ok = 0; i = i + 1; }
        ASSERT_TRUE(ok);
        sgemm_serial(SGEMM_NO_TRANS, SGEMM_NO_TRANS, m, n, k, (float)2.0, (const float*)A, k,
                     (const float*)B, n, (float)0.0, (float*)C, n);
        i = 0;
        while (i < 35) { if (!near_f(C[i] + (float)1.0, R[i], (float)1e-3)) ok = 0; i = i + 1; }
        ASSERT_TRUE(ok);
        ASSERT_TRUE(sgemm_vendor() == 0 || sgemm_vendor() == 1);
        ASSERT_TRUE(str_len(sgemm_backend()) > 0UL);
        int lvl = simd_level();
        ASSERT_TRUE(lvl >= SIMD_SCALAR);
        ASSERT_TRUE(str_len(simd_level_name(lvl)) > 0UL);
        ASSERT_STR_EQ(simd_level_name(SIMD_AVX2), "AVX2");
    }
}

void t_ipc() {
    unsafe {
        struct Pipe p;
        ASSERT_EQ(p.init(), 0);
        ASSERT_EQ(p.set_read_nonblocking(), 0);
        unsigned char buf[16];
        // Empty non-blocking pipe: returns at once instead of blocking.
        ASSERT_EQ(p.read(&buf[0], 16UL), -1);
        ASSERT_TRUE(sock_would_block());
        ASSERT_EQ(p.set_write_nonblocking(), 0);
        ASSERT_EQ(p.write((const unsigned char*)"ping", 4UL), 4);
        ASSERT_EQ(p.read(&buf[0], 16UL), 4);
        ASSERT_EQ(buf[3], 'g');
        ASSERT_EQ(p.close_write(), 0);
        ASSERT_EQ(p.read(&buf[0], 16UL), 0);          // EOF
        ASSERT_EQ(p.close_read(), 0);

        const &char path = "/tmp/safec_std_test_uds.sock";
        ASSERT_EQ(uds_unlink(path), 0);                      // missing is fine
        int lfd = uds_listen_nb(path);
        ASSERT_TRUE(lfd >= 0);
        ASSERT_EQ(uds_accept_nb(lfd), -1);                   // nobody yet
        int cfd = uds_connect_nb(path);
        ASSERT_TRUE(cfd >= 0);
        int sfd = -1;
        int tries = 0;
        while (sfd < 0 && tries < 1000) { sfd = uds_accept_nb(lfd); tries = tries + 1; }
        ASSERT_TRUE(sfd >= 0);
        ASSERT_EQ(uds_listen_nb(path), -1);                  // path still bound
        ASSERT_EQ(sched_htons((unsigned short)0x1234), 0x3412);
        ASSERT_EQ(uds_unlink(path), 0);
        ASSERT_EQ(uds_listen_nb("/tmp/a-path-that-is-far-too-long-for-sun-path-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"), -1);
    }
}

static unsigned long long g_cmtx = 0ULL;
static unsigned long long g_cond = 0ULL;
static int g_ready = 0;

static ?&int signaller(?&int arg) {
    unsafe {
        time_sleep_ns(2000000LL);
        mutex_lock(&g_cmtx);
        g_ready = 1;
        cond_signal(&g_cond);
        mutex_unlock(&g_cmtx);
    }
    return null;
}

void t_cond_trylock() {
    unsafe {
        ASSERT_EQ(mutex_init(&g_cmtx), 0);
        ASSERT_EQ(cond_init(&g_cond), 0);
        unsigned long long tid = 0ULL;
        ASSERT_EQ(thread_create(&tid, signaller, null), 0);
        mutex_lock(&g_cmtx);
        while (g_ready == 0) { cond_wait(&g_cond, &g_cmtx); }
        ASSERT_TRUE(mutex_trylock(&g_cmtx) != 0);   // already held
        mutex_unlock(&g_cmtx);
        thread_join(tid);
        ASSERT_EQ(g_ready, 1);
        ASSERT_EQ(mutex_trylock(&g_cmtx), 0);
        mutex_unlock(&g_cmtx);
        // timed wait with nobody signalling times out
        mutex_lock(&g_cmtx);
        ASSERT_EQ(cond_timedwait_ms(&g_cond, &g_cmtx, 5UL), THREAD_TIMEDOUT);
        mutex_unlock(&g_cmtx);
        cond_destroy(&g_cond);
        mutex_destroy(&g_cmtx);

        unsigned long long rw = 0ULL;
        ASSERT_EQ(rwlock_init(&rw), 0);
        ASSERT_EQ(rwlock_tryrdlock(&rw), 0);
        ASSERT_EQ(rwlock_tryrdlock(&rw), 0);       // shared
        ASSERT_TRUE(rwlock_trywrlock(&rw) != 0);    // readers hold it
        rwlock_rdunlock(&rw);
        rwlock_rdunlock(&rw);
        ASSERT_EQ(rwlock_trywrlock(&rw), 0);
        rwlock_wrunlock(&rw);
        rwlock_destroy(&rw);
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("bump", t_bump);
        t.add("typed allocators", t_typed_alloc);
        t.add("slab", t_slab);
        t.add("pool", t_pool);
        t.add("tlsf", t_tlsf);
        t.add("heap+result", t_heap_result);
        t.add("thread/mutex/atomic/spinlock/rwlock/cond", t_threads);
        t.add("mpsc/lockfree/channel", t_queues);
        t.add("signal", t_signal);
        t.add("task/thread_bare", t_tasks);
        t.add("parallel_for", t_parallel);
        t.add("time", t_time);
        t.add("io_file", t_file);
        t.add("io_traits over File/Pipe", t_io_traits);
        t.add("bench/fuzz", t_bench_fuzz);
        t.add("simd", t_simd);
        t.add("simd ops/vmath/sgemm/cpu", t_simd_more);
        t.add("ipc pipe/uds", t_ipc);
        t.add("cond/trylock/rwlock try", t_cond_trylock);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
