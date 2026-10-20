// std::alloc (bump/slab/pool/tlsf), heap, result, thread/atomic, std::sync
// (spinlock/mpsc/lockfree/channel/task/thread_bare), time, io_file,
// test/bench+fuzz, simd
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
#include <std/io.h>
#include <std/sync/spinlock.h>
#include <std/sync/mpsc.h>
#include <std/sync/lockfree.h>
#include <std/sync/channel.h>
#include <std/sync/task.h>
#include <std/sync/thread_bare.h>
#include <std/simd/simd.h>
#include <std/reactive/signal.h>

struct Pkt { long long id; int len; };

void t_typed_alloc() {
    struct BumpAllocator b = bump_new(256UL);
    ?&heap Pkt p = bump_alloc_t(&b);
    match (p) { case some(x): { x.id = 7LL; x.len = 3; } case null: ASSERT_TRUE(0); }
    ?&heap long long arr = bump_alloc_array_t(&b, 8UL);
    ASSERT_FALSE(arr.is_null());
    unsafe { ASSERT_EQ((unsigned long)arr % 8UL, 0); }
    ?&heap long long tooMany = bump_alloc_array_t(&b, 1000UL);
    ASSERT_TRUE(tooMany.is_null());
    b.destroy();

    struct SlabAllocator s = slab_new(sizeof(struct Pkt), 2UL);
    ?&heap Pkt s1 = slab_alloc_t(&s);
    ?&heap Pkt s2 = slab_alloc_t(&s);
    ?&heap Pkt s3 = slab_alloc_t(&s);
    ASSERT_TRUE(s3.is_null());
    match (s1) { case some(x): slab_free_t(&s, x); case null: ASSERT_TRUE(0); }
    match (s2) { case some(x): slab_free_t(&s, x); case null: ASSERT_TRUE(0); }
    ASSERT_EQ(s.available(), 2);
    s.destroy();

    struct PoolAllocator pl = pool_new(4096UL);
    unsigned long before = pl.available();
    ?&heap Pkt pp = pool_alloc_t(&pl);
    match (pp) {
        case some(x): pool_free_t(&pl, x);
        case null: ASSERT_TRUE(0);
    }
    ASSERT_EQ(pl.available(), before);
    pl.destroy();

    struct TlsfAllocator t = tlsf_new(65536UL);
    ?&heap int ints = tlsf_alloc_array_t(&t, 100UL);
    match (ints) {
        case some(x): { unsafe { x[99] = 5; ASSERT_EQ(x[99], 5); } tlsf_free_t(&t, x); }
        case null: ASSERT_TRUE(0);
    }
    t.destroy();
}

void t_bump() {
    unsafe {
        struct BumpAllocator b = bump_new(1024UL);
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
        b.reset();
        ASSERT_EQ(b.remaining(), 1024);
        b.destroy();
    }
}

void t_slab() {
    unsafe {
        struct SlabAllocator s = slab_new(32UL, 4UL);
        ?&heap void a = s.alloc();
        ?&heap void b = s.alloc();
        ?&heap void c = s.alloc();
        ?&heap void d = s.alloc();
        ASSERT_FALSE(d.is_null());
        ?&heap void none = s.alloc();
        ASSERT_TRUE(none.is_null());
        ASSERT_EQ(s.available(), 0);
        s.dealloc(b);
        ASSERT_EQ(s.available(), 1);
        ?&heap void e = s.alloc();
        ASSERT_TRUE((unsigned long)e == (unsigned long)b);
        s.dealloc(a); s.dealloc(c); s.dealloc(d); s.dealloc(e);
        ASSERT_EQ(s.available(), 4);
        s.destroy();
    }
}

void t_pool() {
    unsafe {
        struct PoolAllocator p = pool_new(4096UL);
        unsigned long before = p.available();
        ?&heap char aq = (?&heap char)p.alloc(100UL);
        ?&heap char bq = (?&heap char)p.alloc(200UL);
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
        ?&heap void huge = p.alloc(1000000UL);
        ASSERT_TRUE(huge.is_null());
        p.destroy();
    }
}

void t_tlsf() {
    unsafe {
        struct TlsfAllocator t = tlsf_new(65536UL);
        ?&heap unsigned char ptrs[64];
        int i = 0;
        while (i < 64) {
            ptrs[i] = (?&heap unsigned char)t.alloc((unsigned long)(16 + i * 13));
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
                case some(blk): { t.free((&heap void)blk); }
            }
            i = i + 1;
        }
        // After freeing everything a large block must fit again (coalescing).
        ?&heap void bigq = t.alloc(50000UL);
        match (bigq) {
            case null: { ASSERT_TRUE(0); }
            case some(big): { t.free(big); }
        }
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

        struct Result<int, long long> r = result_ok(42);
        ASSERT_TRUE(r.ok());
        ASSERT_EQ(r.get_ok().default(0), 42);
        ASSERT_TRUE(r.get_err().is_null());
        struct Result<int, long long> e = result_err(7LL);
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
        g_spin = spinlock_init();
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
    g_mq = mpsc_new(64UL);
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

    struct LFQueue<long long> q = lfq_new(8UL);
    long long i = 0LL;
    while (i < 8LL) { ASSERT_TRUE(q.enqueue(i)); i = i + 1LL; }
    ASSERT_TRUE(q.is_full());
    ASSERT_FALSE(q.enqueue(i));
    ASSERT_EQ(q.dequeue().default(-1LL), 0);
    ASSERT_EQ(q.len(), 7);
    q.destroy();

    struct Chan<int> ch = chan_new(4);
    ch.send(11);
    ch.send(22);
    ASSERT_EQ(ch.recv().default(0), 11);
    ASSERT_EQ(ch.recv().default(0), 22);
    ch.close();
    ASSERT_TRUE(ch.recv().is_none());
    ASSERT_FALSE(ch.send(33));

    // blocking hand-off across threads, closed by the producer
    g_ch = chan_new(2);
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

void t_tasks() {
    unsafe {
        ?&heap TaskScheduler sq = alloc(sizeof(struct TaskScheduler));
        ?&heap ThreadSched tsq = alloc(sizeof(struct ThreadSched));
        ASSERT_FALSE(sq.is_null());
        ASSERT_FALSE(tsq.is_null());
        if (sq.is_null() || tsq.is_null()) return;
        &TaskScheduler s = std::expect_ref(sq, "sq is null");
        *s = task_sched_init();
        ASSERT_TRUE(task_spawn(s, counting_task, null) >= 0);
        task_spawn(s, counting_task, null);
        s.run_all();
        ASSERT_EQ(g_task_hits, 8);
        ASSERT_EQ(s.active_count(), 0);
        dealloc(sq);

        g_task_hits = 0;
        &ThreadSched ts = std::expect_ref(tsq, "tsq is null");
        *ts = thread_sched_init();
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
    struct Signal<int> c = signal_new(5);
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
        ?&File wq = file_open(path, "wb");
        ASSERT_FALSE(wq.is_null());
        if (wq.is_null()) return;
        &File w = std::expect_ref(wq, "write handle");
        file_puts("line1\n", w);
        file_write(w, (const &unsigned char)"abc", 3UL);
        file_close(w);
        &File f = std::expect_ref(file_open(path, "rb"), "read handle");
        ASSERT_EQ(file_size(f), 9);
        char buf[32];
        file_gets(buf, 32, f);
        ASSERT_STR_EQ(buf, "line1\n");
        ASSERT_EQ(file_getc(f), 'a');
        file_seek(f, 0LL, 0);
        ASSERT_EQ(file_tell(f), 0);
        file_close(f);
        ASSERT_EQ(io_remove(path), 0);
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
        *b = bench_suite_init();
        bench_add(b, "noop", bench_fn, null, 1000UL);
        b.run();
        ASSERT_EQ(g_bench_calls, 1000);
        ASSERT_TRUE(b.cases[0].elapsed_s >= 0.0);
        dealloc(bq);
        struct FuzzTarget ft = fuzz_target_init(fuzz_fn, 50UL);
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

int main() {
    struct TestSuite t = test_suite_init();
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
        t.add("time", t_time);
        t.add("io_file", t_file);
        t.add("bench/fuzz", t_bench_fuzz);
        t.add("simd", t_simd);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
