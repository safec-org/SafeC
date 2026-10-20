// SafeC Standard Library — persistent thread pool (see parallel.h).
//
// One job runs at a time (g_par_busy). The caller publishes it under the
// pool mutex and bumps a generation counter; each worker wakes, claims
// chunks from an atomic counter until none are left, then reports done.
// The caller claims chunks too, then waits for every worker's report, so
// no worker can still be inside the job when parallel_for returns.
#pragma once
#include <std/sync/parallel.h>
#include <std/atomic.h>
#include <std/thread.h>
#include <std/sys.h>

namespace std {

static int g_par_busy = 0;               // 1 while a job runs (atomic)
static unsigned long g_par_want = 0UL;   // configured thread count, 0 = not yet decided

static void par_run_(void* body, void* ctx, unsigned long lo, unsigned long hi) {
    unsafe {
        fn void(void*, unsigned long, unsigned long) f = (fn void(void*, unsigned long, unsigned long))body;
        f(ctx, lo, hi);
    }
}

#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__)

unsigned long parallel_threads() { return 1UL; }
void parallel_set_threads(unsigned long n) { g_par_want = 1UL; }

void parallel_for_raw_(unsigned long n, unsigned long grain, void* body, void* ctx) {
    if (n > 0UL) par_run_(body, ctx, 0UL, n);
}

#else

#ifdef _WIN32
extern unsigned long GetActiveProcessorCount(unsigned short group);
#else
extern long sysconf(int name);
#endif

static unsigned long par_cpus_() {
    match (sys_getenv("SAFEC_NUM_THREADS")) {
        case some(env): {
            long long v = sys_atoll(env);
            if (v >= 1LL) return (unsigned long)v;
        }
        case null: { }
    }
    long c = 1L;
    unsafe {
#ifdef _WIN32
        c = (long)GetActiveProcessorCount((unsigned short)0xFFFF);   // ALL_PROCESSOR_GROUPS
#elif defined(__APPLE__)
        c = sysconf(58);                                             // _SC_NPROCESSORS_ONLN
#else
        c = sysconf(84);                                             // _SC_NPROCESSORS_ONLN
#endif
    }
    return c < 1L ? 1UL : (unsigned long)c;
}

static unsigned long par_clamp_(unsigned long n) {
    if (n < 1UL) return 1UL;
    if (n > (unsigned long)PARALLEL_MAX_THREADS) return (unsigned long)PARALLEL_MAX_THREADS;
    return n;
}

// Pool state. Everything but the chunk counter is guarded by g_par_mu.
static int g_par_inited = 0;
static unsigned long long g_par_mu = 0ULL;
static unsigned long long g_par_wake = 0ULL;   // workers wait here for a new generation
static unsigned long long g_par_done = 0ULL;   // the caller waits here for pending == 0
static unsigned long g_par_workers = 0UL;      // workers started
static unsigned long g_par_active = 0UL;       // workers taking part in the current job
static unsigned long g_par_gen = 0UL;
static unsigned long g_par_pending = 0UL;      // workers yet to report on this generation
static void* g_par_body = (void*)0;
static void* g_par_ctx = (void*)0;
static unsigned long g_par_n = 0UL;
static unsigned long g_par_chunk = 1UL;
static long long g_par_next = 0LL;             // next chunk index (atomic)
static unsigned long g_par_ids[PARALLEL_MAX_THREADS];
static unsigned long g_par_born[PARALLEL_MAX_THREADS];   // g_par_gen when each worker was started

static void par_chunks_(void* body, void* ctx, unsigned long n, unsigned long chunk) {
    while (1) {
        unsigned long c = (unsigned long)atomic_fetch_add_ll(&g_par_next, 1LL);
        unsigned long lo = c * chunk;
        if (lo >= n) return;
        unsigned long hi = n - lo < chunk ? n : lo + chunk;
        par_run_(body, ctx, lo, hi);
    }
}

static ?&unsigned long par_worker_(?&unsigned long arg) {
    unsigned long id = 0UL;
    match (arg) { case some(p): { id = *p; } case null: { } }
    mutex_lock(&g_par_mu);
    // Not g_par_gen: a job may already have been published by the time
    // this thread first gets the mutex, and the worker must take part in it.
    unsigned long seen = g_par_born[id];
    while (1) {
        while (g_par_gen == seen) cond_wait(&g_par_wake, &g_par_mu);
        seen = g_par_gen;
        int take = id < g_par_active;
        void* body = g_par_body;
        void* ctx = g_par_ctx;
        unsigned long n = g_par_n;
        unsigned long chunk = g_par_chunk;
        mutex_unlock(&g_par_mu);
        if (take) par_chunks_(body, ctx, n, chunk);
        mutex_lock(&g_par_mu);
        g_par_pending = g_par_pending - 1UL;
        if (g_par_pending == 0UL) cond_signal(&g_par_done);
    }
    return null;
}

unsigned long parallel_threads() {
    if (g_par_want == 0UL) g_par_want = par_clamp_(par_cpus_());
    return g_par_want;
}

void parallel_set_threads(unsigned long n) { g_par_want = par_clamp_(n); }

// Starts workers up to want - 1 (called with g_par_busy held). Returns how
// many are running.
static unsigned long par_start_(unsigned long want) {
    if (!g_par_inited) {
        if (mutex_init(&g_par_mu) != 0) return 0UL;
        if (cond_init(&g_par_wake) != 0 || cond_init(&g_par_done) != 0) return 0UL;
        g_par_inited = 1;
    }
    while (g_par_workers + 1UL < want) {
        unsigned long i = g_par_workers;
        g_par_ids[i] = i;
        unsigned long long tid = 0ULL;
        mutex_lock(&g_par_mu);
        g_par_born[i] = g_par_gen;
        int r = thread_create(&tid, par_worker_, (?&unsigned long)&g_par_ids[i]);
        if (r == 0) g_par_workers = g_par_workers + 1UL;
        mutex_unlock(&g_par_mu);
        if (r != 0) break;
        thread_detach(tid);
    }
    return g_par_workers;
}

void parallel_for_raw_(unsigned long n, unsigned long grain, void* body, void* ctx) {
    if (n == 0UL) return;
    if (grain < 1UL) grain = 1UL;
    unsigned long want = parallel_threads();
    if (n <= grain || want <= 1UL || atomic_exchange_int(&g_par_busy, 1) != 0) {
        par_run_(body, ctx, 0UL, n);
        return;
    }
    unsigned long workers = par_start_(want);
    unsigned long active = workers < want - 1UL ? workers : want - 1UL;
    if (active == 0UL) {
        par_run_(body, ctx, 0UL, n);
        atomic_store_int(&g_par_busy, 0);
        return;
    }
    // Aim for ~4 chunks per thread so a slow chunk doesn't hold everyone up,
    // but never below the caller's grain.
    unsigned long threads = active + 1UL;
    unsigned long chunk = (n + threads * 4UL - 1UL) / (threads * 4UL);
    if (chunk < grain) chunk = grain;

    mutex_lock(&g_par_mu);
    g_par_body = body;
    g_par_ctx = ctx;
    g_par_n = n;
    g_par_chunk = chunk;
    atomic_store_ll(&g_par_next, 0LL);
    g_par_active = active;
    g_par_pending = workers;
    g_par_gen = g_par_gen + 1UL;
    cond_broadcast(&g_par_wake);
    mutex_unlock(&g_par_mu);

    par_chunks_(body, ctx, n, chunk);

    mutex_lock(&g_par_mu);
    while (g_par_pending != 0UL) cond_wait(&g_par_done, &g_par_mu);
    mutex_unlock(&g_par_mu);
    atomic_store_int(&g_par_busy, 0);
}

#endif

} // namespace std
