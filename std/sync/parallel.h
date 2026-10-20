#pragma once
// SafeC Standard Library — data-parallel loops on a persistent thread pool.
//
// parallel_for(n, grain, body, ctx) splits [0, n) into chunks of at least
// 'grain' indices and runs body(ctx, lo, hi) for each, on the calling
// thread plus the pool's workers, returning when every chunk is done.
// Chunks are handed out dynamically, so uneven work balances out.
//
// The workers are started on first use and then kept, parked on a condition
// variable between calls, so a call costs a wake-up (a few microseconds),
// not a thread spawn. Pick 'grain' so one chunk is worth at least ~10-50us
// of work; with n <= grain the body simply runs inline.
//
// Runs serially (body(ctx, 0, n) on the caller) when:
//   - the pool has one thread (SAFEC_NUM_THREADS=1, parallel_set_threads(1),
//     or a single-core machine),
//   - another parallel_for is already running (a nested call from inside a
//     body, or a concurrent call from another thread),
//   - the target has no threads (freestanding / bare-metal, WebAssembly).
//
// Not for real-time code: a call blocks until the slowest chunk finishes
// and wakes threads through the OS scheduler. Audio callbacks and other
// hard-deadline paths should stay single-threaded.
//
// Thread count: SAFEC_NUM_THREADS if set, else the number of online CPUs
// (at most PARALLEL_MAX_THREADS), including the calling thread.

namespace std {

#define PARALLEL_MAX_THREADS 64

// Runtime entry point (bodies are called through an untyped pointer);
// call parallel_for below instead.
void parallel_for_raw_(unsigned long n, unsigned long grain, void* body, void* ctx);

// struct Job { const &float x; &float y; };
// static void work(&Job j, unsigned long lo, unsigned long hi) { ... }
// std::parallel_for(n, 4096UL, work, &job);
generic<T>
void parallel_for(unsigned long n, unsigned long grain,
                  fn void(&T, unsigned long, unsigned long) body, &T ctx) {
    unsafe { parallel_for_raw_(n, grain, (void*)body, (void*)ctx); }
}

// Threads a parallel_for uses, counting the caller (>= 1).
unsigned long parallel_threads();

// Sets the thread count (clamped to 1 .. PARALLEL_MAX_THREADS). Lowering it
// takes effect on the next call; raising it starts more workers lazily.
void parallel_set_threads(unsigned long n);

} // namespace std
