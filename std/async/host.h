// SafeC Standard Library — async: hosted parking and block_on
//
// HostPark is the Park source (std/async/executor.h) for a hosted program
// with no I/O to wait for: the OS monotonic clock, and a condition
// variable to sleep on until the next timer deadline or a wake from
// another thread. block_on(&fut) runs a future to completion on the
// calling thread with a fresh LocalExecutor and a HostPark — the usual way
// into async code from main() (Rust's futures::executor::block_on):
//
//   struct Fetch f = Fetch::new();
//   int n = block_on(&f);
//
// For async I/O use std/async/net.h's Runtime instead (its IoDriver parks
// in the OS readiness queue); for parallelism std/async/pool.h's
// ThreadPool. Hosted targets only (it uses std/thread.h and std/time.h).
#pragma once
#include <std/async/executor.h>
#include <std/thread.h>

namespace std {

struct HostPark {
    unsigned long long mu;       // std/thread.h mutex
    unsigned long long cv;       // and condition variable
    int                permit;   // an unpark() that no park() has consumed yet
    int                ok;       // mu/cv were created

    // Creates the mutex and condition variable; destroy() frees them.
    static struct HostPark new();
    void destroy();
    long long now_ms();
    void park(long long timeout_ms);
    void unpark();
};

// Milliseconds of the OS monotonic clock.
long long async_mono_ms();

// Run 'fut' to completion on this thread and return its value. Tasks it
// spawns (async_spawn) run meanwhile, on the same thread; any still
// running when 'fut' finishes are abandoned with the executor.
generic<F: Future, T>
T block_on(&F fut) {
    struct LocalExecutor ex = LocalExecutor::new();
    struct HostPark park = HostPark::new();
    T r = ex.block_on(&park, fut);
    park.destroy();
    return r;
}

} // namespace std
