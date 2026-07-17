// SafeC Standard Library — Bare-metal cooperative thread backend
//
// Implements the compiler's generic thread-backend hook contract
// (see CodeGen::selectThreadBackend / ThreadBackend::Hook in the compiler)
// using the existing cooperative TaskScheduler (std/sync/task.h) instead of
// any OS thread API. This is what 'spawn'/'join'/'spawn_scoped' compile
// down to automatically on a freestanding/bare-metal target (--freestanding),
// and it's also a legitimate choice on a hosted target for an application
// that wants pure single-threaded cooperative "SafeC threads" instead of
// real OS threads — just link this instead of relying on pthread/CreateThread
// being present.
//
// It's also the reference implementation of the same extension point a
// vendor RTOS shim or a third-party threading library would satisfy to plug
// in an entirely different thread primitive without any compiler changes:
// define '__safec_thread_create'/'__safec_thread_join' matching this exact
// signature, link that instead of this file, and every 'spawn'/'join'
// site targeting ThreadBackend::Hook transparently uses it.
//
// A spawned function is expected to follow TaskScheduler's own yield
// protocol ('int(void* arg, int resume_point)', see task.h) rather than
// run to completion in one call — there's no OS thread underneath to
// preempt it, so a function that never yields blocks the whole cooperative
// scheduler (including whichever task called __safec_thread_join on it).
#pragma once
#include <std/sync/task.sc>

// Deliberately NOT inside 'namespace std' — non-extern functions declared
// inside 'namespace std { ... }' get mangled to 'std_name' (see
// Sema::collectFunction), which would silently fail to link against the
// literal '__safec_thread_create'/'__safec_thread_join' symbol names the
// compiler emits calls to.

static struct TaskScheduler __safec_bare_sched_;
static int __safec_bare_sched_ready_ = 0;

long long __safec_thread_create(void* func, void* arg) {
    if (__safec_bare_sched_ready_ == 0) {
        __safec_bare_sched_ = std::task_sched_init();
        __safec_bare_sched_ready_ = 1;
    }
    int idx = __safec_bare_sched_.spawn_task(func, arg);
    return (long long)idx;
}

void __safec_thread_join(long long handle) {
    int idx = (int)handle;
    if (idx < 0 || idx >= __safec_bare_sched_.count) {
        return;
    }
    // Cooperative "join": keep stepping the *whole* scheduler (not just
    // the target task) until it reaches TASK_DONE — every other live task
    // gets to make progress while we wait too, which is the entire point
    // of cooperative threads sharing one real thread.
    while (__safec_bare_sched_.tasks[idx].state != 2) { // TASK_DONE
        int active = __safec_bare_sched_.tick();
        if (active == 0) {
            break; // nothing left can make progress — avoid spinning forever
        }
    }
}
