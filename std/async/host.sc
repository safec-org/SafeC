// SafeC Standard Library — async: HostPark
#pragma once
#include <std/async/host.h>
#include <std/time.h>

namespace std {

long long async_mono_ms() {
    return time_mono_ns() / 1000000LL;
}

struct HostPark HostPark::new() {
    struct HostPark p;
    p.mu = 0ULL;
    p.cv = 0ULL;
    p.permit = 0;
    p.ok = 0;
    if (mutex_init(&p.mu) == 0 && cond_init(&p.cv) == 0) p.ok = 1;
    return p;
}

void HostPark::destroy() { self.drop(); }

void HostPark::drop() {
    if (self.ok == 0) return;
    cond_destroy(&self.cv);
    mutex_destroy(&self.mu);
    self.ok = 0;
}

long long HostPark::now_ms() { return async_mono_ms(); }

void HostPark::park(long long timeout_ms) {
    if (self.ok == 0) {
        if (timeout_ms > 0LL) thread_sleep_ms((unsigned long)timeout_ms);
        return;
    }
    mutex_lock(&self.mu);
    if (self.permit == 0 && timeout_ms != 0LL) {
        if (timeout_ms < 0LL) cond_wait(&self.cv, &self.mu);
        else cond_timedwait_ms(&self.cv, &self.mu, (unsigned long)timeout_ms);
    }
    self.permit = 0;
    mutex_unlock(&self.mu);
}

void HostPark::unpark() {
    if (self.ok == 0) return;
    mutex_lock(&self.mu);
    self.permit = 1;
    cond_signal(&self.cv);
    mutex_unlock(&self.mu);
}

} // namespace std
