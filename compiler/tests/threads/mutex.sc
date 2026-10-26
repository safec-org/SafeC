// SAFEC-ARGS: -I %root/std -I %root --compat-preprocessor --target x86_64-unknown-linux-gnu
// (Pinned to a Linux target: the CHECKs name pthread_create; std/thread uses
// Win32 threads on Windows.)
// Shared counters: a static Mutex, and atomics lent to scoped threads.
#include <std/thread.h>
#include <std/atomic.h>
static struct Mutex<int> g_m;

?&int bump(?&int unused) {
    struct MutexGuard<int> g = g_m.lock();
    &int v = g.get();
    *v = *v + 1;
    g.drop();
    return null;
}

?&AtomicInt hit(?&AtomicInt a) { match (a) { case some(x): x.fetch_add(1); case null: {} } return null; }

// CHECK-LABEL: define {{.*}}@counters(
// CHECK: call {{.*}}pthread_create
int counters() {
    g_m = Mutex::new(0);
    long long h1 = spawn(bump, null);
    join(h1);
    struct AtomicInt local = AtomicInt::new(0);
    long long h2 = spawn_scoped(hit, &local);
    local.fetch_add(1);                   // a Sync type stays usable while lent
    join(h2);
    struct MutexGuard<int> g = g_m.lock();
    int n = *g.get();
    g.drop();
    return n + local.load();
}

int main() { return counters(); }
