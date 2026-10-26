// SAFEC-ARGS: -I @STD@/std -I @STD@ --compat-preprocessor
// Mutex<T>: the value is reachable only through a guard, and a reference
// from the guard can't outlive it. thread_spawn moves its argument.
#include <std/thread.h>
#include <std/collections/vec.h>
static struct Mutex<int> g_m;

int after_unlock() {
    struct MutexGuard<int> g = g_m.lock();
    &int v = g.get();
    g.drop();
    return *v;                            // EXPECT-ERROR: use of a reference borrowed from 'g' after 'g' was dropped
}

int forgot_unlock() {
    struct MutexGuard<int> g = g_m.lock();
    return *g.get();                      // EXPECT-ERROR: value of Drop type 'MutexGuard<int>' leaks when the function returns
}

int mutex_gone() {
    struct Mutex<int> m = Mutex::new(1);
    struct MutexGuard<int> g = m.lock();
    m.drop();
    int x = *g.get();                     // EXPECT-ERROR: use of 'g', which borrows from 'm', after 'm' was dropped
    // EXPECT-ERROR: reads a &heap reference after std::dealloc() freed it
    g.drop();
    return x;
}

int vlen(struct Vec<int> v) { int n = (int)v.len; v.drop(); return n; }
int moved_to_thread() {
    struct Vec<int> v = Vec::new();
    struct SpawnedThread<struct Vec<int>, int> t = thread_spawn(vlen, v);
    v.push(1);                            // EXPECT-ERROR: use of 'v' after it was moved
    return t.join();
}

struct Raw { int *p; };
int raw_len(struct Raw r) { return 0; }
int not_send() {
    struct Raw r; r.p = (int*)0;
    struct SpawnedThread<struct Raw, int> t = thread_spawn(raw_len, r);   // EXPECT-ERROR: does not satisfy constraint 'Send'
    return t.join();
}

int main() { return 0; }
