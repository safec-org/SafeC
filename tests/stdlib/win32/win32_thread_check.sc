// Windows backends of thread.h and time.h: the DWORD-returning Win32 calls
// they wrap (GetCurrentThreadId, GetLastError, ...) and their timeouts.
#include <std/test/test.h>
#include <std/thread.h>
#include <std/time.h>
#include <std/sys.h>

static unsigned long long g_m = 0ULL;
static unsigned long long g_cv = 0ULL;
static int g_flag = 0;
static unsigned long long g_child_id = 0ULL;

static ?&int child(?&int arg) {
    unsafe {
        g_child_id = thread_self();
        mutex_lock(&g_m);
        g_flag = 1;
        cond_signal(&g_cv);
        mutex_unlock(&g_m);
    }
    return null;
}

void t_win32_thread_time() {
    unsafe {
        unsigned long long me = thread_self();
        ASSERT_TRUE(me != 0ULL);
        ASSERT_TRUE(me < 0x100000000ULL);             // a thread id is a 32-bit DWORD
        ASSERT_EQ(mutex_init(&g_m), 0);
        ASSERT_EQ(cond_init(&g_cv), 0);
        unsigned long long tid = 0ULL;
        ASSERT_EQ(thread_create(&tid, child, null), 0);
        mutex_lock(&g_m);
        while (g_flag == 0) { cond_wait(&g_cv, &g_m); }
        ASSERT_EQ(mutex_trylock(&g_m), 0);          // a CRITICAL_SECTION is recursive: the owner may re-enter
        mutex_unlock(&g_m);
        mutex_unlock(&g_m);
        thread_join(tid);
        ASSERT_TRUE(g_child_id != me && g_child_id < 0x100000000ULL);
        mutex_lock(&g_m);
        ASSERT_EQ(cond_timedwait_ms(&g_cv, &g_m, 10UL), THREAD_TIMEDOUT);
        mutex_unlock(&g_m);

        long long t0 = time_mono_ns();
        time_sleep_ns(20000000LL);
        long long dt = time_mono_ns() - t0;
        ASSERT_TRUE(dt >= 15000000LL && dt < 2000000000LL);
        ASSERT_TRUE(time_wall_ns() / 1000000000LL > 1700000000LL);
        ASSERT_TRUE(time_cpu_ns() >= 0LL);
        ASSERT_TRUE(sys_monotonic_ns() > 0LL);
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe { t.add("win32 thread/time backends", t_win32_thread_time); }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
