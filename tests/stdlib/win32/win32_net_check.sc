// Windows backends of std/sched's reactor and non-blocking sockets.
//  * Every entry point that runs WSAStartup must leave its caller's stack
//    alone. WSAStartup fills a whole WSADATA (408 bytes on Win64); a
//    too-small buffer for it lets the call overwrite the frames above
//    (std/async's ConnectFut once lost its 'self' pointer that way), so
//    each call below runs under a guard-filled frame checked afterwards.
//  * A poll_events callback may remove registrations while several fds
//    are ready at once (std/async's IoDriver does exactly that): the
//    WSAPoll backend must still report each ready (fd, filter) once.
#include <std/test/test.h>
#include <std/sched/reactor.h>
#include <std/sched/io_nb.h>

#define GUARD_LEN 1024UL
#define GUARD_BYTE (unsigned char)0x5A

static int g_listen = -1;
static unsigned short g_port = (unsigned short)0;

// Bytes of 'g' that no longer hold GUARD_BYTE.
static int smashed_(const &unsigned char g) {
    int n = 0;
    unsafe {
        for (unsigned long i = 0UL; i < GUARD_LEN; i = i + 1UL) {
            if (g[i] != GUARD_BYTE) { n = n + 1; }
        }
    }
    return n;
}

// Runs one Winsock-initializing call with a guard area right above its
// frame; the call's result, or -1000 - (smashed bytes) when it wrote into
// the guard.
static int guarded_(int which) {
    unsigned char guard[1024];
    unsafe {
        for (unsigned long i = 0UL; i < GUARD_LEN; i = i + 1UL) { guard[i] = GUARD_BYTE; }
    }
    int r = 0;
    if (which == 0) {
        struct Reactor re = Reactor::new();
        r = re.init();
        re.close_();
        drop(re);   // its (empty) registration tables
    } else if (which == 1) {
        r = tcp_listen_nb((unsigned short)0);
    } else {
        r = tcp_connect_nb(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1), g_port);
    }
    int bad = smashed_(&guard[0]);
    return bad == 0 ? r : -1000 - bad;
}

extern int closesocket(unsigned long long fd);

struct SockName_ {
    unsigned short family;
    unsigned short port;
    unsigned int   addr;
    unsigned long long zero;
};
extern int getsockname(unsigned long long fd, SockName_* addr, int* addrlen);

void t_win32_wsastartup_frame() {
    ASSERT_EQ(guarded_(0), 0);                 // Reactor::init
    g_listen = guarded_(1);                    // tcp_listen_nb
    ASSERT_TRUE(g_listen >= 0);
    struct SockName_ sa;
    sa.family = (unsigned short)0; sa.port = (unsigned short)0; sa.addr = 0U; sa.zero = 0ULL;
    int len = 16;
    int rc = 0;
    unsafe { rc = getsockname((unsigned long long)g_listen, &sa, &len); }
    ASSERT_EQ(rc, 0);
    g_port = sched_htons(sa.port);             // its own inverse
    int c = guarded_(2);                       // tcp_connect_nb
    ASSERT_TRUE(c >= 0);
    ASSERT_EQ(tcp_wait_connected(c), 0);
    unsafe {
        closesocket((unsigned long long)c);
        closesocket((unsigned long long)g_listen);
    }
}

// poll_events callback: records the event and unregisters it, the way
// std/async's IoDriver::on_ready_ does once nobody waits on it any more.
static struct Reactor g_re;
static int g_seen = 0;
static int g_bad = 0;
static int g_fds[2];

static void on_ready_remove_(?&int ctx, int fd, int filter) {
    g_seen = g_seen + 1;
    if (filter != SCHED_WRITE || (fd != g_fds[0] && fd != g_fds[1])) { g_bad = g_bad + 1; }
    g_re.remove(fd, filter);
}

void t_win32_poll_remove_in_callback() {
    g_re = Reactor::new();
    ASSERT_EQ(g_re.init(), 0);
    int l = tcp_listen_nb((unsigned short)0);
    ASSERT_TRUE(l >= 0);
    struct SockName_ sa;
    sa.family = (unsigned short)0; sa.port = (unsigned short)0; sa.addr = 0U; sa.zero = 0ULL;
    int len = 16;
    unsafe { getsockname((unsigned long long)l, &sa, &len); }
    unsigned int lo = sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1);
    // Two connected sockets: both writable, so one WSAPoll reports both.
    g_fds[0] = tcp_connect_nb(lo, sched_htons(sa.port));
    g_fds[1] = tcp_connect_nb(lo, sched_htons(sa.port));
    ASSERT_TRUE(g_fds[0] >= 0 && g_fds[1] >= 0);
    ASSERT_EQ(tcp_wait_connected(g_fds[0]), 0);
    ASSERT_EQ(tcp_wait_connected(g_fds[1]), 0);
    g_re.add(l, SCHED_READ);              // first in the table, ready too (pending accepts)
    g_re.add(g_fds[0], SCHED_WRITE);
    g_re.add(g_fds[1], SCHED_WRITE);
    g_re.remove(l, SCHED_READ);           // keep only the two writable sockets
    g_seen = 0;
    g_bad = 0;
    int n = g_re.poll_events(on_ready_remove_, (?&int)null, 1000LL);
    ASSERT_EQ(n, 2);
    ASSERT_EQ(g_seen, 2);
    ASSERT_EQ(g_bad, 0);
    ASSERT_EQ(g_re.poll_events(on_ready_remove_, (?&int)null, 0LL), 0);   // both unregistered
    unsafe {
        closesocket((unsigned long long)g_fds[0]);
        closesocket((unsigned long long)g_fds[1]);
        closesocket((unsigned long long)l);
    }
    g_re.close_();
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe { t.add("win32 WSAStartup keeps caller frames intact", t_win32_wsastartup_frame);
             t.add("win32 reactor: callback removes fds while several are ready", t_win32_poll_remove_in_callback); }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
