// Windows backends of std/sched's reactor and non-blocking sockets: every
// entry point that runs WSAStartup must leave its caller's stack alone.
// WSAStartup fills a whole WSADATA (408 bytes on Win64); a too-small buffer
// for it lets the call overwrite the frames above (std/async's ConnectFut
// once lost its 'self' pointer that way), so each call below runs under a
// guard-filled frame that is checked afterwards.
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
    } else if (which == 1) {
        r = tcp_listen_nb((unsigned short)0);
    } else {
        r = tcp_connect_nb(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1), g_port);
    }
    int bad = smashed_(&guard[0]);
    return bad == 0 ? r : -1000 - bad;
}

extern int getsockname(unsigned long long fd, void* addr, void* addrlen);
extern int closesocket(unsigned long long fd);

struct SockName_ {
    unsigned short family;
    unsigned short port;
    unsigned int   addr;
    unsigned long long zero;
};

void t_win32_wsastartup_frame() {
    ASSERT_EQ(guarded_(0), 0);                 // Reactor::init
    g_listen = guarded_(1);                    // tcp_listen_nb
    ASSERT_TRUE(g_listen >= 0);
    struct SockName_ sa;
    sa.family = (unsigned short)0; sa.port = (unsigned short)0; sa.addr = 0U; sa.zero = 0ULL;
    int len = 16;
    int rc = 0;
    unsafe { rc = getsockname((unsigned long long)g_listen, (void*)&sa, (void*)&len); }
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

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe { t.add("win32 WSAStartup keeps caller frames intact", t_win32_wsastartup_frame); }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
