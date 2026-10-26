// std/async demo: one single-threaded Runtime (a LocalExecutor parked in
// the OS readiness queue) runs
//   * an HTTP/1.0 server: an accept loop that spawns one handler task per
//     connection, stopped by a CancelToken;
//   * CLIENTS clients that connect, GET a page and check the reply, each
//     under a timeout;
//   * a stats task that reports every 50 ms until the token is cancelled;
//   * a task awaiting a result computed on a ThreadPool (another OS
//     thread), delivered through a Oneshot -- a wake from another thread.
// Every task is a stackless coroutine: its locals live in its struct, and
// ASYNC_BEGIN/AWAIT turn poll() into a state machine over goto labels.
//
//   safeguard run            (exit status 0 when every client got its page)
#include <std/io.h>
#include <std/async/async.h>
#include <std/async/net.h>
#include <std/async/pool.h>

#define CLIENTS  8
#define HANDLERS 8

extern int printf(const char* fmt, ...);

static struct CancelToken g_stop;
static int g_served = 0;

// Copies a C string into 'dst'; its length.
static unsigned long put_str(&unsigned char dst, unsigned long cap, const char* s) {
    unsigned long n = 0UL;
    unsafe {
        while (n < cap && s[n] != (char)0) { dst[n] = (unsigned char)s[n]; n = n + 1UL; }
    }
    return n;
}

static int ends_header(const &unsigned char b, unsigned long n) {
    if (n < 4UL) return 0;
    unsafe {
        return (b[n - 4UL] == (unsigned char)13 && b[n - 3UL] == (unsigned char)10 &&
                b[n - 2UL] == (unsigned char)13 && b[n - 1UL] == (unsigned char)10) ? 1 : 0;
    }
}

// ── server ─────────────────────────────────────────────────────────────────

// One connection: read the request head (1 s timeout), answer, close.
struct Handler {
    int                   st;
    int                   id;
    struct AsyncTcpStream c;
    struct Timeout<struct ReadFut, long long> rd;
    ?long long            n;
    unsigned long         have;
    struct WriteAllFut    wr;
    int                   ok;
    unsigned char         buf[512];
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> Handler::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    self.have = 0UL;
    while (ends_header(&self.buf[0], self.have) == 0 && self.have < 512UL) {
        self.rd = timeout(self.c.read(self.buf[self.have..512]), 1000LL);
        AWAIT_INTO(1, self.n, self.rd);
        if (self.n.default(0LL) <= 0LL) { std::replace(&self.c, AsyncTcpStream::from_fd(-1)).close(); ASYNC_RETURN(0); }   // timed out, EOF or error
        self.have = self.have + (unsigned long)self.n.default(0LL);
    }
    {
        // The reply (built in the buffer: the request is no longer needed).
        unsigned long k = put_str(&self.buf[0], 512UL,
            "HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\nhello from std/async, connection #");
        self.buf[k] = (unsigned char)(48 + self.id / 10 % 10);
        self.buf[k + 1UL] = (unsigned char)(48 + self.id % 10);
        self.buf[k + 2UL] = (unsigned char)10;
        self.have = k + 3UL;
    }
    self.wr = self.c.write_all(self.buf[0..self.have]);
    AWAIT_INTO(2, self.ok, self.wr);
    std::replace(&self.c, AsyncTcpStream::from_fd(-1)).close();   // drop() closes nothing again
    g_served = g_served + 1;
    ASYNC_RETURN(self.ok);
    ASYNC_END;
}

void Handler::drop() {
    if (self.st == 1) self.rd.drop();
    if (self.st == 2) self.wr.drop();
    self.c.close();
    self.st = ASYNC_DONE;
}

// Accepts until g_stop is cancelled: each round is a select between the
// next connection and the cancellation (the loser is dropped).
struct Server {
    int                     st;
    struct AsyncTcpListener l;
    struct Select2<struct AcceptFut, struct Cancelled, struct AsyncTcpStream, Unit> next;
    struct Either<struct AsyncTcpStream, Unit> ev;
    int                     accepted;
    int                     slot;
    struct AsyncTask<struct Handler, int> handlers[HANDLERS];
    int                     busy[HANDLERS];
    struct YieldNow         y;
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> Server::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    self.accepted = 0;
    while (1) {
        self.next = select2(self.l.accept(), g_stop.cancelled_fut());
        AWAIT_INTO(1, self.ev, self.next);
        if (self.ev.which == 2) break;                 // cancelled
        if (self.ev.left.fd < 0) continue;
        // A free handler slot (all busy: wait for one to finish).
        self.slot = -1;
        while (self.slot < 0) {
            for (int i = 0; i < HANDLERS; i = i + 1) {
                if (self.slot < 0 && (self.busy[i] == 0 || self.handlers[i].is_finished() != 0)) self.slot = i;
            }
            if (self.slot < 0) {
                self.y = YieldNow::new();
                AWAIT(2, self.y);
            }
        }
        {
            struct Handler h;
            h.st = ASYNC_INIT;
            h.id = self.accepted + 1;
            h.c = std::replace(&self.ev.left, AsyncTcpStream::from_fd(-1));   // the connection moves to its handler
            struct AsyncTask<struct Handler, int> t = AsyncTask::new(h);
            if (self.busy[self.slot] != 0) std::replace(&self.handlers[self.slot], t).drop();   // the finished one it replaces
            else self.handlers[self.slot] = t;
            self.busy[self.slot] = 1;
            drop(async_spawn(cx, &self.handlers[self.slot]));                  // detached: busy[] tracks it
        }
        self.accepted = self.accepted + 1;
    }
    unsafe { self.l.close(); }   // early, in place: drop() closes nothing again
    ASYNC_RETURN(self.accepted);
    ASYNC_END;
}

void Server::drop() {
    if (self.st == 1) self.next.drop();
    if (self.st == 2) self.y.drop();
    self.ev.drop();   // which == 0 until the first connection
    for (int i = 0; i < HANDLERS; i = i + 1) {
        if (self.busy[i] != 0) { self.handlers[i].drop(); self.busy[i] = 0; }   // finished by now
    }
    self.l.close();
    self.st = ASYNC_DONE;
}

// ── clients ────────────────────────────────────────────────────────────────

struct Client {
    int                   st;
    int                   id;
    unsigned short        port;
    struct Sleep          delay;
    struct ConnectFut     cn;
    struct AsyncTcpStream s;
    struct WriteAllFut    wr;
    struct ReadFut        rd;
    int                   ok;
    long long             n;
    unsigned long         have;
    unsigned char         buf[512];
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> Client::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 4);
    self.delay = Sleep::new((long long)(self.id * 7));         // arrive staggered
    AWAIT(1, self.delay);
    self.cn = AsyncTcpStream::connect(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1), self.port);
    AWAIT_INTO(2, self.s, self.cn);
    if (self.s.fd < 0) ASYNC_RETURN(0);
    self.have = put_str(&self.buf[0], 512UL, "GET / HTTP/1.0\r\nHost: localhost\r\n\r\n");
    self.wr = self.s.write_all(self.buf[0..self.have]);
    AWAIT_INTO(3, self.ok, self.wr);
    self.have = 0UL;
    while (self.have < 511UL) {
        self.rd = self.s.read(self.buf[self.have..511]);
        AWAIT_INTO(4, self.n, self.rd);
        if (self.n <= 0LL) break;                               // HTTP/1.0: closed at the end
        self.have = self.have + (unsigned long)self.n;
    }
    std::replace(&self.s, AsyncTcpStream::from_fd(-1)).close();   // drop() closes nothing again
    // "HTTP/1.0 200"
    if (self.have < 12UL || self.buf[9] != (unsigned char)50 || self.buf[10] != (unsigned char)48 ||
        self.buf[11] != (unsigned char)48) ASYNC_RETURN(0);
    ASYNC_RETURN((int)self.have);
    ASYNC_END;
}

void Client::drop() {
    if (self.st == 1) self.delay.drop();
    if (self.st == 2) self.cn.drop();
    if (self.st == 3) self.wr.drop();
    if (self.st == 4) self.rd.drop();
    self.s.close();
    self.st = ASYNC_DONE;
}

// ── stats ──────────────────────────────────────────────────────────────────

struct Stats {
    int st;
    int ticks;
    struct Select2<struct Sleep, struct Cancelled, Unit, Unit> next;
    struct Either<Unit, Unit> ev;
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> Stats::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    self.ticks = 0;
    while (1) {
        self.next = select2(Sleep::new(50LL), g_stop.cancelled_fut());
        AWAIT_INTO(1, self.ev, self.next);
        if (self.ev.which == 2) break;
        self.ticks = self.ticks + 1;
        printf("  [stats] t+%d ms: %d requests served\n", self.ticks * 50, g_served);
    }
    ASYNC_RETURN(self.ticks);
    ASYNC_END;
}

void Stats::drop() { if (self.st == 1) self.next.drop(); self.st = ASYNC_DONE; }

// ── a CPU job on a ThreadPool, awaited from the Runtime ────────────────────

static struct Oneshot<int> g_primes;

// Counts primes below 'limit' in slices of 2000 numbers, yielding between
// slices (a long computation shouldn't hog a worker), then sends the count.
struct PrimeJob {
    int             st;
    int             limit;
    int             at;
    int             count;
    struct YieldNow y;
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> PrimeJob::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 1);
    self.count = 0;
    for (self.at = 2; self.at < self.limit; self.at = self.at + 2000) {
        for (int n = self.at; n < self.at + 2000 && n < self.limit; n = n + 1) {
            int prime = 1;
            for (int d = 2; d * d <= n && prime != 0; d = d + 1) { if (n % d == 0) prime = 0; }
            self.count = self.count + prime;
        }
        self.y = YieldNow::new();
        AWAIT(1, self.y);
    }
    g_primes.send(self.count);
    ASYNC_RETURN(self.count);
    ASYNC_END;
}

void PrimeJob::drop() { if (self.st == 1) self.y.drop(); self.st = ASYNC_DONE; }

// ── main: wait for every client, then stop the server and the stats ────────

struct Main {
    int                     st;
    struct JoinHandle<int>  clients[CLIENTS];
    ?int                    results[CLIENTS];
    struct JoinAll<struct JoinHandle<int>, ?int> all;
    int                     n;
    struct Select2<struct OneshotRecv<int>, struct Sleep, ?int, Unit> primes;
    struct Either<?int, Unit> got;
    int                     ok;
    struct Poll<int> poll(&Context cx);
    void drop();
};

struct Poll<int> Main::poll(&Context cx) {
    ASYNC_BEGIN(self.st, 2);
    // Every client, concurrently (their JoinHandles are futures too).
    self.all = join_all(self.clients[0..CLIENTS], self.results[0..CLIENTS]);
    AWAIT_INTO(1, self.n, self.all);
    self.ok = 0;
    for (int i = 0; i < CLIENTS; i = i + 1) {
        int bytes = self.results[i].default(0);
        printf("  client %d: %d bytes\n", i + 1, bytes);
        if (bytes > 0) self.ok = self.ok + 1;
    }
    // The pool's result, or give up after 5 s.
    self.primes = select2(g_primes.recv(), Sleep::new(5000LL));
    AWAIT_INTO(2, self.got, self.primes);
    if (self.got.which == 1) printf("  thread pool: %d primes below 200000\n", self.got.left.default(-1));
    else printf("  thread pool: no answer in time\n");
    g_stop.cancel();                                   // server and stats wind down
    ASYNC_RETURN(self.ok);
    ASYNC_END;
}

void Main::drop() {
    if (self.st == 1) self.all.drop();
    if (self.st == 2) self.primes.drop();
    for (int i = 0; i < CLIENTS; i = i + 1) self.clients[i].drop();   // idempotent
    self.st = ASYNC_DONE;
}

static struct Runtime rt;
static struct ThreadPool pool;
static struct AsyncTask<struct Server, int> server_task;
static struct AsyncTask<struct Stats, int> stats_task;
static struct AsyncTask<struct Client, int> client_tasks[CLIENTS];
static struct AsyncTask<struct PrimeJob, int> prime_task;

int main() {
    rt = Runtime::new();
    if (rt.init() != 0) { printf("async_demo: no I/O driver\n"); return 1; }
    g_stop = CancelToken::new();
    g_primes = Oneshot::new();

    struct AsyncTcpListener l = AsyncTcpListener::bind((unsigned short)0);
    if (l.fd < 0) { printf("async_demo: cannot listen\n"); l.close(); return 1; }
    struct Server srv;
    srv.st = ASYNC_INIT;
    srv.l = l;
    srv.ev.which = 0;
    for (int i = 0; i < HANDLERS; i = i + 1) srv.busy[i] = 0;
    unsigned short port = srv.l.local_port();
    printf("async_demo: serving on 127.0.0.1:%d\n", (int)port);
    server_task = AsyncTask::new(srv);
    drop(rt.spawn(&server_task));

    struct Stats st;
    st.st = ASYNC_INIT;
    stats_task = AsyncTask::new(st);
    drop(rt.spawn(&stats_task));

    pool = ThreadPool::new();
    pool.start(2);
    struct PrimeJob pj;
    pj.st = ASYNC_INIT;
    pj.limit = 200000;
    prime_task = AsyncTask::new(pj);
    drop(pool.spawn(&prime_task));

    struct Main m;
    m.st = ASYNC_INIT;
    for (int i = 0; i < CLIENTS; i = i + 1) {
        struct Client c;
        c.st = ASYNC_INIT;
        c.id = i + 1;
        c.port = port;
        client_tasks[i] = AsyncTask::new(c);
        m.clients[i] = rt.spawn(&client_tasks[i]);
    }

    int ok = rt.block_on(&m);   // the clients, then the pool's answer
    m.drop();
    rt.run();                   // the server and the stats notice the cancellation
    pool.shutdown();
    printf("async_demo: %d/%d clients ok, %d requests served, server accepted %d\n",
           ok, CLIENTS, g_served, server_task.take().default(-1));
    rt.close_();
    return ok == CLIENTS ? 0 : 1;
}
