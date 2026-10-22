// SafeC Standard Library — async: I/O driver, Runtime, async TCP
//
// Futures that wait for sockets. They make the non-blocking call at once
// (std/sched/io_nb.h's sockets); when it would block, they register the
// task's waker with the IoDriver for that fd's readiness and are Pending.
// The IoDriver is a std::Reactor (std/sched/reactor.h: epoll on Linux,
// kqueue on macOS/BSD, WSAPoll on Windows) driven through
// Reactor::poll_events, and it is also the executor's Park source: when no
// task is runnable, the executor parks in the OS readiness queue until a
// socket is ready, the next timer is due, or another thread wakes a task.
//
// Runtime bundles a LocalExecutor and an IoDriver (tokio's current-thread
// runtime):
//
//   static struct Runtime rt;           // must not move once initialized
//   rt = Runtime::new();
//   if (rt.init() != 0) { ... }          // opens the OS event queue
//   rt.spawn(&server_task);
//   int r = rt.block_on(&client);        // or rt.run(): until every task is done
//   rt.close_();
//
// Inside tasks:
//
//   self.l = AsyncTcpListener::bind(8080);          // .fd < 0 on failure
//   self.acc = self.l.accept();
//   AWAIT_INTO(1, self.conn, self.acc);             // struct AsyncTcpStream; .fd < 0 on error
//   self.rd = self.conn.read(self.buf[0..512]);
//   AWAIT_INTO(2, self.n, self.rd);                 // bytes read, 0 at EOF, -1 on error
//   self.wr = self.conn.write_all(self.buf[0..self.n]);
//   AWAIT_INTO(3, self.ok, self.wr);                // 1 when all written
//   self.conn.close();
//
//   self.cn = AsyncTcpStream::connect(sched_ipv4(127, 0, 0, 1), 8080);
//   AWAIT_INTO(4, self.conn, self.cn);
//
// Limits: ASYNC_IO_WAITERS concurrent waits per driver (a wait that finds
// the table full re-polls instead); on Linux one fd is awaited for one
// direction at a time (epoll registers an fd once — a read and a write
// waiting on the same socket at the same time make the second one
// re-poll). Wakes from other threads end a park at once on POSIX (a wake
// pipe); on Windows the driver parks at most ASYNC_IO_REMOTE_MS at a time
// so they are seen within that. Hosted targets only.
#pragma once
#include <std/async/executor.h>
#include <std/async/host.h>
#include <std/sched/reactor.h>
#include <std/sched/io_nb.h>

#define ASYNC_IO_WAITERS 256
#define ASYNC_IO_REMOTE_MS 20

namespace std {

struct IoWaiter {
    int          fd;
    int          filter;     // SCHED_READ / SCHED_WRITE
    int          serial;     // 0 = free
    int          armed;      // this entry did Reactor::add(fd, filter)
    struct Waker w;
};

struct IoDriver {
    struct Reactor  reactor;
    int             ok;           // reactor opened
    struct IoWaiter waiters[ASYNC_IO_WAITERS];
    int             nwaiting;
    int             next_serial;
    int             wake_rd;      // wake pipe (POSIX), -1 if none
    int             wake_wr;
    int             wake_pending; // atomic: a byte is in the pipe (or an unpark is pending)
    long long       events;       // statistics: readiness events seen

    static struct IoDriver new();
    // Opens the OS event queue (and the wake pipe). 0 on success.
    int  init();
    void close_();

    // Wake 'w' once 'fd' is ready for 'filter'. 'id' is a previous
    // registration of the same future (refreshed), or 0. Returns the
    // registration id, 0 if the table is full (re-poll instead).
    int  wait(int fd, int filter, const &Waker w, int id);
    void cancel(int id);
    int  waiting() const;

    // Park (std/async/executor.h): the OS clock, and waiting in the reactor.
    long long now_ms();
    void park(long long timeout_ms);
    void unpark();

    // Internal: Reactor::poll_events's callback.
    void on_ready_(int fd, int filter);
};

// The IoDriver of the executor polling 'cx', or null outside a Runtime.
?&IoDriver io_driver(&Context cx);

// A LocalExecutor parked in an IoDriver.
struct Runtime {
    struct LocalExecutor ex;
    struct IoDriver      io;

    static struct Runtime new();
    // Open the I/O driver and attach it (call once the Runtime is in place).
    int  init();
    void close_();
    generic<F, T> struct JoinHandle<T> spawn(&AsyncTask<F, T> task);
    // Until every spawned task has finished.
    void run();
    generic<F: Future, T> T block_on(&F fut);
};

// ── TCP ────────────────────────────────────────────────────────────────────

struct AsyncTcpStream {
    int fd;      // < 0: not connected (a failed connect/accept)

    static struct AsyncTcpStream from_fd(int fd);
    static struct ConnectFut connect(unsigned int addr_network_order, unsigned short port);
    // Up to buf.len bytes: the count, 0 once the peer closed, -1 on error.
    struct ReadFut read([]unsigned char buf);
    // Some of buf: the count written, -1 on error.
    struct WriteFut write([]unsigned char buf);
    // All of buf: 1 when written, 0 on error.
    struct WriteAllFut write_all([]unsigned char buf);
    int close();
    int is_open() const;
};

struct AsyncTcpListener {
    int fd;      // < 0: bind/listen failed

    // Listen on 'port' (0: any free port; see local_port) on every interface.
    static struct AsyncTcpListener bind(unsigned short port);
    // Ready with the next connection (its .fd < 0 on an error).
    struct AcceptFut accept();
    // The port it listens on.
    unsigned short local_port() const;
    int close();
};

// Ready once 'fd' is ready for 'filter' (SCHED_READ / SCHED_WRITE), for
// sockets and pipes these types don't cover.
struct IoReady {
    int fd;
    int filter;
    int wait_id;
    int done;
    ?&IoDriver drv;
    static struct IoReady new(int fd, int filter);
    struct Poll<Unit> poll(&Context cx);
    void drop();
};

struct AcceptFut {
    int lfd;
    int wait_id;
    ?&IoDriver drv;
    struct Poll<struct AsyncTcpStream> poll(&Context cx);
    void drop();
};

struct ConnectFut {
    unsigned int   addr;
    unsigned short port;
    int            fd;        // -1 until started
    int            wait_id;
    int            done;
    ?&IoDriver     drv;
    struct Poll<struct AsyncTcpStream> poll(&Context cx);
    // Dropping an unfinished connect closes its socket.
    void drop();
};

struct ReadFut {
    int               fd;
    []unsigned char   buf;
    int               wait_id;
    ?&IoDriver        drv;
    struct Poll<long long> poll(&Context cx);
    void drop();
};

struct WriteFut {
    int               fd;
    []unsigned char   buf;
    int               wait_id;
    ?&IoDriver        drv;
    struct Poll<long long> poll(&Context cx);
    void drop();
};

struct WriteAllFut {
    int               fd;
    []unsigned char   buf;
    unsigned long     sent;
    int               wait_id;
    ?&IoDriver        drv;
    struct Poll<int> poll(&Context cx);
    void drop();
};

// The local port of a bound socket (0 on error).
unsigned short async_sock_local_port(int fd);
// Socket read/write/close that work on every platform (recv/send/
// closesocket on Windows; no SIGPIPE).
long long async_sock_read(int fd, []unsigned char buf);
long long async_sock_write(int fd, []unsigned char buf);
int       async_sock_close(int fd);

generic<F, T>
struct JoinHandle<T> Runtime::spawn(&AsyncTask<F, T> task) {
    return self.ex.spawn(task);
}

generic<F: Future, T>
T Runtime::block_on(&F fut) {
    T r = self.ex.block_on(&self.io, fut);
    return r;
}

} // namespace std
