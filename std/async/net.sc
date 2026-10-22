// SafeC Standard Library — async: IoDriver, Runtime, async TCP
// safeguard: link windows -lws2_32
#pragma once
#include <std/async/net.h>
#include <std/panic.h>
#include <std/thread.h>
#include <std/stddef.h>

namespace std {

// ── Platform socket calls ──────────────────────────────────────────────────
// Like std/http/http.sc's: Winsock sockets need recv/send/closesocket; on
// POSIX read/close work on any fd, and Linux writes use send(MSG_NOSIGNAL)
// (BSD/macOS sockets from io_nb.h already have SO_NOSIGPIPE).

#ifdef _WIN32
extern int recv(unsigned long long fd, void* buf, int count, int flags);
extern int send(unsigned long long fd, const void* buf, int count, int flags);
extern int closesocket(unsigned long long fd);
extern int getsockname(unsigned long long fd, void* addr, int* len);
extern int getpeername(unsigned long long fd, void* addr, int* len);
extern int getsockopt(unsigned long long fd, int level, int name, void* val, int* len);
#define ASYNC_SOL_SOCKET_ 0xffff
#define ASYNC_SO_ERROR_   0x1007

static long long async_raw_read_(int fd, void* buf, unsigned long n) {
    unsafe { return (long long)recv((unsigned long long)fd, buf, (int)n, 0); }
}
static long long async_raw_write_(int fd, const void* buf, unsigned long n) {
    unsafe { return (long long)send((unsigned long long)fd, buf, (int)n, 0); }
}
int async_sock_close(int fd) {
    unsafe { return closesocket((unsigned long long)fd); }
}
static int async_getsockname_(int fd, void* addr, &int len) {
    unsafe { return getsockname((unsigned long long)fd, addr, (int*)len); }
}
static int async_getpeername_(int fd, void* addr, &int len) {
    unsafe { return getpeername((unsigned long long)fd, addr, (int*)len); }
}
static int async_so_error_(int fd) {
    int err = 0;
    int len = 4;
    unsafe { getsockopt((unsigned long long)fd, ASYNC_SOL_SOCKET_, ASYNC_SO_ERROR_, (void*)&err, &len); }
    return err;
}
#else
extern ssize_t read(int fd, void* buf, size_t count);
extern ssize_t write(int fd, const void* buf, size_t count);
extern int close(int fd);
extern int pipe(int* fds);
extern int getsockname(int fd, void* addr, unsigned int* len);
extern int getpeername(int fd, void* addr, unsigned int* len);
extern int getsockopt(int fd, int level, int name, void* val, unsigned int* len);
#ifdef __linux__
#define ASYNC_SOL_SOCKET_ 1
#define ASYNC_SO_ERROR_   4
extern ssize_t send(int fd, const void* buf, size_t count, int flags);
static long long async_raw_write_(int fd, const void* buf, unsigned long n) {
    unsafe { return (long long)send(fd, buf, (size_t)n, 0x4000); }   // MSG_NOSIGNAL
}
#else
#define ASYNC_SOL_SOCKET_ 0xffff
#define ASYNC_SO_ERROR_   0x1007
static long long async_raw_write_(int fd, const void* buf, unsigned long n) {
    unsafe { return (long long)write(fd, buf, (size_t)n); }
}
#endif
static long long async_raw_read_(int fd, void* buf, unsigned long n) {
    unsafe { return (long long)read(fd, buf, (size_t)n); }
}
int async_sock_close(int fd) {
    unsafe { return close(fd); }
}
static int async_getsockname_(int fd, void* addr, &int len) {
    unsafe { return getsockname(fd, addr, (unsigned int*)len); }
}
static int async_getpeername_(int fd, void* addr, &int len) {
    unsafe { return getpeername(fd, addr, (unsigned int*)len); }
}
static int async_so_error_(int fd) {
    int err = 0;
    unsigned int len = 4U;
    unsafe { getsockopt(fd, ASYNC_SOL_SOCKET_, ASYNC_SO_ERROR_, (void*)&err, &len); }
    return err;
}
#endif

long long async_sock_read(int fd, []unsigned char buf) {
    if (buf.len == 0UL) return 0LL;
    unsafe { return async_raw_read_(fd, (void*)&buf[0], buf.len); }
}

long long async_sock_write(int fd, []unsigned char buf) {
    if (buf.len == 0UL) return 0LL;
    unsafe { return async_raw_write_(fd, (const void*)&buf[0], buf.len); }
}

unsigned short async_sock_local_port(int fd) {
    // sockaddr_in: the port is bytes 2..3 (network order) on every
    // platform (BSD's sin_len + sin_family, or a 2-byte family, before it).
    unsigned char sa[16];
    int i = 0;
    while (i < 16) { sa[i] = (unsigned char)0; i = i + 1; }
    int len = 16;
    unsafe { if (async_getsockname_(fd, (void*)&sa[0], &len) != 0) return (unsigned short)0; }
    return (unsigned short)(((unsigned int)sa[2] << 8) | (unsigned int)sa[3]);
}

// 1 connected, 0 still connecting, -1 failed.
static int async_sock_connected_(int fd) {
    unsigned char sa[16];
    int len = 16;
    int rc;
    unsafe { rc = async_getpeername_(fd, (void*)&sa[0], &len); }
    if (rc == 0) return 1;
    if (async_so_error_(fd) != 0) return -1;
    return 0;
}

// ── IoDriver ───────────────────────────────────────────────────────────────
// Registration id = serial * 512 + slot + 1.

static int io_slot_(int id) { return (id & 511) - 1; }

struct IoDriver IoDriver::new() {
    struct IoDriver d;
    d.reactor = Reactor::new();
    d.ok = 0;
    int i = 0;
    while (i < ASYNC_IO_WAITERS) {
        d.waiters[i].fd = -1;
        d.waiters[i].filter = 0;
        d.waiters[i].serial = 0;
        d.waiters[i].armed = 0;
        d.waiters[i].w = Waker::noop();
        i = i + 1;
    }
    d.nwaiting = 0;
    d.next_serial = 1;
    d.wake_rd = -1;
    d.wake_wr = -1;
    d.wake_pending = 0;
    d.events = 0LL;
    return d;
}

int IoDriver::init() {
    if (self.reactor.init() != 0) return -1;
    self.ok = 1;
#ifndef _WIN32
    int fds[2];
    fds[0] = -1;
    fds[1] = -1;
    int rc;
    unsafe { rc = pipe(&fds[0]); }
    if (rc == 0) {
        fd_set_nonblocking(fds[0]);
        fd_set_nonblocking(fds[1]);
        self.wake_rd = fds[0];
        self.wake_wr = fds[1];
        self.reactor.add(self.wake_rd, SCHED_READ);
    }
#endif
    return 0;
}

void IoDriver::close_() {
    if (self.ok == 0) return;
    int i = 0;
    while (i < ASYNC_IO_WAITERS) {
        if (self.waiters[i].serial != 0) {
            self.waiters[i].serial = 0;
            self.waiters[i].w.drop();
        }
        i = i + 1;
    }
    self.nwaiting = 0;
#ifndef _WIN32
    if (self.wake_rd >= 0) {
        self.reactor.remove(self.wake_rd, SCHED_READ);
        unsafe { close(self.wake_rd); close(self.wake_wr); }
        self.wake_rd = -1;
        self.wake_wr = -1;
    }
#endif
    self.reactor.close_();
    self.ok = 0;
}

static int io_valid_(const &IoDriver d, int id) {
    if (id <= 0) return 0;
    int slot = io_slot_(id);
    if (slot < 0 || slot >= ASYNC_IO_WAITERS) return 0;
    return (d.waiters[slot].serial != 0 && d.waiters[slot].serial == id / 512) ? 1 : 0;
}

// Frees entry 'slot' (without waking it) and keeps the reactor registration
// right: if it was the one that registered (fd, filter), hand that over to
// another entry waiting on the same pair, or unregister it.
static void io_free_slot_(&IoDriver d, int slot) {
    int fd = d.waiters[slot].fd;
    int filter = d.waiters[slot].filter;
    int armed = d.waiters[slot].armed;
    d.waiters[slot].serial = 0;
    d.waiters[slot].armed = 0;
    d.waiters[slot].fd = -1;
    d.nwaiting = d.nwaiting - 1;
    if (armed == 0) return;
    int i = 0;
    while (i < ASYNC_IO_WAITERS) {
        if (d.waiters[i].serial != 0 && d.waiters[i].fd == fd && d.waiters[i].filter == filter) {
            d.waiters[i].armed = 1;
            return;
        }
        i = i + 1;
    }
    d.reactor.remove(fd, filter);
}

int IoDriver::wait(int fd, int filter, const &Waker w, int id) {
    if (self.ok == 0) return 0;
    if (io_valid_(self, id) != 0) {
        int slot = io_slot_(id);
        if (self.waiters[slot].fd == fd && self.waiters[slot].filter == filter) {
            if (self.waiters[slot].w.will_wake(w) == 0) {
                self.waiters[slot].w.drop();
                self.waiters[slot].w = w.clone();
            }
            return id;
        }
        self.cancel(id);
    }
    // Is (fd, filter) registered with the reactor already?
    int have = 0;
    int clash = 0;
    int i = 0;
    while (i < ASYNC_IO_WAITERS) {
        if (self.waiters[i].serial != 0 && self.waiters[i].fd == fd && self.waiters[i].armed != 0) {
            if (self.waiters[i].filter == filter) have = 1;
            else clash = 1;
        }
        i = i + 1;
    }
#ifdef __linux__
    // epoll takes an fd once: a second direction can't be added.
    if (have == 0 && clash != 0) return 0;
#endif
    int slot = -1;
    i = 0;
    while (i < ASYNC_IO_WAITERS && slot < 0) {
        if (self.waiters[i].serial == 0) slot = i;
        i = i + 1;
    }
    if (slot < 0) return 0;
    int ser = self.next_serial;
    self.next_serial = self.next_serial >= 4000000 ? 1 : self.next_serial + 1;
    self.waiters[slot].fd = fd;
    self.waiters[slot].filter = filter;
    self.waiters[slot].serial = ser;
    self.waiters[slot].w = w.clone();
    self.waiters[slot].armed = 0;
    if (have == 0) {
        self.reactor.add(fd, filter);
        self.waiters[slot].armed = 1;
    }
    self.nwaiting = self.nwaiting + 1;
    return ser * 512 + slot + 1;
}

void IoDriver::cancel(int id) {
    if (io_valid_(self, id) == 0) return;
    int slot = io_slot_(id);
    self.waiters[slot].w.drop();
    io_free_slot_(self, slot);
}

int IoDriver::waiting() const { return self.nwaiting; }

void IoDriver::on_ready_(int fd, int filter) {
    self.events = self.events + 1LL;
#ifndef _WIN32
    if (fd == self.wake_rd) {
        unsigned char junk[64];
        long long n = 1LL;
        while (n > 0LL) {
            unsafe { n = async_raw_read_(fd, (void*)&junk[0], 64UL); }
        }
        atomic_store_int(&self.wake_pending, 0);
        return;
    }
#endif
    // Wake everything waiting on this fd, whatever the direction: an
    // error or hang-up may be reported as either, and a spurious wake is
    // harmless (the future retries and registers again).
    int i = 0;
    while (i < ASYNC_IO_WAITERS) {
        if (self.waiters[i].serial != 0 && self.waiters[i].fd == fd) {
            struct Waker w = self.waiters[i].w;
            io_free_slot_(self, i);
            w.wake();
        }
        i = i + 1;
    }
}

static void io_on_ready_cb_(?&IoDriver dq, int fd, int filter) {
    match (dq) {
        case some(d): d.on_ready_(fd, filter);
        case null: { }
    }
}

long long IoDriver::now_ms() { return async_mono_ms(); }

void IoDriver::park(long long timeout_ms) {
    if (self.ok == 0) {
        thread_sleep_ms((unsigned long)(timeout_ms < 0LL ? (long long)ASYNC_IO_REMOTE_MS : timeout_ms));
        return;
    }
    long long t = timeout_ms;
#ifdef _WIN32
    if (atomic_exchange_int(&self.wake_pending, 0) != 0) return;
    if (t < 0LL || t > (long long)ASYNC_IO_REMOTE_MS) t = (long long)ASYNC_IO_REMOTE_MS;
    if (self.nwaiting == 0) {
        if (t > 0LL) thread_sleep_ms((unsigned long)t);
        return;
    }
#endif
    self.reactor.poll_events(io_on_ready_cb_, (?&IoDriver)self, t);
}

void IoDriver::unpark() {
#ifdef _WIN32
    atomic_store_int(&self.wake_pending, 1);
#else
    if (self.wake_wr < 0) return;
    if (atomic_exchange_int(&self.wake_pending, 1) != 0) return;
    unsigned char one = (unsigned char)1;
    unsafe { write(self.wake_wr, (const void*)&one, (size_t)1); }   // a pipe: not send()
#endif
}

?&IoDriver io_driver(&Context cx) {
    unsafe { return (?&IoDriver)cx.io_; }
}

static &IoDriver io_need_driver_(&Context cx) {
    match (io_driver(cx)) {
        case some(d): return d;
        case null: {
            panic_at("std/async: an I/O future was polled outside a Runtime (Context has no IoDriver)",
                     "std/async/net.sc", 0);
        }
    }
    return io_need_driver_(cx);   // unreachable
}

// ── Runtime ────────────────────────────────────────────────────────────────

struct Runtime Runtime::new() {
    struct Runtime rt;
    rt.ex = LocalExecutor::new();
    rt.io = IoDriver::new();
    return rt;
}

int Runtime::init() {
    if (self.io.init() != 0) return -1;
    unsafe { self.ex.io_ = (void*)&self.io; }
    return 0;
}

void Runtime::close_() {
    self.io.close_();
    unsafe { self.ex.io_ = (void*)0; }
}

void Runtime::run() {
    self.ex.run(&self.io);
}

// ── TCP ────────────────────────────────────────────────────────────────────

struct AsyncTcpStream AsyncTcpStream::from_fd(int fd) {
    struct AsyncTcpStream s;
    s.fd = fd;
    return s;
}

int AsyncTcpStream::close() {
    if (self.fd < 0) return -1;
    int rc = async_sock_close(self.fd);
    self.fd = -1;
    return rc;
}

int AsyncTcpStream::is_open() const { return self.fd >= 0 ? 1 : 0; }

struct ConnectFut AsyncTcpStream::connect(unsigned int addr_network_order, unsigned short port) {
    struct ConnectFut c;
    c.addr = addr_network_order;
    c.port = port;
    c.fd = -1;
    c.wait_id = 0;
    c.done = 0;
    c.drv = null;
    return c;
}

struct ReadFut AsyncTcpStream::read([]unsigned char buf) {
    struct ReadFut r;
    r.fd = self.fd;
    r.buf = buf;
    r.wait_id = 0;
    r.drv = null;
    return r;
}

struct WriteFut AsyncTcpStream::write([]unsigned char buf) {
    struct WriteFut w;
    w.fd = self.fd;
    w.buf = buf;
    w.wait_id = 0;
    w.drv = null;
    return w;
}

struct WriteAllFut AsyncTcpStream::write_all([]unsigned char buf) {
    struct WriteAllFut w;
    w.fd = self.fd;
    w.buf = buf;
    w.sent = 0UL;
    w.wait_id = 0;
    w.drv = null;
    return w;
}

struct AsyncTcpListener AsyncTcpListener::bind(unsigned short port) {
    struct AsyncTcpListener l;
    l.fd = tcp_listen_nb(port);
    return l;
}

struct AcceptFut AsyncTcpListener::accept() {
    struct AcceptFut a;
    a.lfd = self.fd;
    a.wait_id = 0;
    a.drv = null;
    return a;
}

unsigned short AsyncTcpListener::local_port() const {
    if (self.fd < 0) return (unsigned short)0;
    return async_sock_local_port(self.fd);
}

int AsyncTcpListener::close() {
    if (self.fd < 0) return -1;
    int rc = async_sock_close(self.fd);
    self.fd = -1;
    return rc;
}

// Cancel a registration held by a future (drop()).
static void io_cancel_(?&IoDriver dq, &int wait_id) {
    match (dq) {
        case some(d): d.cancel(*wait_id);
        case null: { }
    }
    *wait_id = 0;
}

// Register for readiness; a full table means re-poll.
static void io_wait_(&IoDriver d, &Context cx, int fd, int filter, &int wait_id) {
    *wait_id = d.wait(fd, filter, &cx.waker, *wait_id);
    if (*wait_id == 0) cx.wake_by_ref();
}

struct IoReady IoReady::new(int fd, int filter) {
    struct IoReady r;
    r.fd = fd;
    r.filter = filter;
    r.wait_id = 0;
    r.done = 0;
    r.drv = null;
    return r;
}

struct Poll<Unit> IoReady::poll(&Context cx) {
    if (self.done != 0) return Poll::ready(UNIT);
    &IoDriver d = io_need_driver_(cx);
    self.drv = d;
    // Registered before and woken since (the registration is gone): ready.
    if (self.wait_id != 0 && io_valid_(d, self.wait_id) == 0) {
        self.wait_id = 0;
        self.done = 1;
        return Poll::ready(UNIT);
    }
    io_wait_(d, cx, self.fd, self.filter, &self.wait_id);
    return Poll::pending();
}

void IoReady::drop() { io_cancel_(self.drv, &self.wait_id); self.done = 1; }

struct Poll<struct AsyncTcpStream> AcceptFut::poll(&Context cx) {
    &IoDriver d = io_need_driver_(cx);
    self.drv = d;
    if (self.lfd < 0) return Poll::ready(AsyncTcpStream::from_fd(-1));
    int fd = tcp_accept_nb(self.lfd);
    if (fd >= 0) {
        io_cancel_(self.drv, &self.wait_id);
        return Poll::ready(AsyncTcpStream::from_fd(fd));
    }
    if (sock_would_block() == 0) {
        io_cancel_(self.drv, &self.wait_id);
        return Poll::ready(AsyncTcpStream::from_fd(-1));
    }
    io_wait_(d, cx, self.lfd, SCHED_READ, &self.wait_id);
    return Poll::pending();
}

void AcceptFut::drop() { io_cancel_(self.drv, &self.wait_id); }

struct Poll<struct AsyncTcpStream> ConnectFut::poll(&Context cx) {
    if (self.done != 0) return Poll::ready(AsyncTcpStream::from_fd(-1));
    &IoDriver d = io_need_driver_(cx);
    self.drv = d;
    if (self.fd < 0) {
        self.fd = tcp_connect_nb(self.addr, self.port);
        if (self.fd < 0) {
            self.done = 1;
            return Poll::ready(AsyncTcpStream::from_fd(-1));
        }
    }
    int st = async_sock_connected_(self.fd);
    if (st != 0) {
        io_cancel_(self.drv, &self.wait_id);
        self.done = 1;
        int fd = self.fd;
        self.fd = -1;
        if (st < 0) {
            async_sock_close(fd);
            return Poll::ready(AsyncTcpStream::from_fd(-1));
        }
        return Poll::ready(AsyncTcpStream::from_fd(fd));
    }
    io_wait_(d, cx, self.fd, SCHED_WRITE, &self.wait_id);
    return Poll::pending();
}

void ConnectFut::drop() {
    io_cancel_(self.drv, &self.wait_id);
    if (self.done == 0 && self.fd >= 0) async_sock_close(self.fd);
    self.fd = -1;
    self.done = 1;
}

struct Poll<long long> ReadFut::poll(&Context cx) {
    &IoDriver d = io_need_driver_(cx);
    self.drv = d;
    if (self.fd < 0) return Poll::ready(-1LL);
    long long n = async_sock_read(self.fd, self.buf);
    if (n >= 0LL) {
        io_cancel_(self.drv, &self.wait_id);
        return Poll::ready(n);
    }
    if (sock_would_block() == 0) {
        io_cancel_(self.drv, &self.wait_id);
        return Poll::ready(-1LL);
    }
    io_wait_(d, cx, self.fd, SCHED_READ, &self.wait_id);
    return Poll::pending();
}

void ReadFut::drop() { io_cancel_(self.drv, &self.wait_id); }

struct Poll<long long> WriteFut::poll(&Context cx) {
    &IoDriver d = io_need_driver_(cx);
    self.drv = d;
    if (self.fd < 0) return Poll::ready(-1LL);
    long long n = async_sock_write(self.fd, self.buf);
    if (n >= 0LL) {
        io_cancel_(self.drv, &self.wait_id);
        return Poll::ready(n);
    }
    if (sock_would_block() == 0) {
        io_cancel_(self.drv, &self.wait_id);
        return Poll::ready(-1LL);
    }
    io_wait_(d, cx, self.fd, SCHED_WRITE, &self.wait_id);
    return Poll::pending();
}

void WriteFut::drop() { io_cancel_(self.drv, &self.wait_id); }

struct Poll<int> WriteAllFut::poll(&Context cx) {
    &IoDriver d = io_need_driver_(cx);
    self.drv = d;
    if (self.fd < 0) return Poll::ready(0);
    while (self.sent < self.buf.len) {
        long long n = async_sock_write(self.fd, self.buf[self.sent..self.buf.len]);
        if (n > 0LL) {
            self.sent = self.sent + (unsigned long)n;
        } else if (n < 0LL && sock_would_block() != 0) {
            io_wait_(d, cx, self.fd, SCHED_WRITE, &self.wait_id);
            return Poll::pending();
        } else {
            io_cancel_(self.drv, &self.wait_id);
            return Poll::ready(0);
        }
    }
    io_cancel_(self.drv, &self.wait_id);
    return Poll::ready(1);
}

void WriteAllFut::drop() { io_cancel_(self.drv, &self.wait_id); }

} // namespace std
