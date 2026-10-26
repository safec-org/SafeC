// SafeC Standard Library — async: channels (oneshot, bounded MPMC)
//
// Channels whose send and receive are futures: a receiver waiting for a
// value, or a sender waiting for room, is Pending (with its waker
// registered) instead of blocking a thread — tokio::sync::oneshot and
// tokio::sync::mpsc (bounded). They work across executors and threads
// (every operation takes the channel's spinlock), and allocate nothing:
// an AsyncChannel's buffer is a slice you provide.
//
//   struct Oneshot<int> done = Oneshot::new();
//   // a task:   AWAIT_INTO(1, self.v, self.rx);    with self.rx = done.recv()
//   // elsewhere: done.send(42);
//
//   int slots[8];
//   struct AsyncChannel<int> ch = AsyncChannel::new(slots[0..8]);
//   self.tx = ch.send(7);     AWAIT_INTO(1, self.ok, self.tx);   // 1 sent, 0 closed
//   self.rx = ch.recv();      AWAIT_INTO(2, self.got, self.rx);  // ?int: null once closed and drained
//   ch.close();
//
// A channel must stay where it is while futures from send()/recv() exist
// (they point at it). Send and receive futures are each polled to
// completion once; make a new one for every value. Dropping a pending
// send future doesn't send its value.
#pragma once
#include <std/async/future.h>
#include <std/sync/spinlock.h>

namespace std {

// ── Oneshot ────────────────────────────────────────────────────────────────

#define ONESHOT_EMPTY  0
#define ONESHOT_FULL   1
#define ONESHOT_CLOSED 2   // closed by the sender with no value
#define ONESHOT_TAKEN  3   // the value was received

// One value, sent once, received once.
generic<T>
struct Oneshot {
    T                value[1];
    int              state;     // ONESHOT_*
    struct WakerSlot rx;
    struct Spinlock  lk;

    static struct Oneshot<T> new();
    // Store the value and wake the receiver. 1 on success, 0 if a value was
    // already sent or the channel was closed.
    int  send(T v);
    // No value will come: the receiver gets null.
    void close();
    // Closes it, releases the receiver's waker, and drops a value nobody
    // received. Nothing may use the channel (or a future from it) after.
    void drop();
    // The receive future: Ready(some(v)), or Ready(null) once closed.
    struct OneshotRecv<T> recv();
    // The value if it has arrived (and takes it), else null.
    ?T   try_recv();
    int  is_closed() const;
};

generic<T>
struct OneshotRecv {
    ?&Oneshot<T> ch;
    struct Poll<?T> poll(&Context cx);
    void drop();
};

// ── AsyncChannel ───────────────────────────────────────────────────────────

// A bounded multi-producer, multi-consumer FIFO over a caller's buffer.
generic<T>
struct AsyncChannel {
    []T              buf;
    unsigned long    head;      // index of the oldest value
    unsigned long    count;     // values in the buffer
    int              closed;
    struct WaitList  senders;   // waiting for room
    struct WaitList  receivers; // waiting for a value
    struct Spinlock  lk;

    // A channel holding at most storage.len values.
    static struct AsyncChannel<T> new([]T storage);
    // Future: Ready(1) once 'v' is in the channel, Ready(0) if it's closed.
    struct ChanSend<T> send(T v);
    // Future: Ready(some(v)) with the oldest value, Ready(null) once the
    // channel is closed and empty.
    struct ChanRecv<T> recv();
    // Without waiting: 1 sent, 0 full, -1 closed.
    int  try_send(T v);
    // Without waiting: the oldest value, or null if empty (or closed and empty).
    ?T   try_recv();
    // No more sends: pending and future sends get 0, receivers drain what
    // is buffered and then get null.
    void close();
    // Closes it, releases the waiting wakers, and drops the values still
    // buffered. Nothing may use the channel (or a future from it) after.
    void drop();
    int  is_closed() const;
    unsigned long len() const;
    unsigned long capacity() const;
};

generic<T>
struct ChanSend {
    ?&AsyncChannel<T> ch;
    T                 value;
    int               wait_id;
    int               done;
    int               sent;      // 'value' moved into the channel
    struct Poll<int> poll(&Context cx);
    void drop();               // and 'value', if it was never sent
};

generic<T>
struct ChanRecv {
    ?&AsyncChannel<T> ch;
    int               wait_id;
    int               done;
    struct Poll<?T> poll(&Context cx);
    void drop();
};

// ── Oneshot ────────────────────────────────────────────────────────────────

generic<T>
struct Oneshot<T> Oneshot::new() {
    struct Oneshot<T> o;
    o.state = ONESHOT_EMPTY;
    o.rx = WakerSlot::new();
    o.lk = Spinlock::new();
    return o;
}

generic<T>
int Oneshot::send(T v) {
    self.lk.lock();
    if (self.state != ONESHOT_EMPTY) { self.lk.unlock(); drop(v); return 0; }
    self.value[0] = v;
    self.state = ONESHOT_FULL;
    self.rx.wake();
    self.lk.unlock();
    return 1;
}

generic<T>
void Oneshot::close() {
    self.lk.lock();
    if (self.state == ONESHOT_EMPTY) {
        self.state = ONESHOT_CLOSED;
        self.rx.wake();
    }
    self.lk.unlock();
}

generic<T>
void Oneshot::drop() {
    self.close();
    if (self.state == ONESHOT_FULL) {
        self.state = ONESHOT_TAKEN;
        unsafe { drop(self.value[0]); }
    }
    unsafe { forget(self.value); }   // empty, or dropped just above
    self.rx.drop();
}

generic<T>
struct OneshotRecv<T> Oneshot::recv() {
    struct OneshotRecv<T> r;
    r.ch = self;
    return r;
}

generic<T>
?T Oneshot::try_recv() {
    self.lk.lock();
    if (self.state == ONESHOT_FULL) {
        self.state = ONESHOT_TAKEN;
        ?T v = null;
        unsafe { v = self.value[0]; }   // moves out: TAKEN marks the slot empty
        self.lk.unlock();
        return v;
    }
    self.lk.unlock();
    return null;
}

generic<T>
int Oneshot::is_closed() const { return self.state == ONESHOT_CLOSED ? 1 : 0; }

generic<T>
struct Poll<?T> OneshotRecv::poll(&Context cx) {
    match (self.ch) {
        case null: { ?T none = null; return Poll::ready(none); }
        case some(o): {
            o.lk.lock();
            if (o.state == ONESHOT_FULL) {
                o.state = ONESHOT_TAKEN;
                ?T v = null;
                unsafe { v = o.value[0]; }   // moves out: TAKEN marks the slot empty
                o.lk.unlock();
                return Poll::ready(v);
            }
            if (o.state != ONESHOT_EMPTY) {
                o.lk.unlock();
                ?T none = null;
                return Poll::ready(none);
            }
            o.rx.register_(&cx.waker);
            o.lk.unlock();
            return Poll::pending();
        }
    }
    ?T none = null;
    return Poll::ready(none);
}

generic<T>
void OneshotRecv::drop() {
    match (self.ch) {
        case some(o): { o.lk.lock(); o.rx.clear(); o.lk.unlock(); }
        case null: { }
    }
    self.ch = null;
}

// ── AsyncChannel ───────────────────────────────────────────────────────────

generic<T>
struct AsyncChannel<T> AsyncChannel::new([]T storage) {
    struct AsyncChannel<T> c;
    c.buf = storage;
    c.head = 0UL;
    c.count = 0UL;
    c.closed = 0;
    c.senders = WaitList::new();
    c.receivers = WaitList::new();
    c.lk = Spinlock::new();
    return c;
}

// With the lock held: push if there is room. 1 pushed (*v moved into the
// buffer), 0 full (*v untouched).
generic<T>
int async_chan_push_(&AsyncChannel<T> c, &T v) {
    if (c.count >= c.buf.len) return 0;
    unsigned long at = (c.head + c.count) % c.buf.len;
    unsafe { c.buf[at] = *v; }
    c.count = c.count + 1UL;
    c.receivers.wake_all();
    return 1;
}

generic<T>
struct ChanSend<T> AsyncChannel::send(T v) {
    struct ChanSend<T> s;
    s.ch = self;
    s.value = v;
    s.wait_id = 0;
    s.done = 0;
    s.sent = 0;
    return s;
}

generic<T>
struct ChanRecv<T> AsyncChannel::recv() {
    struct ChanRecv<T> r;
    r.ch = self;
    r.wait_id = 0;
    r.done = 0;
    return r;
}

generic<T>
int AsyncChannel::try_send(T v) {
    self.lk.lock();
    if (self.closed != 0) { self.lk.unlock(); drop(v); return -1; }
    int ok = async_chan_push_(self, &v);
    self.lk.unlock();
    if (ok) { unsafe { forget(v); } }   // moved into the buffer
    else drop(v);                       // full: the value is dropped
    return ok;
}

generic<T>
?T AsyncChannel::try_recv() {
    self.lk.lock();
    if (self.count == 0UL) { self.lk.unlock(); return null; }
    ?T v = null;
    unsafe { v = self.buf[self.head]; }   // moves out of the buffer
    self.head = (self.head + 1UL) % self.buf.len;
    self.count = self.count - 1UL;
    self.senders.wake_all();
    self.lk.unlock();
    return v;
}

generic<T>
void AsyncChannel::close() {
    self.lk.lock();
    self.closed = 1;
    self.senders.wake_all();
    self.receivers.wake_all();
    self.lk.unlock();
}

generic<T>
void AsyncChannel::drop() {
    self.close();
    if const (needs_drop(T)) {
        while (self.count > 0UL) {
            unsafe { drop(self.buf[self.head]); }
            self.head = (self.head + 1UL) % self.buf.len;
            self.count = self.count - 1UL;
        }
    }
    self.senders.drop();
    self.receivers.drop();
}

generic<T>
int AsyncChannel::is_closed() const { return self.closed; }

generic<T>
unsigned long AsyncChannel::len() const { return self.count; }

generic<T>
unsigned long AsyncChannel::capacity() const { return self.buf.len; }

generic<T>
struct Poll<int> ChanSend::poll(&Context cx) {
    if (self.done != 0) return Poll::ready(0);
    match (self.ch) {
        case null: { self.done = 1; return Poll::ready(0); }
        case some(c): {
            c.lk.lock();
            if (c.closed != 0) {
                c.senders.remove(self.wait_id);
                self.wait_id = 0;
                self.done = 1;
                c.lk.unlock();
                return Poll::ready(0);
            }
            if (async_chan_push_(c, &self.value) != 0) {
                self.sent = 1;
                c.senders.remove(self.wait_id);
                self.wait_id = 0;
                self.done = 1;
                c.lk.unlock();
                return Poll::ready(1);
            }
            self.wait_id = c.senders.add(self.wait_id, &cx.waker);
            c.lk.unlock();
            if (self.wait_id == 0) cx.wake_by_ref();   // no waiter slot left: re-poll
            return Poll::pending();
        }
    }
    return Poll::ready(0);
}

generic<T>
void ChanSend::drop() {
    match (self.ch) {
        case some(c): { c.lk.lock(); c.senders.remove(self.wait_id); c.lk.unlock(); }
        case null: { }
    }
    self.wait_id = 0;
    self.done = 1;
    if (self.sent == 0) { self.sent = 1; unsafe { drop(self.value); } }
    unsafe { forget(self.value); }   // sent (the channel owns it) or dropped
}

generic<T>
struct Poll<?T> ChanRecv::poll(&Context cx) {
    match (self.ch) {
        case null: { ?T none = null; return Poll::ready(none); }
        case some(c): {
            c.lk.lock();
            if (c.count > 0UL) {
                ?T v = null;
                unsafe { v = c.buf[c.head]; }   // moves out of the buffer
                c.head = (c.head + 1UL) % c.buf.len;
                c.count = c.count - 1UL;
                c.receivers.remove(self.wait_id);
                self.wait_id = 0;
                self.done = 1;
                c.senders.wake_all();
                c.lk.unlock();
                return Poll::ready(v);
            }
            if (c.closed != 0) {
                c.receivers.remove(self.wait_id);
                self.wait_id = 0;
                self.done = 1;
                c.lk.unlock();
                ?T none = null;
                return Poll::ready(none);
            }
            self.wait_id = c.receivers.add(self.wait_id, &cx.waker);
            c.lk.unlock();
            if (self.wait_id == 0) cx.wake_by_ref();
            return Poll::pending();
        }
    }
    ?T none = null;
    return Poll::ready(none);
}

generic<T>
void ChanRecv::drop() {
    match (self.ch) {
        case some(c): { c.lk.lock(); c.receivers.remove(self.wait_id); c.lk.unlock(); }
        case null: { }
    }
    self.wait_id = 0;
    self.done = 1;
}

} // namespace std
