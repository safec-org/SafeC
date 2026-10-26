// SafeC Standard Library — async: combinators (join, select, timeout, map, and_then)
//
// Futures built from other futures, as in Rust's futures::future and
// tokio's join!/select!/timeout. Each takes its children by value (move
// them in before the first poll: futures are pinned once polled) and owns
// them: it polls them, and drops the ones it abandons — select drops the
// loser, timeout drops the inner future when time runs out, and drop() of
// a combinator drops every child still running.
//
// Rust infers a combinator's output from its children; SafeC generics
// can't name "the T of F's Poll<T>", so the output types are type
// parameters too, written in the declaration and inferred from it:
//
//   struct Join2<struct Sleep, struct Fetch, Unit, int> j =
//       join2(Sleep::new(50LL), Fetch::new());
//   AWAIT_INTO(1, self.both, self.j);        // self.both: struct Both<Unit, int>
//
//   struct Timeout<struct Fetch, int> t = timeout(Fetch::new(), 100LL);
//   AWAIT_INTO(2, self.res, self.t);          // self.res: ?int (null = timed out)
//
// Nothing here allocates. JoinAll and SelectAny take caller-provided
// slices of futures (and of output slots), up to 64 futures.
#pragma once
#include <std/async/future.h>
#include <std/async/timer.h>

namespace std {

// Both outputs of a Join2 (Rust's (A, B)).
generic<TA, TB>
struct Both {
    TA first;
    TB second;
};

// Which future a Select2 picked, and its output (Rust's Either; the other
// field is unset). It owns that output when its type owns something.
generic<TA, TB>
struct Either {
    int which;     // 1: 'left' holds the first future's output, 2: 'right' the second's
    TA  left;
    TB  right;
    int is_left() const;
    void drop() if (needs_drop(TA) || needs_drop(TB));
};

// One output of a SelectAny, and whose it is.
generic<T>
struct Picked {
    int index;     // position in the slice
    T   value;
};

// Polls both until both are Ready (tokio::join!). Output: Both<TA, TB>.
generic<A: Future, B: Future, TA, TB>
struct Join2 {
    A   a;
    B   b;
    TA  ra[1];
    TB  rb[1];
    int done_a;     // 1: ra holds a's output
    int done_b;
    int delivered;  // the outputs moved out in the Ready Both
    struct Poll<struct Both<TA, TB>> poll(&Context cx);
    void drop();    // and outputs nobody received
};

// Three at once. Output: Both<TA, Both<TB, TC>> -- (a, (b, c)).
generic<A: Future, B: Future, C: Future, TA, TB, TC>
struct Join3 {
    A   a;
    B   b;
    C   c;
    TA  ra[1];
    TB  rb[1];
    TC  rc[1];
    int done;       // bit 0: a, 1: b, 2: c (its output is stored)
    int delivered;  // the outputs moved out in the Ready Both
    struct Poll<struct Both<TA, struct Both<TB, TC>>> poll(&Context cx);
    void drop();    // and outputs nobody received
};

// Polls every future in 'futs' until all are Ready, storing future i's
// output in outs[i] (futures::future::join_all, without the heap). Output:
// how many (= futs.len). At most 64 futures.
generic<F: Future, T>
struct JoinAll {
    []F                futs;
    []T                outs;
    unsigned long long done;     // bit i: futs[i] finished
    struct Poll<int> poll(&Context cx);
    void drop();
};

// The first of two to be Ready wins; the other is dropped (tokio::select!
// with two branches, polled in order). Output: Either<TA, TB>.
generic<A: Future, B: Future, TA, TB>
struct Select2 {
    A   a;
    B   b;
    int finished;
    struct Poll<struct Either<TA, TB>> poll(&Context cx);
    void drop();
};

// The first in 'futs' to be Ready wins (lowest index on a tie); the rest
// are dropped (futures::future::select_all). Output: Picked<T>.
generic<F: Future, T>
struct SelectAny {
    []F futs;
    int finished;
    struct Poll<struct Picked<T>> poll(&Context cx);
    void drop();
};

// 'inner', unless 'ms' milliseconds pass first: then 'inner' is dropped
// (tokio::time::timeout). Output: ?T, null on timeout. Needs timers.
generic<F: Future, T>
struct Timeout {
    F            inner;
    struct Sleep sleep;
    int          finished;
    struct Poll<?T> poll(&Context cx);
    void drop();
};

// 'inner''s output passed through f (FutureExt::map). Output: U.
generic<F: Future, T, U>
struct MapFut {
    F         inner;
    fn U(T)   f;
    struct Poll<U> poll(&Context cx);
    void drop();
};

// Runs 'first', then the future f makes from its output (and_then /
// .then). Output: U, the second future's.
generic<F: Future, T, G: Future, U>
struct AndThen {
    F        first;
    fn G(T)  f;
    G        second[1];
    int      stage;      // 0: running first, 1: running second, 2: done
    struct Poll<U> poll(&Context cx);
    void drop();
};

// ── constructors (defined below) ───────────────────────────────────────────
//
//   join2(a, b)            -> Join2<A, B, TA, TB>
//   join3(a, b, c)         -> Join3<A, B, C, TA, TB, TC>
//   join_all(futs, outs)   -> JoinAll<F, T>
//   select2(a, b)          -> Select2<A, B, TA, TB>
//   select_any(futs)       -> SelectAny<F, T>
//   timeout(inner, ms)     -> Timeout<F, T>
//   map_fut(inner, f)      -> MapFut<F, T, U>
//   and_then(first, f)     -> AndThen<F, T, G, U>

// ── definitions ────────────────────────────────────────────────────────────

generic<TA, TB>
int Either::is_left() const { return self.which == 1 ? 1 : 0; }

generic<A, B, TA, TB>
struct Join2<A, B, TA, TB> join2(A a, B b) {
    struct Join2<A, B, TA, TB> j;
    j.a = a;
    j.b = b;
    j.done_a = 0;
    j.done_b = 0;
    j.delivered = 0;
    return j;
}

generic<A: Future, B: Future, TA, TB>
struct Poll<struct Both<TA, TB>> Join2::poll(&Context cx) {
    if (self.done_a == 0 && poll_take_into_(self.a.poll(cx), &self.ra[0]) != 0) self.done_a = 1;
    if (self.done_b == 0 && poll_take_into_(self.b.poll(cx), &self.rb[0]) != 0) self.done_b = 1;
    if (self.done_a == 0 || self.done_b == 0) return Poll::pending();
    struct Both<TA, TB> r;
    unsafe {   // the outputs move out of their slots (once: 'delivered')
        r.first = self.ra[0];
        r.second = self.rb[0];
    }
    self.delivered = 1;
    return Poll::ready(r);
}

generic<A: Future, B: Future, TA, TB>
void Join2::drop() {
    if (self.done_a == 0) { self.a.drop(); self.done_a = 2; }
    else if (self.done_a == 1 && self.delivered == 0) { unsafe { drop(self.ra[0]); } }
    if (self.done_b == 0) { self.b.drop(); self.done_b = 2; }
    else if (self.done_b == 1 && self.delivered == 0) { unsafe { drop(self.rb[0]); } }
    self.delivered = 1;
    unsafe { forget(self.ra); forget(self.rb); }   // empty, moved out, or dropped above
}

generic<A, B, C, TA, TB, TC>
struct Join3<A, B, C, TA, TB, TC> join3(A a, B b, C c) {
    struct Join3<A, B, C, TA, TB, TC> j;
    j.a = a;
    j.b = b;
    j.c = c;
    j.done = 0;
    j.delivered = 0;
    return j;
}

generic<A: Future, B: Future, C: Future, TA, TB, TC>
struct Poll<struct Both<TA, struct Both<TB, TC>>> Join3::poll(&Context cx) {
    if ((self.done & 1) == 0 && poll_take_into_(self.a.poll(cx), &self.ra[0]) != 0) self.done = self.done | 1;
    if ((self.done & 2) == 0 && poll_take_into_(self.b.poll(cx), &self.rb[0]) != 0) self.done = self.done | 2;
    if ((self.done & 4) == 0 && poll_take_into_(self.c.poll(cx), &self.rc[0]) != 0) self.done = self.done | 4;
    if (self.done != 7) return Poll::pending();
    struct Both<TB, TC> bc;
    struct Both<TA, struct Both<TB, TC>> r;
    unsafe {   // the outputs move out of their slots (once: 'delivered')
        bc.first = self.rb[0];
        bc.second = self.rc[0];
        r.first = self.ra[0];
    }
    r.second = bc;
    self.delivered = 1;
    return Poll::ready(r);
}

generic<A: Future, B: Future, C: Future, TA, TB, TC>
void Join3::drop() {
    int stored = self.delivered == 0 ? self.done : 0;
    if ((self.done & 1) == 0) self.a.drop();
    if ((self.done & 2) == 0) self.b.drop();
    if ((self.done & 4) == 0) self.c.drop();
    unsafe {
        if ((stored & 1) != 0) drop(self.ra[0]);
        if ((stored & 2) != 0) drop(self.rb[0]);
        if ((stored & 4) != 0) drop(self.rc[0]);
        forget(self.ra); forget(self.rb); forget(self.rc);   // empty, moved out, or dropped above
    }
    self.done = 7;
    self.delivered = 1;
}

generic<F, T>
struct JoinAll<F, T> join_all([]F futs, []T outs) {
    struct JoinAll<F, T> j;
    j.futs = futs;
    j.outs = outs;
    j.done = 0ULL;
    return j;
}

generic<F: Future, T>
struct Poll<int> JoinAll::poll(&Context cx) {
    unsigned long n = self.futs.len;
    if (n > 64UL) n = 64UL;
    if (self.outs.len < n) n = self.outs.len;
    unsigned long long all = n == 64UL ? ~0ULL : ((1ULL << n) - 1ULL);
    unsigned long i = 0UL;
    while (i < n) {
        unsigned long long bit = 1ULL << i;
        if ((self.done & bit) == 0ULL) {
            if (poll_take_into_(self.futs[i].poll(cx), &self.outs[i]) != 0) self.done = self.done | bit;
        }
        i = i + 1UL;
    }
    if (self.done != all) return Poll::pending();
    return Poll::ready((int)n);
}

generic<F: Future, T>
void JoinAll::drop() {
    unsigned long n = self.futs.len;
    if (n > 64UL) n = 64UL;
    unsigned long i = 0UL;
    while (i < n) {
        unsigned long long bit = 1ULL << i;
        if ((self.done & bit) == 0ULL) { self.futs[i].drop(); self.done = self.done | bit; }
        i = i + 1UL;
    }
}

generic<A, B, TA, TB>
struct Select2<A, B, TA, TB> select2(A a, B b) {
    struct Select2<A, B, TA, TB> s;
    s.a = a;
    s.b = b;
    s.finished = 0;
    return s;
}

generic<A: Future, B: Future, TA, TB>
struct Poll<struct Either<TA, TB>> Select2::poll(&Context cx) {
    struct Poll<TA> pa = self.a.poll(cx);
    if (pa.ready_ != 0) {
        self.b.drop();
        self.finished = 1;
        struct Either<TA, TB> e;
        e.which = 1;
        e.left = pa.unwrap();
        return Poll::ready(e);
    }
    pa.drop();
    struct Poll<TB> pb = self.b.poll(cx);
    if (pb.ready_ != 0) {
        self.a.drop();
        self.finished = 1;
        struct Either<TA, TB> e;
        e.which = 2;
        e.right = pb.unwrap();
        return Poll::ready(e);
    }
    pb.drop();
    return Poll::pending();
}

generic<TA, TB>
void Either::drop() {
    unsafe {   // only the slot 'which' names holds a value
        if (self.which == 1) drop(self.left);
        if (self.which == 2) drop(self.right);
        forget(self.left); forget(self.right);
    }
    self.which = 0;
}

generic<A: Future, B: Future, TA, TB>
void Select2::drop() {
    if (self.finished == 0) { self.a.drop(); self.b.drop(); self.finished = 1; }
}

generic<F, T>
struct SelectAny<F, T> select_any([]F futs) {
    struct SelectAny<F, T> s;
    s.futs = futs;
    s.finished = 0;
    return s;
}

generic<F: Future, T>
struct Poll<struct Picked<T>> SelectAny::poll(&Context cx) {
    unsigned long n = self.futs.len;
    unsigned long i = 0UL;
    while (i < n) {
        struct Poll<T> p = self.futs[i].poll(cx);
        if (p.ready_ == 0) p.drop();
        else {
            unsigned long k = 0UL;
            while (k < n) {
                if (k != i) self.futs[k].drop();
                k = k + 1UL;
            }
            self.finished = 1;
            struct Picked<T> r;
            r.index = (int)i;
            r.value = p.unwrap();
            return Poll::ready(r);
        }
        i = i + 1UL;
    }
    return Poll::pending();
}

generic<F: Future, T>
void SelectAny::drop() {
    if (self.finished != 0) return;
    unsigned long i = 0UL;
    while (i < self.futs.len) { self.futs[i].drop(); i = i + 1UL; }
    self.finished = 1;
}

generic<F, T>
struct Timeout<F, T> timeout(F inner, long long ms) {
    struct Timeout<F, T> t;
    t.inner = inner;
    t.sleep = Sleep::new(ms);
    t.finished = 0;
    return t;
}

generic<F: Future, T>
struct Poll<?T> Timeout::poll(&Context cx) {
    struct Poll<T> p = self.inner.poll(cx);
    if (p.ready_ != 0) {
        self.sleep.drop();
        self.finished = 1;
        ?T v = p.take();
        return Poll::ready(v);
    }
    p.drop();
    struct Poll<Unit> s = self.sleep.poll(cx);
    if (s.ready_ != 0) {
        self.inner.drop();
        self.finished = 1;
        ?T none = null;
        return Poll::ready(none);
    }
    return Poll::pending();
}

generic<F: Future, T>
void Timeout::drop() {
    if (self.finished == 0) { self.inner.drop(); self.sleep.drop(); self.finished = 1; }
}

generic<F, T, U>
struct MapFut<F, T, U> map_fut(F inner, fn U(T) f) {
    struct MapFut<F, T, U> m;
    m.inner = inner;
    m.f = f;
    return m;
}

generic<F: Future, T, U>
struct Poll<U> MapFut::poll(&Context cx) {
    struct Poll<T> p = self.inner.poll(cx);
    if (p.ready_ == 0) { p.drop(); return Poll::pending(); }
    return Poll::ready(self.f(p.unwrap()));
}

generic<F: Future, T, U>
void MapFut::drop() { self.inner.drop(); }

generic<F, T, G, U>
struct AndThen<F, T, G, U> and_then(F first, fn G(T) f) {
    struct AndThen<F, T, G, U> a;
    a.first = first;
    a.f = f;
    a.stage = 0;
    return a;
}

generic<F: Future, T, G: Future, U>
struct Poll<U> AndThen::poll(&Context cx) {
    if (self.stage == 0) {
        struct Poll<T> p = self.first.poll(cx);
        if (p.ready_ == 0) { p.drop(); return Poll::pending(); }
        self.second[0] = self.f(p.unwrap());
        self.stage = 1;
    }
    if (self.stage == 1) {
        struct Poll<U> q = self.second[0].poll(cx);
        if (q.ready_ == 0) { q.drop(); return Poll::pending(); }
        self.stage = 2;
        return q;
    }
    return Poll::pending();
}

generic<F: Future, T, G: Future, U>
void AndThen::drop() {
    if (self.stage == 0) self.first.drop();
    if (self.stage == 1) self.second[0].drop();
    self.stage = 2;
}

} // namespace std
