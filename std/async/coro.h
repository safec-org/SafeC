// SafeC Standard Library — async: stackless coroutines with goto
//
// SafeC has no async fn. A coroutine here is a future (std/async/future.h)
// whose poll() is written as straight-line code with await points, and
// whose state — where to resume, and every variable that lives across an
// await — sits in the future's own struct. Each poll() jumps back to where
// the previous one stopped: a switch on the saved resume point dispatches
// to a goto label at the top level of the function body. That is exactly
// what rustc generates for an async fn (a state machine over the await
// points), written out with the preprocessor.
//
//   struct Fetch {
//       int          st;        // resume point: ASYNC_INIT before the first poll
//       struct Sleep delay;     // a child future awaited at point 1
//       int          i;         // a local that lives across awaits
//       int          total;
//       static struct Fetch new();
//       struct Poll<int> poll(&Context cx);
//       void drop();
//   };
//
//   struct Poll<int> Fetch::poll(&Context cx) {
//       ASYNC_BEGIN(self.st, 2);              // 2 await points: 1 and 2
//       self.total = 0;
//       for (self.i = 0; self.i < 3; self.i = self.i + 1) {
//           self.delay = Sleep::new(10LL);    // a fresh future for each await
//           AWAIT(1, self.delay);             // suspends until it's Ready
//           self.total = self.total + self.i;
//       }
//       ASYNC_YIELD(2);                       // let other tasks run once
//       ASYNC_RETURN(self.total);             // Ready(total); the coroutine is done
//       ASYNC_END;
//   }
//
//   void Fetch::drop() {                      // abandon it: drop what is awaited
//       if (self.st == 1) self.delay.drop();
//       self.st = ASYNC_DONE;
//   }
//
// The macros (all need --compat-preprocessor; a safeguard package sets
// '[build] compat_preprocessor = true'):
//
//   ASYNC_BEGIN(state, n)  first statement of poll(). 'state' is the int
//                          field holding the resume point (ASYNC_INIT = 0
//                          before the first poll); n is the number of await
//                          points, numbered 1..n (at most ASYNC_MAX_POINTS).
//   AWAIT(k, fut)          await point k: poll 'fut' (a future lvalue, normally
//                          a field of self) until it is Ready; its value is
//                          discarded.
//   AWAIT_INTO(k, dst, fut) the same, storing the value in 'dst' (a field).
//   ASYNC_YIELD(k)         await point k: be Pending once (after waking
//                          ourselves), so the executor runs others first.
//   ASYNC_RETURN(v)        finish with Poll::ready(v). Any number of them.
//   ASYNC_END              last statement of poll(): reaching it (no
//                          ASYNC_RETURN on that path) panics.
//
// poll()'s Context parameter must be named 'cx'.
//
// RULES — the compiler enforces the first two, the rest are yours:
//
//  1. A local that lives across an await point must be a field of the
//     state struct. A local declared after ASYNC_BEGIN in a scope that is
//     still open at a later await point is a compile error — the resuming
//     goto would jump over its declaration ("goto '__async_resume_1' jumps
//     over the declaration of 'n'"):
//
//         ASYNC_BEGIN(self.st, 1);
//         int n = compute();         // error at AWAIT(1, ...) below
//         AWAIT(1, self.delay);
//         use(n);                    // 'n' would be garbage after resuming
//
//     Fix: make it 'self.n'. A local whose block ends before the next await
//     ('{ int t = f(); self.n = t * 2; }') is fine, as are locals declared
//     *before* ASYNC_BEGIN — but those are re-initialized on every poll.
//     Loop counters too: 'for (int i = 0; ...) { AWAIT(...) }' is an error,
//     'for (self.i = 0; ...)' isn't.
//  2. Await points are numbered once each, 1..n. A duplicate number is a
//     "duplicate label" error, a number above n an undeclared identifier
//     '__async_point_k', a declared point that is never awaited an
//     "undeclared label '__async_resume_k'".
//     AWAIT inside a 'match' arm is not possible (its labels aren't goto
//     targets: "use of undeclared label"); use if/else, or switch.
//  3. 'defer' inside a coroutine runs at every suspension (each Pending
//     is a 'return'), not once at the end. Clean up explicitly before
//     ASYNC_RETURN, and in drop().
//  4. Borrows don't survive an await. Don't keep a reference to a local,
//     a stack value, or an arena region across one; keep indices or owned
//     values in the state struct. A pointer into the struct itself is only
//     valid because the future is pinned (it doesn't move once polled —
//     see future.h).
//  5. Initialize each child future right before awaiting it (as above): a
//     future is polled to completion once. Await the same field again only
//     after re-creating it.
//  6. drop(): a coroutine abandoned at await point k must drop the future
//     it was awaiting there (only that one is live). Then set the state to
//     ASYNC_DONE.
//  7. Polling a finished coroutine (state ASYNC_DONE) panics, like Rust's
//     "`async fn` resumed after completion".
#pragma once
#include <std/async/future.h>
#include <std/panic.h>

// The resume-point type. The macros declare their names with it rather
// than with 'int': the preprocessor's hygiene pass renames identifiers a
// macro body declares after a type keyword, and these must stay visible
// from one macro expansion (ASYNC_BEGIN) to the next (AWAIT).
typedef int AsyncResume;

#define ASYNC_INIT 0
#define ASYNC_DONE (-1)
#define ASYNC_MAX_POINTS 32

namespace std {
// ASYNC_RETURN's helper: marks the coroutine finished, passes 'p' through.
generic<P>
P async_finish_(&AsyncResume st, P p) {
    *st = ASYNC_DONE;
    return p;
}
// AWAIT's helper: 1 if 'p' is Ready, dropping the value it discards.
generic<T>
int async_poll_done_(struct Poll<T> p) {
    int r = p.ready_;
    p.drop();
    return r;
}
} // namespace std

#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macros: compat mode only

// One dispatch case and one marker per declared await point. The marker
// ('AsyncResume __async_point_k = k') is what AWAIT(k, ...) stores as the
// resume point, so awaiting an undeclared k fails to compile.
#define ASYNC_P_(k) AsyncResume __async_point_##k = k;
#define ASYNC_C_(k) case k: goto __async_resume_##k;

#define ASYNC_POINTS_0_
#define ASYNC_POINTS_1_  ASYNC_POINTS_0_  ASYNC_P_(1)
#define ASYNC_POINTS_2_  ASYNC_POINTS_1_  ASYNC_P_(2)
#define ASYNC_POINTS_3_  ASYNC_POINTS_2_  ASYNC_P_(3)
#define ASYNC_POINTS_4_  ASYNC_POINTS_3_  ASYNC_P_(4)
#define ASYNC_POINTS_5_  ASYNC_POINTS_4_  ASYNC_P_(5)
#define ASYNC_POINTS_6_  ASYNC_POINTS_5_  ASYNC_P_(6)
#define ASYNC_POINTS_7_  ASYNC_POINTS_6_  ASYNC_P_(7)
#define ASYNC_POINTS_8_  ASYNC_POINTS_7_  ASYNC_P_(8)
#define ASYNC_POINTS_9_  ASYNC_POINTS_8_  ASYNC_P_(9)
#define ASYNC_POINTS_10_ ASYNC_POINTS_9_  ASYNC_P_(10)
#define ASYNC_POINTS_11_ ASYNC_POINTS_10_ ASYNC_P_(11)
#define ASYNC_POINTS_12_ ASYNC_POINTS_11_ ASYNC_P_(12)
#define ASYNC_POINTS_13_ ASYNC_POINTS_12_ ASYNC_P_(13)
#define ASYNC_POINTS_14_ ASYNC_POINTS_13_ ASYNC_P_(14)
#define ASYNC_POINTS_15_ ASYNC_POINTS_14_ ASYNC_P_(15)
#define ASYNC_POINTS_16_ ASYNC_POINTS_15_ ASYNC_P_(16)
#define ASYNC_POINTS_17_ ASYNC_POINTS_16_ ASYNC_P_(17)
#define ASYNC_POINTS_18_ ASYNC_POINTS_17_ ASYNC_P_(18)
#define ASYNC_POINTS_19_ ASYNC_POINTS_18_ ASYNC_P_(19)
#define ASYNC_POINTS_20_ ASYNC_POINTS_19_ ASYNC_P_(20)
#define ASYNC_POINTS_21_ ASYNC_POINTS_20_ ASYNC_P_(21)
#define ASYNC_POINTS_22_ ASYNC_POINTS_21_ ASYNC_P_(22)
#define ASYNC_POINTS_23_ ASYNC_POINTS_22_ ASYNC_P_(23)
#define ASYNC_POINTS_24_ ASYNC_POINTS_23_ ASYNC_P_(24)
#define ASYNC_POINTS_25_ ASYNC_POINTS_24_ ASYNC_P_(25)
#define ASYNC_POINTS_26_ ASYNC_POINTS_25_ ASYNC_P_(26)
#define ASYNC_POINTS_27_ ASYNC_POINTS_26_ ASYNC_P_(27)
#define ASYNC_POINTS_28_ ASYNC_POINTS_27_ ASYNC_P_(28)
#define ASYNC_POINTS_29_ ASYNC_POINTS_28_ ASYNC_P_(29)
#define ASYNC_POINTS_30_ ASYNC_POINTS_29_ ASYNC_P_(30)
#define ASYNC_POINTS_31_ ASYNC_POINTS_30_ ASYNC_P_(31)
#define ASYNC_POINTS_32_ ASYNC_POINTS_31_ ASYNC_P_(32)

#define ASYNC_CASES_0_
#define ASYNC_CASES_1_  ASYNC_CASES_0_  ASYNC_C_(1)
#define ASYNC_CASES_2_  ASYNC_CASES_1_  ASYNC_C_(2)
#define ASYNC_CASES_3_  ASYNC_CASES_2_  ASYNC_C_(3)
#define ASYNC_CASES_4_  ASYNC_CASES_3_  ASYNC_C_(4)
#define ASYNC_CASES_5_  ASYNC_CASES_4_  ASYNC_C_(5)
#define ASYNC_CASES_6_  ASYNC_CASES_5_  ASYNC_C_(6)
#define ASYNC_CASES_7_  ASYNC_CASES_6_  ASYNC_C_(7)
#define ASYNC_CASES_8_  ASYNC_CASES_7_  ASYNC_C_(8)
#define ASYNC_CASES_9_  ASYNC_CASES_8_  ASYNC_C_(9)
#define ASYNC_CASES_10_ ASYNC_CASES_9_  ASYNC_C_(10)
#define ASYNC_CASES_11_ ASYNC_CASES_10_ ASYNC_C_(11)
#define ASYNC_CASES_12_ ASYNC_CASES_11_ ASYNC_C_(12)
#define ASYNC_CASES_13_ ASYNC_CASES_12_ ASYNC_C_(13)
#define ASYNC_CASES_14_ ASYNC_CASES_13_ ASYNC_C_(14)
#define ASYNC_CASES_15_ ASYNC_CASES_14_ ASYNC_C_(15)
#define ASYNC_CASES_16_ ASYNC_CASES_15_ ASYNC_C_(16)
#define ASYNC_CASES_17_ ASYNC_CASES_16_ ASYNC_C_(17)
#define ASYNC_CASES_18_ ASYNC_CASES_17_ ASYNC_C_(18)
#define ASYNC_CASES_19_ ASYNC_CASES_18_ ASYNC_C_(19)
#define ASYNC_CASES_20_ ASYNC_CASES_19_ ASYNC_C_(20)
#define ASYNC_CASES_21_ ASYNC_CASES_20_ ASYNC_C_(21)
#define ASYNC_CASES_22_ ASYNC_CASES_21_ ASYNC_C_(22)
#define ASYNC_CASES_23_ ASYNC_CASES_22_ ASYNC_C_(23)
#define ASYNC_CASES_24_ ASYNC_CASES_23_ ASYNC_C_(24)
#define ASYNC_CASES_25_ ASYNC_CASES_24_ ASYNC_C_(25)
#define ASYNC_CASES_26_ ASYNC_CASES_25_ ASYNC_C_(26)
#define ASYNC_CASES_27_ ASYNC_CASES_26_ ASYNC_C_(27)
#define ASYNC_CASES_28_ ASYNC_CASES_27_ ASYNC_C_(28)
#define ASYNC_CASES_29_ ASYNC_CASES_28_ ASYNC_C_(29)
#define ASYNC_CASES_30_ ASYNC_CASES_29_ ASYNC_C_(30)
#define ASYNC_CASES_31_ ASYNC_CASES_30_ ASYNC_C_(31)
#define ASYNC_CASES_32_ ASYNC_CASES_31_ ASYNC_C_(32)

// '__async_st' aliases the state field; it is declared before the
// dispatching goto, so it is in scope at every resume label.
#define ASYNC_BEGIN(state, n)                                                  \
    &AsyncResume __async_st = &(state);                                        \
    ASYNC_POINTS_##n##_                                                        \
    switch (*__async_st) {                                                     \
    case 0: break;                                                             \
    ASYNC_CASES_##n##_                                                         \
    default:                                                                   \
        std::panic_at("std/async: coroutine polled after it completed",        \
                      __FILE__, __LINE__);                                     \
        return Poll::pending();                                                \
    }

#define AWAIT(k, fut)                                                          \
    *__async_st = __async_point_##k;                                           \
__async_resume_##k:                                                            \
    if (std::async_poll_done_((fut).poll(cx)) == 0) return Poll::pending()

#define AWAIT_INTO(k, dst, fut)                                                \
    *__async_st = __async_point_##k;                                           \
__async_resume_##k:                                                            \
    {                                                                          \
        auto __async_p = (fut).poll(cx);                                       \
        if (__async_p.is_pending() != 0) { __async_p.drop(); return Poll::pending(); } \
        (dst) = __async_p.unwrap();                                            \
    }

#define ASYNC_YIELD(k)                                                         \
    *__async_st = __async_point_##k;                                           \
    cx.wake_by_ref();                                                          \
    return Poll::pending();                                                    \
__async_resume_##k:                                                            \
    { }

// One statement (usable as a match arm or an unbraced if body).
#define ASYNC_RETURN(v)                                                        \
    return std::async_finish_(__async_st, Poll::ready(v))

#define ASYNC_END                                                              \
    std::panic_at("std/async: coroutine reached ASYNC_END without ASYNC_RETURN", \
                  __FILE__, __LINE__);                                         \
    return Poll::pending()

#endif // __SAFEC_COMPAT_PREPROCESSOR__
