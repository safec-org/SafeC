// SafeC Standard Library — Typed channel wrappers
//
// The language's own channel syntax — chan_create(capacity)/chan_send(ch,
// value_ptr)/chan_recv(ch, out_ptr)/chan_close(ch) — are compiler built-ins
// (no #include needed to use them), but they're deliberately untyped at
// the compiler level: chan_create only ever takes a capacity, never an
// element size or type, so the runtime backing them (channel.sc's
// __safec_chan_* functions) has no way to know what's actually being
// sent. It picks one fixed convention instead: every channel carries
// 8-byte ("machine word") payload slots, and chan_send/chan_recv always
// copy exactly 8 bytes at *value_ptr/*out_ptr — passing a pointer to
// something smaller directly (e.g. a bare 'int') is a real
// out-of-bounds-read/write hazard, not just a style concern.
//
// chan_send_t<T>/chan_recv_t<T> below are the safe way to use a channel
// for any T that actually fits (static_assert'd at compile time, so an
// oversized T is a compile error here rather than silent corruption at
// the raw built-ins): they zero-pad into/read back from a real, fully-
// valid 8-byte local, so there's no out-of-bounds access regardless of
// T's actual size (1 to 8 bytes).
//
//   void* ch = chan_create(16);              // still the raw built-in
//   std::chan_send_t(ch, 42);                // T=int inferred from the argument
//   int v = 0;
//   int ok;
//   unsafe { ok = std::chan_recv_t(ch, (int*)&v); }
//   if (ok) { ... }
//   chan_close(ch);                          // still the raw built-in
//
// The explicit '(int*)&v' cast (inside unsafe) on the receive side is not
// optional today — generic type inference doesn't unify a plain '&v'
// reference argument against a 'T*' pointer parameter (a pre-existing
// compiler limitation, also affecting e.g. collections/vec.h's
// vec_pop_t(v, T* out) despite its own docs showing 'vec_pop_t(&v, &last)'
// with no cast; that call does not actually compile as written). Passing
// '&v' alone here fails with "cannot infer type arguments"; the explicit
// pointer cast is what makes T resolvable.
//
// For a T larger than 8 bytes, box it on the heap and send the pointer
// instead (a pointer is always exactly 8 bytes on every target this
// compiler supports, so 'chan_send_t(ch, myStructPtr)' with T inferred as
// 'MyStruct*' still fits the fixed slot size).
#pragma once

namespace std {

generic<T>
int chan_send_t(void* ch, T val);

generic<T>
int chan_recv_t(void* ch, T* out);

} // namespace std
