// SafeC Standard Library — Chan<T>: typed channel over the channel built-ins
//
//   struct Chan<int> ch = std::chan_new(16);
//   ch.send(42);                    // blocks while full; 0 once closed
//   match (ch.recv()) {             // blocks while empty; null once closed and drained
//       case some(v): ...
//       case null: ...
//   }
//   ch.close();
//
// The language's chan_create/chan_send/chan_recv/chan_close built-ins are
// untyped: the runtime (channel.sc's __safec_chan_* functions) carries
// fixed 8-byte payload slots and copies exactly 8 bytes per send/recv.
// Chan<T> fixes the element type and pads T into a full 8-byte slot, so
// nothing reads or writes past a smaller T. sizeof(T) <= 8 is checked at
// compile time; for a larger T, send a reference to it (a '?&T' fits on
// every supported target) and keep the referent alive until received.
#pragma once

// Channel runtime (defined in channel.sc, deliberately outside namespace std
// — see there); declared here because Chan<T>'s generic methods are
// instantiated in the caller's translation unit.
void* __safec_chan_create(int requestedCap);
bool  __safec_chan_send(void* chPtr, void* valPtr);
bool  __safec_chan_recv(void* chPtr, void* outPtr);
void  __safec_chan_close(void* chPtr);

// A T moves into / out of the runtime's 8-byte slot by memcpy: strict
// aliasing forbids reading a T through the slot's integer type.
#include <std/stddef.h>
extern void* memcpy(void* dst, const void* src, size_t n);   // size_t: 32-bit on wasm32 / 32-bit targets

namespace std {

generic<T>
struct Chan {
    ?&void raw;   // runtime handle from __safec_chan_create; null if creation failed

    int  send(T val);     // 1 on success, 0 if closed (or never created)
    ?T   recv();          // null once closed and drained
    void close();         // wakes every blocked sender/receiver
};

generic<T>
struct Chan<T> chan_new(int cap) {
    static_assert(sizeof(T) <= 8UL, "Chan<T>: T must fit in 8 bytes (channels carry fixed 8-byte payload slots) — send a ?&T to a larger value instead");
    struct Chan<T> c;
    c.raw = null;
    unsafe { c.raw = (?&void)__safec_chan_create(cap); }
    return c;
}

generic<T>
int Chan::send(T val) {
    if (self.raw.is_null()) return 0;
    unsigned long long slot = 0ULL;
    bool ok;
    unsafe {
        memcpy((void*)&slot, (const void*)&val, (size_t)sizeof(T));
        ok = __safec_chan_send((void*)self.raw, (void*)&slot);
    }
    return ok ? 1 : 0;
}

generic<T>
?T Chan::recv() {
    if (self.raw.is_null()) return null;
    unsigned long long slot = 0ULL;
    bool ok;
    unsafe { ok = __safec_chan_recv((void*)self.raw, (void*)&slot); }
    if (!ok) return null;
    T v[1];   // filled by memcpy, which definite-init can't see through
    unsafe { memcpy((void*)&v[0], (const void*)&slot, (size_t)sizeof(T)); }
    return v[0];
}

generic<T>
void Chan::close() {
    if (self.raw.is_null()) return;
    unsafe { __safec_chan_close((void*)self.raw); }
}

} // namespace std
