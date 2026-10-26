// SafeC Standard Library — Channel built-ins runtime (see channel.h for
// the typed Chan<T> wrapper most callers want)
//
// Implements the compiler's channel built-in contract — chan_create/
// chan_send/chan_recv/chan_close, see CodeGen.cpp's "Channel built-ins"
// comment for the exact ABI this must match:
//   SafecChan_* __safec_chan_create(int capacity);
//   bool  __safec_chan_send(SafecChan_* ch, unsigned long long* val_slot);
//   bool  __safec_chan_recv(SafecChan_* ch, unsigned long long* out_slot);
//   void  __safec_chan_close(SafecChan_* ch);
// — as a bounded, blocking, multi-producer/multi-consumer queue of fixed
// 8-byte payload slots, built on std::thread.h's mutex/condvar (so it's
// cross-platform for free: pthread on POSIX, Win32 CRITICAL_SECTION/
// CONDITION_VARIABLE on Windows, both already implemented and verified
// elsewhere in this stdlib — no new platform-specific code needed here).
//
// capacity < 1 is clamped to 1 (a channel needs at least one slot; this
// is a bounded/buffered channel only, not Go's unbuffered/rendezvous
// capacity-0 semantics). chan_send blocks while the buffer is full;
// chan_recv blocks while it's empty; chan_close wakes every blocked
// sender/receiver — a blocked send() returns false once closed, and
// recv() returns false once closed *and* drained (still-buffered items
// are delivered first, matching the "close signals no more sends are
// coming" rather than "discard everything" convention most channel
// implementations use).
//
// Deliberately NOT inside 'namespace std' for the four __safec_chan_*
// functions — see bare_spawn.sc's comment for why (namespaced non-extern
// functions get name-mangled to 'std_name' and would silently fail to
// link against the literal symbol names the compiler's CodeGen emits
// calls to). Chan<T> (channel.h) calls these by their literal names.
// safeguard: runtime-symbol __safec_chan_   (what chan_create & co. compile to)
#pragma once
#include <std/sync/channel.h>
#include <std/thread.h>
#include <std/mem.h>

// struct SafecChan_ is in channel.h (Chan<T> holds one).


SafecChan_* __safec_chan_create(int requestedCap) {
    int cap = requestedCap;
    if (cap < 1) { cap = 1; }
    struct SafecChan_* ch;
    unsafe { ch = (struct SafecChan_*)std::alloc_raw_(sizeof(struct SafecChan_)); }
    if (ch == (struct SafecChan_*)0) {
        return ch;
    }
    unsafe {
        ch->buf = (unsigned long long*)std::alloc_raw_(std::checked_mul_size((unsigned long)cap, 8UL));
        ch->cap = cap;
        ch->head = 0;
        ch->tail = 0;
        ch->count = 0;
        ch->closed = 0;
        std::mutex_init(&ch->mtx);
        std::cond_init(&ch->not_empty);
        std::cond_init(&ch->not_full);
    }
    return ch;
}

bool __safec_chan_send(SafecChan_* ch, unsigned long long* valPtr) {
    if (ch == (struct SafecChan_*)0) {
        return false;
    }
    unsafe {
        std::mutex_lock(&ch->mtx);
        while (ch->count == ch->cap && ch->closed == 0) {
            std::cond_wait(&ch->not_full, &ch->mtx);
        }
        if (ch->closed != 0) {
            std::mutex_unlock(&ch->mtx);
            return false;
        }
        unsigned long long val;
        val = *valPtr;
        ch->buf[ch->tail] = val;
        ch->tail = (ch->tail + 1) % ch->cap;
        ch->count = ch->count + 1;
        std::cond_signal(&ch->not_empty);
        std::mutex_unlock(&ch->mtx);
    }
    return true;
}

bool __safec_chan_recv(SafecChan_* ch, unsigned long long* outPtr) {
    if (ch == (struct SafecChan_*)0) {
        return false;
    }
    unsafe {
        std::mutex_lock(&ch->mtx);
        while (ch->count == 0 && ch->closed == 0) {
            std::cond_wait(&ch->not_empty, &ch->mtx);
        }
        if (ch->count == 0) {
            // Closed and fully drained — no more items will ever arrive.
            std::mutex_unlock(&ch->mtx);
            return false;
        }
        unsigned long long val;
        val = ch->buf[ch->head];
        ch->head = (ch->head + 1) % ch->cap;
        ch->count = ch->count - 1;
        *outPtr = val;
        std::cond_signal(&ch->not_full);
        std::mutex_unlock(&ch->mtx);
    }
    return true;
}

void __safec_chan_close(SafecChan_* ch) {
    if (ch == (struct SafecChan_*)0) {
        return;
    }
    unsafe {
        std::mutex_lock(&ch->mtx);
        ch->closed = 1;
        std::cond_broadcast(&ch->not_empty);
        std::cond_broadcast(&ch->not_full);
        std::mutex_unlock(&ch->mtx);
    }
}

namespace std {



} // namespace std
