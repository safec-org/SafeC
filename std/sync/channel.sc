// SafeC Standard Library — Channel built-ins runtime (see channel.h for
// the typed chan_send_t<T>/chan_recv_t<T> wrappers most callers want)
//
// Implements the compiler's channel built-in contract — chan_create/
// chan_send/chan_recv/chan_close, see CodeGen.cpp's "Channel built-ins"
// comment for the exact ABI this must match:
//   void* __safec_chan_create(int capacity);
//   bool  __safec_chan_send(void* ch, void* val_ptr);
//   bool  __safec_chan_recv(void* ch, void* out_ptr);
//   void  __safec_chan_close(void* ch);
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
// calls to). chan_send_t/chan_recv_t (channel.h) call these by their
// literal names below and are themselves ordinary 'namespace std'
// members, same as every other generic<T> wrapper in this stdlib.
#pragma once
#include <std/sync/channel.h>
#include <std/thread.sc>
#include <std/mem.sc>

struct SafecChan_ {
    unsigned long long* buf;   // ring buffer, 'cap' 8-byte slots
    int cap;
    int head;                  // next slot to read
    int tail;                  // next slot to write
    int count;                 // pending items
    int closed;                // 0 = open, 1 = closed
    unsigned long long mtx;
    unsigned long long not_empty; // signaled when count > 0, or on close
    unsigned long long not_full;  // signaled when count < cap, or on close
};

void* __safec_chan_create(int requestedCap) {
    int cap = requestedCap;
    if (cap < 1) { cap = 1; }
    struct SafecChan_* ch;
    unsafe { ch = (struct SafecChan_*)std::alloc(sizeof(struct SafecChan_)); }
    if (ch == (struct SafecChan_*)0) {
        return (void*)0;
    }
    unsafe {
        ch->buf = (unsigned long long*)std::alloc((unsigned long)cap * 8UL);
        ch->cap = cap;
        ch->head = 0;
        ch->tail = 0;
        ch->count = 0;
        ch->closed = 0;
        std::mutex_init(&ch->mtx);
        std::cond_init(&ch->not_empty);
        std::cond_init(&ch->not_full);
    }
    return (void*)ch;
}

bool __safec_chan_send(void* chPtr, void* valPtr) {
    struct SafecChan_* ch;
    unsafe { ch = (struct SafecChan_*)chPtr; }
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
        val = *(unsigned long long*)valPtr;
        ch->buf[ch->tail] = val;
        ch->tail = (ch->tail + 1) % ch->cap;
        ch->count = ch->count + 1;
        std::cond_signal(&ch->not_empty);
        std::mutex_unlock(&ch->mtx);
    }
    return true;
}

bool __safec_chan_recv(void* chPtr, void* outPtr) {
    struct SafecChan_* ch;
    unsafe { ch = (struct SafecChan_*)chPtr; }
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
        *(unsigned long long*)outPtr = val;
        std::cond_signal(&ch->not_full);
        std::mutex_unlock(&ch->mtx);
    }
    return true;
}

void __safec_chan_close(void* chPtr) {
    struct SafecChan_* ch;
    unsafe { ch = (struct SafecChan_*)chPtr; }
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

generic<T>
int chan_send_t(void* ch, T val) {
    static_assert(sizeof(T) <= 8UL, "chan_send_t: T must fit in 8 bytes (channels carry fixed 8-byte payload slots) — box larger types on the heap and send the pointer instead");
    unsigned long long slot = 0ULL;
    unsafe { *(T*)&slot = val; }
    bool ok;
    unsafe { ok = __safec_chan_send(ch, (void*)&slot); }
    return ok ? 1 : 0;
}

generic<T>
int chan_recv_t(void* ch, T* out) {
    static_assert(sizeof(T) <= 8UL, "chan_recv_t: T must fit in 8 bytes (channels carry fixed 8-byte payload slots) — box larger types on the heap and send the pointer instead");
    unsigned long long slot = 0ULL;
    bool ok;
    unsafe { ok = __safec_chan_recv(ch, (void*)&slot); }
    if (ok) {
        unsafe { *out = *(T*)&slot; }
    }
    return ok ? 1 : 0;
}

} // namespace std
