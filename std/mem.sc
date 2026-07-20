// SafeC Standard Library — Memory
// Safe wrappers around malloc / free / realloc / mem* from libc.
#pragma once
#include <std/mem.h>

// ── Explicit extern declarations for libc memory functions ───────────────────
namespace std {

extern void* malloc(unsigned long size);
extern void* calloc(unsigned long count, unsigned long size);
extern void  free(void* ptr);
extern void* realloc(void* ptr, unsigned long new_size);
extern void* memcpy(void* dst, const void* src, unsigned long n);
extern void* memmove(void* dst, const void* src, unsigned long n);
extern void* memset(void* ptr, int val, unsigned long n);
extern int   memcmp(const void* a, const void* b, unsigned long n);

// Not '#include <std/panic.h>' — that header also defines function-like
// macros (PANIC/PANIC_ASSERT) that error out in safe mode without
// --compat-preprocessor, and pulling it in here would force every
// consumer of alloc()/dealloc() (effectively the whole stdlib) to also
// link a --compat-preprocessor build of panic.sc. Diagnostic-then-abort
// inlined directly instead, matching panic_at's own hosted-mode default
// behavior (see panic.sc) without the extra link dependency.
extern void  abort();
extern int   fprintf(void* stream, const char* fmt, ...);
// macOS/BSD: stderr is a macro expanding to __stderrp, not a plain extern
// symbol (same reason std/io.sc declares it this way — see io.sc:31-36).
extern void* __stderrp;

// ── Use-after-free / double-free / mismatched-deallocation detection ────────
// alloc()/alloc_zeroed() prefix every allocation with a small header
// { magic, reserved } (16 bytes — 2 x 8-byte words, so the payload after
// it stays 16-byte aligned, same as the raw malloc/calloc block, since
// malloc already guarantees at least that alignment on every hosted
// target this runs on — SIMD-typed payloads (std/simd's vec<T,N>) depend
// on that). The second word only exists to keep that alignment; nothing
// reads it, and it used to be written on every alloc() call regardless
// (a real, measured cost: microbenchmarked at ~5% of alloc/dealloc's
// total time on a binarytrees-shaped allocation-heavy workload for a
// store nothing ever consumed) — removed. dealloc() checks the magic
// before freeing:
//   - already ALLOC_FREED_MAGIC_  → double free (or a stale/UAF pointer
//     being freed a second time) → diagnosed and aborted, not silently
//     corrupting the allocator's free list.
//   - neither magic value          → not a pointer alloc()/alloc_zeroed()
//     ever returned (wrong allocator, a stack/arena pointer, corrupted
//     memory, or an offset into the middle of a real allocation) →
//     diagnosed and aborted instead of freeing garbage.
//   - NULL                         → safe no-op, matching free(NULL).
// This catches mismatched-deallocator bugs deterministically (the exact
// bug class std/str.sc's str_dup/convert.sc's int_to_str-family used to
// risk before they were switched to call alloc() instead of raw malloc())
// and use-after-free specifically in the "freed pointer handed back to
// dealloc/realloc_buf again" shape. It does NOT catch a UAF that only
// reads/writes the payload without ever re-deallocating — that needs
// shadow-memory instrumentation (ASan-style), well beyond a fixed-size
// header's reach, and isn't attempted here.
//
// A magic header alone isn't enough to catch double-free reliably: this
// platform's malloc overwrites a freed block's first bytes with its own
// free-list bookkeeping immediately on free() (confirmed empirically —
// ALLOC_FREED_MAGIC_ was gone by the very next read), so a *second*
// dealloc() would usually see neither magic value and misreport a
// same-cost-but-differently-worded diagnosis rather than reliably
// recognizing "freed twice". alloc_quarantine_push_ below fixes this by
// deferring the real free() for a bounded number of calls (a small ring
// of raw pointers) so the header — and thus the double-free check — stays
// reliable for any re-free that happens within that window, which covers
// the common case (the two dealloc() calls are close together: same
// function, both branches of an if/else, a destructor invoked twice,
// etc). A double-free separated by 64+ *other* frees can still fall back
// to the weaker "neither magic" diagnosis — a known, documented limit,
// not a silent gap.
//
// The ring is per-thread (thread_local), not one shared global structure.
// Two earlier versions of this both serialized every thread's dealloc()
// calls against each other, just by different mechanisms:
//   1. A single global spinlock around one shared ring -- every dealloc()
//      in the process, on any thread, waited for the same lock. Measured:
//      an 8-thread allocation-heavy workload got ~15x *slower* than
//      single-threaded, not faster, with all 8 threads spinning.
//   2. A lock-free shared ring (atomic_fetch_add for the slot claim,
//      atomic_exchange for the swap, each ring slot padded to its own
//      cache line to stop false sharing between unrelated slots). This
//      removed the software lock, but a *shared* ring still means a block
//      one thread allocated is often evicted (and actually free()'d) by a
//      *different* thread than the one that allocated it -- profiled with
//      'sample' on real hardware: worker threads were spending real,
//      measurable time inside libsystem_malloc's own internal zone lock
//      (_os_unfair_lock_lock_slow, reached from _xzm_free /
//      _xzm_xzone_malloc_freelist_outlined) -- the platform allocator's
//      cross-thread free path is exactly the slow path for a per-thread-
//      cache allocator like macOS's. Removing our own lock didn't remove
//      that one.
// Giving each thread its own quarantine ring means a thread only ever
// evicts (and really free()s) blocks *it* previously quarantined -- for
// code where a thread mostly allocates and frees its own data (the common
// case, and exactly the shape of e.g. a per-thread work-stealing or
// partitioned workload), this keeps every real free() on the same thread
// that malloc'd it, which is the fast path every mainstream allocator
// optimizes for. No atomics are needed here at all: 'thread_local' means
// no other thread can ever observe or race this data, so the same
// slot-claim-then-swap logic that used to need atomic_fetch_add/
// atomic_exchange for correctness is now plain sequential code.
//
// This does NOT weaken double-free detection: the magic-value check that
// actually catches a double-free lives in the allocated block's own
// header (see dealloc() below), set the instant *any* thread calls
// dealloc() on it, independent of which thread's ring later evicts (and
// genuinely frees) the underlying memory. What per-thread rings change is
// only the bookkeeping question of *when* a quarantined block's memory
// is handed back to the platform allocator -- and if anything this makes
// the detection window longer in aggregate (up to 64 per thread now
// live at once, rather than 64 total across every thread), at the cost of
// proportionally more memory held in quarantine at any given moment.
#define ALLOC_HDR_SIZE_          ((unsigned long)16)
#define ALLOC_LIVE_MAGIC_        ((unsigned long)0x5AFEC0DE5AFEC0DE)
#define ALLOC_FREED_MAGIC_       ((unsigned long)0xDEADDEADDEADDEAD)
#define ALLOC_QUARANTINE_SIZE_   ((unsigned long)64)
// Must stay a power of two: alloc_quarantine_push_ uses '& (SIZE_ - 1)'
// instead of '% SIZE_' for the slot index specifically so the wrap is a
// single guaranteed-cheap AND regardless of optimization level — '%' by
// a compile-time-constant power of two gets strength-reduced to the same
// AND at -O2 by any competent backend, but this makes it true at -O0/-O1
// too instead of depending on that pass having run, on a call this hot
// (every single dealloc()).
#define ALLOC_QUARANTINE_MASK_   ((unsigned long)63)

static void alloc_abort_(const char* msg) {
    unsafe { fprintf(__stderrp, "std::alloc fatal: %s\n", msg); }
    unsafe { abort(); }
}

thread_local static unsigned long allocQuarantineRing_[64];
thread_local static unsigned long allocQuarantineHead_ = (unsigned long)0;

static void alloc_quarantine_push_(void* raw) {
    unsafe {
        unsigned long slot = allocQuarantineHead_ & ALLOC_QUARANTINE_MASK_;
        unsigned long evictedBits = allocQuarantineRing_[slot];
        allocQuarantineRing_[slot] = (unsigned long)raw;
        allocQuarantineHead_ = allocQuarantineHead_ + (unsigned long)1;

        if (evictedBits != (unsigned long)0) {
            free((void*)evictedBits);
        }
    }
}

// Allocate `size` bytes.  Returns NULL on failure; callers should check.
// Use inside unsafe{} when storing the result in a raw pointer.
void* alloc(unsigned long size) {
    unsafe {
        void* raw = malloc(size + ALLOC_HDR_SIZE_);
        if (raw == (void*)0) { return (void*)0; }
        unsigned long* hdr = (unsigned long*)raw;
        hdr[0] = ALLOC_LIVE_MAGIC_;
        // hdr[1] is the reserved alignment-padding word (see this
        // section's header comment) — deliberately left unwritten, not
        // an oversight.
        return (void*)((unsigned long)raw + ALLOC_HDR_SIZE_);
    }
}

// Allocate `size` bytes and zero-initialize them.
void* alloc_zeroed(unsigned long size) {
    unsafe {
        void* raw = calloc((unsigned long)1, size + ALLOC_HDR_SIZE_);
        if (raw == (void*)0) { return (void*)0; }
        unsigned long* hdr = (unsigned long*)raw;
        hdr[0] = ALLOC_LIVE_MAGIC_;
        // hdr[1] already 0 from calloc — no write needed.
        return (void*)((unsigned long)raw + ALLOC_HDR_SIZE_);
    }
}

// Free memory previously returned by alloc / alloc_zeroed / realloc_buf.
void dealloc(void* ptr) {
    unsafe {
        if (ptr == (void*)0) { return; }
        void* raw = (void*)((unsigned long)ptr - ALLOC_HDR_SIZE_);
        unsigned long* hdr = (unsigned long*)raw;
        if (hdr[0] == ALLOC_FREED_MAGIC_) {
            alloc_abort_("dealloc() called twice on the same pointer (double free)");
            return;
        }
        if (hdr[0] != ALLOC_LIVE_MAGIC_) {
            alloc_abort_("dealloc() called on a pointer alloc()/alloc_zeroed() never "
                         "returned (mismatched allocator or corrupted heap)");
            return;
        }
        hdr[0] = ALLOC_FREED_MAGIC_;
        alloc_quarantine_push_(raw);
    }
}

// Resize an allocation.  Returns NULL on failure (old block is NOT freed).
void* realloc_buf(void* ptr, unsigned long new_size) {
    unsafe {
        if (ptr == (void*)0) { return alloc(new_size); }
        void* raw = (void*)((unsigned long)ptr - ALLOC_HDR_SIZE_);
        unsigned long* hdr = (unsigned long*)raw;
        if (hdr[0] == ALLOC_FREED_MAGIC_) {
            alloc_abort_("realloc_buf() called on an already-freed pointer (use after free)");
            return (void*)0;
        }
        if (hdr[0] != ALLOC_LIVE_MAGIC_) {
            alloc_abort_("realloc_buf() called on a pointer alloc()/alloc_zeroed() never "
                         "returned (mismatched allocator or corrupted heap)");
            return (void*)0;
        }
        void* newRaw = realloc(raw, new_size + ALLOC_HDR_SIZE_);
        if (newRaw == (void*)0) { return (void*)0; }
        // newHdr[0] is already ALLOC_LIVE_MAGIC_ -- realloc() preserves
        // the original block's leading bytes (which already passed the
        // live-magic check above) up to min(old size, new size), and the
        // header itself is always within that preserved region.
        return (void*)((unsigned long)newRaw + ALLOC_HDR_SIZE_);
    }
}

unsigned long checked_mul_size(unsigned long a, unsigned long b) {
    if (a != (unsigned long)0 && b > (~(unsigned long)0) / a) {
        alloc_abort_("checked_mul_size: allocation size overflowed (count * element "
                     "size too large for this platform's unsigned long)");
        return (unsigned long)0;
    }
    return a * b;
}

unsigned long checked_add_size(unsigned long a, unsigned long b) {
    if (b > (~(unsigned long)0) - a) {
        alloc_abort_("checked_add_size: allocation size overflowed (a + b "
                     "too large for this platform's unsigned long)");
        return (unsigned long)0;
    }
    return a + b;
}

// Copy `n` bytes from `src` to `dst`.  Regions must not overlap.
inline void safe_memcpy(void* dst, const void* src, unsigned long n) {
    unsafe { memcpy(dst, src, n); }
}

// Copy `n` bytes from `src` to `dst`.  Handles overlapping regions.
inline void safe_memmove(void* dst, const void* src, unsigned long n) {
    unsafe { memmove(dst, src, n); }
}

// Fill `n` bytes starting at `ptr` with byte value `val`.
inline void safe_memset(void* ptr, int val, unsigned long n) {
    unsafe { memset(ptr, val, n); }
}

// Compare `n` bytes of `a` and `b`.
// Returns <0, 0, or >0 (same semantics as C memcmp).
inline int safe_memcmp(const void* a, const void* b, unsigned long n) {
    unsafe { return memcmp(a, b, n); }
}

// ── Cache-line helpers ────────────────────────────────────────────────────────

inline const unsigned long mem_align_up(unsigned long addr, unsigned long align) {
    unsigned long mask = align - (unsigned long)1;
    return (addr + mask) & ~mask;
}

inline const unsigned long mem_align_down(unsigned long addr, unsigned long align) {
    unsigned long mask = align - (unsigned long)1;
    return addr & ~mask;
}

inline const int mem_is_aligned(unsigned long addr, unsigned long align) {
    unsigned long mask = align - (unsigned long)1;
    if ((addr & mask) == (unsigned long)0) { return 1; }
    return 0;
}

inline void mem_prefetch(const void* addr, int write, int locality) {
    unsafe {
#ifdef __GNUC__
        __builtin_prefetch(addr, write, locality);
#else
        (void)addr; (void)write; (void)locality;
#endif
    }
}

inline void mem_zero_secure(void* ptr, unsigned long n) {
    // Volatile write-through to prevent compiler elimination — the whole
    // point of a "secure" zero (e.g. wiping a key before it goes out of
    // scope) is that it survives dead-store elimination, so the bulk word
    // path below must stay just as volatile as the byte loop it replaces.
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)ptr;
        unsigned long i = (unsigned long)0;

        // Byte-at-a-time until the write pointer is 8-byte aligned (no-op
        // if it already is), so the bulk loop never issues a misaligned
        // volatile access.
        while (i < n && mem_is_aligned((unsigned long)(p + i), (unsigned long)8) == 0) {
            p[i] = (unsigned char)0;
            i = i + (unsigned long)1;
        }

        volatile unsigned long* pw = (volatile unsigned long*)(p + i);
        unsigned long words = (n - i) / (unsigned long)8;
        unsigned long w = (unsigned long)0;
        while (w < words) {
            pw[w] = (unsigned long)0;
            w = w + (unsigned long)1;
        }
        i = i + words * (unsigned long)8;

        while (i < n) {
            p[i] = (unsigned char)0;
            i = i + (unsigned long)1;
        }
    }
}

inline void mem_clflush(const void* addr) {
    unsafe {
#ifdef __x86_64__
        asm volatile ("clflush (%0)" : : "r"(addr) : "memory");
#else
        (void)addr;
#endif
    }
}

// ── Alignment utilities ───────────────────────────────────────────────────────

inline void* mem_align_ptr(void* ptr, unsigned long align) {
    unsafe {
        unsigned long p = (unsigned long)ptr;
        unsigned long a = mem_align_up(p, align);
        return (void*)a;
    }
}

inline const int mem_fits_page(unsigned long addr, unsigned long size) {
    unsigned long page_base = mem_align_down(addr, (unsigned long)4096);
    if (addr + size <= page_base + (unsigned long)4096) { return 1; }
    return 0;
}

} // namespace std
