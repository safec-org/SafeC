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
// { magic, size } (16 bytes — 2 x 8-byte fields, so the payload after it
// stays 16-byte aligned, same as the raw malloc/calloc block, since
// malloc already guarantees at least that alignment on every hosted
// target this runs on). dealloc() checks the magic before freeing:
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
// of raw pointers, spinlocked since dealloc() is already used from
// multithreaded code elsewhere in std/, e.g. sync/channel.sc) so the
// header — and thus the double-free check — stays reliable for any
// re-free that happens within that window, which covers the common case
// (the two dealloc() calls are close together: same function, both
// branches of an if/else, a destructor invoked twice, etc). A double-free
// separated by 64+ *other* frees can still fall back to the weaker
// "neither magic" diagnosis — a known, documented limit, not a silent gap.
#define ALLOC_HDR_SIZE_          ((unsigned long)16)
#define ALLOC_LIVE_MAGIC_        ((unsigned long)0x5AFEC0DE5AFEC0DE)
#define ALLOC_FREED_MAGIC_       ((unsigned long)0xDEADDEADDEADDEAD)
#define ALLOC_QUARANTINE_SIZE_   ((unsigned long)64)

static void alloc_abort_(const char* msg) {
    unsafe { fprintf(__stderrp, "std::alloc fatal: %s\n", msg); }
    unsafe { abort(); }
}

static void* allocQuarantineRing_[64];
static unsigned long allocQuarantineHead_ = (unsigned long)0;
static int allocQuarantineLock_ = 0;

// Defers the real free() of 'raw' (the header-prefixed block, not the
// user-facing payload pointer) by pushing it into a fixed-size ring;
// whatever pointer that eviction displaces (if any — the ring starts
// zeroed, so early calls displace nothing) gets actually freed now. Same
// lock/spin shape as std/sync/spinlock.sc's Spinlock — inlined directly
// via the atomic_exchange/atomic_load/atomic_store builtins rather than
// linking that module, since this is used from mem.sc's alloc()/dealloc(),
// which is itself a dependency of nearly everything in std/.
static void alloc_quarantine_push_(void* raw) {
    unsafe {
        while (atomic_exchange(&allocQuarantineLock_, 1) != 0) {
            while (atomic_load(&allocQuarantineLock_) != 0) {}
        }
        unsigned long slot = allocQuarantineHead_ % ALLOC_QUARANTINE_SIZE_;
        void* evicted = allocQuarantineRing_[slot];
        allocQuarantineRing_[slot] = raw;
        allocQuarantineHead_ = allocQuarantineHead_ + (unsigned long)1;
        atomic_store(&allocQuarantineLock_, 0);

        if (evicted != (void*)0) {
            free(evicted);
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
        hdr[1] = size;
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
        hdr[1] = size;
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
        unsigned long* newHdr = (unsigned long*)newRaw;
        newHdr[0] = ALLOC_LIVE_MAGIC_;
        newHdr[1] = new_size;
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
