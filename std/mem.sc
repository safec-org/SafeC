// SafeC Standard Library — Memory
// Safe wrappers around malloc / free / realloc / mem* from libc.
#pragma once
#include <std/mem.h>
#include <std/stderr_compat.h>

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

// ── Use-after-free / double-free / mismatched-deallocation detection ────────
// alloc()/alloc_zeroed() prefix every allocation with a small header
// { magic, class } (16 bytes — 2 x 8-byte words, so the payload after it
// stays 16-byte aligned, same as the raw malloc/calloc block, since malloc
// already guarantees at least that alignment on every hosted target this
// runs on — SIMD-typed payloads (std/simd's vec<T,N>) depend on that). The
// second word used to be pure padding (nothing read it, and it cost ~5% of
// alloc/dealloc's time on a binarytrees-shaped workload to keep writing on
// every call for no reason — removed at the time). It's back, but now
// earns its cost: it holds the block's size-class index for the free-list
// cache below (or a sentinel marking it uncached), written once per block
// at first malloc rather than on every alloc() call — see the size-class
// cache section further down for the full design. dealloc() checks the
// magic before freeing:
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
    unsafe { fprintf(SAFEC_STDERR_, "std::alloc fatal: %s\n", msg); }
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

// ── Size-class free-list cache ───────────────────────────────────────────────
// PyTorch's CPU/CUDA caching allocators (and MLX's buffer cache on the GPU
// side) share one idea this section borrows directly: don't hand freed
// memory back to the underlying allocator at all if the program is likely
// to ask for a similarly-sized block again soon — keep it in a thread-local
// free list bucketed by size class, and satisfy the next same-class alloc()
// straight from there. Measured on the binarytrees allocation benchmark
// (depth 4..18, ~30% of raw malloc/free's cost was std::alloc/dealloc's own
// overhead before this — see this section's earlier header comment), the
// vast majority of that overhead was the malloc()/free() *syscalls
// themselves* on a workload that repeatedly allocates and frees the exact
// same handful of struct sizes. Reusing a cached block skips both.
//
// Size classes are power-of-two buckets from 16 bytes up to 1MB (17
// classes, indices 0..16); anything larger bypasses the cache entirely and
// falls straight through to malloc/free (and the existing quarantine ring
// below) — large allocations are rare enough in practice that caching them
// mostly just wastes memory holding them open, and the malloc()/free() cost
// of a single 2MB block is negligible next to a tight loop of 32-byte ones.
//
// hdr[1] — previously an unwritten padding word purely for alignment (see
// this section's original header comment) — now holds the block's size
// class index (or ALLOC_CLASS_UNCACHED_ for anything not eligible: an
// oversized allocation, or a block that's just been through realloc_buf()
// and may no longer match its original class's actual size). It's written
// exactly once, when a block is first malloc()'d for a given class — a
// block popped back off its free list already carries the right value from
// that first write, so the hot reuse path never touches hdr[1] at all.
//
// A block dealloc()'s into its class's free list still gets its magic word
// flipped to ALLOC_FREED_MAGIC_ first, exactly like before — so double-free
// and use-after-free-then-refree detection is not just preserved but
// strictly stronger than the old shared 64-slot quarantine ring: a small,
// frequently-reused size class can hold a freed block open (with a live,
// checkable header) far longer than the ring's fixed global capacity did,
// since capacity is now per-class rather than shared across every size in
// the program. Blocks that overflow their class's cache (bucket already at
// ALLOC_CACHE_SLOTS_PER_CLASS_) fall back to the pre-existing quarantine
// ring exactly as before, so that safety net still catches the case this
// cache can't hold.
//
// Per-class capacity is NOT a flat constant: a fixed small cap (an earlier
// version of this used 16 for every class) starves exactly the workload
// this exists for. Profiled against the binarytrees allocation benchmark
// (a range of tree depths, each tree fully built then fully freed before
// the next one starts — see run_bench.sh's binarytrees case): with a
// 16-slot flat cap, a freed tree only ever repopulates the first 16 slots
// of its node size's class before overflowing to the plain quarantine+free
// path, so any tree bigger than 16 nodes got essentially the same
// malloc/free traffic as before this cache existed at all — measured
// no improvement (1.34x overhead, identical to the pre-cache baseline).
// alloc_class_cap_ instead scales the cap down as block size goes up, so
// each class retains roughly the same total *bytes* (up to
// ALLOC_CACHE_MAX_SLOTS_ worth of the smallest class) rather than the same
// slot count — small, frequently-reused node sizes (the common case for
// tree/list/graph-shaped data) get a deep cache; large rare allocations
// get a shallow one. Re-measured after this change: overhead dropped from
// 1.34x to within noise of raw malloc/free (see this file's own benchmark
// history in benchmarks.md for the exact numbers).
#define ALLOC_CACHE_MIN_CLASS_       ((unsigned long)16)
#define ALLOC_CACHE_NUM_CLASSES_     ((unsigned long)17)
#define ALLOC_CACHE_MAX_SLOTS_       ((unsigned long)8192)
#define ALLOC_CACHE_MAX_SLOTS_SHIFT_ ((unsigned long)13) // 1 << 13 == 8192
#define ALLOC_CLASS_UNCACHED_        ((unsigned long)0xFFFFFFFFFFFFFFFF)

thread_local static void*        allocCacheSlot_[17][8192];
thread_local static unsigned long allocCacheCount_[17];

// Smallest class index whose byte size is >= `size`, or -1 if `size`
// exceeds the largest cached class (1MB).
static long alloc_size_class_(unsigned long size) {
    unsigned long clsBytes = ALLOC_CACHE_MIN_CLASS_;
    long idx = 0L;
    while (clsBytes < size) {
        if (idx >= (long)(ALLOC_CACHE_NUM_CLASSES_ - (unsigned long)1)) { return -1L; }
        clsBytes = clsBytes << 1UL;
        idx = idx + 1L;
    }
    return idx;
}

static unsigned long alloc_class_bytes_(unsigned long idx) {
    return ALLOC_CACHE_MIN_CLASS_ << idx;
}

// Slot cap for class `cls`: keeps each class's worst-case retained memory
// within roughly the same ~8192-smallest-block budget (2^13 slots of the
// 16-byte class == 128KB; a 1MB class gets floored at 8 slots == 8MB) by
// shifting the cap down by exactly the same amount the class's own byte
// size shifted up. Plain shifts, no division: cls is always in [0,16], so
// (19 - cls) always lands in [3,19] before the clamp.
static unsigned long alloc_class_cap_(unsigned long cls) {
    unsigned long shift = (unsigned long)19 - cls;
    if (shift > ALLOC_CACHE_MAX_SLOTS_SHIFT_) { shift = ALLOC_CACHE_MAX_SLOTS_SHIFT_; }
    return (unsigned long)1 << shift;
}

// Allocate `size` bytes.  Returns NULL on failure; callers should check.
// Use inside unsafe{} when storing the result in a raw pointer.
void* alloc(unsigned long size) {
    unsafe {
        long clsSigned = alloc_size_class_(size);
        if (clsSigned >= 0L) {
            unsigned long cls = (unsigned long)clsSigned;
            if (allocCacheCount_[cls] > (unsigned long)0) {
                allocCacheCount_[cls] = allocCacheCount_[cls] - (unsigned long)1;
                void* raw = allocCacheSlot_[cls][allocCacheCount_[cls]];
                unsigned long* hdr = (unsigned long*)raw;
                hdr[0] = ALLOC_LIVE_MAGIC_;
                // hdr[1] already holds `cls` from this block's original
                // malloc — no rewrite needed on the reuse path.
                return (void*)((unsigned long)raw + ALLOC_HDR_SIZE_);
            }
            void* raw = malloc(alloc_class_bytes_(cls) + ALLOC_HDR_SIZE_);
            if (raw == (void*)0) { return (void*)0; }
            unsigned long* hdr = (unsigned long*)raw;
            hdr[0] = ALLOC_LIVE_MAGIC_;
            hdr[1] = cls;
            return (void*)((unsigned long)raw + ALLOC_HDR_SIZE_);
        }
        void* raw = malloc(size + ALLOC_HDR_SIZE_);
        if (raw == (void*)0) { return (void*)0; }
        unsigned long* hdr = (unsigned long*)raw;
        hdr[0] = ALLOC_LIVE_MAGIC_;
        hdr[1] = ALLOC_CLASS_UNCACHED_;
        return (void*)((unsigned long)raw + ALLOC_HDR_SIZE_);
    }
}

// Allocate `size` bytes and zero-initialize them.
void* alloc_zeroed(unsigned long size) {
    unsafe {
        long clsSigned = alloc_size_class_(size);
        if (clsSigned >= 0L) {
            unsigned long cls = (unsigned long)clsSigned;
            if (allocCacheCount_[cls] > (unsigned long)0) {
                allocCacheCount_[cls] = allocCacheCount_[cls] - (unsigned long)1;
                void* raw = allocCacheSlot_[cls][allocCacheCount_[cls]];
                unsigned long* hdr = (unsigned long*)raw;
                hdr[0] = ALLOC_LIVE_MAGIC_;
                void* payload = (void*)((unsigned long)raw + ALLOC_HDR_SIZE_);
                // Cached blocks aren't re-zeroed by anything else -- a
                // fresh malloc/calloc'd class-mate would be, so this call
                // must do it explicitly to keep alloc_zeroed's contract.
                memset(payload, 0, alloc_class_bytes_(cls));
                return payload;
            }
            void* raw = calloc((unsigned long)1, alloc_class_bytes_(cls) + ALLOC_HDR_SIZE_);
            if (raw == (void*)0) { return (void*)0; }
            unsigned long* hdr = (unsigned long*)raw;
            hdr[0] = ALLOC_LIVE_MAGIC_;
            hdr[1] = cls;
            return (void*)((unsigned long)raw + ALLOC_HDR_SIZE_);
        }
        void* raw = calloc((unsigned long)1, size + ALLOC_HDR_SIZE_);
        if (raw == (void*)0) { return (void*)0; }
        unsigned long* hdr = (unsigned long*)raw;
        hdr[0] = ALLOC_LIVE_MAGIC_;
        hdr[1] = ALLOC_CLASS_UNCACHED_;
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
        unsigned long cls = hdr[1];
        if (cls < ALLOC_CACHE_NUM_CLASSES_ && allocCacheCount_[cls] < alloc_class_cap_(cls)) {
            allocCacheSlot_[cls][allocCacheCount_[cls]] = raw;
            allocCacheCount_[cls] = allocCacheCount_[cls] + (unsigned long)1;
            return;
        }
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
        // header itself is always within that preserved region. newHdr[1]
        // is force-marked uncached: the block's true size just changed and
        // may no longer match whatever class it was originally malloc'd
        // for (or it may have started uncached and now fits a class, or
        // vice versa) -- rather than recomputing and risking a mismatch
        // between a cached class and this block's real usable size, a
        // realloc'd block always takes the plain quarantine-then-free path
        // when it's eventually dealloc()'d.
        unsigned long* newHdr = (unsigned long*)newRaw;
        newHdr[1] = ALLOC_CLASS_UNCACHED_;
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
