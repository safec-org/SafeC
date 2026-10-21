// std::alloc size classes, and a CPU convolution followed by large
// allocations. Sizes in (1MB, 1.5MB] used to map to a size class 33 one past
// the 33-entry cache tables, so alloc_raw_ read allocCacheCount_[33], which is
// whatever thread_local the linker put next — cnn.sc's im2col buffer pointer
// once Conv2D::forward had run, giving a garbage "cached" block and a segfault.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/thread.h>
#include <std/collections/vec.h>
#include <std/ml/cnn.h>
#include <std/sync/parallel.h>
#include <std/alloc/bump.h>
#include <std/alloc/pool.h>
#include <std/alloc/slab.h>
#include <std/alloc/tlsf.h>

static int near(float a, float b) {
    float d = a - b;
    if (d < (float)0.0) d = (float)0.0 - d;
    return d <= (float)1.0e-3;
}

// Header word 1 holds the block's size class (all-ones when uncached).
static unsigned long block_class(void* p) {
    unsafe { return ((unsigned long*)((unsigned long)p - 16UL))[1]; }
}

void t_size_classes() {
    unsigned long sizes[9] = { 1048560UL, 1048576UL, 1048577UL, 1048580UL, 1310720UL,
                               1572864UL, 1572865UL, 2097152UL, 1572864UL - 8UL };
    unsafe {
        unsigned long i = 0UL;
        while (i < 9UL) {
            void* p = alloc_raw_(sizes[i]);
            ASSERT_TRUE(p != (void*)0);
            unsigned long cls = block_class(p);
            if (sizes[i] <= 1048576UL) ASSERT_EQ(cls, 32UL);
            else ASSERT_TRUE(cls == 0xFFFFFFFFFFFFFFFFUL);
            ((char*)p)[0] = (char)1;
            ((char*)p)[sizes[i] - 1UL] = (char)2;
            dealloc_raw_(p);
            i = i + 1UL;
        }
    }
}

// One conv against a direct reference, then allocations around 1MB.
static void conv_then_alloc(unsigned long inC, unsigned long outC, unsigned long k,
                            unsigned long stride, unsigned long pad,
                            unsigned long h, unsigned long w, int relu) {
    struct Conv2D l = Conv2D::new(inC, outC, k, k, stride, pad);
    struct FeatureMap x = FeatureMap::new(inC, h, w);
    unsafe {
        unsigned long i = 0UL;
        while (i < inC * h * w) { x.data[i] = (float)((i * 7UL) % 11UL) * (float)0.1 - (float)0.5; i = i + 1UL; }
        i = 0UL;
        while (i < outC) { l.bias[i] = (float)i * (float)0.05 - (float)0.1; i = i + 1UL; }
    }
    struct FeatureMap y = relu ? l.relu_forward(&x) : l.forward(&x);
    unsigned long oh = (h + 2UL * pad - k) / stride + 1UL;
    unsigned long ow = (w + 2UL * pad - k) / stride + 1UL;
    ASSERT_EQ(y.channels, outC);
    ASSERT_EQ(y.height, oh);
    ASSERT_EQ(y.width, ow);
    unsafe {
        unsigned long oc = 0UL;
        while (oc < outC) {
            unsigned long p = 0UL;
            while (p < oh * ow) {
                float s = l.bias[oc];
                unsigned long ic = 0UL;
                while (ic < inC) {
                    unsigned long kh = 0UL;
                    while (kh < k) {
                        unsigned long kw = 0UL;
                        while (kw < k) {
                            long ih = (long)((p / ow) * stride + kh) - (long)pad;
                            long iw = (long)((p % ow) * stride + kw) - (long)pad;
                            if (ih >= 0L && iw >= 0L && (unsigned long)ih < h && (unsigned long)iw < w)
                                s = s + l.weight[((oc * inC + ic) * k + kh) * k + kw]
                                      * x.data[(ic * h + (unsigned long)ih) * w + (unsigned long)iw];
                            kw = kw + 1UL;
                        }
                        kh = kh + 1UL;
                    }
                    ic = ic + 1UL;
                }
                if (relu && s < (float)0.0) s = (float)0.0;
                ASSERT_TRUE(near(y.data[oc * oh * ow + p], s));
                p = p + 1UL;
            }
            oc = oc + 1UL;
        }
    }
    // The original repro: just over 1MB of floats, then a few more sizes,
    // each twice so the second round can come from the size-class cache.
    int round = 0;
    while (round < 2) {
        struct Vec<float> v = vec_with_cap(262145UL);
        unsigned long i = 0UL;
        while (i < 262145UL) { v.push((float)i); i = i + 1UL; }
        ASSERT_EQ(v.length(), 262145);
        ASSERT_TRUE(v.get(262144UL).default((float)-1.0) == (float)262144.0);
        v.free();
        t_size_classes();
        round = round + 1;
    }
    y.free();
    x.free();
    l.free();
}

void t_conv_then_alloc() {
    unsigned long threads[3] = { 1UL, 2UL, 4UL };
    unsafe {
        unsigned long saved = parallel_threads();
        unsigned long t = 0UL;
        while (t < 3UL) {
            parallel_set_threads(threads[t]);
            conv_then_alloc(3UL, 8UL, 3UL, 1UL, 1UL, 17UL, 13UL, 0);   // the original repro
            conv_then_alloc(4UL, 6UL, 5UL, 2UL, 0UL, 19UL, 23UL, 1);
            conv_then_alloc(16UL, 32UL, 3UL, 1UL, 1UL, 40UL, 36UL, 1); // several blocks, threaded
            conv_then_alloc(8UL, 5UL, 1UL, 1UL, 0UL, 9UL, 7UL, 0);     // 1x1 path
            conv_then_alloc(2UL, 3UL, 3UL, 1UL, 1UL, 1UL, 1UL, 0);     // one pixel
            t = t + 1UL;
        }
        parallel_set_threads(saved);
    }
}

// Slabs (blocks up to 1 KB): every size keeps its data and 16-byte payload
// alignment, freed blocks are reused, realloc moves slab blocks correctly,
// and frees from another thread come back to the owner.
#define SLAB_FLAG 0x4000000000000000UL

static void* g_far[20000];

static ?&int free_all_(?&int arg) {
    unsafe {
        unsigned long i = 0UL;
        while (i < 20000UL) { dealloc_raw_(g_far[i]); i = i + 1UL; }
    }
    return null;
}

static ?&int alloc_all_(?&int arg) {
    unsafe {
        unsigned long i = 0UL;
        while (i < 20000UL) { g_far[i] = alloc_raw_(40UL); ((int*)g_far[i])[0] = (int)i; i = i + 1UL; }
    }
    return null;
}

void t_slabs() {
    unsafe {
        unsigned long sizes[12] = { 1UL, 8UL, 16UL, 17UL, 24UL, 33UL, 100UL, 200UL, 384UL, 513UL, 768UL, 1024UL };
        void* ps[600];
        unsigned long si = 0UL;
        while (si < 12UL) {
            unsigned long sz = sizes[si];
            unsigned long i = 0UL;
            while (i < 600UL) {
                unsigned char* p = (unsigned char*)alloc_raw_(sz);
                ASSERT_TRUE(p != (unsigned char*)0);
                ASSERT_EQ((unsigned long)p % 16UL, 0UL);
                ASSERT_TRUE((block_class((void*)p) & SLAB_FLAG) != 0UL);
                unsigned long k = 0UL;
                while (k < sz) { p[k] = (unsigned char)(i + k); k = k + 1UL; }
                ps[i] = (void*)p;
                i = i + 1UL;
            }
            int bad = 0;
            i = 0UL;
            while (i < 600UL) {
                unsigned char* p = (unsigned char*)ps[i];
                unsigned long k = 0UL;
                while (k < sz) { if (p[k] != (unsigned char)(i + k)) bad = 1; k = k + 1UL; }
                i = i + 1UL;
            }
            ASSERT_EQ(bad, 0);
            // free the even ones, reallocate them: they come from those slots
            i = 0UL;
            while (i < 600UL) { dealloc_raw_(ps[i]); i = i + 2UL; }
            void* again = alloc_raw_(sz);
            int reused = 0;
            i = 0UL;
            while (i < 600UL) { if (ps[i] == again) reused = 1; i = i + 2UL; }
            ASSERT_EQ(reused, 1);
            dealloc_raw_(again);
            i = 1UL;
            while (i < 600UL) { dealloc_raw_(ps[i]); i = i + 2UL; }
            si = si + 1UL;
        }

        // realloc: slab -> larger slab -> uncached-size block, data kept
        unsigned char* r = (unsigned char*)alloc_raw_(40UL);
        unsigned long k = 0UL;
        while (k < 40UL) { r[k] = (unsigned char)(k * 3UL); k = k + 1UL; }
        r = (unsigned char*)realloc_buf_raw_((void*)r, 700UL);
        ASSERT_TRUE((block_class((void*)r) & SLAB_FLAG) != 0UL);
        r = (unsigned char*)realloc_buf_raw_((void*)r, 5000UL);
        ASSERT_TRUE(r != (unsigned char*)0);
        int same = 1;
        k = 0UL;
        while (k < 40UL) { if (r[k] != (unsigned char)(k * 3UL)) same = 0; k = k + 1UL; }
        ASSERT_EQ(same, 1);
        dealloc_raw_((void*)r);

        // blocks made here, freed on another thread, then reused here
        unsigned long i = 0UL;
        while (i < 20000UL) { g_far[i] = alloc_raw_(40UL); i = i + 1UL; }
        unsigned long lo = (unsigned long)g_far[0];
        unsigned long hi = lo;
        i = 0UL;
        while (i < 20000UL) {
            unsigned long a = (unsigned long)g_far[i];
            if (a < lo) lo = a;
            if (a > hi) hi = a;
            i = i + 1UL;
        }
        unsigned long long tid = 0ULL;
        ASSERT_EQ(thread_create(&tid, free_all_, (?&int)null), 0);
        thread_join(tid);
        int inOld = 0;
        i = 0UL;
        while (i < 20000UL) {
            unsigned long a = (unsigned long)alloc_raw_(40UL);
            ASSERT_TRUE(a != 0UL);
            if (a >= lo && a <= hi) inOld = inOld + 1;
            g_far[i] = (void*)a;
            i = i + 1UL;
        }
        ASSERT_TRUE(inOld > 10000);   // most come back from the remotely freed slabs
        i = 0UL;
        while (i < 20000UL) { dealloc_raw_(g_far[i]); i = i + 1UL; }

        // blocks made on a thread that has exited, freed here
        ASSERT_EQ(thread_create(&tid, alloc_all_, (?&int)null), 0);
        thread_join(tid);
        int okv = 1;
        i = 0UL;
        while (i < 20000UL) { if (((int*)g_far[i])[0] != (int)i) okv = 0; dealloc_raw_(g_far[i]); i = i + 1UL; }
        ASSERT_EQ(okv, 1);
    }
}

// ── std/alloc's Allocator trait ──────────────────────────────────────────────
// One generic routine run over bump, pool, slab and tlsf, and over an
// allocator defined here, which conforms just by having alloc and dealloc.

struct Rec { long long id; double v; int tag; };

// alloc_t / alloc_array_t / dealloc_t and the raw methods through the trait.
// n: how many Recs one array allocation asks for. 1 when all went well.
generic<A: Allocator>
int exercise_allocator(&A a, unsigned long n) {
    int ok = 1;
    ?&heap Rec r = alloc_t(a);
    ?&heap Rec arr = alloc_array_t(a, n);
    match (r) {
        case null: { ok = 0; }
        case some(x): {
            x.id = 42LL; x.v = 1.5; x.tag = 7;
            unsafe { if ((unsigned long)x % alignof(struct Rec) != 0UL) ok = 0; }
        }
    }
    match (arr) {
        case null: { ok = 0; }
        case some(xs): {
            unsafe {
                if ((unsigned long)xs % alignof(struct Rec) != 0UL) ok = 0;
                unsigned long i = 0UL;
                while (i < n) { xs[i].id = (long long)i; xs[i].tag = (int)i * 3; i = i + 1UL; }
                i = 0UL;
                while (i < n) { if (xs[i].id != (long long)i || xs[i].tag != (int)i * 3) ok = 0; i = i + 1UL; }
            }
        }
    }
    match (r) { case null: { } case some(x): { if (x.id != 42LL || x.tag != 7) ok = 0; dealloc_t(a, x); } }
    match (arr) { case null: { } case some(xs): { dealloc_t(a, xs); } }
    if (!a.alloc(8UL, 3UL).is_null()) ok = 0;        // align not a power of two
    a.dealloc((?&heap void)null);                      // null: a no-op everywhere
    // The raw method: 8 bytes, 8-aligned (every allocator here honours 8).
    ?&heap void raw = a.alloc(8UL, 8UL);
    match (raw) {
        case null: { ok = 0; }
        case some(p): { unsafe { if ((unsigned long)p % 8UL != 0UL) ok = 0; } a.dealloc(p); }
    }
    return ok;
}

// A user's allocator: std::alloc's heap, counting live blocks. Conforms to
// Allocator by defining alloc and dealloc with the trait's signatures.
struct CountingHeap {
    unsigned long live;
    unsigned long total;
    ?&heap void alloc(unsigned long size, unsigned long align);
    void        dealloc(?&heap void ptr);
};

?&heap void CountingHeap::alloc(unsigned long size, unsigned long align) {
    // std::alloc's blocks are 16-aligned.
    if ((align & (align - 1UL)) != 0UL || align > 16UL) return null;
    void* p;
    unsafe { p = alloc_raw_(size); }
    if (p == (void*)0) return null;
    self.live = self.live + 1UL;
    self.total = self.total + 1UL;
    unsafe { return (?&heap void)p; }
}

void CountingHeap::dealloc(?&heap void ptr) {
    void* p = ptr;
    if (p == (void*)0) return;
    unsafe { dealloc_raw_(p); }
    self.live = self.live - 1UL;
}

void t_allocator_trait() {
    struct BumpAllocator b = bump_new(4096UL);
    ASSERT_EQ(exercise_allocator(&b, 16UL), 1);
    b.destroy();

    struct PoolAllocator p = pool_new(4096UL);
    unsigned long poolFree = p.available();
    ASSERT_EQ(exercise_allocator(&p, 16UL), 1);
    ASSERT_EQ(p.available(), poolFree);              // every block came back
    p.destroy();

    // A slab hands out one object size: the array is one Rec, a second won't fit.
    struct SlabAllocator s = slab_new(sizeof(struct Rec), 4UL);
    ASSERT_EQ(exercise_allocator(&s, 1UL), 1);
    ASSERT_EQ(s.available(), 4UL);
    ?&heap Rec two = alloc_array_t(&s, 2UL);
    ASSERT_TRUE(two.is_null());
    s.destroy();

    struct TlsfAllocator t = tlsf_new(65536UL);
    ASSERT_EQ(exercise_allocator(&t, 16UL), 1);
    ?&heap Rec big = alloc_array_t(&t, 2000UL);      // fits again: all coalesced
    ASSERT_FALSE(big.is_null());
    match (big) { case null: { } case some(x): { dealloc_t(&t, x); } }
    t.destroy();

    struct CountingHeap h;
    h.live = 0UL;
    h.total = 0UL;
    ASSERT_EQ(exercise_allocator(&h, 16UL), 1);
    ASSERT_EQ(h.live, 0UL);
    ASSERT_EQ(h.total, 3UL);
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("alloc size classes around 1MB", t_size_classes);
        t.add("conv2d then large allocations", t_conv_then_alloc);
        t.add("slabs: reuse, alignment, realloc, cross-thread frees", t_slabs);
        t.add("std/alloc: one generic routine over every Allocator", t_allocator_trait);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
