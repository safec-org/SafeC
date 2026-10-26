// SafeC Standard Library — device registry, selection and memory cache
// (see device.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/time.h>
#include <std/str.h>

namespace std {

extern char* getenv(const char* name);
extern int   strcmp(const char* a, const char* b);
extern unsigned char* malloc(size_t size);
extern void  free(unsigned char* ptr);

static struct DeviceOps g_dev_ops[DEVICE_COUNT];
static int g_dev_linked = 0;
static int g_dev_default = DEVICE_AUTO;
static int g_dev_auto = -2;          // resolved device_auto(), -2 = not yet
static int g_dev_avail[DEVICE_COUNT];   // 0 = unknown, 1 = yes, 2 = no
static struct DeviceStats g_dev_stats[DEVICE_COUNT];

// The default device_link_backends_ (no backends), for programs whose units
// enable none: a weak definition rather than a weak undefined reference,
// which Mach-O's linker rejects unless every link passes -U for it. A unit
// with a backend feature defines its own (device.h) and, listed before the
// std archive, is the one the linker keeps.
#if !defined(SAFEC_FEATURE_ROCM) && !defined(SAFEC_FEATURE_CUDA) && !defined(SAFEC_FEATURE_MPS) && \
    !defined(SAFEC_FEATURE_VULKAN) && !defined(SAFEC_FEATURE_WEBGPU)
void device_link_backends_() __attribute__((weak)) {}
#endif

static void device_link_() {
    if (g_dev_linked) return;
    g_dev_linked = 1;
    device_link_backends_();
}

void device_register_ops_(int device, struct DeviceOps ops) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT) return;
    unsafe { g_dev_ops[device] = ops; }
}

int device_compiled(int device) {
    if (device == DEVICE_CPU) return 1;
    if (device < 0 || device >= DEVICE_COUNT) return 0;
    device_link_();
    unsafe { return g_dev_ops[device].registered; }
}

int device_available(int device) {
    if (device == DEVICE_CPU) return 1;
    if (!device_compiled(device)) return 0;
    unsafe {
        if (g_dev_avail[device] == 0) g_dev_avail[device] = g_dev_ops[device].available() ? 1 : 2;
        return g_dev_avail[device] == 1;
    }
}

static int device_from_name_(const char* s) {
    unsafe {
        if (strcmp(s, "cpu") == 0) return DEVICE_CPU;
        if (strcmp(s, "rocm") == 0 || strcmp(s, "hip") == 0) return DEVICE_ROCM;
        if (strcmp(s, "cuda") == 0) return DEVICE_CUDA;
        if (strcmp(s, "mps") == 0 || strcmp(s, "metal") == 0) return DEVICE_MPS;
        if (strcmp(s, "vulkan") == 0 || strcmp(s, "spirv") == 0) return DEVICE_VULKAN;
        if (strcmp(s, "webgpu") == 0 || strcmp(s, "wgpu") == 0) return DEVICE_WEBGPU;
    }
    return -1;
}

int device_auto() {
    if (g_dev_auto != -2) return g_dev_auto;
    int pick = -1;
    unsafe {
        char* env = getenv("SAFEC_DEVICE");
        if (env != (char*)0) {
            int d = device_from_name_((const char*)env);
            if (d >= 0 && device_available(d)) pick = d;
        }
    }
    if (pick < 0) {
        if (device_available(DEVICE_ROCM)) pick = DEVICE_ROCM;
        else if (device_available(DEVICE_CUDA)) pick = DEVICE_CUDA;
        else if (device_available(DEVICE_MPS)) pick = DEVICE_MPS;
        else if (device_available(DEVICE_VULKAN)) pick = DEVICE_VULKAN;
        else if (device_available(DEVICE_WEBGPU)) pick = DEVICE_WEBGPU;
        else pick = DEVICE_CPU;
    }
    g_dev_auto = pick;
    return pick;
}

void device_set_default(int device) { g_dev_default = device; }

int device_default() {
    if (g_dev_default >= 0 && device_available(g_dev_default)) return g_dev_default;
    return device_auto();
}

const &char device_name(int device) {
    if (device == DEVICE_ROCM) return "rocm";
    if (device == DEVICE_CUDA) return "cuda";
    if (device == DEVICE_MPS) return "mps";
    if (device == DEVICE_VULKAN) return "vulkan";
    if (device == DEVICE_WEBGPU) return "webgpu";
    return "cpu";
}

int device_sync(int device) {
    if (device == DEVICE_CPU) return 1;
    if (!device_available(device)) return 0;
    unsafe { return g_dev_ops[device].sync(); }
}

// ── Memory cache ─────────────────────────────────────────────────────────────
// Size classes of powers of two from 256 bytes. A freed block goes on its
// class's free list (a small heap array of handles) and is handed out again
// for the next request of that class, so a training loop allocates from the
// driver only during its first step.
#define DEV_CLASSES 40
#define DEV_MIN_SHIFT 8UL

struct DevFreeList {
    unsigned long* items;
    unsigned long  count;
    unsigned long  cap;
};

static struct DevFreeList g_dev_free[DEVICE_COUNT * DEV_CLASSES];
static unsigned long g_dev_reserved[DEVICE_COUNT];

static unsigned long dev_class_(unsigned long bytes) {
    unsigned long c = 0UL;
    unsigned long s = 1UL << DEV_MIN_SHIFT;
    while (s < bytes && c + 1UL < (unsigned long)DEV_CLASSES) { s = s << 1UL; c = c + 1UL; }
    return c;
}

static unsigned long dev_class_bytes_(unsigned long c) { return (1UL << DEV_MIN_SHIFT) << c; }

unsigned long device_mem_alloc_(int device, unsigned long bytes) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT || !device_available(device)) return 0UL;
    unsigned long c = dev_class_(bytes == 0UL ? 1UL : bytes);
    unsafe {
        struct DevFreeList* fl = &g_dev_free[(unsigned long)device * (unsigned long)DEV_CLASSES + c];
        if (fl->count > 0UL) {
            fl->count = fl->count - 1UL;
            return fl->items[fl->count];
        }
        unsigned long h = g_dev_ops[device].alloc(dev_class_bytes_(c));
        if (h == 0UL) {
            // out of memory: give the cached blocks back and try once more
            device_cache_clear(device);
            h = g_dev_ops[device].alloc(dev_class_bytes_(c));
        }
        if (h != 0UL) g_dev_reserved[device] = g_dev_reserved[device] + dev_class_bytes_(c);
        return h;
    }
}

void device_mem_free_(int device, unsigned long handle, unsigned long bytes) {
    handle = dev_addr_(handle);
    device_shadow_drop_(device, handle);
    if (handle == 0UL || device <= DEVICE_CPU || device >= DEVICE_COUNT) return;
    unsigned long c = dev_class_(bytes == 0UL ? 1UL : bytes);
    unsafe {
        struct DevFreeList* fl = &g_dev_free[(unsigned long)device * (unsigned long)DEV_CLASSES + c];
        if (fl->count == fl->cap) {
            unsigned long cap = fl->cap == 0UL ? 64UL : fl->cap * 2UL;
            unsigned long* items = (unsigned long*)malloc(sizeof(unsigned long) * cap);
            if (items == (unsigned long*)0) { g_dev_ops[device].release(handle); return; }
            unsigned long i = 0UL;
            while (i < fl->count) { items[i] = fl->items[i]; i = i + 1UL; }
            if (fl->items != (unsigned long*)0) free((unsigned char*)fl->items);
            fl->items = items;
            fl->cap = cap;
        }
        fl->items[fl->count] = handle;
        fl->count = fl->count + 1UL;
    }
}

void device_cache_clear(int device) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT || !device_compiled(device)) return;
    device_sync(device);
    unsafe {
        unsigned long c = 0UL;
        while (c < (unsigned long)DEV_CLASSES) {
            struct DevFreeList* fl = &g_dev_free[(unsigned long)device * (unsigned long)DEV_CLASSES + c];
            unsigned long i = 0UL;
            while (i < fl->count) { g_dev_ops[device].release(fl->items[i]); i = i + 1UL; }
            g_dev_reserved[device] = g_dev_reserved[device] - fl->count * dev_class_bytes_(c);
            fl->count = 0UL;
            c = c + 1UL;
        }
    }
}

unsigned long device_bytes_reserved(int device) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT) return 0UL;
    unsafe { return g_dev_reserved[device]; }
}

// ── Forwarding to the backend ────────────────────────────────────────────────

struct DeviceStats device_stats(int device) {
    struct DeviceStats z;
    z.launches = 0UL; z.gemms = 0UL; z.uploads = 0UL; z.downloads = 0UL; z.uploadBytes = 0UL; z.downloadBytes = 0UL;
    z.cpuLaunches = 0UL;
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT) return z;
    unsafe { return g_dev_stats[device]; }
}

void device_stats_reset(int device) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT) return;
    unsafe {
        g_dev_stats[device].launches = 0UL; g_dev_stats[device].gemms = 0UL;
        g_dev_stats[device].uploads = 0UL; g_dev_stats[device].downloads = 0UL;
        g_dev_stats[device].uploadBytes = 0UL; g_dev_stats[device].downloadBytes = 0UL;
        g_dev_stats[device].cpuLaunches = 0UL;
    }
}

void device_note_cpu_launch_(int device) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT) return;
    unsafe { g_dev_stats[device].cpuLaunches = g_dev_stats[device].cpuLaunches + 1UL; }
}

int device_upload_(int device, unsigned long dst, const &float src, unsigned long n) {
    if (!device_available(device)) return 0;
    unsafe { g_dev_stats[device].uploads = g_dev_stats[device].uploads + 1UL; g_dev_stats[device].uploadBytes = g_dev_stats[device].uploadBytes + 4UL * n; }
    unsafe { return g_dev_ops[device].upload(dev_addr_(dst), src, n); }
}

int device_download_(int device, &float dst, unsigned long src, unsigned long n) {
    if (!device_available(device)) return 0;
    unsafe { g_dev_stats[device].downloads = g_dev_stats[device].downloads + 1UL; g_dev_stats[device].downloadBytes = g_dev_stats[device].downloadBytes + 4UL * n; }
    unsafe { return g_dev_ops[device].download(dst, dev_addr_(src), n); }
}

// n elements: a raw copy between buffers of one type, else a converting one.
int device_copy_(int device, unsigned long dst, unsigned long src, unsigned long n) {
    if (!device_available(device)) return 0;
    if (dev_tag_(dst) != dev_tag_(src)) {
        struct DevArgs a = dev_args_();
        unsafe { a.p[0] = src; a.p[2] = dst; a.u[0] = 6U; }   // EW_COPY
        return device_launch_(device, DK_EW, n, &a);
    }
    unsigned long words = dev_bytes_(dev_tag_(dst), n) / 4UL;
    unsafe { return g_dev_ops[device].copy(dev_addr_(dst), dev_addr_(src), words); }
}

static int dev_gemm_raw_(int device, int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                         unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                         float beta, unsigned long c, unsigned long ldc) {
    unsafe { g_dev_stats[device].gemms = g_dev_stats[device].gemms + 1UL; }
    unsafe { return g_dev_ops[device].gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

// Split-K: a GEMM with a small output and a long k (a convolution's weight
// gradient: k = batch * pixels) leaves most of the GPU idle, so it runs as
// S GEMMs over slices of k into a scratch [S, m, n] in one batched call,
// then a sum over the slices into c.
#define DEV_SPLITK_MIN_K   4096UL
#define DEV_SPLITK_SLICE   2048UL
#define DEV_SPLITK_MAX_OUT 262144UL
#define DEV_SPLITK_SCRATCH 4194304UL

static int dev_gemm_splitk_(int device, int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                            unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                            float beta, unsigned long c, unsigned long ldc) {
    unsigned long mn = m * n;
    unsigned long big = m > n ? m : n;
    if (k < DEV_SPLITK_MIN_K || mn > DEV_SPLITK_MAX_OUT || k < 8UL * big || ldc != n) return -1;
    unsigned long S = k / DEV_SPLITK_SLICE;
    if (S > 64UL) S = 64UL;
    if (S * mn > DEV_SPLITK_SCRATCH) S = DEV_SPLITK_SCRATCH / mn;
    if (S < 2UL) return -1;
    unsigned long Ks = k / S;
    unsigned long rem = k - S * Ks;
    // a slice along k of op(a) / op(b), in elements
    unsigned long sa = ta ? Ks * lda : Ks;
    unsigned long sb = tb ? Ks : Ks * ldb;
    unsigned long scratch = device_mem_alloc_(device, sizeof(float) * S * mn);
    if (scratch == 0UL) return -1;
    struct DevGemm g;
    g.dtype = DEVICE_FP32;
    g.ctype = DEVICE_FP32;
    g.ta = ta; g.tb = tb; g.m = m; g.n = n; g.k = Ks; g.alpha = alpha;
    g.a = a; g.lda = lda; g.sa = sa; g.b = b; g.ldb = ldb; g.sb = sb;
    g.beta = (float)0.0; g.c = scratch; g.ldc = n; g.sc = mn; g.batch = S;
    int ok = device_gemm_batched_(device, &g);
    if (ok && beta != (float)1.0) {
        struct DevArgs e = dev_args_();
        unsafe {
            e.p[0] = c; e.p[2] = c;
            e.u[0] = beta == (float)0.0 ? 5U : 3U;   // EW_FILL 0 / EW_SCALE beta
            e.f[0] = beta == (float)0.0 ? (float)0.0 : beta;
        }
        ok = device_launch_(device, DK_EW, mn, &e);
    }
    if (ok) {
        struct DevArgs r = dev_args_();
        unsafe { r.p[0] = scratch; r.p[1] = c; r.u[0] = (unsigned int)mn; r.u[1] = (unsigned int)S; }
        ok = device_launch_(device, DK_COLSUM_ACC, mn, &r);
    }
    if (ok && rem != 0UL) {
        unsigned long ao = device_offset_(device, a, sizeof(float) * S * sa);
        unsigned long bo = device_offset_(device, b, sizeof(float) * S * sb);
        ok = dev_gemm_raw_(device, ta, tb, m, n, rem, alpha, ao, lda, bo, ldb, (float)1.0, c, ldc);
    }
    device_mem_free_(device, scratch, sizeof(float) * S * mn);
    return ok;
}

static int dev_gemm32_(int device, int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                       unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                       float beta, unsigned long c, unsigned long ldc) {
    int r = dev_gemm_splitk_(device, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if (r >= 0) return r;
    return dev_gemm_raw_(device, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ── 16-bit copies of marked buffers ──────────────────────────────────────────
#define DEV_SHADOWS 512
struct DevShadow_ {
    int device;
    unsigned long src;      // the marked buffer (address)
    unsigned long n;        // elements
    unsigned long tag;
    unsigned long copy;     // its copy (tagged), 0 when none yet
};
static struct DevShadow_ g_dev_shadow[DEV_SHADOWS];
static int g_dev_shadow_count = 0;

static int dev_shadow_find_(int device, unsigned long src) {
    int i = 0;
    while (i < g_dev_shadow_count) {
        unsafe { if (g_dev_shadow[i].device == device && g_dev_shadow[i].src == src) return i; }
        i = i + 1;
    }
    return -1;
}

void device_shadow_mark_(int device, unsigned long handle) {
    unsigned long src = dev_addr_(handle);
    if (src == 0UL || dev_tag_(handle) != 0UL || dev_shadow_find_(device, src) >= 0 || g_dev_shadow_count >= DEV_SHADOWS) return;
    unsafe {
        g_dev_shadow[g_dev_shadow_count].device = device;
        g_dev_shadow[g_dev_shadow_count].src = src;
        g_dev_shadow[g_dev_shadow_count].n = 0UL;
        g_dev_shadow[g_dev_shadow_count].tag = 0UL;
        g_dev_shadow[g_dev_shadow_count].copy = 0UL;
    }
    g_dev_shadow_count = g_dev_shadow_count + 1;
}

void device_shadow_drop_(int device, unsigned long handle) {
    if (g_dev_shadow_count == 0) return;
    int i = dev_shadow_find_(device, dev_addr_(handle));
    if (i < 0) return;
    unsafe {
        struct DevShadow_ e = g_dev_shadow[i];
        g_dev_shadow[i] = g_dev_shadow[g_dev_shadow_count - 1];
        g_dev_shadow_count = g_dev_shadow_count - 1;
        if (e.copy != 0UL) device_mem_free_(device, e.copy, dev_bytes_(e.tag, e.n));
    }
}

// The cached 'tag' copy of marked src (n elements), made now if missing; 0
// when src isn't marked.
static unsigned long dev_retype_(int device, unsigned long src, unsigned long n, unsigned long tag);
static unsigned long dev_shadow_get_(int device, unsigned long src, unsigned long n, unsigned long tag) {
    if (g_dev_shadow_count == 0 || dev_tag_(src) != 0UL) return 0UL;
    int i = dev_shadow_find_(device, dev_addr_(src));
    if (i < 0) return 0UL;
    unsafe {
        if (g_dev_shadow[i].copy != 0UL && g_dev_shadow[i].n == n && g_dev_shadow[i].tag == tag) return g_dev_shadow[i].copy;
        if (g_dev_shadow[i].copy != 0UL) {
            device_mem_free_(device, g_dev_shadow[i].copy, dev_bytes_(g_dev_shadow[i].tag, g_dev_shadow[i].n));
            i = dev_shadow_find_(device, dev_addr_(src));   // freeing never drops another entry, but re-find
            if (i < 0) return 0UL;
        }
        unsigned long c = dev_retype_(device, src, n, tag);
        g_dev_shadow[i].copy = c;
        g_dev_shadow[i].n = n;
        g_dev_shadow[i].tag = tag;
        return c;
    }
}

// A scratch copy of n elements of src as 'tag' (0 on failure).
static unsigned long dev_retype_(int device, unsigned long src, unsigned long n, unsigned long tag) {
    unsigned long h = device_mem_alloc_(device, dev_bytes_(tag, n));
    if (h == 0UL) return 0UL;
    h = dev_tagged_(h, tag);
    if (!device_copy_(device, h, src, n)) { device_mem_free_(device, h, dev_bytes_(tag, n)); return 0UL; }
    return h;
}

// A fresh scratch buffer of n elements of 'tag'.
static unsigned long dev_scratch_(int device, unsigned long n, unsigned long tag) {
    return dev_tagged_(device_mem_alloc_(device, dev_bytes_(tag, n)), tag);
}

static void dev_scratch_free_(int device, unsigned long h, unsigned long n) {
    if (h != 0UL) device_mem_free_(device, h, dev_bytes_(dev_tag_(h), n));
}

// Any of a, b, c may be 16-bit: the product runs in their 16-bit type (an
// fp32 operand converted to a scratch copy), accumulating in fp32 into c of
// whatever type; in fp32 when the backend has no such GEMM.
int device_gemm_(int device, int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                 unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                 float beta, unsigned long c, unsigned long ldc) {
    if (!device_available(device)) return 0;
    unsigned long xa = dev_tag_(a);
    unsigned long xb = dev_tag_(b);
    unsigned long xc = dev_tag_(c);
    if (xa == 0UL && xb == 0UL && xc == 0UL) return dev_gemm32_(device, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if (m == 0UL || n == 0UL) return 1;
    unsigned long t16 = xa != 0UL ? xa : (xb != 0UL ? xb : xc);
    unsigned long na = (ta ? k : m) * lda;
    unsigned long nb = (tb ? n : k) * ldb;
    int ok = 0;
    if (xc == 0UL || xc == t16) {
        unsigned long sa16 = xa == t16 ? 0UL : dev_shadow_get_(device, a, na, t16);
        unsigned long sb16 = xb == t16 ? 0UL : dev_shadow_get_(device, b, nb, t16);
        unsigned long a16 = xa == t16 ? a : (sa16 != 0UL ? sa16 : dev_retype_(device, a, na, t16));
        unsigned long b16 = xb == t16 ? b : (sb16 != 0UL ? sb16 : dev_retype_(device, b, nb, t16));
        if (a16 != 0UL && b16 != 0UL) {
            struct DevGemm g;
            g.dtype = (int)t16; g.ctype = (int)xc;
            g.ta = ta; g.tb = tb; g.m = m; g.n = n; g.k = k; g.alpha = alpha;
            g.a = a16; g.lda = lda; g.sa = 0UL; g.b = b16; g.ldb = ldb; g.sb = 0UL;
            g.beta = beta; g.c = c; g.ldc = ldc; g.sc = 0UL; g.batch = 1UL;
            ok = device_gemm_batched_(device, &g);
        }
        if (a16 != a && a16 != sa16) dev_scratch_free_(device, a16, na);
        if (b16 != b && b16 != sb16) dev_scratch_free_(device, b16, nb);
    }
    if (ok) return 1;
    // fp32 on widened copies
    unsigned long nc = m * ldc;
    unsigned long a32 = xa == 0UL ? a : dev_retype_(device, a, na, 0UL);
    unsigned long b32 = xb == 0UL ? b : dev_retype_(device, b, nb, 0UL);
    unsigned long c32 = xc == 0UL ? c : dev_retype_(device, c, nc, 0UL);
    if (a32 != 0UL && b32 != 0UL && c32 != 0UL) {
        ok = dev_gemm32_(device, ta, tb, m, n, k, alpha, a32, lda, b32, ldb, beta, c32, ldc);
        if (ok && c32 != c) ok = device_copy_(device, c, c32, nc);
    }
    if (a32 != a) dev_scratch_free_(device, a32, na);
    if (b32 != b) dev_scratch_free_(device, b32, nb);
    if (c32 != c) dev_scratch_free_(device, c32, nc);
    return ok;
}

int device_gemm_batched_(int device, const &DevGemm g) {
    if (!device_available(device)) return 0;
    if (g.batch == 0UL || g.m == 0UL || g.n == 0UL) return 1;
    struct DevGemm r = *g;
    r.a = dev_addr_(g.a); r.b = dev_addr_(g.b); r.c = dev_addr_(g.c);
    unsafe { g_dev_stats[device].gemms = g_dev_stats[device].gemms + 1UL; }
    unsafe { return g_dev_ops[device].gemm_batched(&r); }
}

// The convolution in element type t: operands of another type go through
// scratch copies (inputs converted, the output converted back).
static int dev_conv_as_(int device, const &DevConv c, unsigned long t) {
    unsigned long nx = c.batch * c.inC * c.inH * c.inW;
    unsigned long ny = c.batch * c.outC * c.outH * c.outW;
    unsigned long nw = c.outC * c.inC * c.kH * c.kW;
    // which operand the op writes: FWD y, BWD_DATA x, BWD_WEIGHTS w
    int wx = c.op == DEV_CONV_BWD_DATA;
    int ww = c.op == DEV_CONV_BWD_WEIGHTS;
    int wy = c.op == DEV_CONV_FWD;
    unsigned long x = c.x; unsigned long w = c.w; unsigned long y = c.y;
    unsigned long ws = 0UL;   // a cached copy of the weights (not freed here)
    int ok = 1;
    if (dev_tag_(x) != t) { x = wx ? dev_scratch_(device, nx, t) : dev_retype_(device, c.x, nx, t); ok = ok && x != 0UL; }
    if (ok && dev_tag_(w) != t) {
        if (!ww) ws = dev_shadow_get_(device, c.w, nw, t);
        w = ww ? dev_scratch_(device, nw, t) : (ws != 0UL ? ws : dev_retype_(device, c.w, nw, t));
        ok = w != 0UL;
    }
    if (ok && dev_tag_(y) != t) { y = wy ? dev_scratch_(device, ny, t) : dev_retype_(device, c.y, ny, t); ok = y != 0UL; }
    if (ok) {
        struct DevConv r = *c;
        r.dtype = (int)t;
        r.x = dev_addr_(x); r.w = dev_addr_(w); r.y = dev_addr_(y);
        unsafe { ok = g_dev_ops[device].conv(&r); }
    }
    if (ok && wx && x != c.x) ok = device_copy_(device, c.x, x, nx);
    if (ok && ww && w != c.w) ok = device_copy_(device, c.w, w, nw);
    if (ok && wy && y != c.y) ok = device_copy_(device, c.y, y, ny);
    if (x != c.x) dev_scratch_free_(device, x, nx);
    if (w != c.w && w != ws) dev_scratch_free_(device, w, nw);
    if (y != c.y) dev_scratch_free_(device, y, ny);
    return ok;
}

// Per shape, direction and 16-bit type: whether 16 bits beat fp32 (timed
// on first sight, each after an untimed run; library algorithm searches
// happen in those).
#define DEV_TUNE_MAX 1024
#define DEV_TUNE_KEY 20
static unsigned long g_dev_tune_key[DEV_TUNE_MAX * DEV_TUNE_KEY];
static int g_dev_tune_pick[DEV_TUNE_MAX];
static int g_dev_tune_count = 0;

static double dev_time_conv_(int device, const &DevConv c, unsigned long t) {
    if (!dev_conv_as_(device, c, t) || !device_sync(device)) return 1e30;
    double best = 1e30;
    int r = 0;
    while (r < 2) {
        long long t0 = time_mono_ns();
        if (!dev_conv_as_(device, c, t) || !device_sync(device)) return 1e30;
        double ms = (double)(time_mono_ns() - t0) * 1e-6;
        if (ms < best) best = ms;
        r = r + 1;
    }
    return best;
}

static int dev_conv_pick16_(int device, const &DevConv c, unsigned long t16) {
    if (device_precision_pick() == DEVICE_PICK_ALWAYS) return 1;
    unsigned long key[DEV_TUNE_KEY];
    unsafe {
        key[0] = (unsigned long)c.op; key[1] = (unsigned long)c.transpose; key[2] = c.batch; key[3] = c.inC;
        key[4] = c.inH; key[5] = c.inW; key[6] = c.outC; key[7] = c.outH; key[8] = c.outW; key[9] = c.kH;
        key[10] = c.kW; key[11] = c.sH; key[12] = c.sW; key[13] = c.pH; key[14] = c.pW; key[15] = c.dH;
        key[16] = c.dW; key[17] = t16; key[18] = (unsigned long)device;
        key[19] = dev_tag_(c.x) | (dev_tag_(c.w) << 2UL) | (dev_tag_(c.y) << 4UL);
    }
    int i = 0;
    while (i < g_dev_tune_count) {
        int same = 1;
        int j = 0;
        unsafe { while (same && j < DEV_TUNE_KEY) { if (g_dev_tune_key[i * DEV_TUNE_KEY + j] != key[j]) same = 0; j = j + 1; } }
        if (same) return g_dev_tune_pick[i];
        i = i + 1;
    }
    double t16ms = dev_time_conv_(device, c, t16);
    double t32ms = dev_time_conv_(device, c, 0UL);
    int pick = t16ms < 1e29 && t16ms <= t32ms;
    if (g_dev_tune_count < DEV_TUNE_MAX) {
        int j = 0;
        unsafe { while (j < DEV_TUNE_KEY) { g_dev_tune_key[g_dev_tune_count * DEV_TUNE_KEY + j] = key[j]; j = j + 1; } }
        g_dev_tune_pick[g_dev_tune_count] = pick;
        g_dev_tune_count = g_dev_tune_count + 1;
    }
    return pick;
}

// Operands of any element types; a 16-bit precision (or 16-bit operands)
// runs the library in 16 bits where that is faster for the shape.
int device_conv_(int device, const &DevConv c) {
    if (!device_available(device)) return 0;
    unsigned long t16 = dev_tag_(c.x);
    if (t16 == 0UL) t16 = dev_tag_(c.y);
    if (t16 == 0UL) t16 = dev_tag_(c.w);
    if (t16 == 0UL && device_store_tag_(device) != DEVICE_FP32) t16 = (unsigned long)device_store_tag_(device);
    if (t16 == 0UL) return dev_conv_as_(device, c, 0UL);
    if (dev_conv_pick16_(device, c, t16) && dev_conv_as_(device, c, t16)) return 1;
    return dev_conv_as_(device, c, 0UL);
}

int device_gemm16_(int device, int dtype, int ta, int tb, unsigned long m, unsigned long n, unsigned long k,
                   unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb, float beta,
                   unsigned long c, unsigned long ldc) {
    struct DevGemm g;
    g.dtype = dtype;
    g.ctype = DEVICE_FP32;
    g.ta = ta; g.tb = tb; g.m = m; g.n = n; g.k = k; g.alpha = (float)1.0;
    g.a = a; g.lda = lda; g.sa = 0UL; g.b = b; g.ldb = ldb; g.sb = 0UL;
    g.beta = beta; g.c = c; g.ldc = ldc; g.sc = 0UL; g.batch = 1UL;
    return device_gemm_batched_(device, &g);
}

// ── Precision and the fp16 loss scale ────────────────────────────────────────
static int g_dev_prec[DEVICE_COUNT];
static unsigned long g_dev_amp[DEVICE_COUNT];

int device_set_precision(int device, int precision) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT) return precision == DEVICE_FP32;
    if (precision < DEVICE_FP32 || precision > DEVICE_BF16) return 0;
    unsafe { g_dev_prec[device] = precision; }
    return 1;
}

static int g_dev_pick = DEVICE_PICK_TIMED;

int device_set_precision_pick(int mode) {
    int prev = g_dev_pick;
    g_dev_pick = mode == DEVICE_PICK_ALWAYS ? DEVICE_PICK_ALWAYS : DEVICE_PICK_TIMED;
    return prev;
}

int device_precision_pick() { return g_dev_pick; }

int device_precision(int device) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT) return DEVICE_FP32;
    unsafe { return g_dev_prec[device]; }
}

unsigned long device_amp_state_(int device) {
    if (device_precision(device) != DEVICE_FP16 || !device_available(device)) return 0UL;
    unsafe {
        if (g_dev_amp[device] == 0UL) {
            unsigned long h = device_mem_alloc_(device, 16UL);
            if (h == 0UL) return 0UL;
            float init[4];
            init[0] = (float)65536.0; init[1] = (float)1.0 / (float)65536.0; init[2] = (float)0.0; init[3] = (float)0.0;
            if (!device_upload_(device, h, (const &float)&init[0], 4UL)) { device_mem_free_(device, h, 16UL); return 0UL; }
            g_dev_amp[device] = h;
        }
        return g_dev_amp[device];
    }
}

float device_loss_scale(int device) {
    unsigned long h = device_amp_state_(device);
    if (h == 0UL) return (float)1.0;
    float v[4];
    unsafe { if (!device_download_(device, (&float)&v[0], h, 4UL)) return (float)1.0; }
    return v[0];
}

int device_loop_gemm_batched_(int device, const &DevGemm g) {
    if (g.dtype != DEVICE_FP32) return 0;
    unsigned long i = 0UL;
    while (i < g.batch) {
        unsigned long a = device_offset_(device, g.a, sizeof(float) * i * g.sa);
        unsigned long b = device_offset_(device, g.b, sizeof(float) * i * g.sb);
        unsigned long c = device_offset_(device, g.c, sizeof(float) * i * g.sc);
        unsafe {
            if (!g_dev_ops[device].gemm(g.ta, g.tb, g.m, g.n, g.k, g.alpha, a, g.lda, b, g.ldb, g.beta, c, g.ldc)) return 0;
        }
        i = i + 1UL;
    }
    return 1;
}

int device_launch_(int device, int kernel, unsigned long n, const &DevArgs args) {
    if (!device_available(device)) return 0;
    if (n == 0UL) return 1;
    struct DevArgs a = *args;
    unsafe { a.u[7] = (unsigned int)n; }   // the kernels' thread count
    if (kernel == DK_EW) { unsafe { kernel = device_ew_kernel_((int)a.u[0]); } }   // the op's own kernel
    unsafe { g_dev_stats[device].launches = g_dev_stats[device].launches + 1UL; }
    unsafe { return g_dev_ops[device].launch(kernel, n, &a); }
}

unsigned long device_offset_(int device, unsigned long handle, unsigned long bytes) {
    if (bytes == 0UL) return handle;
    if (!device_available(device)) return 0UL;
    unsafe { return dev_tagged_(g_dev_ops[device].offset(dev_addr_(handle), bytes), dev_tag_(handle)); }
}

unsigned long device_offset_el_(int device, unsigned long handle, unsigned long n) {
    return device_offset_(device, handle, n * dev_esize_(dev_tag_(handle)));
}

int device_store_tag_(int device) {
    int p = device_precision(device);
    if (p == DEVICE_FP32 || !device_available(device)) return DEVICE_FP32;
    unsafe { return g_dev_ops[device].storage16() ? p : DEVICE_FP32; }
}

struct DevArgs dev_args_() {
    struct DevArgs a;
    unsigned long i = 0UL;
    unsafe {
        while (i < (unsigned long)DEVK_PTRS) { a.p[i] = 0UL; i = i + 1UL; }
        i = 0UL;
        while (i < (unsigned long)DEVK_INTS) { a.u[i] = 0U; i = i + 1UL; }
        i = 0UL;
        while (i < (unsigned long)DEVK_FLOATS) { a.f[i] = (float)0.0; i = i + 1UL; }
    }
    return a;
}

} // namespace std
