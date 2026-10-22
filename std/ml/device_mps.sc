// SafeC Standard Library — the Metal (Apple Silicon) device backend (see device.h).
// MpsBackend: tensor buffers are gpu_mps.h MpsBufs in shared storage, so the
// CPU and the GPU address the same memory. GEMMs run on the GPU through
// mps_gemm (falling back to sgemm on the shared memory for the forms it
// doesn't take: beta != 0, strided operands, both operands transposed), the
// optimizer step through the nn_opt_step Metal kernel, and every other
// kernel through device_ref.sc on the shared memory -- no copies, only the
// GPU work before it finished. Kernels added to gpu_mps_kernels.metal (and
// the metallib regenerated with gen_mps_metallib.sh on a Mac) can take over
// from the CPU versions one by one in MpsBackend::launch. A program enables
// it with its mps feature; link -framework Metal -framework Foundation -lobjc.
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/gpu_mps.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, size_t n);

// A device handle: the buffer, and a byte offset into it (views made by
// offset() share their base's buffer and are never freed).
struct MpsDevBuf_ {
    struct MpsBuf buf;
    unsigned long off;
    int view;
};

// Views live in a ring: a handle from offset() is used by the call it was
// made for, long before the ring comes round to it again.
#define DMPS_VIEWS 4096UL
static struct MpsDevBuf_ g_dmps_views[DMPS_VIEWS];
static unsigned long g_dmps_view_next = 0UL;

// NnOpt in gpu_mps_kernels.metal.
struct DmpsOpt_ {
    unsigned int n;
    int kind;
    unsigned int nTensors;
    float lr;
    float mu;
    float b1;
    float b2;
    float eps;
    float wd;
    float ibc1;
    float isbc2;
    float clip;
    float clampW;
};

static float* dmps_ptr_(unsigned long h) {
    h = dev_addr_(h);
    if (h == 0UL) return (float*)0;
    unsafe {
        struct MpsDevBuf_* d = (struct MpsDevBuf_*)h;
        match (d->buf.contents()) {
            case some(p): { return (float*)((unsigned char*)(float*)p + d->off); }
            case null: { return (float*)0; }
        }
    }
    return (float*)0;
}

static int dmps_finish_() {
    // ops outside a batch run and wait; inside one, end it before the CPU looks
    if (mps_batch_is_active()) mps_batch_end();
    return 1;
}

struct MpsBackend {
    int unused_;

    static int available();
    static unsigned long alloc(unsigned long bytes);
    static void release(unsigned long handle);
    static int upload(unsigned long dst, const &float src, unsigned long n);
    static int download(&float dst, unsigned long src, unsigned long n);
    static int copy(unsigned long dst, unsigned long src, unsigned long n);
    static int sync();
    static int gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                    unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                    float beta, unsigned long c, unsigned long ldc);
    static int gemm_batched(const &DevGemm g);
    static int conv(const &DevConv c);
    static int storage16();
    static int launch(int kernel, unsigned long n, const &DevArgs args);
    static unsigned long offset(unsigned long handle, unsigned long bytes);
};

int MpsBackend::available() { return mps_available(); }

unsigned long MpsBackend::alloc(unsigned long bytes) {
    struct MpsBuf b = MpsBuf::alloc((bytes + 3UL) / 4UL, 0);
    if (!b.valid()) return 0UL;
    unsafe {
        struct MpsDevBuf_* d = (struct MpsDevBuf_*)malloc(sizeof(struct MpsDevBuf_));
        if (d == (struct MpsDevBuf_*)0) { b.free(); return 0UL; }
        d->buf = b;
        d->off = 0UL;
        d->view = 0;
        return (unsigned long)d;
    }
}

void MpsBackend::release(unsigned long handle) {
    if (handle == 0UL) return;
    unsafe {
        struct MpsDevBuf_* d = (struct MpsDevBuf_*)handle;
        if (d->view) return;
        dmps_finish_();
        d->buf.free();
        free((void*)d);
    }
}

int MpsBackend::upload(unsigned long dst, const &float src, unsigned long n) {
    dmps_finish_();
    float* p = dmps_ptr_(dst);
    if (p == (float*)0) return 0;
    unsafe { memcpy((void*)p, (const void*)src, 4UL * n); }
    return 1;
}

int MpsBackend::download(&float dst, unsigned long src, unsigned long n) {
    dmps_finish_();
    float* p = dmps_ptr_(src);
    if (p == (float*)0) return 0;
    unsafe { memcpy((void*)dst, (const void*)p, 4UL * n); }
    return 1;
}

int MpsBackend::copy(unsigned long dst, unsigned long src, unsigned long n) {
    dmps_finish_();
    float* d = dmps_ptr_(dst);
    float* s = dmps_ptr_(src);
    if (d == (float*)0 || s == (float*)0) return 0;
    unsafe { memcpy((void*)d, (const void*)s, 4UL * n); }
    return 1;
}

int MpsBackend::sync() { return dmps_finish_(); }

int MpsBackend::gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                     unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                     float beta, unsigned long c, unsigned long ldc) {
    if (a == 0UL || b == 0UL || c == 0UL) return 0;
    unsafe {
        struct MpsDevBuf_* da = (struct MpsDevBuf_*)a;
        struct MpsDevBuf_* db = (struct MpsDevBuf_*)b;
        struct MpsDevBuf_* dc = (struct MpsDevBuf_*)c;
        int plain = alpha == (float)1.0 && beta == (float)0.0 && !(ta && tb) &&
                    lda == (ta ? m : k) && ldb == (tb ? k : n) && ldc == n &&
                    da->off == 0UL && db->off == 0UL && dc->off == 0UL;
        if (plain && mps_gemm(da->buf, ta, db->buf, tb, dc->buf, m, n, k)) return 1;
        dmps_finish_();
        return device_ref_gemm_(ta, tb, m, n, k, alpha, (unsigned long)dmps_ptr_(a), lda, (unsigned long)dmps_ptr_(b), ldb,
                                beta, (unsigned long)dmps_ptr_(c), ldc);
    }
}

int MpsBackend::gemm_batched(const &DevGemm g) { return device_loop_gemm_batched_(DEVICE_MPS, g); }

int MpsBackend::conv(const &DevConv c) { return 0; }
int MpsBackend::storage16() { return 0; }
int MpsBackend::launch(int kernel, unsigned long n, const &DevArgs args) {
    unsafe {
        if (kernel == DK_OPT && args.p[4] == 0UL) {
            // the optimizer step on the GPU (nn_opt_step bounds-checks n)
            struct MpsDevBuf_* w = (struct MpsDevBuf_*)args.p[0];
            struct MpsDevBuf_* g = (struct MpsDevBuf_*)args.p[1];
            struct MpsDevBuf_* m = (struct MpsDevBuf_*)args.p[2];
            struct MpsDevBuf_* v = (struct MpsDevBuf_*)args.p[3];
            if (w->off == 0UL && g->off == 0UL && m->off == 0UL && v->off == 0UL) {
                struct MpsBuf bufs[5];
                bufs[0] = w->buf; bufs[1] = g->buf; bufs[2] = m->buf; bufs[3] = v->buf; bufs[4] = g->buf;
                struct DmpsOpt_ p;
                p.n = (unsigned int)n; p.kind = (int)args.u[0]; p.nTensors = 0U;
                p.lr = args.f[0]; p.wd = args.f[1]; p.mu = args.f[2]; p.b1 = args.f[3]; p.b2 = args.f[4];
                p.eps = args.f[5]; p.ibc1 = args.f[6]; p.isbc2 = args.f[7]; p.clip = (float)0.0; p.clampW = (float)0.0;
                if (mps_dispatch("nn_opt_step", (const &MpsBuf)&bufs[0], 5UL, (const &void)(const void*)&p,
                                 sizeof(struct DmpsOpt_), (n + 255UL) / 256UL, 256UL)) return 1;
            }
        }
        if (kernel == DK_OPT_MULTI || kernel == DK_AMP_UNSCALE_MULTI) return 0;   // its table holds handles: one launch per tensor instead
        // everything else: the CPU version on the shared memory
        dmps_finish_();
        struct DevArgs h = *args;
        unsigned long i = 0UL;
        while (i < (unsigned long)DEVK_PTRS) { h.p[i] = (unsigned long)dmps_ptr_(args.p[i]); i = i + 1UL; }
        return device_ref_launch_(kernel, n, &h);
    }
}

unsigned long MpsBackend::offset(unsigned long handle, unsigned long bytes) {
    if (handle == 0UL) return 0UL;
    unsafe {
        struct MpsDevBuf_* base = (struct MpsDevBuf_*)handle;
        struct MpsDevBuf_* v = &g_dmps_views[g_dmps_view_next % DMPS_VIEWS];
        g_dmps_view_next = g_dmps_view_next + 1UL;
        v->buf = base->buf;
        v->off = base->off + bytes;
        v->view = 1;
        return (unsigned long)v;
    }
}

void device_mps_register_() {
    struct MpsBackend tag;
    tag.unused_ = 0;
    device_register_backend_(DEVICE_MPS, &tag);
}

} // namespace std
