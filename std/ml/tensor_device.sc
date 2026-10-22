// SafeC Standard Library — tensors on devices: keeping the host and device
// copies of a tensor's values and gradient coherent (see tensor.h's TS_*
// bits and device.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/tensor.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/float16.h>

namespace std {

static float* tdev_host_grad_(FloatTensor* t) {
    unsafe {
        if (t->grad == (float*)0) {
            float* g;
            match (t->owner) {
                case some(a): { g = (float*)tensor_arena_alloc_raw_(a, sizeof(float) * t->size); }
                case null: { g = (float*)malloc(sizeof(float) * t->size); }
            }
            __tensor_ew(EW_FILL, (const float*)0, (const float*)0, (float)0.0, g, t->size);
            t->grad = (&heap float)g;
        }
        return (float*)t->grad;
    }
}

static int tdev_fill_(int device, unsigned long buf, unsigned long n, float v) {
    struct DevArgs a = dev_args_();
    unsafe {
        a.p[2] = buf;
        a.u[0] = (unsigned int)EW_FILL;
        a.f[0] = v;
    }
    return device_launch_(device, DK_EW, n, &a);
}

// A device buffer of any element type to and from n fp32 host values.
static int tdev_down_(int device, unsigned long h, float* dst, unsigned long n) {
    unsafe {
        unsigned long tag = dev_tag_(h);
        if (tag == 0UL) return device_download_(device, (&float)dst, h, n);
        unsigned long words = dev_bytes_(tag, n) / 4UL;
        unsigned short* t = (unsigned short*)malloc(4UL * words);
        if (t == (unsigned short*)0) return 0;
        int ok = device_download_(device, (&float)(float*)t, h, words);
        if (ok) {
            if (tag == (unsigned long)DEVICE_FP16) fp16_to_f32_bulk((const &unsigned short)t, (&float)dst, n);
            else bf16_to_f32_bulk((const &unsigned short)t, (&float)dst, n);
        }
        free((void*)t);
        return ok;
    }
}

static int tdev_up_(int device, unsigned long h, const float* src, unsigned long n) {
    unsafe {
        unsigned long tag = dev_tag_(h);
        if (tag == 0UL) return device_upload_(device, h, (const &float)src, n);
        unsigned long words = dev_bytes_(tag, n) / 4UL;
        unsigned short* t = (unsigned short*)malloc(4UL * words);
        if (t == (unsigned short*)0) return 0;
        if (tag == (unsigned long)DEVICE_FP16) f32_to_fp16_bulk((const &float)src, (&unsigned short)t, n);
        else f32_to_bf16_bulk((const &float)src, (&unsigned short)t, n);
        int ok = device_upload_(device, h, (const &float)(const float*)t, words);
        free((void*)t);
        return ok;
    }
}

void tensor_host_(void* p) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (t->dev == DEVICE_CPU || (t->coh & TS_HOST) != 0 || t->dbuf == 0UL) return;
        int ok = tdev_down_(t->dev, t->dbuf, (float*)t->data, t->size);
        if (ok) t->coh = t->coh | TS_HOST;
    }
}

void tensor_host_write_(void* p) {
    tensor_host_(p);
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (t->dev != DEVICE_CPU) t->coh = (t->coh | TS_HOST) & ~TS_DEV;
    }
}

void tensor_grad_host_(void* p) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (t->dev == DEVICE_CPU || (t->coh & TS_GHOST) != 0) return;
        float* g = tdev_host_grad_(t);
        if ((t->coh & TS_GZERO) != 0) {
            __tensor_ew(EW_FILL, (const float*)0, (const float*)0, (float)0.0, g, t->size);
            t->coh = t->coh | TS_GHOST;
            return;
        }
        if (t->dgrad != 0UL && tdev_down_(t->dev, t->dgrad, g, t->size)) t->coh = t->coh | TS_GHOST;
    }
}

void tensor_grad_host_write_(void* p) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        tensor_grad_host_(p);
        tdev_host_grad_(t);   // the caller writes it: it must exist
        if (t->dev != DEVICE_CPU) t->coh = (t->coh | TS_GHOST) & ~(TS_GDEV | TS_GZERO);
    }
}

int tensor_dev_(void* p, int device) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (device == DEVICE_CPU) { tensor_host_(p); return 1; }
        if (t->dev != DEVICE_CPU && t->dev != device) {
            // another GPU: bring the values home first, then move
            tensor_host_(p);
            tensor_release_device_(p);
        }
        if (t->dbuf == 0UL) {
            t->dbuf = device_mem_alloc_(device, sizeof(float) * t->size);
            if (t->dbuf == 0UL) return 0;
            t->dev = device;
            t->coh = t->coh & ~TS_DEV;
            // the gradient (if any) is on the host for now
            t->coh = (t->coh | TS_GHOST) & ~TS_GDEV;
            if (t->grad == (float*)0) t->coh = t->coh | TS_GDEV;   // none yet: zeros everywhere
        }
        if ((t->coh & TS_DEV) == 0) {
            device_shadow_drop_(device, t->dbuf);
            int ok = tdev_up_(device, t->dbuf, (const float*)t->data, t->size);
            if (!ok) return 0;
            t->coh = t->coh | TS_DEV;
        }
        return 1;
    }
}

// fp16's unscale table on each device, uploaded again only when it changes.
#define TDEV_AMP_TABLE 512UL
static unsigned long g_tdev_amp_dev[DEVICE_COUNT];
static unsigned long g_tdev_amp_cap[DEVICE_COUNT];
static unsigned long g_tdev_amp_cnt[DEVICE_COUNT];
static unsigned long g_tdev_amp_host[DEVICE_COUNT * TDEV_AMP_TABLE * 3UL];

int tensor_amp_unscale_table_(int device, unsigned long amp, const &unsigned long tab, unsigned long cnt,
                              unsigned long total) {
    if (device <= DEVICE_CPU || device >= DEVICE_COUNT || cnt == 0UL || cnt > TDEV_AMP_TABLE) return 0;
    unsafe {
        unsigned long* host = &g_tdev_amp_host[(unsigned long)device * TDEV_AMP_TABLE * 3UL];
        int same = g_tdev_amp_cnt[device] == cnt && g_tdev_amp_dev[device] != 0UL;
        unsigned long k = 0UL;
        while (same && k < cnt * 3UL) { if (host[k] != tab[k]) same = 0; k = k + 1UL; }
        if (!same) {
            if (g_tdev_amp_cap[device] < cnt) {
                if (g_tdev_amp_dev[device] != 0UL) device_mem_free_(device, g_tdev_amp_dev[device], 24UL * g_tdev_amp_cap[device]);
                g_tdev_amp_dev[device] = device_mem_alloc_(device, 24UL * cnt);
                g_tdev_amp_cap[device] = g_tdev_amp_dev[device] != 0UL ? cnt : 0UL;
                if (g_tdev_amp_dev[device] == 0UL) { g_tdev_amp_cnt[device] = 0UL; return 0; }
            }
            if (!device_upload_(device, g_tdev_amp_dev[device], (const &float)(const float*)tab, 6UL * cnt)) {
                g_tdev_amp_cnt[device] = 0UL;
                return 0;
            }
            k = 0UL;
            while (k < cnt * 3UL) { host[k] = tab[k]; k = k + 1UL; }
            g_tdev_amp_cnt[device] = cnt;
        }
        struct DevArgs a = dev_args_();
        a.p[0] = g_tdev_amp_dev[device]; a.p[1] = amp; a.u[1] = (unsigned int)cnt;
        return device_launch_(device, DK_AMP_UNSCALE_MULTI, total, &a);
    }
}

unsigned long tensor_grad_atomic_begin_(void* p, int device) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (dev_tag_(t->dgrad) == 0UL) return t->dgrad;
        unsigned long s = device_mem_alloc_(device, sizeof(float) * t->size);
        if (s == 0UL) return 0UL;
        if (!tdev_fill_(device, s, t->size, (float)0.0)) { device_mem_free_(device, s, sizeof(float) * t->size); return 0UL; }
        return s;
    }
}

void tensor_grad_atomic_end_(void* p, int device, unsigned long target) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (target == 0UL || target == t->dgrad) return;
        struct DevArgs a = dev_args_();
        a.p[0] = target; a.p[2] = t->dgrad; a.u[0] = (unsigned int)EW_ACC;
        device_launch_(device, DK_EW, t->size, &a);
        device_mem_free_(device, target, sizeof(float) * t->size);
    }
}

// The device gradient current, leaving a zero one unwritten (TS_GZERO).
static int tdev_grad_current_(FloatTensor* t, int device) {
    unsafe {
        if (t->dev != device && !tensor_dev_((void*)t, device)) return 0;
        if (t->dgrad == 0UL) {
            // a 16-bit activation's gradient is 16-bit too
            unsigned long tag = dev_tag_(t->dbuf);
            t->dgrad = dev_tagged_(device_mem_alloc_(device, dev_bytes_(tag, t->size)), tag);
            if (t->dgrad == 0UL) return 0;
            if (t->grad == (float*)0 || (t->coh & TS_GHOST) == 0) {
                // no host gradient to bring over: zeros, written when first needed
                if (t->grad == (float*)0) t->coh = t->coh | TS_GHOST;
                t->coh = t->coh | TS_GDEV | TS_GZERO;
            } else {
                t->coh = t->coh & ~TS_GDEV;
            }
        }
        if ((t->coh & TS_GDEV) == 0) {
            if (!tdev_up_(device, t->dgrad, (const float*)t->grad, t->size)) return 0;
            t->coh = (t->coh | TS_GDEV) & ~TS_GZERO;
        }
        return 1;
    }
}

int tensor_grad_dev_(void* p, int device) {
    if (device == DEVICE_CPU) return 1;
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (!tdev_grad_current_(t, device)) return 0;
        if ((t->coh & TS_GZERO) != 0) {
            if (!tdev_fill_(device, t->dgrad, t->size, (float)0.0)) return 0;
            t->coh = t->coh & ~TS_GZERO;
        }
        return 1;
    }
}

int tensor_grad_dev_acc_(void* p, int device) {
    if (device == DEVICE_CPU) return 1;
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (!tdev_grad_current_(t, device)) return 0;
        if ((t->coh & TS_GZERO) != 0) { t->coh = t->coh & ~TS_GZERO; return 2; }
        return 1;
    }
}

void tensor_dev_wrote_(void* p) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (t->dbuf != 0UL) device_shadow_drop_(t->dev, t->dbuf);
        t->coh = (t->coh | TS_DEV) & ~TS_HOST;
    }
}

void tensor_grad_dev_wrote_(void* p) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        t->coh = (t->coh | TS_GDEV) & ~(TS_GHOST | TS_GZERO);
    }
}

// Returns the device buffers to the cache; 'keep' first brings the values
// and gradient home (a tensor moving to the CPU), else they are dropped (a
// tensor being freed).
static void tdev_release_(FloatTensor* t, int keep) {
    unsafe {
        if (t->dev == DEVICE_CPU) return;
        if (keep && (t->coh & TS_HOST) == 0 && t->dbuf != 0UL) tensor_host_((void*)t);
        if (keep && (t->coh & TS_GHOST) == 0 && t->dgrad != 0UL) tensor_grad_host_((void*)t);
        device_mem_free_(t->dev, t->dbuf, dev_bytes_(dev_tag_(t->dbuf), t->size));
        device_mem_free_(t->dev, t->dgrad, dev_bytes_(dev_tag_(t->dgrad), t->size));
        t->dbuf = 0UL;
        t->dgrad = 0UL;
        t->dev = DEVICE_CPU;
        t->coh = (t->coh | TS_HOST | TS_GHOST | TS_GDEV) & ~(TS_DEV | TS_DEVOP | TS_GZERO);
    }
}

void tensor_release_device_(void* p) { unsafe { tdev_release_((FloatTensor*)p, 1); } }

void tensor_free_device_(void* p) { unsafe { tdev_release_((FloatTensor*)p, 0); } }

int tensor_place_(void* p, int device) {
    unsafe {
        FloatTensor* t = (FloatTensor*)p;
        if (device == t->dev) return 1;
        if (device == DEVICE_CPU) { tensor_release_device_(p); return 1; }
        if (!device_available(device)) return 0;
        if (!tensor_dev_(p, device)) return 0;
        if (t->grad != (float*)0 && !tensor_grad_dev_(p, device)) return 0;
        return 1;
    }
}

int tensor_op_device_(void* a, void* b, void* c) {
    int d = DEVICE_CPU;
    unsafe {
        FloatTensor* xs[3];
        xs[0] = (FloatTensor*)a; xs[1] = (FloatTensor*)b; xs[2] = (FloatTensor*)c;
        unsigned long i = 0UL;
        while (i < 3UL) {
            FloatTensor* t = xs[i];
            if (t != (FloatTensor*)0 && t->dev != DEVICE_CPU) {
                if (d == DEVICE_CPU) d = t->dev;
                else if (d != t->dev) return DEVICE_CPU;   // two GPUs: the host decides
            }
            i = i + 1UL;
        }
    }
    return d;
}

?&FloatTensor tensor_dev_new_(const &unsigned long shape, unsigned long ndim, int device) {
    return tensor_dev_new_t_(shape, ndim, device, DEVICE_FP32);
}

?&FloatTensor tensor_dev_new_t_(const &unsigned long shape, unsigned long ndim, int device, int tag) {
    &FloatTensor out;
    unsafe { out = __tensor_alloc((const unsigned long*)shape, ndim, 0); }
    unsafe {
        out->dbuf = dev_tagged_(device_mem_alloc_(device, dev_bytes_((unsigned long)tag, out->size)), (unsigned long)tag);
        if (out->dbuf == 0UL) { out->free(); return null; }
        out->dev = device;
        out->coh = TS_DEV | TS_DEVOP | TS_GHOST | TS_GDEV;
    }
    return (?&FloatTensor)out;
}


// ══ Device ops ════════════════════════════════════════════════════════════════
// Each returns the op's result computed on the operands' device, or null when
// the operands are on the CPU or the device can't run it -- the caller then
// computes on the CPU. Results are graph nodes like the CPU ops' (TS_DEVOP:
// their backward functions below run on the device too).

static unsigned long tdev_rows_(FloatTensor* t) {
    unsafe { return t->ndim >= 2UL ? t->shape[0] : 1UL; }
}

static unsigned long tdev_cols_(FloatTensor* t) {
    unsafe { return t->shape[t->ndim - 1UL]; }
}

// Results are stored as 'tag': ops that start 16-bit activations (matmul,
// convolutions, normalizations) take the device's storage type
// (device_store_tag_), the others keep their operands' -- 16 bits when one
// is -- and losses, sums and statistics stay fp32.
static int tdev_tag1_(int device, FloatTensor* a) {
    unsafe { return dev_tag_(a->dbuf) != 0UL ? device_store_tag_(device) : DEVICE_FP32; }
}

static int tdev_tag2_(int device, FloatTensor* a, FloatTensor* b) {
    unsafe { return (dev_tag_(a->dbuf) != 0UL || dev_tag_(b->dbuf) != 0UL) ? device_store_tag_(device) : DEVICE_FP32; }
}

static ?&FloatTensor tdev_like_(FloatTensor* t, int device, int tag) {
    unsafe { return tensor_dev_new_t_((const &unsigned long)t->shape, t->ndim, device, tag); }
}

static ?&FloatTensor tdev_2d_(unsigned long rows, unsigned long cols, int device, int tag) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    return tensor_dev_new_t_((const &unsigned long)&shape[0], 2UL, device, tag);
}

static ?&FloatTensor tdev_1d_(unsigned long n, int device) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    return tensor_dev_new_((const &unsigned long)&shape[0], 1UL, device);
}

static int tdev_ew_launch_(int device, int op, unsigned long x, unsigned long y, unsigned long out, unsigned long z,
                           float k, float k2, unsigned long n) {
    struct DevArgs a = dev_args_();
    unsafe {
        a.p[0] = x; a.p[1] = y; a.p[2] = out; a.p[3] = z;
        a.u[0] = (unsigned int)op;
        a.f[0] = k; a.f[1] = k2;
    }
    return device_launch_(device, DK_EW, n, &a);
}

// An accumulating EW op into a gradient: fresh (tensor_grad_dev_acc_ gave 2)
// writes it instead.
static int tdev_ew_acc_(int device, int op, unsigned long x, unsigned long y, unsigned long out, unsigned long z,
                        float k, int fresh, unsigned long n) {
    struct DevArgs a = dev_args_();
    unsafe {
        a.p[0] = x; a.p[1] = y; a.p[2] = out; a.p[3] = z;
        a.u[0] = (unsigned int)op;
        a.u[1] = fresh ? 1U : 0U;
        a.f[0] = k;
    }
    return device_launch_(device, DK_EW, n, &a);
}

// A small host-side tensor carrying an op's constants to its backward
// function (parent 1 of the node; never on a device, never trained).
static &FloatTensor tdev_aux_(float a0, float a1, float a2, float a3) {
    unsigned long shape[1];
    unsafe { shape[0] = 4UL; }
    &FloatTensor aux;
    unsafe {
        aux = __tensor_alloc((const unsigned long*)&shape[0], 1UL, 0);
        aux->data[0] = a0; aux->data[1] = a1; aux->data[2] = a2; aux->data[3] = a3;
    }
    return aux;
}

static void tdev_drop_(FloatTensor* t) {
    unsafe {
        match (t->owner) {
            case some(a): { }
            case null: { tensor_release_device_((void*)t); t->free(); free((void*)t); }
        }
    }
}

// ── Unary elementwise ────────────────────────────────────────────────────────
// aux: [backward op, y source (0 none, 1 input, 2 output), k, z source (0 none, 2 output)]
static void tdev_unary_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        FloatTensor* aux = __tensor_parent(s, 1UL);
        if (!a->requiresGrad) return;
        int d = s->dev;
        int bop = (int)aux->data[0];
        int ysrc = (int)aux->data[1];
        int zsrc = (int)aux->data[3];
        if (ysrc == 1 && !tensor_dev_((void*)a, d)) return;
        int m = tensor_grad_dev_acc_((void*)a, d);
        if (!m) return;
        unsigned long y = ysrc == 1 ? a->dbuf : (ysrc == 2 ? s->dbuf : 0UL);
        unsigned long z = zsrc == 2 ? s->dbuf : 0UL;
        tdev_ew_acc_(d, bop, s->dgrad, y, a->dgrad, z, aux->data[2], m == 2, s->size);
        tensor_grad_dev_wrote_((void*)a);
    }
}

?&FloatTensor tdev_unary_(const &FloatTensor a, int fwd, float k, float k2, int bop, int ysrc, float bk, int zsrc) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)0, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe { if (!tensor_dev_((void*)a, d)) return null; }
    unsafe {
        match (tdev_like_((FloatTensor*)a, d, tdev_tag1_(d, (FloatTensor*)a))) {
            case some(o): {
                if (!tdev_ew_launch_(d, fwd, a->dbuf, 0UL, o->dbuf, 0UL, k, k2, a->size)) { tdev_drop_((FloatTensor*)o); return null; }
                &FloatTensor aux = tdev_aux_((float)bop, (float)ysrc, bk, (float)zsrc);
                __tensor_link2((FloatTensor*)o, (FloatTensor*)a, (FloatTensor*)aux, (void*)tdev_unary_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Binary elementwise (same shape) ──────────────────────────────────────────
// extraScalar: 0 add, 1 sub, 2 mul, 3 minimum
static void tdev_binary_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        FloatTensor* b = __tensor_parent(s, 1UL);
        int d = s->dev;
        int kind = (int)s->extraScalar;
        unsigned long n = s->size;
        if (kind >= 2) { if (!tensor_dev_((void*)a, d) || !tensor_dev_((void*)b, d)) return; }
        int ma = a->requiresGrad ? tensor_grad_dev_acc_((void*)a, d) : 0;
        if (ma) {
            int f = ma == 2;
            if (kind == 2) tdev_ew_acc_(d, EW_ACC_MUL, s->dgrad, b->dbuf, a->dgrad, 0UL, (float)0.0, f, n);
            else if (kind == 3) tdev_ew_acc_(d, EW_ACC_MINA, s->dgrad, a->dbuf, a->dgrad, b->dbuf, (float)0.0, f, n);
            else tdev_ew_acc_(d, EW_ACC, s->dgrad, 0UL, a->dgrad, 0UL, (float)0.0, f, n);
            tensor_grad_dev_wrote_((void*)a);
        }
        int mb = b->requiresGrad ? tensor_grad_dev_acc_((void*)b, d) : 0;
        if (mb) {
            int f = mb == 2;
            if (kind == 2) tdev_ew_acc_(d, EW_ACC_MUL, s->dgrad, a->dbuf, b->dgrad, 0UL, (float)0.0, f, n);
            else if (kind == 3) tdev_ew_acc_(d, EW_ACC_MINB, s->dgrad, b->dbuf, b->dgrad, a->dbuf, (float)0.0, f, n);
            else if (kind == 1) tdev_ew_acc_(d, EW_DEC, s->dgrad, 0UL, b->dgrad, 0UL, (float)0.0, f, n);
            else tdev_ew_acc_(d, EW_ACC, s->dgrad, 0UL, b->dgrad, 0UL, (float)0.0, f, n);
            tensor_grad_dev_wrote_((void*)b);
        }
    }
}

?&FloatTensor tdev_binary_(const &FloatTensor a, const &FloatTensor b, int kind) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)b, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (a->size != b->size) return null;
        if (!tensor_dev_((void*)a, d) || !tensor_dev_((void*)b, d)) return null;
        int op = kind == 0 ? EW_ADD : (kind == 1 ? EW_SUB : (kind == 2 ? EW_MUL : EW_MIN));
        match (tdev_like_((FloatTensor*)a, d, tdev_tag2_(d, (FloatTensor*)a, (FloatTensor*)b))) {
            case some(o): {
                if (!tdev_ew_launch_(d, op, a->dbuf, b->dbuf, o->dbuf, 0UL, (float)0.0, (float)0.0, a->size)) {
                    tdev_drop_((FloatTensor*)o); return null;
                }
                o->extraScalar = (float)kind;
                __tensor_link2((FloatTensor*)o, (FloatTensor*)a, (FloatTensor*)b, (void*)tdev_binary_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Sum ──────────────────────────────────────────────────────────────────────
static void tdev_sum_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        int m = a->requiresGrad ? tensor_grad_dev_acc_((void*)a, s->dev) : 0;
        if (!m) return;
        // scale (extraScalar, 1 for sum and 1/n for mean) times the seed s.grad[0]
        tdev_ew_acc_(s->dev, EW_ACC_PSCALE, 0UL, s->dgrad, a->dgrad, 0UL, s->extraScalar, m == 2, a->size);
        tensor_grad_dev_wrote_((void*)a);
    }
}

// sum (scale 1) or mean (scale 1/n) to a 1-element tensor
?&FloatTensor tdev_sum_(const &FloatTensor a, float scale) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)0, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)a, d)) return null;
        match (tdev_1d_(1UL, d)) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = a->dbuf; g.p[1] = o->dbuf;
                if (!tdev_fill_(d, o->dbuf, 1UL, (float)0.0) || !device_launch_(d, DK_SUM, a->size, &g)) { tdev_drop_((FloatTensor*)o); return null; }
                if (scale != (float)1.0) tdev_ew_launch_(d, EW_SCALE, o->dbuf, 0UL, o->dbuf, 0UL, scale, (float)0.0, 1UL);
                o->extraScalar = scale;
                __tensor_link1((FloatTensor*)o, (FloatTensor*)a, (void*)tdev_sum_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Matmul ───────────────────────────────────────────────────────────────────
static void tdev_matmul_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        FloatTensor* b = __tensor_parent(s, 1UL);
        int d = s->dev;
        unsigned long m = a->shape[0];
        unsigned long k = a->shape[1];
        unsigned long n = b->shape[1];
        // dA += dC B^T, dB += A^T dC (beta 1: accumulate)
        int ma = a->requiresGrad && tensor_dev_((void*)b, d) ? tensor_grad_dev_acc_((void*)a, d) : 0;
        if (ma) {
            device_gemm_(d, 0, 1, m, k, n, (float)1.0, s->dgrad, n, b->dbuf, n, ma == 2 ? (float)0.0 : (float)1.0, a->dgrad, k);
            tensor_grad_dev_wrote_((void*)a);
        }
        int mb = b->requiresGrad && tensor_dev_((void*)a, d) ? tensor_grad_dev_acc_((void*)b, d) : 0;
        if (mb) {
            device_gemm_(d, 1, 0, k, n, m, (float)1.0, a->dbuf, k, s->dgrad, n, mb == 2 ? (float)0.0 : (float)1.0, b->dgrad, n);
            tensor_grad_dev_wrote_((void*)b);
        }
    }
}


?&FloatTensor tdev_matmul_(const &FloatTensor a, const &FloatTensor b) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)b, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        unsigned long m = a->shape[0];
        unsigned long k = a->shape[1];
        unsigned long n = b->shape[1];
        if (!tensor_dev_((void*)a, d) || !tensor_dev_((void*)b, d)) return null;
        // a parameter (a leaf) keeps its 16-bit copy until the optimizer writes it
        if (device_store_tag_(d) != DEVICE_FP32) {
            if (a->gradFn == (void*)0) device_shadow_mark_(d, a->dbuf);
            if (b->gradFn == (void*)0) device_shadow_mark_(d, b->dbuf);
        }
        match (tdev_2d_(m, n, d, device_store_tag_(d))) {
            case some(o): {
                if (!device_gemm_(d, 0, 0, m, n, k, (float)1.0, a->dbuf, k, b->dbuf, n, (float)0.0, o->dbuf, n)) {
                    tdev_drop_((FloatTensor*)o); return null;
                }
                __tensor_link2((FloatTensor*)o, (FloatTensor*)a, (FloatTensor*)b, (void*)tdev_matmul_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// gb[c] += sum_r d[r, c] (* x[r, c] when x is set): blocks of 256 columns
// by slices of the rows.
static int tdev_colsum_(int d, unsigned long src, unsigned long dst, unsigned long x, unsigned long cols, unsigned long rows) {
    unsigned long blocks = (cols + 255UL) / 256UL;
    unsigned long S = rows / 4UL;
    if (S * blocks > 2048UL) S = 2048UL / blocks;
    if (S < 1UL) S = 1UL;
    struct DevArgs g = dev_args_();
    unsafe {
        g.p[0] = src; g.p[1] = dst; g.p[2] = x;
        g.u[0] = (unsigned int)cols; g.u[1] = (unsigned int)rows; g.u[2] = (unsigned int)S;
    }
    return device_launch_(d, DK_COLSUM_ACC, blocks * S * 256UL, &g);
}

// ── Bias add: x [m, n] + b [n] on every row ──────────────────────────────────
static void tdev_bias_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* b = __tensor_parent(s, 1UL);
        int d = s->dev;
        unsigned long n = b->size;
        int mx = x->requiresGrad ? tensor_grad_dev_acc_((void*)x, d) : 0;
        if (mx) {
            tdev_ew_acc_(d, EW_ACC, s->dgrad, 0UL, x->dgrad, 0UL, (float)0.0, mx == 2, s->size);
            tensor_grad_dev_wrote_((void*)x);
        }
        if (b->requiresGrad && tensor_grad_dev_((void*)b, d)) {
            tdev_colsum_(d, s->dgrad, b->dgrad, 0UL, n, s->size / n);
            tensor_grad_dev_wrote_((void*)b);
        }
    }
}

?&FloatTensor tdev_add_bias_(const &FloatTensor x, const &FloatTensor b) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)b, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)b, d)) return null;
        match (tdev_like_((FloatTensor*)x, d, tdev_tag2_(d, (FloatTensor*)x, (FloatTensor*)b))) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = x->dbuf; g.p[1] = b->dbuf; g.p[2] = o->dbuf;
                g.u[0] = (unsigned int)b->size;
                if (!device_launch_(d, DK_BIAS_ROWS, x->size, &g)) { tdev_drop_((FloatTensor*)o); return null; }
                __tensor_link2((FloatTensor*)o, (FloatTensor*)x, (FloatTensor*)b, (void*)tdev_bias_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Row multiply: x [m, n] * g [n] on every row ──────────────────────────────
static void tdev_mul_row_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* g = __tensor_parent(s, 1UL);
        int d = s->dev;
        unsigned long n = g->size;
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)g, d)) return;
        int mx = x->requiresGrad ? tensor_grad_dev_acc_((void*)x, d) : 0;
        if (mx) {
            struct DevArgs a = dev_args_();
            a.p[0] = s->dgrad; a.p[1] = g->dbuf; a.p[2] = x->dgrad; a.u[0] = (unsigned int)n; a.u[6] = mx == 2 ? 1U : 0U;
            device_launch_(d, DK_MUL_ROW_ACC, s->size, &a);
            tensor_grad_dev_wrote_((void*)x);
        }
        if (g->requiresGrad && tensor_grad_dev_((void*)g, d)) {
            tdev_colsum_(d, s->dgrad, g->dgrad, x->dbuf, n, s->size / n);
            tensor_grad_dev_wrote_((void*)g);
        }
    }
}

?&FloatTensor tdev_mul_row_(const &FloatTensor x, const &FloatTensor g) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)g, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)g, d)) return null;
        match (tdev_like_((FloatTensor*)x, d, tdev_tag2_(d, (FloatTensor*)x, (FloatTensor*)g))) {
            case some(o): {
                struct DevArgs a = dev_args_();
                a.p[0] = x->dbuf; a.p[1] = g->dbuf; a.p[2] = o->dbuf; a.u[0] = (unsigned int)g->size;
                if (!device_launch_(d, DK_MUL_ROW, x->size, &a)) { tdev_drop_((FloatTensor*)o); return null; }
                __tensor_link2((FloatTensor*)o, (FloatTensor*)x, (FloatTensor*)g, (void*)tdev_mul_row_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Row sums: [m, n] -> [m, 1] ───────────────────────────────────────────────
static void tdev_rowsum_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        int m = a->requiresGrad ? tensor_grad_dev_acc_((void*)a, s->dev) : 0;
        if (!m) return;
        struct DevArgs g = dev_args_();
        g.p[0] = s->dgrad; g.p[1] = a->dgrad; g.u[0] = (unsigned int)(a->size / s->size); g.u[6] = m == 2 ? 1U : 0U;
        device_launch_(s->dev, DK_ROWBCAST_ACC, a->size, &g);
        tensor_grad_dev_wrote_((void*)a);
    }
}

?&FloatTensor tdev_sum_rows_(const &FloatTensor a) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)0, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)a, d)) return null;
        unsigned long rows = tdev_rows_((FloatTensor*)a);
        unsigned long cols = tdev_cols_((FloatTensor*)a);
        match (tdev_2d_(rows, 1UL, d, tdev_tag1_(d, (FloatTensor*)a))) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = a->dbuf; g.p[1] = o->dbuf; g.u[0] = (unsigned int)cols;
                if (!device_launch_(d, DK_ROWSUM, rows, &g)) { tdev_drop_((FloatTensor*)o); return null; }
                __tensor_link1((FloatTensor*)o, (FloatTensor*)a, (void*)tdev_rowsum_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Columns: concat and slice ────────────────────────────────────────────────
static void tdev_concat_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        FloatTensor* b = __tensor_parent(s, 1UL);
        int d = s->dev;
        unsigned long rows = s->shape[0];
        unsigned long p = a->size / rows;
        unsigned long q = b->size / rows;
        // slice the wide gradient back: dk_slice with accumulate into each side
        int ma = a->requiresGrad ? tensor_grad_dev_acc_((void*)a, d) : 0;
        if (ma) {
            struct DevArgs g = dev_args_();
            g.p[0] = s->dgrad; g.p[1] = a->dgrad;
            g.u[0] = (unsigned int)(p + q); g.u[1] = 0U; g.u[2] = (unsigned int)p; g.u[3] = 2U; g.u[6] = ma == 2 ? 1U : 0U;
            device_launch_(d, DK_SLICE, rows * p, &g);
            tensor_grad_dev_wrote_((void*)a);
        }
        int mb = b->requiresGrad ? tensor_grad_dev_acc_((void*)b, d) : 0;
        if (mb) {
            struct DevArgs g = dev_args_();
            g.p[0] = s->dgrad; g.p[1] = b->dgrad;
            g.u[0] = (unsigned int)(p + q); g.u[1] = (unsigned int)p; g.u[2] = (unsigned int)q; g.u[3] = 2U; g.u[6] = mb == 2 ? 1U : 0U;
            device_launch_(d, DK_SLICE, rows * q, &g);
            tensor_grad_dev_wrote_((void*)b);
        }
    }
}

?&FloatTensor tdev_concat_cols_(const &FloatTensor a, const &FloatTensor b) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)b, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)a, d) || !tensor_dev_((void*)b, d)) return null;
        unsigned long rows = tdev_rows_((FloatTensor*)a);
        unsigned long p = tdev_cols_((FloatTensor*)a);
        unsigned long q = tdev_cols_((FloatTensor*)b);
        match (tdev_2d_(rows, p + q, d, tdev_tag2_(d, (FloatTensor*)a, (FloatTensor*)b))) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = a->dbuf; g.p[1] = b->dbuf; g.p[2] = o->dbuf;
                g.u[0] = (unsigned int)p; g.u[1] = (unsigned int)q;
                if (!device_launch_(d, DK_CONCAT, rows * (p + q), &g)) { tdev_drop_((FloatTensor*)o); return null; }
                __tensor_link2((FloatTensor*)o, (FloatTensor*)a, (FloatTensor*)b, (void*)tdev_concat_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

static void tdev_slice_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        if (!a->requiresGrad || !tensor_grad_dev_((void*)a, s->dev)) return;
        struct DevArgs g = dev_args_();
        g.p[0] = s->dgrad; g.p[1] = a->dgrad;
        g.u[0] = (unsigned int)a->shape[1]; g.u[1] = (unsigned int)s->extraScalar; g.u[2] = (unsigned int)s->shape[1];
        g.u[3] = 1U;
        device_launch_(s->dev, DK_SLICE, s->size, &g);
        tensor_grad_dev_wrote_((void*)a);
    }
}

?&FloatTensor tdev_slice_cols_(const &FloatTensor a, unsigned long start, unsigned long len) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)0, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)a, d)) return null;
        unsigned long rows = a->shape[0];
        match (tdev_2d_(rows, len, d, tdev_tag1_(d, (FloatTensor*)a))) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = a->dbuf; g.p[1] = o->dbuf;
                g.u[0] = (unsigned int)a->shape[1]; g.u[1] = (unsigned int)start; g.u[2] = (unsigned int)len; g.u[3] = 0U;
                if (!device_launch_(d, DK_SLICE, rows * len, &g)) { tdev_drop_((FloatTensor*)o); return null; }
                o->extraScalar = (float)start;
                __tensor_link1((FloatTensor*)o, (FloatTensor*)a, (void*)tdev_slice_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Row gather (embeddings, codebooks) ───────────────────────────────────────
static void tdev_gather_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* table = __tensor_parent(s, 0UL);
        FloatTensor* idx = __tensor_parent(s, 1UL);
        int d = s->dev;
        if (!table->requiresGrad || !tensor_dev_((void*)idx, d) || !tensor_grad_dev_((void*)table, d)) return;
        struct DevArgs g = dev_args_();
        unsigned long tgt = tensor_grad_atomic_begin_((void*)table, d);
        if (tgt == 0UL) return;
        g.p[0] = s->dgrad; g.p[1] = idx->dbuf; g.p[2] = tgt; g.u[0] = (unsigned int)table->shape[1];
        device_launch_(d, DK_SCATTER_ROWS, s->size, &g);
        tensor_grad_atomic_end_((void*)table, d, tgt);
        tensor_grad_dev_wrote_((void*)table);
    }
}

?&FloatTensor tdev_gather_rows_(const &FloatTensor table, const &FloatTensor idx) {
    int d;
    unsafe { d = tensor_op_device_((void*)table, (void*)idx, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)table, d) || !tensor_dev_((void*)idx, d)) return null;
        unsigned long dim = table->shape[1];
        unsigned long m = idx->size;
        match (tdev_2d_(m, dim, d, tdev_tag1_(d, (FloatTensor*)table))) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = table->dbuf; g.p[1] = idx->dbuf; g.p[2] = o->dbuf; g.u[0] = (unsigned int)dim;
                if (!device_launch_(d, DK_GATHER_ROWS, m * dim, &g)) { tdev_drop_((FloatTensor*)o); return null; }
                __tensor_link2((FloatTensor*)o, (FloatTensor*)table, (FloatTensor*)idx, (void*)tdev_gather_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Softmax / log-softmax over the last axis ─────────────────────────────────
static void tdev_softmax_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        if (!a->requiresGrad || !tensor_grad_dev_((void*)a, s->dev)) return;
        unsigned long cols = s->shape[s->ndim - 1UL];
        struct DevArgs g = dev_args_();
        g.p[0] = s->dgrad; g.p[1] = s->dbuf; g.p[2] = a->dgrad;
        g.u[0] = (unsigned int)cols; g.u[1] = (unsigned int)s->extraScalar;
        device_launch_(s->dev, DK_SOFTMAX_BWD, s->size / cols, &g);
        tensor_grad_dev_wrote_((void*)a);
    }
}

?&FloatTensor tdev_softmax_(const &FloatTensor a, int logMode) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)0, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)a, d)) return null;
        unsigned long cols = a->shape[a->ndim - 1UL];
        match (tdev_like_((FloatTensor*)a, d, tdev_tag1_(d, (FloatTensor*)a))) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = a->dbuf; g.p[1] = o->dbuf; g.u[0] = (unsigned int)cols; g.u[1] = (unsigned int)logMode;
                if (!device_launch_(d, DK_SOFTMAX_ROWS, a->size / cols, &g)) { tdev_drop_((FloatTensor*)o); return null; }
                o->extraScalar = (float)logMode;
                __tensor_link1((FloatTensor*)o, (FloatTensor*)a, (void*)tdev_softmax_backward_);
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Fused losses ─────────────────────────────────────────────────────────────
// kind: 0 cross-entropy (t = class indices), 1 BCE on logits, 2 Huber (k = delta)
static void tdev_loss_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* t = __tensor_parent(s, 1UL);
        FloatTensor* aux = __tensor_parent(s, 2UL);
        int d = s->dev;
        int m = x->requiresGrad ? tensor_grad_dev_acc_((void*)x, d) : 0;
        if (!m) return;
        int kind = (int)aux->data[0];
        struct DevArgs g = dev_args_();
        g.p[0] = x->dbuf; g.p[1] = t->dbuf; g.p[2] = x->dgrad; g.p[3] = s->dgrad;
        g.u[6] = m == 2 ? 1U : 0U;
        if (kind == 0) {
            unsigned long cols = x->shape[x->ndim - 1UL];
            unsigned long rows = x->size / cols;
            g.u[0] = (unsigned int)cols; g.f[0] = (float)1.0 / (float)rows;
            device_launch_(d, DK_CE_BWD, rows, &g);
        } else {
            g.u[0] = (unsigned int)(kind - 1); g.f[0] = (float)1.0 / (float)x->size; g.f[1] = aux->data[1];
            device_launch_(d, DK_EWLOSS_BWD, x->size, &g);
        }
        tensor_grad_dev_wrote_((void*)x);
    }
}

?&FloatTensor tdev_loss_(const &FloatTensor x, const &FloatTensor t, int kind, float k) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)t, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)t, d)) return null;
        match (tdev_1d_(1UL, d)) {
            case some(o): {
                struct DevArgs g = dev_args_();
                g.p[0] = x->dbuf; g.p[1] = t->dbuf; g.p[2] = o->dbuf;
                int ok = tdev_fill_(d, o->dbuf, 1UL, (float)0.0);
                if (kind == 0) {
                    unsigned long cols = x->shape[x->ndim - 1UL];
                    unsigned long rows = x->size / cols;
                    g.u[0] = (unsigned int)cols; g.f[0] = (float)1.0 / (float)rows;
                    ok = ok && device_launch_(d, DK_CE_FWD, rows, &g);
                } else {
                    g.u[0] = (unsigned int)(kind - 1); g.f[0] = (float)1.0 / (float)x->size; g.f[1] = k;
                    ok = ok && device_launch_(d, DK_EWLOSS_FWD, x->size, &g);
                }
                if (!ok) { tdev_drop_((FloatTensor*)o); return null; }
                if (Autograd::get().enabled() && x->requiresGrad) {
                    &FloatTensor aux = tdev_aux_((float)kind, k, (float)0.0, (float)0.0);
                    o->requiresGrad = 1;
                    o->parents.push(x);
                    o->parents.push(t);
                    o->parents.push(aux);
                    o->gradFn = (void*)tdev_loss_backward_;
                }
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Layer norm (no affine) ───────────────────────────────────────────────────
static void tdev_layernorm_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* inv = __tensor_parent(s, 1UL);
        int d = s->dev;
        if (!x->requiresGrad || !tensor_grad_dev_((void*)x, d)) return;
        unsigned long cols = s->shape[s->ndim - 1UL];
        struct DevArgs g = dev_args_();
        g.p[0] = s->dgrad; g.p[1] = s->dbuf; g.p[2] = inv->dbuf; g.p[3] = x->dgrad; g.u[0] = (unsigned int)cols;
        device_launch_(d, DK_LAYERNORM_BWD, s->size / cols, &g);
        tensor_grad_dev_wrote_((void*)x);
    }
}

?&FloatTensor tdev_layer_norm_(const &FloatTensor x, float eps) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)0, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)x, d)) return null;
        unsigned long cols = x->shape[x->ndim - 1UL];
        unsigned long rows = x->size / cols;
        match (tdev_like_((FloatTensor*)x, d, device_store_tag_(d))) {
            case some(o): {
                match (tdev_1d_(rows, d)) {
                    case some(inv): {
                        struct DevArgs g = dev_args_();
                        g.p[0] = x->dbuf; g.p[1] = o->dbuf; g.p[2] = inv->dbuf; g.u[0] = (unsigned int)cols; g.f[0] = eps;
                        if (!device_launch_(d, DK_LAYERNORM, rows, &g)) {
                            tdev_drop_((FloatTensor*)o); tdev_drop_((FloatTensor*)inv); return null;
                        }
                        inv->coh = inv->coh & ~TS_DEVOP;   // a constant of the node
                        __tensor_link2((FloatTensor*)o, (FloatTensor*)x, (FloatTensor*)inv, (void*)tdev_layernorm_backward_);
                        return (?&FloatTensor)o;
                    }
                    case null: { tdev_drop_((FloatTensor*)o); return null; }
                }
            }
            case null: { return null; }
        }
    }
    return null;
}

// A copy that is not part of the graph, on the same device.
?&FloatTensor tdev_detach_(const &FloatTensor a) {
    int d;
    unsafe { d = tensor_op_device_((void*)a, (void*)0, (void*)0); }
    if (d == DEVICE_CPU) return null;
    unsafe {
        if (!tensor_dev_((void*)a, d)) return null;
        match (tdev_like_((FloatTensor*)a, d, (int)dev_tag_(a->dbuf))) {
            case some(o): {
                if (!device_copy_(d, o->dbuf, a->dbuf, a->size)) { tdev_drop_((FloatTensor*)o); return null; }
                o->coh = o->coh & ~TS_DEVOP;
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}


// ── Convolution and transposed convolution (convnet.h's layout) ──────────────
// In convolution terms: the image side is C x H x W, the column side
// oC x oH x oW, the weight [oC, C*kH*kW] -- which is also a transposed
// convolution's [inC, outC*kH*kW] with its input on the column side. Per
// image: im2col / col2im (device_kernels.cu) around one GEMM.
// aux: [transpose, C, H, W, oC, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, batch]
struct TdevConv_ {
    unsigned long C; unsigned long H; unsigned long W;
    unsigned long oC; unsigned long oH; unsigned long oW;
    unsigned long kH; unsigned long kW; unsigned long sH; unsigned long sW;
    unsigned long pH; unsigned long pW; unsigned long dH; unsigned long dW;
    unsigned long batch;
    int transpose;
};

static struct DevArgs tdev_col_args_(const &TdevConv_ g, unsigned long img, unsigned long col, unsigned long nb) {
    struct DevArgs a = dev_args_();
    unsafe {
        a.p[0] = img; a.p[1] = col;
        a.u[0] = (unsigned int)g.C; a.u[1] = (unsigned int)g.H; a.u[2] = (unsigned int)g.W;
        a.u[3] = (unsigned int)g.kH; a.u[4] = (unsigned int)g.kW; a.u[5] = (unsigned int)g.oH; a.u[6] = (unsigned int)g.oW;
        a.f[0] = (float)g.sH; a.f[1] = (float)g.sW; a.f[2] = (float)g.pH; a.f[3] = (float)g.pW;
        a.f[4] = (float)g.dH; a.f[5] = (float)g.dW; a.f[6] = (float)nb;
    }
    return a;
}

// col [C*kH*kW, nb*oP] = im2col of nb images starting at 'img'.
static int tdev_im2col_(int d, const &TdevConv_ g, unsigned long img, unsigned long col, unsigned long nb) {
    struct DevArgs a = tdev_col_args_(g, img, col, nb);
    return device_launch_(d, DK_IM2COL, g.C * g.kH * g.kW * nb * g.oH * g.oW, &a);
}

// images at 'img' += col2im('col') for nb images.
static int tdev_col2im_(int d, const &TdevConv_ g, unsigned long col, unsigned long img, unsigned long nb) {
    struct DevArgs a = tdev_col_args_(g, col, img, nb);
    return device_launch_(d, DK_COL2IM, g.C * g.kH * g.kW * nb * g.oH * g.oW, &a);
}

// out [B2, A, P] (+)= in [A, B2, P]
static int tdev_swap01_(int d, unsigned long in, unsigned long out, unsigned long A, unsigned long B2, unsigned long P,
                        int acc) {
    struct DevArgs a = dev_args_();
    unsafe {
        a.p[0] = in; a.p[1] = out;
        a.u[0] = (unsigned int)A; a.u[1] = (unsigned int)B2; a.u[2] = (unsigned int)P; a.u[3] = (unsigned int)acc;
    }
    return device_launch_(d, DK_SWAP01, A * B2 * P, &a);
}

// Images per chunk: the column matrix stays under ~32M floats.
static unsigned long tdev_conv_chunk_(const &TdevConv_ g) {
    unsigned long per = g.C * g.kH * g.kW * g.oH * g.oW;
    unsigned long c = per == 0UL ? g.batch : 33554432UL / per;
    if (c < 1UL) c = 1UL;
    if (c > g.batch) c = g.batch;
    return c;
}

static int tdev_bias_ch_(int d, unsigned long out, unsigned long bias, unsigned long channels, unsigned long plane,
                         unsigned long batch) {
    struct DevArgs a = dev_args_();
    unsafe { a.p[0] = out; a.p[1] = bias; a.u[0] = (unsigned int)channels; a.u[1] = (unsigned int)plane; }
    return device_launch_(d, DK_BIAS_CHANNELS, batch * channels * plane, &a);
}

static struct TdevConv_ tdev_conv_geom_(FloatTensor* aux) {
    struct TdevConv_ g;
    unsafe {
        float* v = (float*)aux->data;
        g.transpose = (int)v[0];
        g.C = (unsigned long)v[1]; g.H = (unsigned long)v[2]; g.W = (unsigned long)v[3];
        g.oC = (unsigned long)v[4]; g.oH = (unsigned long)v[5]; g.oW = (unsigned long)v[6];
        g.kH = (unsigned long)v[7]; g.kW = (unsigned long)v[8]; g.sH = (unsigned long)v[9]; g.sW = (unsigned long)v[10];
        g.pH = (unsigned long)v[11]; g.pW = (unsigned long)v[12]; g.dH = (unsigned long)v[13]; g.dW = (unsigned long)v[14];
        g.batch = (unsigned long)v[15];
    }
    return g;
}

// The layer's convolution for the backend's library (device.h's DevConv).
static struct DevConv tdev_devconv_(const &TdevConv_ g, int op, unsigned long x, unsigned long w, unsigned long y) {
    struct DevConv c;
    c.op = op;
    c.dtype = DEVICE_FP32;
    c.transpose = g.transpose;
    c.batch = g.batch;
    if (g.transpose) {
        c.inC = g.oC; c.inH = g.oH; c.inW = g.oW; c.outC = g.C; c.outH = g.H; c.outW = g.W;
    } else {
        c.inC = g.C; c.inH = g.H; c.inW = g.W; c.outC = g.oC; c.outH = g.oH; c.outW = g.oW;
    }
    c.kH = g.kH; c.kW = g.kW; c.sH = g.sH; c.sW = g.sW; c.pH = g.pH; c.pW = g.pW; c.dH = g.dH; c.dW = g.dW;
    c.x = x; c.w = w; c.y = y;
    return c;
}

// grad += the library's gradient for 'op' (through a scratch: the library
// overwrites); 0 when the backend has no library convolution for it.
static int tdev_lib_grad_(int d, const &TdevConv_ g, int op, FloatTensor* s, FloatTensor* x, FloatTensor* w, int fresh) {
    unsafe {
        FloatTensor* t = op == DEV_CONV_BWD_DATA ? x : w;
        if (fresh) {
            // the gradient was all zeros: the library writes it
            struct DevConv c = op == DEV_CONV_BWD_DATA ? tdev_devconv_(g, op, x->dgrad, w->dbuf, s->dgrad)
                                                       : tdev_devconv_(g, op, x->dbuf, w->dgrad, s->dgrad);
            if (device_conv_(d, &c)) return 1;
            return tdev_fill_(d, t->dgrad, t->size, (float)0.0) ? 0 : -1;   // im2col adds to zeros
        }
        unsigned long gt = dev_tag_(t->dgrad);
        unsigned long tmp = dev_tagged_(device_mem_alloc_(d, dev_bytes_(gt, t->size)), gt);
        if (tmp == 0UL) return 0;
        struct DevConv c = op == DEV_CONV_BWD_DATA ? tdev_devconv_(g, op, tmp, w->dbuf, s->dgrad)
                                                   : tdev_devconv_(g, op, x->dbuf, tmp, s->dgrad);
        int ok = device_conv_(d, &c) && tdev_ew_launch_(d, EW_ACC, tmp, 0UL, t->dgrad, 0UL, (float)0.0, (float)0.0, t->size);
        device_mem_free_(d, tmp, dev_bytes_(gt, t->size));
        return ok;
    }
}

// The fp32 backward; skipX / skipW: that gradient was already added.
static void tdev_conv_bwd_(void* selfPtr, int skipX, int skipW) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* w = __tensor_parent(s, 1UL);
        FloatTensor* b = __tensor_parent(s, 2UL);
        FloatTensor* aux = __tensor_parent(s, 3UL);
        int d = s->dev;
        struct TdevConv_ g = tdev_conv_geom_(aux);
        unsigned long CK = g.C * g.kH * g.kW;
        unsigned long oP = g.oH * g.oW;
        unsigned long iP = g.H * g.W;
        unsigned long imgSz = g.C * iP;       // image side, per image
        unsigned long colSz = g.oC * oP;      // column side, per image
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)w, d)) return;
        int mx = x->requiresGrad && !skipX ? tensor_grad_dev_acc_((void*)x, d) : 0;
        int mw = w->requiresGrad && !skipW ? tensor_grad_dev_acc_((void*)w, d) : 0;
        int gx = mx != 0;
        int gw = mw != 0;
        int gb = b->requiresGrad && tensor_grad_dev_((void*)b, d);
        int doneX = gx && tdev_lib_grad_(d, &g, DEV_CONV_BWD_DATA, s, x, w, mx == 2) == 1;
        int doneW = gw && tdev_lib_grad_(d, &g, DEV_CONV_BWD_WEIGHTS, s, x, w, mw == 2) == 1;
        int fx = gx && !doneX;
        int fw = gw && !doneW;
        unsigned long chunk = tdev_conv_chunk_(&g);
        unsigned long col = 0UL;
        unsigned long cm = 0UL;
        if (fx || fw) {
            col = device_mem_alloc_(d, sizeof(float) * CK * chunk * oP);
            cm = device_mem_alloc_(d, sizeof(float) * g.oC * chunk * oP);   // column side, channel-major
            if (col == 0UL || cm == 0UL) return;
        }
        // col2im adds with atomics: a 16-bit input gradient goes through fp32
        unsigned long xg = fx && !g.transpose ? tensor_grad_atomic_begin_((void*)x, d) : x->dgrad;
        if (fx && xg == 0UL) return;
        unsigned long i = 0UL;
        while ((fx || fw) && i < g.batch) {
            unsigned long nb = g.batch - i < chunk ? g.batch - i : chunk;
            unsigned long BP = nb * oP;
            if (!g.transpose) {
                // the output gradient [nb, oC, oP] -> [oC, nb*oP]
                tdev_swap01_(d, device_offset_el_(d, s->dgrad, i * colSz), cm, nb, g.oC, oP, 0);
                if (fw) {
                    tdev_im2col_(d, &g, device_offset_el_(d, x->dbuf, i * imgSz), col, nb);
                    device_gemm_(d, 0, 1, g.oC, CK, BP, (float)1.0, cm, BP, col, BP, (float)1.0, w->dgrad, CK);
                }
                if (fx) {
                    device_gemm_(d, 1, 0, CK, BP, g.oC, (float)1.0, w->dbuf, CK, cm, BP, (float)0.0, col, BP);
                    tdev_col2im_(d, &g, col, device_offset_el_(d, xg, i * imgSz), nb);
                }
            } else {
                // the output gradient is on the image side
                tdev_im2col_(d, &g, device_offset_el_(d, s->dgrad, i * imgSz), col, nb);
                if (fx) {
                    device_gemm_(d, 0, 0, g.oC, BP, CK, (float)1.0, w->dbuf, CK, col, BP, (float)0.0, cm, BP);
                    tdev_swap01_(d, cm, device_offset_el_(d, x->dgrad, i * colSz), g.oC, nb, oP, 1);
                }
                if (fw) {
                    tdev_swap01_(d, device_offset_el_(d, x->dbuf, i * colSz), cm, nb, g.oC, oP, 0);
                    device_gemm_(d, 0, 1, g.oC, CK, BP, (float)1.0, cm, BP, col, BP, (float)1.0, w->dgrad, CK);
                }
            }
            i = i + nb;
        }
        if (fx && !g.transpose) tensor_grad_atomic_end_((void*)x, d, xg);
        if (col != 0UL) device_mem_free_(d, col, sizeof(float) * CK * chunk * oP);
        if (cm != 0UL) device_mem_free_(d, cm, sizeof(float) * g.oC * chunk * oP);
        if (gb) {
            struct DevArgs a = dev_args_();
            unsigned long ch = g.transpose ? g.C : g.oC;
            unsigned long plane = g.transpose ? iP : oP;
            a.p[0] = s->dgrad; a.p[1] = b->dgrad;
            unsigned long S = tdev_slices_(ch, g.batch * plane);
            a.u[0] = (unsigned int)ch; a.u[1] = (unsigned int)plane; a.u[2] = (unsigned int)g.batch; a.u[3] = (unsigned int)S;
            device_launch_(d, DK_BIAS_CHANNELS_BWD, ch * S * 256UL, &a);
        }
        if (gx) tensor_grad_dev_wrote_((void*)x);
        if (gw) tensor_grad_dev_wrote_((void*)w);
        if (gb) tensor_grad_dev_wrote_((void*)b);
    }
}

static void tdev_conv_backward_(void* selfPtr) { tdev_conv_bwd_(selfPtr, 0, 0); }

?&FloatTensor tdev_conv_(const &FloatTensor x, const &FloatTensor w, const &FloatTensor b, int transpose,
                         unsigned long C, unsigned long H, unsigned long W,
                         unsigned long oC, unsigned long oH, unsigned long oW,
                         unsigned long kH, unsigned long kW, unsigned long sH, unsigned long sW,
                         unsigned long pH, unsigned long pW, unsigned long dH, unsigned long dW) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)w, (void*)b); }
    if (d == DEVICE_CPU) return null;
    struct TdevConv_ g;
    g.C = C; g.H = H; g.W = W; g.oC = oC; g.oH = oH; g.oW = oW;
    g.kH = kH; g.kW = kW; g.sH = sH; g.sW = sW; g.pH = pH; g.pW = pW; g.dH = dH; g.dW = dW;
    g.transpose = transpose;
    unsigned long CK = C * kH * kW;
    unsigned long oP = oH * oW;
    unsigned long iP = H * W;
    unsafe {
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)w, d) || !tensor_dev_((void*)b, d)) return null;
        unsigned long inSz = transpose ? oC * oP : C * iP;
        g.batch = x->size / inSz;
        unsigned long outSz = transpose ? C * iP : oC * oP;
        int outTag = device_store_tag_(d);
        if (outTag != DEVICE_FP32 && w->gradFn == (void*)0) device_shadow_mark_(d, w->dbuf);
        match (tdev_2d_(g.batch, outSz, d, outTag)) {
            case some(o): {
                struct DevConv lib = tdev_devconv_(&g, DEV_CONV_FWD, x->dbuf, w->dbuf, o->dbuf);
                int viaLib = device_conv_(d, &lib);
                unsigned long acc32 = 0UL;
                unsigned long chunk = tdev_conv_chunk_(&g);
                unsigned long col = 0UL;
                unsigned long cm = 0UL;
                int ok = 1;
                if (!viaLib) {
                    col = device_mem_alloc_(d, sizeof(float) * CK * chunk * oP);
                    cm = device_mem_alloc_(d, sizeof(float) * oC * chunk * oP);
                    ok = col != 0UL && cm != 0UL;
                    // col2im accumulates (atomics: into fp32, converted after)
                    if (ok && transpose && dev_tag_(o->dbuf) != 0UL) {
                        acc32 = device_mem_alloc_(d, sizeof(float) * o->size);
                        ok = acc32 != 0UL;
                    }
                    if (ok && transpose) ok = tdev_fill_(d, acc32 != 0UL ? acc32 : o->dbuf, o->size, (float)0.0);
                }
                unsigned long i = 0UL;
                while (ok && !viaLib && i < g.batch) {
                    unsigned long nb = g.batch - i < chunk ? g.batch - i : chunk;
                    unsigned long BP = nb * oP;
                    if (!transpose) {
                        // [oC, nb*oP] = W col, then -> [nb, oC, oP]
                        ok = tdev_im2col_(d, &g, device_offset_el_(d, x->dbuf, i * C * iP), col, nb) &&
                             device_gemm_(d, 0, 0, oC, BP, CK, (float)1.0, w->dbuf, CK, col, BP, (float)0.0, cm, BP) &&
                             tdev_swap01_(d, cm, device_offset_el_(d, o->dbuf, i * outSz), oC, nb, oP, 0);
                    } else {
                        // x [nb, oC, oP] -> [oC, nb*oP]; col = W^T x; images += col2im(col)
                        ok = tdev_swap01_(d, device_offset_el_(d, x->dbuf, i * inSz), cm, nb, oC, oP, 0) &&
                             device_gemm_(d, 1, 0, CK, BP, oC, (float)1.0, w->dbuf, CK, cm, BP, (float)0.0, col, BP) &&
                             tdev_col2im_(d, &g, col, device_offset_el_(d, acc32 != 0UL ? acc32 : o->dbuf, i * outSz), nb);
                    }
                    i = i + nb;
                }
                if (col != 0UL) device_mem_free_(d, col, sizeof(float) * CK * chunk * oP);
                if (cm != 0UL) device_mem_free_(d, cm, sizeof(float) * oC * chunk * oP);
                if (acc32 != 0UL) {
                    if (ok) ok = device_copy_(d, o->dbuf, acc32, o->size);
                    device_mem_free_(d, acc32, sizeof(float) * o->size);
                }
                if (ok) ok = transpose ? tdev_bias_ch_(d, o->dbuf, b->dbuf, C, iP, g.batch)
                                       : tdev_bias_ch_(d, o->dbuf, b->dbuf, oC, oP, g.batch);
                if (!ok) { tdev_drop_((FloatTensor*)o); return null; }
                if (Autograd::get().enabled() && (x->requiresGrad || w->requiresGrad || b->requiresGrad)) {
                    unsigned long shape[1];
                    shape[0] = 16UL;
                    &FloatTensor aux = __tensor_alloc((const unsigned long*)&shape[0], 1UL, 0);
                    float* v = (float*)aux->data;
                    v[0] = (float)transpose; v[1] = (float)C; v[2] = (float)H; v[3] = (float)W;
                    v[4] = (float)oC; v[5] = (float)oH; v[6] = (float)oW; v[7] = (float)kH; v[8] = (float)kW;
                    v[9] = (float)sH; v[10] = (float)sW; v[11] = (float)pH; v[12] = (float)pW; v[13] = (float)dH;
                    v[14] = (float)dW; v[15] = (float)g.batch;
                    o->requiresGrad = 1;
                    o->parents.push(x);
                    o->parents.push(w);
                    o->parents.push(b);
                    o->parents.push(aux);
                    o->gradFn = (void*)tdev_conv_backward_;
                }
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

// Slices for a per-channel reduction over M values (device_kernels.cu's
// SLICE_LOOP): enough blocks to fill the GPU.
static unsigned long tdev_slices_(unsigned long C, unsigned long M) {
    unsigned long S = M / 4096UL;
    unsigned long cap = C == 0UL ? 1UL : 2048UL / C;
    if (S > cap) S = cap;
    if (S < 1UL) S = 1UL;
    return S;
}

// ── Batch and group normalization ────────────────────────────────────────────
#define TDEV_LEAK ((float)0.2)   // ACT_LEAKY's slope
// Node parents: x, gamma, beta, stats (device [2S]: mean, invstd), aux (host:
// C, plane, groups (0: batch norm), training).
static int tdev_any_grad_(FloatTensor* a, FloatTensor* b, FloatTensor* c) {
    unsafe { return Autograd::get().enabled() && (a->requiresGrad || b->requiresGrad || c->requiresGrad); }
}

static void tdev_norm_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* gm = __tensor_parent(s, 1UL);
        FloatTensor* bt = __tensor_parent(s, 2UL);
        FloatTensor* st = __tensor_parent(s, 3UL);
        FloatTensor* aux = __tensor_parent(s, 4UL);
        int d = s->dev;
        unsigned long C = (unsigned long)aux->data[0];
        unsigned long P = (unsigned long)aux->data[1];
        unsigned long G = (unsigned long)aux->data[2];
        int training = aux->data[3] != (float)0.0;
        unsigned int act = (unsigned int)aux->data[4];
        unsigned long B = s->size / (C * P);
        unsigned long S = st->size / 2UL;
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)gm, d)) return;
        int mx = x->requiresGrad ? tensor_grad_dev_acc_((void*)x, d) : 0;
        int gx = mx != 0;
        if (mx == 2 && G != 0UL && !tdev_fill_(d, x->dgrad, x->size, (float)0.0)) return;   // dk_gn_bwd adds
        int gg = gm->requiresGrad && tensor_grad_dev_((void*)gm, d);
        int gb = bt->requiresGrad && tensor_grad_dev_((void*)bt, d);
        unsigned long mean = st->dbuf;
        unsigned long inv = device_offset_el_(d, st->dbuf, S);
        unsigned long sums = 0UL;
        if (gx && G == 0UL) {
            sums = device_mem_alloc_(d, sizeof(float) * 2UL * C);
            if (sums != 0UL) tdev_fill_(d, sums, 2UL * C, (float)0.0);
        }
        if (gg || gb || sums != 0UL) {
            unsigned long S = tdev_slices_(C, B * P);
            struct DevArgs a = dev_args_();
            a.p[0] = s->dgrad; a.p[1] = x->dbuf; a.p[2] = mean; a.p[3] = inv;
            a.p[4] = gg ? gm->dgrad : 0UL; a.p[5] = gb ? bt->dgrad : 0UL; a.p[6] = sums;
            a.u[0] = (unsigned int)C; a.u[1] = (unsigned int)P; a.u[2] = (unsigned int)G; a.u[3] = (unsigned int)B;
            a.u[4] = (unsigned int)S;
            a.u[5] = act; a.f[1] = TDEV_LEAK; a.p[7] = s->dbuf;
            device_launch_(d, DK_NORM_SUMS, C * S * 256UL, &a);
        }
        if (gx) {
            struct DevArgs a = dev_args_();
            a.p[0] = s->dgrad; a.p[1] = x->dbuf; a.p[2] = mean; a.p[3] = inv; a.p[4] = gm->dbuf;
            a.u[0] = (unsigned int)C; a.u[1] = (unsigned int)P;
            a.u[5] = act; a.f[1] = TDEV_LEAK; a.p[7] = s->dbuf;
            if (G == 0UL) {
                a.p[5] = sums; a.p[6] = x->dgrad; a.u[2] = (unsigned int)training; a.u[6] = mx == 2 ? 1U : 0U;
                a.f[0] = (float)1.0 / (float)(B * P);
                device_launch_(d, DK_BN_BWD_DX, s->size, &a);
            } else {
                a.p[5] = x->dgrad; a.u[2] = (unsigned int)G; a.u[3] = (unsigned int)S;
                device_launch_(d, DK_GN_BWD, S * 256UL, &a);
            }
        }
        if (sums != 0UL) device_mem_free_(d, sums, sizeof(float) * 2UL * C);
        if (gx) tensor_grad_dev_wrote_((void*)x);
        if (gg) tensor_grad_dev_wrote_((void*)gm);
        if (gb) tensor_grad_dev_wrote_((void*)bt);
    }
}

// The normalized output from the statistics in st, linked into the graph.
static ?&FloatTensor tdev_norm_finish_(int d, FloatTensor* x, FloatTensor* gamma, FloatTensor* beta, FloatTensor* st,
                                       unsigned long C, unsigned long P, unsigned long G, int training, int act) {
    unsafe {
        unsigned long S = st->size / 2UL;
        match (tdev_like_(x, d, device_store_tag_(d))) {
            case some(o): {
                struct DevArgs a = dev_args_();
                a.p[0] = x->dbuf; a.p[1] = o->dbuf; a.p[2] = st->dbuf; a.p[3] = device_offset_el_(d, st->dbuf, S);
                a.p[4] = gamma->dbuf; a.p[5] = beta->dbuf;
                a.u[0] = (unsigned int)C; a.u[1] = (unsigned int)P; a.u[2] = (unsigned int)G;
                a.u[5] = (unsigned int)act; a.f[1] = TDEV_LEAK;
                if (!device_launch_(d, DK_NORM_APPLY, x->size, &a)) { tdev_drop_((FloatTensor*)o); tdev_drop_(st); return null; }
                if (tdev_any_grad_(x, gamma, beta)) {
                    st->coh = st->coh & ~TS_DEVOP;   // a constant of the node
                    unsigned long ashape[1];
                    ashape[0] = 5UL;
                    &FloatTensor aux = __tensor_alloc((const unsigned long*)&ashape[0], 1UL, 0);
                    aux->data[0] = (float)C; aux->data[1] = (float)P; aux->data[2] = (float)G;
                    aux->data[3] = training ? (float)1.0 : (float)0.0; aux->data[4] = (float)act;
                    o->requiresGrad = 1;
                    o->parents.push(x);
                    o->parents.push(gamma);
                    o->parents.push(beta);
                    o->parents.push(st);
                    o->parents.push(aux);
                    o->gradFn = (void*)tdev_norm_backward_;
                } else {
                    tdev_drop_(st);
                }
                return (?&FloatTensor)o;
            }
            case null: { tdev_drop_(st); return null; }
        }
    }
    return null;
}

?&FloatTensor tdev_batch_norm_(const &FloatTensor x, const &FloatTensor gamma, const &FloatTensor beta,
                               &FloatTensor runMean, &FloatTensor runVar, unsigned long C, unsigned long P,
                               int training, float momentum, float eps, int act) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)gamma, (void*)beta); }
    if (d == DEVICE_CPU || C == 0UL || P == 0UL || act < 0 || act > 2) return null;
    unsafe {
        if (gamma->size != C || beta->size != C || runMean->size != C || runVar->size != C || x->size % (C * P) != 0UL)
            return null;
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)gamma, d) || !tensor_dev_((void*)beta, d) ||
            !tensor_dev_((void*)runMean, d) || !tensor_dev_((void*)runVar, d)) return null;
        match (tdev_1d_(2UL * C, d)) {
            case some(st): {
                unsigned long B = x->size / (C * P);
                unsigned long S = tdev_slices_(C, B * P);
                unsigned long part = 0UL;
                int ok = 1;
                if (training) {
                    part = device_mem_alloc_(d, sizeof(double) * 3UL * C * S);
                    struct DevArgs p = dev_args_();
                    p.p[0] = x->dbuf; p.p[1] = part;
                    p.u[0] = (unsigned int)C; p.u[1] = (unsigned int)P; p.u[2] = (unsigned int)B; p.u[3] = (unsigned int)S;
                    ok = part != 0UL && device_launch_(d, DK_BN_PARTIAL, C * S * 256UL, &p);
                }
                struct DevArgs a = dev_args_();
                a.p[0] = part; a.p[1] = st->dbuf; a.p[2] = device_offset_el_(d, st->dbuf, C);
                a.p[3] = runMean->dbuf; a.p[4] = runVar->dbuf;
                a.u[0] = (unsigned int)C; a.u[3] = (unsigned int)S; a.u[4] = (unsigned int)training;
                a.f[0] = momentum; a.f[1] = eps;
                if (ok) ok = device_launch_(d, DK_BN_FINISH, C, &a);
                if (part != 0UL) device_mem_free_(d, part, sizeof(double) * 3UL * C * S);
                if (!ok) { tdev_drop_((FloatTensor*)st); return null; }
                if (training) { tensor_dev_wrote_((void*)runMean); tensor_dev_wrote_((void*)runVar); }
                return tdev_norm_finish_(d, (FloatTensor*)x, (FloatTensor*)gamma, (FloatTensor*)beta, (FloatTensor*)st,
                                         C, P, 0UL, training, act);
            }
            case null: { return null; }
        }
    }
    return null;
}

?&FloatTensor tdev_group_norm_(const &FloatTensor x, const &FloatTensor gamma, const &FloatTensor beta,
                               unsigned long C, unsigned long P, unsigned long G, float eps, int act) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)gamma, (void*)beta); }
    if (d == DEVICE_CPU || C == 0UL || P == 0UL || G == 0UL || C % G != 0UL || act < 0 || act > 2) return null;
    unsafe {
        if (gamma->size != C || beta->size != C || x->size % (C * P) != 0UL) return null;
        if (!tensor_dev_((void*)x, d) || !tensor_dev_((void*)gamma, d) || !tensor_dev_((void*)beta, d)) return null;
        unsigned long U = (x->size / (C * P)) * G;
        match (tdev_1d_(2UL * U, d)) {
            case some(st): {
                struct DevArgs a = dev_args_();
                a.p[0] = x->dbuf; a.p[1] = st->dbuf; a.p[2] = device_offset_el_(d, st->dbuf, U);
                a.u[0] = (unsigned int)((C / G) * P); a.u[1] = (unsigned int)U;
                a.f[0] = eps;
                if (!device_launch_(d, DK_GN_STATS, U * 256UL, &a)) { tdev_drop_((FloatTensor*)st); return null; }
                return tdev_norm_finish_(d, (FloatTensor*)x, (FloatTensor*)gamma, (FloatTensor*)beta, (FloatTensor*)st,
                                         C, P, G, 1, act);
            }
            case null: { return null; }
        }
    }
    return null;
}

// ── Pooling, upsampling, resizing ────────────────────────────────────────────
// aux (host): the TdevSpatial_ as floats.
#define TDEV_SP_AUX 14UL

static struct TdevSpatial_ tdev_sp_of_(FloatTensor* aux) {
    struct TdevSpatial_ g;
    unsafe {
        float* v = (float*)aux->data;
        g.kind = (int)v[0];
        g.C = (unsigned long)v[1]; g.H = (unsigned long)v[2]; g.W = (unsigned long)v[3];
        g.oH = (unsigned long)v[4]; g.oW = (unsigned long)v[5];
        g.kH = (unsigned long)v[6]; g.kW = (unsigned long)v[7]; g.sH = (unsigned long)v[8]; g.sW = (unsigned long)v[9];
        g.pH = (unsigned long)v[10]; g.pW = (unsigned long)v[11];
        g.mode = (int)v[12]; g.align = (int)v[13];
    }
    return g;
}

// One launch of the op's kernel: forward (src x -> dst out) or backward
// (src the output's gradient, dst += into x's gradient; xv = x's values).
static int tdev_sp_launch_(int d, const &TdevSpatial_ g, unsigned long src, unsigned long dst, unsigned long xv,
                           unsigned long planes, int bwd, int fresh) {
    struct DevArgs a = dev_args_();
    unsafe {
        if (g.kind == TDEV_POOL) {
            a.p[0] = bwd ? xv : src; a.p[1] = bwd ? src : dst; a.p[2] = bwd ? dst : 0UL;
            a.u[0] = (unsigned int)g.C; a.u[1] = (unsigned int)g.H; a.u[2] = (unsigned int)g.W;
            a.u[3] = (unsigned int)g.oH; a.u[4] = (unsigned int)g.oW; a.u[5] = (unsigned int)g.mode; a.u[6] = (unsigned int)bwd;
            a.f[0] = (float)g.kH; a.f[1] = (float)g.kW; a.f[2] = (float)g.sH; a.f[3] = (float)g.sW;
            a.f[4] = (float)g.pH; a.f[5] = (float)g.pW;
            if (bwd) { a.u[6] = fresh ? 1U : 0U; return device_launch_(d, DK_POOL_BWD, planes * g.H * g.W, &a); }
            return device_launch_(d, DK_POOL, planes * g.oH * g.oW, &a);
        }
        if (g.kind == TDEV_GPOOL) {
            a.p[0] = xv; a.p[1] = src; a.p[2] = dst;
            if (!bwd) { a.p[0] = src; a.p[1] = dst; a.p[2] = 0UL; }
            a.u[0] = (unsigned int)(g.H * g.W); a.u[1] = (unsigned int)g.mode; a.u[2] = (unsigned int)bwd;
            a.u[6] = fresh ? 1U : 0U;
            return device_launch_(d, DK_GPOOL, planes * 256UL, &a);
        }
        if (g.kind == TDEV_UPSAMPLE) {
            a.p[0] = src; a.p[1] = dst;
            a.u[0] = (unsigned int)g.W; a.u[1] = (unsigned int)g.H; a.u[2] = (unsigned int)g.kH; a.u[3] = (unsigned int)bwd;
            a.u[6] = fresh ? 1U : 0U;
            return device_launch_(d, DK_UPSAMPLE, bwd ? planes * g.H * g.W : planes * g.oH * g.oW, &a);
        }
        a.p[0] = src; a.p[1] = dst;
        a.u[0] = (unsigned int)g.H; a.u[1] = (unsigned int)g.W; a.u[2] = (unsigned int)g.oH; a.u[3] = (unsigned int)g.oW;
        a.u[4] = (unsigned int)g.mode; a.u[5] = (unsigned int)g.align;
        if (bwd) {
            // two passes through t [planes, oH, W]
            unsigned long t = device_mem_alloc_(d, sizeof(float) * planes * g.oH * g.W);
            if (t == 0UL) return 0;
            a.p[1] = t;
            int ok = device_launch_(d, DK_RESIZE_BWD, planes * g.oH * g.W, &a);
            a.p[0] = t; a.p[1] = dst; a.f[0] = (float)1.0; a.u[6] = fresh ? 1U : 0U;
            ok = ok && device_launch_(d, DK_RESIZE_BWD, planes * g.H * g.W, &a);
            device_mem_free_(d, t, sizeof(float) * planes * g.oH * g.W);
            return ok;
        }
        return device_launch_(d, DK_RESIZE, planes * g.oH * g.oW, &a);
    }
}

static void tdev_spatial_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        int d = s->dev;
        if (!x->requiresGrad || !tensor_dev_((void*)x, d)) return;
        struct TdevSpatial_ g = tdev_sp_of_(__tensor_parent(s, 1UL));
        // every op here writes all of dx except global max pooling (one value
        // per plane) and non-tiling window pools: those need it zeroed
        int whole = !(g.kind == TDEV_GPOOL && g.mode == 0) &&
                    !(g.kind == TDEV_POOL && g.kH == g.sH && g.kW == g.sW && g.pH == 0UL && g.pW == 0UL &&
                      (g.oH * g.kH != g.H || g.oW * g.kW != g.W));
        int m = whole ? tensor_grad_dev_acc_((void*)x, d) : tensor_grad_dev_((void*)x, d);
        if (!m) return;
        tdev_sp_launch_(d, &g, s->dgrad, x->dgrad, x->dbuf, x->size / (g.H * g.W), 1, whole && m == 2);
        tensor_grad_dev_wrote_((void*)x);
    }
}

?&FloatTensor tdev_spatial_(const &FloatTensor x, const &TdevSpatial_ g) {
    int d;
    unsafe { d = tensor_op_device_((void*)x, (void*)0, (void*)0); }
    unsigned long plane = g.H * g.W;
    if (d == DEVICE_CPU || g.C == 0UL || plane == 0UL || g.oH * g.oW == 0UL) return null;
    unsafe {
        if (x->size % (g.C * plane) != 0UL) return null;
        if (!tensor_dev_((void*)x, d)) return null;
        unsigned long batch = x->size / (g.C * plane);
        unsigned long planes = batch * g.C;
        match (tdev_2d_(batch, g.C * g.oH * g.oW, d, tdev_tag1_(d, (FloatTensor*)x))) {
            case some(o): {
                if (!tdev_sp_launch_(d, g, x->dbuf, o->dbuf, x->dbuf, planes, 0, 0)) { tdev_drop_((FloatTensor*)o); return null; }
                if (Autograd::get().enabled() && x->requiresGrad) {
                    unsigned long shape[1];
                    shape[0] = TDEV_SP_AUX;
                    &FloatTensor aux = __tensor_alloc((const unsigned long*)&shape[0], 1UL, 0);
                    float* v = (float*)aux->data;
                    v[0] = (float)g.kind; v[1] = (float)g.C; v[2] = (float)g.H; v[3] = (float)g.W;
                    v[4] = (float)g.oH; v[5] = (float)g.oW; v[6] = (float)g.kH; v[7] = (float)g.kW;
                    v[8] = (float)g.sH; v[9] = (float)g.sW; v[10] = (float)g.pH; v[11] = (float)g.pW;
                    v[12] = (float)g.mode; v[13] = (float)g.align;
                    __tensor_link2((FloatTensor*)o, (FloatTensor*)x, aux, (void*)tdev_spatial_backward_);
                }
                return (?&FloatTensor)o;
            }
            case null: { return null; }
        }
    }
    return null;
}

} // namespace std
