// SafeC Standard Library — datasets and mini-batch loading (see datasets.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/datasets.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/math.h>
#include <std/io_file.h>
#include <std/convert.h>

namespace std {

#define DS_PI ((float)3.141592653589793)

// Datasets always live on the heap, also when an arena is active.
static ?&TensorArena ds_heap_begin_() {
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    return cur;
}

static void ds_heap_end_(?&TensorArena cur) {
    match (cur) {
        case some(a): { a.begin(); }
        case null: { }
    }
}

static struct Dataset ds_new_(unsigned long n, unsigned long features, unsigned long targets, unsigned long classes) {
    ?&TensorArena cur = ds_heap_begin_();
    struct Dataset d;
    d.x = tensor_new_2d(n, features, 0);
    d.y = tensor_new_2d(n, targets, 0);
    d.n = n;
    d.features = features;
    d.targets = targets;
    d.classes = classes;
    ds_heap_end_(cur);
    return d;
}

static void ds_set_(&Dataset d, unsigned long i, float a, float b, float label) {
    unsafe {
        d.x->data[i * 2UL] = a;
        d.x->data[i * 2UL + 1UL] = b;
        d.y->data[i] = label;
    }
}

struct Dataset dataset_moons(unsigned long n, float noise, &Rng rng) {
    struct Dataset d = ds_new_(n, 2UL, 1UL, 2UL);
    unsigned long i = 0UL;
    while (i < n) {
        unsigned long c = i % 2UL;
        float t = DS_PI * rng.uniform();
        float a = c == 0UL ? cos_f(t) : (float)1.0 - cos_f(t);
        float b = c == 0UL ? sin_f(t) : (float)0.5 - sin_f(t);
        ds_set_(&d, i, a + noise * rng.normal(), b + noise * rng.normal(), (float)c);
        i = i + 1UL;
    }
    return d;
}

struct Dataset dataset_circles(unsigned long n, float noise, float factor, &Rng rng) {
    struct Dataset d = ds_new_(n, 2UL, 1UL, 2UL);
    unsigned long i = 0UL;
    while (i < n) {
        unsigned long c = i % 2UL;
        float t = (float)2.0 * DS_PI * rng.uniform();
        float r = c == 0UL ? (float)1.0 : factor;
        ds_set_(&d, i, r * cos_f(t) + noise * rng.normal(), r * sin_f(t) + noise * rng.normal(), (float)c);
        i = i + 1UL;
    }
    return d;
}

struct Dataset dataset_spirals(unsigned long n, unsigned long classes, float noise, &Rng rng) {
    struct Dataset d = ds_new_(n, 2UL, 1UL, classes);
    unsigned long i = 0UL;
    while (i < n) {
        unsigned long c = i % classes;
        float r = rng.uniform();
        float t = (float)c * (float)2.0 * DS_PI / (float)classes + r * (float)4.0 + noise * rng.normal();
        ds_set_(&d, i, r * cos_f(t), r * sin_f(t), (float)c);
        i = i + 1UL;
    }
    return d;
}

struct Dataset dataset_blobs(unsigned long n, unsigned long classes, unsigned long features,
                             float spread, &Rng rng) {
    struct Dataset d = ds_new_(n, features, 1UL, classes);
    &Tensor centers = tensor_new_2d(classes, features, 0);
    rng.fill_uniform(centers, (float)-5.0, (float)5.0);
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            unsigned long c = i % classes;
            unsigned long f = 0UL;
            while (f < features) {
                d.x->data[i * features + f] = centers->data[c * features + f] + spread * rng.normal();
                f = f + 1UL;
            }
            d.y->data[i] = (float)c;
            i = i + 1UL;
        }
        centers->free();
        free((void*)centers);
    }
    return d;
}

struct Dataset dataset_xor(unsigned long n, float noise, &Rng rng) {
    struct Dataset d = ds_new_(n, 2UL, 1UL, 2UL);
    unsigned long i = 0UL;
    while (i < n) {
        float a = (float)2.0 * rng.uniform() - (float)1.0;
        float b = (float)2.0 * rng.uniform() - (float)1.0;
        float label = (a > (float)0.0) != (b > (float)0.0) ? (float)1.0 : (float)0.0;
        ds_set_(&d, i, a + noise * rng.normal(), b + noise * rng.normal(), label);
        i = i + 1UL;
    }
    return d;
}

struct Dataset dataset_sine(unsigned long n, float noise, &Rng rng) {
    struct Dataset d = ds_new_(n, 1UL, 1UL, 0UL);
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            float x = DS_PI * ((float)2.0 * rng.uniform() - (float)1.0);
            d.x->data[i] = x;
            d.y->data[i] = sin_f(x) + noise * rng.normal();
            i = i + 1UL;
        }
    }
    return d;
}

struct Dataset dataset_gaussian_ring(unsigned long n, unsigned long modes, float radius, float std, &Rng rng) {
    struct Dataset d = ds_new_(n, 2UL, 1UL, modes);
    unsigned long i = 0UL;
    while (i < n) {
        unsigned long c = rng.below(modes);
        float t = (float)2.0 * DS_PI * (float)c / (float)modes;
        ds_set_(&d, i, radius * cos_f(t) + std * rng.normal(), radius * sin_f(t) + std * rng.normal(), (float)c);
        i = i + 1UL;
    }
    return d;
}

struct Dataset dataset_shapes(unsigned long n, unsigned long side, &Rng rng) {
    struct Dataset d = ds_new_(n, side * side, 1UL, 4UL);
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            unsigned long c = i % 4UL;
            float* img = (float*)&d.x->data[i * side * side];
            // size between side/3 and side-2, centered somewhere it fits
            unsigned long lo = side / 3UL;
            unsigned long sz = lo + rng.below(side - 2UL - lo + 1UL);
            unsigned long x0 = 1UL + rng.below(side - sz - 1UL);
            unsigned long y0 = 1UL + rng.below(side - sz - 1UL);
            float half = (float)(sz - 1UL) * (float)0.5;
            float cx = (float)x0 + half;
            float cy = (float)y0 + half;
            unsigned long y = 0UL;
            while (y < side) {
                unsigned long x = 0UL;
                while (x < side) {
                    int inBox = x >= x0 && x < x0 + sz && y >= y0 && y < y0 + sz;
                    float dx = (float)x - cx;
                    float dy = (float)y - cy;
                    float adx = dx < (float)0.0 ? -dx : dx;
                    float ady = dy < (float)0.0 ? -dy : dy;
                    int on = 0;
                    if (c == 0UL) on = inBox && (x == x0 || x == x0 + sz - 1UL || y == y0 || y == y0 + sz - 1UL);
                    else if (c == 1UL) on = inBox && (adx <= (float)0.5 || ady <= (float)0.5);
                    else if (c == 2UL) on = dx * dx + dy * dy <= (half + (float)0.5) * (half + (float)0.5);
                    else {
                        float e = adx - ady;
                        on = inBox && e <= (float)0.5 && e >= (float)-0.5;
                    }
                    img[y * side + x] = on ? (float)1.0 : (float)0.0;
                    x = x + 1UL;
                }
                y = y + 1UL;
            }
            d.y->data[i] = (float)c;
            i = i + 1UL;
        }
    }
    return d;
}

// ── Files ────────────────────────────────────────────────────────────────────

static struct Dataset ds_empty_() { return ds_new_(0UL, 1UL, 1UL, 0UL); }

// Reads a whole file; returns the byte count (0 on error) and the buffer
// through 'out' (caller frees).
static unsigned long ds_read_all_(const &char path, &&heap unsigned char out) {
    match (file_open(path, "rb")) {
        case some(f): {
            long long sz = file_size(f);
            if (sz <= 0LL) { file_close(f); return 0UL; }
            unsigned long n = (unsigned long)sz;
            unsafe {
                unsigned char* buf = (unsigned char*)malloc(n + 1UL);
                unsigned long got = file_read(f, (&unsigned char)buf, n);
                file_close(f);
                if (got != n) { free((void*)buf); return 0UL; }
                buf[n] = (unsigned char)0;
                *out = (&heap unsigned char)buf;
            }
            return n;
        }
        case null: { return 0UL; }
    }
    return 0UL;
}

static unsigned long ds_be32_(const unsigned char* p) {
    unsafe {
        return ((unsigned long)p[0] << 24UL) | ((unsigned long)p[1] << 16UL) |
               ((unsigned long)p[2] << 8UL) | (unsigned long)p[3];
    }
}

struct Dataset dataset_load_mnist(const &char imagesPath, const &char labelsPath, unsigned long limit) {
    &heap unsigned char img;
    &heap unsigned char lab;
    unsafe { img = (&heap unsigned char)0; lab = (&heap unsigned char)0; }
    unsigned long ni = ds_read_all_(imagesPath, &img);
    unsigned long nl = ds_read_all_(labelsPath, &lab);
    struct Dataset d = ds_empty_();
    unsafe {
        const unsigned char* ip = (const unsigned char*)img;
        const unsigned char* lp = (const unsigned char*)lab;
        int ok = ni >= 16UL && nl >= 8UL && ds_be32_(ip) == 2051UL && ds_be32_(lp) == 2049UL;
        if (ok) {
            unsigned long count = ds_be32_(&ip[4]);
            unsigned long rows = ds_be32_(&ip[8]);
            unsigned long cols = ds_be32_(&ip[12]);
            unsigned long lcount = ds_be32_(&lp[4]);
            if (lcount < count) count = lcount;
            if (limit > 0UL && limit < count) count = limit;
            unsigned long px = rows * cols;
            if (16UL + count * px <= ni && 8UL + count <= nl) {
                d.free();
                d = ds_new_(count, px, 1UL, 10UL);
                unsigned long i = 0UL;
                while (i < count * px) { d.x->data[i] = (float)ip[16UL + i] / (float)255.0; i = i + 1UL; }
                i = 0UL;
                while (i < count) { d.y->data[i] = (float)lp[8UL + i]; i = i + 1UL; }
            }
        }
        if (img != (unsigned char*)0) free((void*)img);
        if (lab != (unsigned char*)0) free((void*)lab);
    }
    return d;
}

static int ds_is_space_(unsigned char c) {
    return c == (unsigned char)' ' || c == (unsigned char)'\t' || c == (unsigned char)'\r';
}

// Parses field [lo, hi) of buf as a float (whitespace trimmed).
static float ds_field_(unsigned char* buf, unsigned long lo, unsigned long hi, &int ok) {
    unsafe {
        while (lo < hi && ds_is_space_(buf[lo])) lo = lo + 1UL;
        while (hi > lo && ds_is_space_(buf[hi - 1UL])) hi = hi - 1UL;
        unsigned char saved = buf[hi];
        buf[hi] = (unsigned char)0;
        int good = 0;
        double v = str_to_float((const &char)&buf[lo], (?&int)&good);
        buf[hi] = saved;
        if (!good) *ok = 0;
        return (float)v;
    }
}

struct Dataset dataset_load_csv(const &char path, long labelColumn, int hasHeader, int classification) {
    &heap unsigned char raw;
    unsafe { raw = (&heap unsigned char)0; }
    unsigned long n = ds_read_all_(path, &raw);
    if (n == 0UL) return ds_empty_();
    struct Dataset d = ds_empty_();
    unsafe {
        unsigned char* buf = (unsigned char*)raw;
        // pass 1: rows and columns
        unsigned long rows = 0UL;
        unsigned long cols = 0UL;
        unsigned long pos = 0UL;
        int skip = hasHeader;
        while (pos < n) {
            unsigned long end = pos;
            while (end < n && buf[end] != (unsigned char)'\n') end = end + 1UL;
            unsigned long k = pos;
            int blank = 1;
            while (k < end) { if (!ds_is_space_(buf[k])) blank = 0; k = k + 1UL; }
            if (!blank) {
                if (skip) skip = 0;
                else {
                    unsigned long c = 1UL;
                    k = pos;
                    while (k < end) { if (buf[k] == (unsigned char)',') c = c + 1UL; k = k + 1UL; }
                    if (cols == 0UL) cols = c;
                    if (c == cols) rows = rows + 1UL;
                }
            }
            pos = end + 1UL;
        }
        int labeled = labelColumn >= 0L && (unsigned long)labelColumn < cols;
        unsigned long features = labeled ? cols - 1UL : cols;
        if (rows > 0UL && features > 0UL) {
            d.free();
            d = ds_new_(rows, features, 1UL, 0UL);
            int ok = 1;
            float maxLabel = (float)0.0;
            unsigned long r = 0UL;
            pos = 0UL;
            skip = hasHeader;
            while (pos < n && r < rows) {
                unsigned long end = pos;
                while (end < n && buf[end] != (unsigned char)'\n') end = end + 1UL;
                unsigned long k = pos;
                int blank = 1;
                unsigned long c = 1UL;
                while (k < end) {
                    if (!ds_is_space_(buf[k])) blank = 0;
                    if (buf[k] == (unsigned char)',') c = c + 1UL;
                    k = k + 1UL;
                }
                if (!blank) {
                    if (skip) skip = 0;
                    else if (c == cols) {
                        unsigned long col = 0UL;
                        unsigned long f = 0UL;
                        unsigned long start = pos;
                        k = pos;
                        while (k <= end) {
                            if (k == end || buf[k] == (unsigned char)',') {
                                float v = ds_field_(buf, start, k, &ok);
                                if (labeled && col == (unsigned long)labelColumn) {
                                    d.y->data[r] = v;
                                    if (v > maxLabel) maxLabel = v;
                                } else {
                                    d.x->data[r * features + f] = v;
                                    f = f + 1UL;
                                }
                                col = col + 1UL;
                                start = k + 1UL;
                            }
                            k = k + 1UL;
                        }
                        r = r + 1UL;
                    }
                }
                pos = end + 1UL;
            }
            if (!ok) { d.free(); d = ds_empty_(); }
            else if (labeled && classification) d.classes = (unsigned long)maxLabel + 1UL;
        }
        free((void*)buf);
    }
    return d;
}

// ── Utilities ────────────────────────────────────────────────────────────────

static void ds_copy_row_(&Dataset dst, unsigned long di, const &Dataset src, unsigned long si) {
    unsafe {
        unsigned long f = 0UL;
        while (f < src.features) { dst.x->data[di * src.features + f] = src.x->data[si * src.features + f]; f = f + 1UL; }
        unsigned long t = 0UL;
        while (t < src.targets) { dst.y->data[di * src.targets + t] = src.y->data[si * src.targets + t]; t = t + 1UL; }
    }
}

struct DatasetSplit Dataset::split(float testFraction, &Rng rng) const {
    unsigned long n = self.n;
    unsigned long nTest = (unsigned long)((float)n * testFraction);
    if (nTest > n) nTest = n;
    struct DatasetSplit s;
    s.train = ds_new_(n - nTest, self.features, self.targets, self.classes);
    s.test = ds_new_(nTest, self.features, self.targets, self.classes);
    unsafe {
        unsigned long* order = (unsigned long*)malloc(sizeof(unsigned long) * (n + 1UL));
        unsigned long i = 0UL;
        while (i < n) { order[i] = i; i = i + 1UL; }
        i = n;
        while (i > 1UL) {
            unsigned long j = rng.below(i);
            unsigned long t = order[i - 1UL]; order[i - 1UL] = order[j]; order[j] = t;
            i = i - 1UL;
        }
        i = 0UL;
        while (i < nTest) { ds_copy_row_(&s.test, i, self, order[i]); i = i + 1UL; }
        while (i < n) { ds_copy_row_(&s.train, i - nTest, self, order[i]); i = i + 1UL; }
        free((void*)order);
    }
    return s;
}

static void ds_apply_stats_(&Dataset d, const float* mean, const float* inv) {
    unsafe {
        unsigned long i = 0UL;
        while (i < d.n) {
            unsigned long f = 0UL;
            while (f < d.features) {
                unsigned long k = i * d.features + f;
                d.x->data[k] = (d.x->data[k] - mean[f]) * inv[f];
                f = f + 1UL;
            }
            i = i + 1UL;
        }
    }
}

static void ds_standardize_(&Dataset fit, ?&Dataset other) {
    unsigned long F = fit.features;
    unsafe {
        float* mean = (float*)malloc(sizeof(float) * (F + 1UL));
        float* inv = (float*)malloc(sizeof(float) * (F + 1UL));
        unsigned long f = 0UL;
        while (f < F) {
            double s = 0.0; double sq = 0.0;
            unsigned long i = 0UL;
            while (i < fit.n) {
                double v = (double)fit.x->data[i * F + f];
                s = s + v; sq = sq + v * v;
                i = i + 1UL;
            }
            double m = fit.n > 0UL ? s / (double)fit.n : 0.0;
            double var = fit.n > 0UL ? sq / (double)fit.n - m * m : 0.0;
            mean[f] = (float)m;
            inv[f] = var > 1e-12 ? (float)(1.0 / sqrt_d(var)) : (float)1.0;
            f = f + 1UL;
        }
        ds_apply_stats_(fit, (const float*)mean, (const float*)inv);
        match (other) {
            case some(o): { ds_apply_stats_(o, (const float*)mean, (const float*)inv); }
            case null: { }
        }
        free((void*)mean);
        free((void*)inv);
    }
}

void Dataset::standardize() { ds_standardize_(self, null); }
void Dataset::standardize_pair(&Dataset test) { ds_standardize_(self, (?&Dataset)test); }

void Dataset::free() {
    self.x->free(); self.y->free();
    unsafe { free((void*)self.x); free((void*)self.y); }
    self.n = 0UL;
}

// ── Mini-batches ─────────────────────────────────────────────────────────────

static void dl_shuffle_(&DataLoader dl) {
    unsafe {
        unsigned long i = dl.n;
        while (i > 1UL) {
            unsigned long j = dl.rng.below(i);
            unsigned long t = dl.order[i - 1UL]; dl.order[i - 1UL] = dl.order[j]; dl.order[j] = t;
            i = i - 1UL;
        }
    }
}

struct DataLoader dataloader_new(const &Dataset ds, unsigned long batchSize, int shuffle, unsigned long seed) {
    struct DataLoader dl;
    dl.x = ds.x;
    dl.y = ds.y;
    dl.n = ds.n;
    dl.batchSize = batchSize == 0UL ? 1UL : batchSize;
    dl.pos = 0UL;
    dl.shuffle = shuffle;
    dl.rng = rng_new(seed);
    unsafe {
        unsigned long* order = (unsigned long*)malloc(sizeof(unsigned long) * (ds.n + 1UL));
        unsigned long i = 0UL;
        while (i < ds.n) { order[i] = i; i = i + 1UL; }
        dl.order = (&heap unsigned long)order;
    }
    if (shuffle) dl_shuffle_(&dl);
    return dl;
}

int DataLoader::next(&Batch out) {
    if (self.pos >= self.n) {
        self.pos = 0UL;
        if (self.shuffle) dl_shuffle_(self);
        return 0;
    }
    unsigned long rows = self.n - self.pos;
    if (rows > self.batchSize) rows = self.batchSize;
    unsigned long F; unsigned long T;
    unsafe { F = self.x->shape[1]; T = self.y->shape[1]; }
    out.x = tensor_new_2d(rows, F, 0);
    out.y = tensor_new_2d(rows, T, 0);
    out.rows = rows;
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long src = self.order[self.pos + r];
            unsigned long f = 0UL;
            while (f < F) { out.x->data[r * F + f] = self.x->data[src * F + f]; f = f + 1UL; }
            unsigned long t = 0UL;
            while (t < T) { out.y->data[r * T + t] = self.y->data[src * T + t]; t = t + 1UL; }
            r = r + 1UL;
        }
    }
    self.pos = self.pos + rows;
    return 1;
}

unsigned long DataLoader::batches() const {
    return (self.n + self.batchSize - 1UL) / self.batchSize;
}

void DataLoader::free() {
    unsafe { free((void*)self.order); }
}

} // namespace std
