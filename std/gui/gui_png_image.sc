// SafeC Standard Library — GUI: GuiImage helpers (see gui_png.h): blank
// images, screenshots, PNG files, alpha conversion and resampling.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_png.h>
#include <std/mem.h>
#include <std/io_file.h>
#include <std/io_traits.h>
extern void* memcpy(void* dst, const void* src, size_t n);

namespace std {

struct GuiImage GuiImage::new(int width, int height) {
    struct GuiImage img;
    if (width < 1) { width = 1; }
    if (height < 1) { height = 1; }
    unsafe { img.pixels = (&heap unsigned char)alloc_zeroed_raw_((unsigned long)width * (unsigned long)height * 4UL); }
    img.width = width;
    img.height = height;
    img.premultiplied = 1;
    return img;
}

struct GuiImage GuiImage::from_window(const &GuiWindow win) {
    struct GuiImage img = GuiImage::new(win.width, win.height);
    unsafe {
        memcpy((void*)img.pixels, (const void*)win.pixels, (unsigned long)win.width * (unsigned long)win.height * 4UL);
    }
    return img;
}

int GuiImage::load_png_file(const &char path) {
    unsafe {
        self.pixels = (&heap unsigned char)0;
        self.width = 0;
        self.height = 0;
        self.premultiplied = 0;
    }
    ?&File fq = File::open(path, "rb");
    match (fq) {
        case null: return 0;
        case some(f): {
            long long n = f.size();
            if (n <= 0LL || n > 0x40000000LL) { f.close(); return 0; }
            unsigned char* buf;
            unsafe { buf = (unsigned char*)alloc_raw_((unsigned long)n); }
            int ok;
            unsafe { ok = read_exact(f, ((unsigned char*)buf)[0..(unsigned long)n]); }
            f.close();
            int r = 0;
            if (ok) { unsafe { r = self.load_png((const unsigned char*)buf, (unsigned long)n); } }
            unsafe { dealloc_raw_((void*)buf); }
            return r;
        }
    }
    return 0;
}

// ── PNG writing ───────────────────────────────────────────────────────────

static unsigned int gGuiCrcTable[256];
static int gGuiCrcReady = 0;

static unsigned int __gui_crc(unsigned int crc, const unsigned char* p, unsigned long n) {
    if (!gGuiCrcReady) {
        unsigned int i = 0U;
        while (i < 256U) {
            unsigned int c = i;
            int k = 0;
            while (k < 8) {
                c = (c & 1U) != 0U ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
                k = k + 1;
            }
            gGuiCrcTable[i] = c;
            i = i + 1U;
        }
        gGuiCrcReady = 1;
    }
    unsigned long i = 0UL;
    unsafe {
        while (i < n) {
            crc = gGuiCrcTable[(crc ^ (unsigned int)p[i]) & 0xFFU] ^ (crc >> 8);
            i = i + 1UL;
        }
    }
    return crc;
}

static void __gui_be32(unsigned char* p, unsigned int v) {
    unsafe {
        p[0] = (unsigned char)(v >> 24);
        p[1] = (unsigned char)(v >> 16);
        p[2] = (unsigned char)(v >> 8);
        p[3] = (unsigned char)v;
    }
}

static int __gui_png_chunk(&File f, const char* type, const unsigned char* data, unsigned long n) {
    unsigned char head[8];
    unsafe {
        __gui_be32(&head[0], (unsigned int)n);
        head[4] = (unsigned char)type[0]; head[5] = (unsigned char)type[1];
        head[6] = (unsigned char)type[2]; head[7] = (unsigned char)type[3];
    }
    unsigned int crc = __gui_crc(0xFFFFFFFFU, &head[4], 4UL);
    crc = __gui_crc(crc, data, n) ^ 0xFFFFFFFFU;
    unsigned char tail[4];
    __gui_be32(&tail[0], crc);
    int ok = 1;
    unsafe {
        if (write_all(f, head[0..8UL]) == 0) { ok = 0; }
        if (n > 0UL && write_all(f, ((const unsigned char*)data)[0..n]) == 0) { ok = 0; }
        if (write_all(f, tail[0..4UL]) == 0) { ok = 0; }
    }
    return ok;
}

int GuiImage::save_png(const &char path) const {
    int w = self.width;
    int h = self.height;
    if (w < 1 || h < 1) { return 0; }
    // Raw scanlines: filter byte 0 + RGBA (straight alpha).
    unsigned long rowBytes = (unsigned long)w * 4UL + 1UL;
    unsigned long rawLen = rowBytes * (unsigned long)h;
    unsigned char* raw;
    unsafe { raw = (unsigned char*)alloc_raw_(rawLen); }
    int y = 0;
    unsafe {
        while (y < h) {
            unsigned char* dst = raw + (unsigned long)y * rowBytes;
            dst[0] = (unsigned char)0;
            const unsigned char* src = (const unsigned char*)self.pixels + (unsigned long)y * (unsigned long)w * 4UL;
            int x = 0;
            while (x < w) {
                unsigned int a = (unsigned int)src[3];
                unsigned int r = (unsigned int)src[0];
                unsigned int g = (unsigned int)src[1];
                unsigned int b = (unsigned int)src[2];
                if (self.premultiplied && a > 0U && a < 255U) {
                    r = (r * 255U + a / 2U) / a; if (r > 255U) { r = 255U; }
                    g = (g * 255U + a / 2U) / a; if (g > 255U) { g = 255U; }
                    b = (b * 255U + a / 2U) / a; if (b > 255U) { b = 255U; }
                }
                dst[1 + x * 4] = (unsigned char)r;
                dst[2 + x * 4] = (unsigned char)g;
                dst[3 + x * 4] = (unsigned char)b;
                dst[4 + x * 4] = (unsigned char)a;
                src = src + 4;
                x = x + 1;
            }
            y = y + 1;
        }
    }
    // zlib stream of stored DEFLATE blocks (<= 65535 bytes each).
    unsigned long nblocks = (rawLen + 65534UL) / 65535UL;
    unsigned long zLen = 2UL + rawLen + nblocks * 5UL + 4UL;
    unsigned char* z;
    unsafe { z = (unsigned char*)alloc_raw_(zLen); }
    unsigned long o = 0UL;
    unsigned int s1 = 1U;
    unsigned int s2 = 0U;
    unsafe {
        z[0] = (unsigned char)0x78; z[1] = (unsigned char)0x01;
        o = 2UL;
        unsigned long pos = 0UL;
        while (pos < rawLen) {
            unsigned long n = rawLen - pos;
            if (n > 65535UL) { n = 65535UL; }
            z[o] = (unsigned char)(pos + n >= rawLen ? 1 : 0);
            z[o + 1UL] = (unsigned char)(n & 0xFFUL);
            z[o + 2UL] = (unsigned char)((n >> 8) & 0xFFUL);
            z[o + 3UL] = (unsigned char)(~n & 0xFFUL);
            z[o + 4UL] = (unsigned char)((~n >> 8) & 0xFFUL);
            o = o + 5UL;
            memcpy((void*)(z + o), (const void*)(raw + pos), n);
            unsigned long k = 0UL;
            while (k < n) {
                s1 = (s1 + (unsigned int)raw[pos + k]) % 65521U;
                s2 = (s2 + s1) % 65521U;
                k = k + 1UL;
            }
            o = o + n;
            pos = pos + n;
        }
        __gui_be32(z + o, (s2 << 16) | s1);
        o = o + 4UL;
    }
    ?&File fq = File::open(path, "wb");
    int ok = 0;
    match (fq) {
        case null: { }
        case some(f): {
            unsigned char sig[8];
            sig[0] = (unsigned char)137; sig[1] = (unsigned char)80; sig[2] = (unsigned char)78; sig[3] = (unsigned char)71;
            sig[4] = (unsigned char)13; sig[5] = (unsigned char)10; sig[6] = (unsigned char)26; sig[7] = (unsigned char)10;
            unsigned char ihdr[13];
            __gui_be32(&ihdr[0], (unsigned int)w);
            __gui_be32(&ihdr[4], (unsigned int)h);
            ihdr[8] = (unsigned char)8;     // bit depth
            ihdr[9] = (unsigned char)6;     // RGBA
            ihdr[10] = (unsigned char)0; ihdr[11] = (unsigned char)0; ihdr[12] = (unsigned char)0;
            ok = 1;
            unsafe {
                if (write_all(f, sig[0..8UL]) == 0) { ok = 0; }
            }
            if (!__gui_png_chunk(f, "IHDR", &ihdr[0], 13UL)) { ok = 0; }
            if (!__gui_png_chunk(f, "IDAT", z, o)) { ok = 0; }
            unsafe { if (!__gui_png_chunk(f, "IEND", (const unsigned char*)&ihdr[0], 0UL)) { ok = 0; } }
            f.close();
        }
    }
    unsafe {
        dealloc_raw_((void*)raw);
        dealloc_raw_((void*)z);
    }
    return ok;
}

// ── alpha and copies ──────────────────────────────────────────────────────

void GuiImage::premultiply() {
    if (self.premultiplied) { return; }
    unsigned long n = (unsigned long)self.width * (unsigned long)self.height;
    unsafe {
        unsigned char* p = (unsigned char*)self.pixels;
        unsigned long i = 0UL;
        while (i < n) {
            unsigned int a = (unsigned int)p[3];
            if (a != 255U) {
                p[0] = (unsigned char)(((unsigned int)p[0] * a + 127U) / 255U);
                p[1] = (unsigned char)(((unsigned int)p[1] * a + 127U) / 255U);
                p[2] = (unsigned char)(((unsigned int)p[2] * a + 127U) / 255U);
            }
            p = p + 4;
            i = i + 1UL;
        }
    }
    self.premultiplied = 1;
}

void GuiImage::unpremultiply() {
    if (!self.premultiplied) { return; }
    unsigned long n = (unsigned long)self.width * (unsigned long)self.height;
    unsafe {
        unsigned char* p = (unsigned char*)self.pixels;
        unsigned long i = 0UL;
        while (i < n) {
            unsigned int a = (unsigned int)p[3];
            if (a != 255U && a != 0U) {
                unsigned int r = ((unsigned int)p[0] * 255U + a / 2U) / a;
                unsigned int g = ((unsigned int)p[1] * 255U + a / 2U) / a;
                unsigned int b = ((unsigned int)p[2] * 255U + a / 2U) / a;
                p[0] = (unsigned char)(r > 255U ? 255U : r);
                p[1] = (unsigned char)(g > 255U ? 255U : g);
                p[2] = (unsigned char)(b > 255U ? 255U : b);
            }
            p = p + 4;
            i = i + 1UL;
        }
    }
    self.premultiplied = 0;
}

struct GuiImage GuiImage::copy() const {
    struct GuiImage img = GuiImage::new(self.width, self.height);
    unsafe {
        memcpy((void*)img.pixels, (const void*)self.pixels, (unsigned long)self.width * (unsigned long)self.height * 4UL);
    }
    img.premultiplied = self.premultiplied;
    return img;
}

unsigned int GuiImage::get_pixel(int x, int y) const {
    if (x < 0 || y < 0 || x >= self.width || y >= self.height) { return 0U; }
    unsafe {
        unsigned long i = ((unsigned long)y * (unsigned long)self.width + (unsigned long)x) * 4UL;
        return ((unsigned int)self.pixels[i] << 24) | ((unsigned int)self.pixels[i + 1UL] << 16) |
               ((unsigned int)self.pixels[i + 2UL] << 8) | (unsigned int)self.pixels[i + 3UL];
    }
}

void GuiImage::set_pixel(int x, int y, unsigned int rgba) {
    if (x < 0 || y < 0 || x >= self.width || y >= self.height) { return; }
    unsafe {
        unsigned long i = ((unsigned long)y * (unsigned long)self.width + (unsigned long)x) * 4UL;
        self.pixels[i] = (unsigned char)(rgba >> 24);
        self.pixels[i + 1UL] = (unsigned char)(rgba >> 16);
        self.pixels[i + 2UL] = (unsigned char)(rgba >> 8);
        self.pixels[i + 3UL] = (unsigned char)rgba;
    }
}

// Resampling works on premultiplied values (so transparent pixels don't
// bleed their color into the result).
struct GuiImage GuiImage::scaled(int width, int height) const {
    if (width < 1) { width = 1; }
    if (height < 1) { height = 1; }
    struct GuiImage src = self.copy();
    src.premultiply();
    struct GuiImage dst = GuiImage::new(width, height);
    int sw = src.width;
    int sh = src.height;
    double fx = (double)sw / (double)width;
    double fy = (double)sh / (double)height;
    unsafe {
        const unsigned char* sp = (const unsigned char*)src.pixels;
        unsigned char* dp = (unsigned char*)dst.pixels;
        int y = 0;
        while (y < height) {
            int x = 0;
            while (x < width) {
                double acc0 = 0.0; double acc1 = 0.0; double acc2 = 0.0; double acc3 = 0.0;
                if (fx > 1.0 || fy > 1.0) {
                    // Area average over the source box this pixel covers.
                    double bx0 = (double)x * fx; double bx1 = bx0 + fx;
                    double by0 = (double)y * fy; double by1 = by0 + fy;
                    if (fx < 1.0) { double c = (bx0 + bx1) * 0.5; bx0 = c - 0.5; bx1 = c + 0.5; }
                    if (fy < 1.0) { double c = (by0 + by1) * 0.5; by0 = c - 0.5; by1 = c + 0.5; }
                    double total = 0.0;
                    int iy = (int)by0;
                    while ((double)iy < by1) {
                        double wy = ((double)(iy + 1) < by1 ? (double)(iy + 1) : by1) - ((double)iy > by0 ? (double)iy : by0);
                        int cy = iy < 0 ? 0 : (iy >= sh ? sh - 1 : iy);
                        int ix = (int)bx0;
                        while ((double)ix < bx1) {
                            double wx = ((double)(ix + 1) < bx1 ? (double)(ix + 1) : bx1) - ((double)ix > bx0 ? (double)ix : bx0);
                            int cx = ix < 0 ? 0 : (ix >= sw ? sw - 1 : ix);
                            double wgt = wx * wy;
                            const unsigned char* q = sp + ((unsigned long)cy * (unsigned long)sw + (unsigned long)cx) * 4UL;
                            acc0 = acc0 + (double)q[0] * wgt; acc1 = acc1 + (double)q[1] * wgt;
                            acc2 = acc2 + (double)q[2] * wgt; acc3 = acc3 + (double)q[3] * wgt;
                            total = total + wgt;
                            ix = ix + 1;
                        }
                        iy = iy + 1;
                    }
                    if (total > 0.0) { acc0 = acc0 / total; acc1 = acc1 / total; acc2 = acc2 / total; acc3 = acc3 / total; }
                } else {
                    // Bilinear between the four nearest source pixels.
                    double sx = ((double)x + 0.5) * fx - 0.5;
                    double sy = ((double)y + 0.5) * fy - 0.5;
                    if (sx < 0.0) { sx = 0.0; }
                    if (sy < 0.0) { sy = 0.0; }
                    int x0 = (int)sx; int y0 = (int)sy;
                    int x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
                    int y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
                    double tx = sx - (double)x0; double ty = sy - (double)y0;
                    const unsigned char* q00 = sp + ((unsigned long)y0 * (unsigned long)sw + (unsigned long)x0) * 4UL;
                    const unsigned char* q10 = sp + ((unsigned long)y0 * (unsigned long)sw + (unsigned long)x1) * 4UL;
                    const unsigned char* q01 = sp + ((unsigned long)y1 * (unsigned long)sw + (unsigned long)x0) * 4UL;
                    const unsigned char* q11 = sp + ((unsigned long)y1 * (unsigned long)sw + (unsigned long)x1) * 4UL;
                    double w00 = (1.0 - tx) * (1.0 - ty); double w10 = tx * (1.0 - ty);
                    double w01 = (1.0 - tx) * ty; double w11 = tx * ty;
                    acc0 = (double)q00[0] * w00 + (double)q10[0] * w10 + (double)q01[0] * w01 + (double)q11[0] * w11;
                    acc1 = (double)q00[1] * w00 + (double)q10[1] * w10 + (double)q01[1] * w01 + (double)q11[1] * w11;
                    acc2 = (double)q00[2] * w00 + (double)q10[2] * w10 + (double)q01[2] * w01 + (double)q11[2] * w11;
                    acc3 = (double)q00[3] * w00 + (double)q10[3] * w10 + (double)q01[3] * w01 + (double)q11[3] * w11;
                }
                unsigned char* o = dp + ((unsigned long)y * (unsigned long)width + (unsigned long)x) * 4UL;
                o[0] = (unsigned char)(int)(acc0 + 0.5);
                o[1] = (unsigned char)(int)(acc1 + 0.5);
                o[2] = (unsigned char)(int)(acc2 + 0.5);
                o[3] = (unsigned char)(int)(acc3 + 0.5);
                x = x + 1;
            }
            y = y + 1;
        }
    }
    src.free();
    if (!self.premultiplied) { dst.unpremultiply(); }
    return dst;
}

} // namespace std
