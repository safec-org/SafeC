// SafeC Standard Library — GUI: the canvas' scan converter and compositor
// (see gui_canvas.h for the model).
//
// Pipeline: a path is flattened into polylines (curves subdivided to a
// device-space tolerance), the polylines become edges, and the edges are
// scan-converted row by row: 16 sample lines per pixel row, each crossing
// sorted by x and walked with the fill rule, every inside span adding its
// exact horizontal coverage to a difference buffer (+ at the span's start
// pixel, split by the fractional position, - at its end), and one prefix
// sum per row turns that into per-pixel coverage. The coverage row is then
// composited through the clip mask with the paint (solid / gradient LUT /
// image sampler) and the blend mode, four channels at a time.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_path.h>
#include <std/gui/gui_png.h>
#include <std/mem.h>
#include <std/math.h>
#include <std/collections/vec.h>

namespace std {

#define GUI_AA_SAMPLES 16

// ── scratch buffers ───────────────────────────────────────────────────────

static float* __gui_vf(&Vec<float> v, unsigned long n) {
    if (v.cap < n) { v.reserve(n + n / 2UL + 16UL); }
    v.len = n;
    unsafe { return (float*)v.data; }
}

static int* __gui_vi(&Vec<int> v, unsigned long n) {
    if (v.cap < n) { v.reserve(n + n / 2UL + 16UL); }
    v.len = n;
    unsafe { return (int*)v.data; }
}

// ── flattening ────────────────────────────────────────────────────────────

static void __gui_poly_begin(&Vec<float> pts, &Vec<int> cont) {
    cont.push((int)(pts.len / 2UL));
    cont.push(0);
    cont.push(0);
}

static void __gui_poly_point(&Vec<float> pts, &Vec<int> cont, float x, float y) {
    unsigned long n = cont.len;
    if (n < 3UL) { __gui_poly_begin(pts, cont); n = cont.len; }
    pts.push(x);
    pts.push(y);
    unsafe {
        int* c = (int*)cont.data;
        c[n - 2UL] = c[n - 2UL] + 1;
    }
}

static void __gui_poly_close(&Vec<int> cont) {
    unsigned long n = cont.len;
    if (n < 3UL) { return; }
    unsafe {
        int* c = (int*)cont.data;
        c[n - 1UL] = 1;
    }
}

// Flattens 'p' transformed by 'm' into polylines with at most 'tol' error.
void __gui_canvas_flatten(const &GuiPath p, struct GuiMatrix m, float tol, &Vec<float> pts, &Vec<int> cont) {
    pts.clear();
    cont.clear();
    if (tol < 0.001f) { tol = 0.001f; }
    float cx = 0.0f; float cy = 0.0f;   // current point, device space
    float sx = 0.0f; float sy = 0.0f;
    int open = 0;
    unsigned long i = 0UL;
    while (i < p.segs.len) {
        struct GuiPathSeg s = *p.segs.at(i);
        i = i + 1UL;
        if (s.verb == GUI_PATH_MOVE) {
            cx = m.apply_x(s.x1, s.y1);
            cy = m.apply_y(s.x1, s.y1);
            sx = cx; sy = cy;
            __gui_poly_begin(pts, cont);
            __gui_poly_point(pts, cont, cx, cy);
            open = 1;
            continue;
        }
        if (s.verb == GUI_PATH_CLOSE) {
            if (open) { __gui_poly_close(cont); }
            open = 0;
            cx = sx; cy = sy;
            continue;
        }
        if (!open) {
            // A segment after a close starts a new subpath at the old start.
            __gui_poly_begin(pts, cont);
            __gui_poly_point(pts, cont, cx, cy);
            open = 1;
        }
        if (s.verb == GUI_PATH_LINE) {
            cx = m.apply_x(s.x1, s.y1);
            cy = m.apply_y(s.x1, s.y1);
            __gui_poly_point(pts, cont, cx, cy);
        } else if (s.verb == GUI_PATH_QUAD) {
            float x1 = m.apply_x(s.x1, s.y1); float y1 = m.apply_y(s.x1, s.y1);
            float x2 = m.apply_x(s.x2, s.y2); float y2 = m.apply_y(s.x2, s.y2);
            float ddx = cx - 2.0f * x1 + x2;
            float ddy = cy - 2.0f * y1 + y2;
            float dd = sqrt_f(ddx * ddx + ddy * ddy);
            int n = (int)ceil_f(sqrt_f(dd / (4.0f * tol)));
            if (n < 1) { n = 1; }
            if (n > 200) { n = 200; }
            int k = 1;
            while (k <= n) {
                float t = (float)k / (float)n;
                float u = 1.0f - t;
                float px = u * u * cx + 2.0f * u * t * x1 + t * t * x2;
                float py = u * u * cy + 2.0f * u * t * y1 + t * t * y2;
                __gui_poly_point(pts, cont, px, py);
                k = k + 1;
            }
            cx = x2; cy = y2;
        } else if (s.verb == GUI_PATH_CUBIC) {
            float x1 = m.apply_x(s.x1, s.y1); float y1 = m.apply_y(s.x1, s.y1);
            float x2 = m.apply_x(s.x2, s.y2); float y2 = m.apply_y(s.x2, s.y2);
            float x3 = m.apply_x(s.x3, s.y3); float y3 = m.apply_y(s.x3, s.y3);
            float ax = cx - 2.0f * x1 + x2; float ay = cy - 2.0f * y1 + y2;
            float bx = x1 - 2.0f * x2 + x3; float by = y1 - 2.0f * y2 + y3;
            float d1 = sqrt_f(ax * ax + ay * ay);
            float d2 = sqrt_f(bx * bx + by * by);
            float dd = d1 > d2 ? d1 : d2;
            int n = (int)ceil_f(sqrt_f(0.75f * dd / tol));
            if (n < 1) { n = 1; }
            if (n > 300) { n = 300; }
            int k = 1;
            while (k <= n) {
                float t = (float)k / (float)n;
                float u = 1.0f - t;
                float px = u * u * u * cx + 3.0f * u * u * t * x1 + 3.0f * u * t * t * x2 + t * t * t * x3;
                float py = u * u * u * cy + 3.0f * u * u * t * y1 + 3.0f * u * t * t * y2 + t * t * t * y3;
                __gui_poly_point(pts, cont, px, py);
                k = k + 1;
            }
            cx = x3; cy = y3;
        }
    }
}

// Polylines (x,y pairs + triples) to device-space edges; every contour is
// closed for filling. 'm' transforms the points first (identity: none).
void __gui_canvas_edges(&GuiCanvas c, &Vec<float> pts, &Vec<int> cont, struct GuiMatrix m, int transform) {
    c.edges.clear();
    unsafe {
        const float* p = (const float*)pts.data;
        const int* ct = (const int*)cont.data;
        unsigned long nc = cont.len / 3UL;
        unsigned long k = 0UL;
        while (k < nc) {
            int first = ct[k * 3UL];
            int count = ct[k * 3UL + 1UL];
            k = k + 1UL;
            if (count < 2) { continue; }
            int i = 0;
            while (i < count) {
                int j = i + 1 < count ? i + 1 : 0;
                float ax = p[(first + i) * 2];
                float ay = p[(first + i) * 2 + 1];
                float bx = p[(first + j) * 2];
                float by = p[(first + j) * 2 + 1];
                if (transform) {
                    float tax = m.apply_x(ax, ay); float tay = m.apply_y(ax, ay);
                    float tbx = m.apply_x(bx, by); float tby = m.apply_y(bx, by);
                    ax = tax; ay = tay; bx = tbx; by = tby;
                }
                i = i + 1;
                if (ay == by) { continue; }
                if (ay != ay || by != by || ax != ax || bx != bx) { continue; }   // NaN
                struct GuiEdge e;
                if (ay < by) {
                    e.x0 = ax; e.y0 = ay; e.x1 = bx; e.y1 = by; e.dir = 1;
                } else {
                    e.x0 = bx; e.y0 = by; e.x1 = ax; e.y1 = ay; e.dir = -1;
                }
                e.dxdy = (e.x1 - e.x0) / (e.y1 - e.y0);
                c.edges.push(e);
            }
        }
    }
}

// ── paint evaluation ──────────────────────────────────────────────────────

struct GuiPaintCtx {
    int kind;
    vec<float, 4> solid;          // premultiplied 0..1
    float ia; float ib; float ic; float id; float ie; float iF;   // device -> paint space
    float alpha;                  // global alpha * paint opacity
    int spread;
    int filter;
    const float* lut;             // 256 x RGBA, premultiplied
    const unsigned char* img;
    int iw; int ih;
    int imgPremul;
    // radial
    float cdx; float cdy; float dr; float aa;
    float x0; float y0; float r0;
};

static float __gui_spread(float t, int spread) {
    if (spread == GUI_SPREAD_REPEAT) { return t - floor_f(t); }
    if (spread == GUI_SPREAD_REFLECT) {
        float u = t - 2.0f * floor_f(t * 0.5f);   // 0..2
        return u > 1.0f ? 2.0f - u : u;
    }
    if (t < 0.0f) { return 0.0f; }
    if (t > 1.0f) { return 1.0f; }
    return t;
}

static vec<float, 4> __gui_premul(struct GuiColor c) {
    float a = (float)c.a / 255.0f;
    vec<float, 4> v = { (float)c.r / 255.0f * a, (float)c.g / 255.0f * a, (float)c.b / 255.0f * a, a };
    return v;
}

static void __gui_build_lut(&GuiCanvas c, const &GuiPaint p) {
    float* lut = __gui_vf(&c.lut, 1024UL);
    int n = p.nstops;
    // Stops sorted by offset (stable insertion sort of a copy).
    struct GuiColorStop st[GUI_GRADIENT_MAX_STOPS];
    int i = 0;
    while (i < n) { st[i] = p.stops[i]; i = i + 1; }
    i = 1;
    while (i < n) {
        struct GuiColorStop key = st[i];
        int j = i - 1;
        while (j >= 0 && st[j].offset > key.offset) { st[j + 1] = st[j]; j = j - 1; }
        st[j + 1] = key;
        i = i + 1;
    }
    unsafe {
        int k = 0;
        while (k < 256) {
            float t = (float)k / 255.0f;
            vec<float, 4> col = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (n == 1 || (n > 0 && t <= st[0].offset)) {
                col = __gui_premul(st[0].color);
            } else if (n > 0 && t >= st[n - 1].offset) {
                col = __gui_premul(st[n - 1].color);
            } else if (n > 1) {
                int s = 0;
                while (s < n - 1 && st[s + 1].offset < t) { s = s + 1; }
                float o0 = st[s].offset;
                float o1 = st[s + 1].offset;
                float f = o1 > o0 ? (t - o0) / (o1 - o0) : 0.0f;
                vec<float, 4> a = __gui_premul(st[s].color);
                vec<float, 4> b = __gui_premul(st[s + 1].color);
                vec<float, 4> fv = { f, f, f, f };
                col = a + (b - a) * fv;
            }
            lut[k * 4] = col[0];
            lut[k * 4 + 1] = col[1];
            lut[k * 4 + 2] = col[2];
            lut[k * 4 + 3] = col[3];
            k = k + 1;
        }
    }
}

// Prepares 'pc' for painting with 'p' under the canvas transform.
static int __gui_paint_ctx(&GuiCanvas c, const &GuiPaint p, &GuiPaintCtx pc) {
    pc.kind = p.kind;
    pc.alpha = c.st.alpha * p.opacity;
    pc.spread = p.spread;
    pc.filter = p.filter;
    pc.lut = (const float*)0;
    pc.img = (const unsigned char*)0;
    pc.iw = 0; pc.ih = 0; pc.imgPremul = 1;
    if (p.kind == GUI_PAINT_SOLID) {
        pc.solid = __gui_premul(p.color);
        return 1;
    }
    struct GuiMatrix total = c.st.m.mul(p.matrix);
    struct GuiMatrix inv;
    if (!total.invert(&inv)) { return 0; }
    pc.ia = inv.a; pc.ib = inv.b; pc.ic = inv.c; pc.id = inv.d; pc.ie = inv.e; pc.iF = inv.f;
    if (p.kind == GUI_PAINT_IMAGE) {
        match (p.image) {
            case null: return 0;
            case some(img): {
                unsafe { pc.img = (const unsigned char*)img.pixels; }
                pc.iw = img.width;
                pc.ih = img.height;
                pc.imgPremul = img.premultiplied;
            }
        }
        if (pc.iw < 1 || pc.ih < 1) { return 0; }
        return 1;
    }
    if (p.nstops <= 0) { return 0; }
    __gui_build_lut(c, p);
    unsafe { pc.lut = (const float*)c.lut.data; }
    pc.x0 = p.x0; pc.y0 = p.y0; pc.r0 = p.r0;
    if (p.kind == GUI_PAINT_LINEAR) {
        float dx = p.x1 - p.x0;
        float dy = p.y1 - p.y0;
        float len2 = dx * dx + dy * dy;
        if (len2 <= 0.0f) { len2 = 1e-12f; }
        pc.cdx = dx / len2;
        pc.cdy = dy / len2;
    } else if (p.kind == GUI_PAINT_RADIAL) {
        pc.cdx = p.x1 - p.x0;
        pc.cdy = p.y1 - p.y0;
        pc.dr = p.r1 - p.r0;
        pc.aa = pc.cdx * pc.cdx + pc.cdy * pc.cdy - pc.dr * pc.dr;
    }
    return 1;
}

static vec<float, 4> __gui_lut_at(const &GuiPaintCtx pc, float t) {
    unsafe {
        int k = (int)(t * 255.0f + 0.5f);
        if (k < 0) { k = 0; }
        if (k > 255) { k = 255; }
        const float* l = pc.lut + k * 4;
        vec<float, 4> v = { l[0], l[1], l[2], l[3] };
        return v;
    }
}

static vec<float, 4> __gui_img_texel(const &GuiPaintCtx pc, int x, int y) {
    vec<float, 4> zero = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (pc.spread == GUI_SPREAD_NONE) {
        if (x < 0 || y < 0 || x >= pc.iw || y >= pc.ih) { return zero; }
    } else if (pc.spread == GUI_SPREAD_REPEAT) {
        x = x % pc.iw; if (x < 0) { x = x + pc.iw; }
        y = y % pc.ih; if (y < 0) { y = y + pc.ih; }
    } else if (pc.spread == GUI_SPREAD_REFLECT) {
        int pw = pc.iw * 2; int ph = pc.ih * 2;
        x = x % pw; if (x < 0) { x = x + pw; }
        y = y % ph; if (y < 0) { y = y + ph; }
        if (x >= pc.iw) { x = pw - 1 - x; }
        if (y >= pc.ih) { y = ph - 1 - y; }
    } else {
        if (x < 0) { x = 0; }
        if (y < 0) { y = 0; }
        if (x >= pc.iw) { x = pc.iw - 1; }
        if (y >= pc.ih) { y = pc.ih - 1; }
    }
    unsafe {
        const unsigned char* q = pc.img + ((unsigned long)y * (unsigned long)pc.iw + (unsigned long)x) * 4UL;
        float a = (float)q[3] / 255.0f;
        if (pc.imgPremul) {
            vec<float, 4> v = { (float)q[0] / 255.0f, (float)q[1] / 255.0f, (float)q[2] / 255.0f, a };
            return v;
        }
        vec<float, 4> v = { (float)q[0] / 255.0f * a, (float)q[1] / 255.0f * a, (float)q[2] / 255.0f * a, a };
        return v;
    }
}

// The paint's premultiplied color at device pixel center (px, py).
static vec<float, 4> __gui_paint_at(const &GuiPaintCtx pc, float px, float py) {
    if (pc.kind == GUI_PAINT_SOLID) { return pc.solid; }
    float u = pc.ia * px + pc.ic * py + pc.ie;
    float v = pc.ib * px + pc.id * py + pc.iF;
    if (pc.kind == GUI_PAINT_LINEAR) {
        float t = (u - pc.x0) * pc.cdx + (v - pc.y0) * pc.cdy;
        return __gui_lut_at(pc, __gui_spread(t, pc.spread));
    }
    if (pc.kind == GUI_PAINT_RADIAL) {
        // Largest t with |p - c(t)| = r(t), r(t) >= 0 (two-circle gradient).
        float qx = u - pc.x0;
        float qy = v - pc.y0;
        float b = qx * pc.cdx + qy * pc.cdy + pc.r0 * pc.dr;
        float cc = qx * qx + qy * qy - pc.r0 * pc.r0;
        float t = 0.0f;
        if (pc.aa > -1e-6f && pc.aa < 1e-6f) {
            if (b > -1e-9f && b < 1e-9f) { vec<float, 4> z = { 0.0f, 0.0f, 0.0f, 0.0f }; return z; }
            t = cc / (2.0f * b);
        } else {
            float disc = b * b - pc.aa * cc;
            if (disc < 0.0f) { vec<float, 4> z = { 0.0f, 0.0f, 0.0f, 0.0f }; return z; }
            float sq = sqrt_f(disc);
            float t1 = (b + sq) / pc.aa;
            float t2 = (b - sq) / pc.aa;
            t = t1 > t2 ? t1 : t2;
            if (pc.r0 + t * pc.dr < 0.0f) {
                t = t1 > t2 ? t2 : t1;
                if (pc.r0 + t * pc.dr < 0.0f) { vec<float, 4> z = { 0.0f, 0.0f, 0.0f, 0.0f }; return z; }
            }
        }
        return __gui_lut_at(pc, __gui_spread(t, pc.spread));
    }
    if (pc.kind == GUI_PAINT_SWEEP) {
        float ang = atan2_f(v - pc.y0, u - pc.x0) - pc.r0;
        float t = ang / (2.0f * 3.14159265358979f);
        t = t - floor_f(t);
        return __gui_lut_at(pc, t);
    }
    // GUI_PAINT_IMAGE
    if (pc.filter == GUI_FILTER_NEAREST) {
        return __gui_img_texel(pc, (int)floor_f(u), (int)floor_f(v));
    }
    float fu = u - 0.5f;
    float fv = v - 0.5f;
    int x0 = (int)floor_f(fu);
    int y0 = (int)floor_f(fv);
    float tx = fu - (float)x0;
    float ty = fv - (float)y0;
    vec<float, 4> a = __gui_img_texel(pc, x0, y0);
    vec<float, 4> b = __gui_img_texel(pc, x0 + 1, y0);
    vec<float, 4> c = __gui_img_texel(pc, x0, y0 + 1);
    vec<float, 4> d = __gui_img_texel(pc, x0 + 1, y0 + 1);
    vec<float, 4> txv = { tx, tx, tx, tx };
    vec<float, 4> tyv = { ty, ty, ty, ty };
    vec<float, 4> top = a + (b - a) * txv;
    vec<float, 4> bot = c + (d - c) * txv;
    return top + (bot - top) * tyv;
}

// ── blending ──────────────────────────────────────────────────────────────

static float __gui_blend_channel(int mode, float cs, float cd) {
    if (mode == GUI_BLEND_MULTIPLY) { return cs * cd; }
    if (mode == GUI_BLEND_SCREEN) { return cs + cd - cs * cd; }
    if (mode == GUI_BLEND_OVERLAY) {
        return cd <= 0.5f ? cs * 2.0f * cd : cs + (2.0f * cd - 1.0f) - cs * (2.0f * cd - 1.0f);
    }
    if (mode == GUI_BLEND_DARKEN) { return cs < cd ? cs : cd; }
    if (mode == GUI_BLEND_LIGHTEN) { return cs > cd ? cs : cd; }
    if (mode == GUI_BLEND_COLOR_DODGE) {
        if (cd <= 0.0f) { return 0.0f; }
        if (cs >= 1.0f) { return 1.0f; }
        float r = cd / (1.0f - cs);
        return r > 1.0f ? 1.0f : r;
    }
    if (mode == GUI_BLEND_COLOR_BURN) {
        if (cd >= 1.0f) { return 1.0f; }
        if (cs <= 0.0f) { return 0.0f; }
        float r = (1.0f - cd) / cs;
        return 1.0f - (r > 1.0f ? 1.0f : r);
    }
    if (mode == GUI_BLEND_HARD_LIGHT) {
        return cs <= 0.5f ? cd * 2.0f * cs : cd + (2.0f * cs - 1.0f) - cd * (2.0f * cs - 1.0f);
    }
    if (mode == GUI_BLEND_SOFT_LIGHT) {
        if (cs <= 0.5f) { return cd - (1.0f - 2.0f * cs) * cd * (1.0f - cd); }
        float dd = cd <= 0.25f ? ((16.0f * cd - 12.0f) * cd + 4.0f) * cd : sqrt_f(cd);
        return cd + (2.0f * cs - 1.0f) * (dd - cd);
    }
    if (mode == GUI_BLEND_DIFFERENCE) { return cs > cd ? cs - cd : cd - cs; }
    if (mode == GUI_BLEND_EXCLUSION) { return cs + cd - 2.0f * cs * cd; }
    return cs;
}

// The blended result of premultiplied source s over destination d (no
// coverage applied).
static vec<float, 4> __gui_blend(int mode, vec<float, 4> s, vec<float, 4> d) {
    float sa = s[3];
    float da = d[3];
    vec<float, 4> isa = { 1.0f - sa, 1.0f - sa, 1.0f - sa, 1.0f - sa };
    vec<float, 4> ida = { 1.0f - da, 1.0f - da, 1.0f - da, 1.0f - da };
    vec<float, 4> vsa = { sa, sa, sa, sa };
    vec<float, 4> vda = { da, da, da, da };
    if (mode == GUI_BLEND_SRC_OVER) { return s + d * isa; }
    if (mode == GUI_BLEND_SRC) { return s; }
    if (mode == GUI_BLEND_DST) { return d; }
    if (mode == GUI_BLEND_DST_OVER) { return s * ida + d; }
    if (mode == GUI_BLEND_SRC_IN) { return s * vda; }
    if (mode == GUI_BLEND_DST_IN) { return d * vsa; }
    if (mode == GUI_BLEND_SRC_OUT) { return s * ida; }
    if (mode == GUI_BLEND_DST_OUT) { return d * isa; }
    if (mode == GUI_BLEND_SRC_ATOP) {
        vec<float, 4> r = s * vda + d * isa;
        r[3] = da;
        return r;
    }
    if (mode == GUI_BLEND_DST_ATOP) {
        vec<float, 4> r = d * vsa + s * ida;
        r[3] = sa;
        return r;
    }
    if (mode == GUI_BLEND_XOR) { return s * ida + d * isa; }
    if (mode == GUI_BLEND_CLEAR) { vec<float, 4> z = { 0.0f, 0.0f, 0.0f, 0.0f }; return z; }
    if (mode == GUI_BLEND_PLUS) {
        vec<float, 4> r = s + d;
        int i = 0;
        while (i < 4) { if (r[i] > 1.0f) { r[i] = 1.0f; } i = i + 1; }
        return r;
    }
    // Separable blend modes: Cs, Cd un-premultiplied; result
    // s*(1-da) + d*(1-sa) + sa*da*B(Cs, Cd), alpha sa + da - sa*da.
    vec<float, 4> r = s * ida + d * isa;
    float k = sa * da;
    if (k > 0.0f) {
        int i = 0;
        while (i < 3) {
            float cs = s[i] / sa;
            float cd = d[i] / da;
            if (cs > 1.0f) { cs = 1.0f; }
            if (cd > 1.0f) { cd = 1.0f; }
            r[i] = r[i] + k * __gui_blend_channel(mode, cs, cd);
            i = i + 1;
        }
    }
    r[3] = sa + da - k;
    return r;
}

// ── compositing one row ───────────────────────────────────────────────────

// Composites coverage cov[0 .. x1-x0) onto row y, pixels x0..x1-1 (already
// clipped to the clip rectangle), through the clip mask.
static void __gui_composite_row(&GuiCanvas c, const &GuiPaintCtx pc, int y, int x0, int x1, const float* cov) {
    unsafe {
        unsigned char* row = (unsigned char*)c.px + (unsigned long)y * (unsigned long)c.stride;
        const unsigned char* mrow = (const unsigned char*)0;
        if (!c.st.mask.is_null()) {
            mrow = (const unsigned char*)c.st.mask + (unsigned long)y * (unsigned long)c.width;
        }
        int mode = c.st.blend;
        float galpha = pc.alpha;
        int aa = c.st.antiAlias;
        vec<float, 4> k255 = { 255.0f, 255.0f, 255.0f, 255.0f };
        vec<float, 4> inv255 = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
        // Solid, opaque, source-over: full-coverage pixels are a plain store.
        int fastSolid = (pc.kind == GUI_PAINT_SOLID && mode == GUI_BLEND_SRC_OVER && galpha >= 1.0f &&
                         pc.solid[3] >= 1.0f) ? 1 : 0;
        unsigned char sr = (unsigned char)(int)(pc.solid[0] * 255.0f + 0.5f);
        unsigned char sg = (unsigned char)(int)(pc.solid[1] * 255.0f + 0.5f);
        unsigned char sb = (unsigned char)(int)(pc.solid[2] * 255.0f + 0.5f);
        float py = (float)y + 0.5f;
        // Linear gradients: t is linear in x, so step it instead of
        // evaluating the paint per pixel.
        int linear = pc.kind == GUI_PAINT_LINEAR ? 1 : 0;
        float lt = 0.0f;
        float ldt = 0.0f;
        if (linear) {
            float u0 = pc.ia * ((float)x0 + 0.5f) + pc.ic * py + pc.ie;
            float v0 = pc.ib * ((float)x0 + 0.5f) + pc.id * py + pc.iF;
            lt = (u0 - pc.x0) * pc.cdx + (v0 - pc.y0) * pc.cdy;
            ldt = pc.ia * pc.cdx + pc.ib * pc.cdy;
        }
        int x = x0;
        while (x < x1) {
            float cv = cov[x - x0];
            if (!aa) { cv = cv >= 0.5f ? 1.0f : 0.0f; }
            if (mrow != (const unsigned char*)0) { cv = cv * (float)mrow[x] / 255.0f; }
            if (cv <= 0.0f) { x = x + 1; continue; }
            if (cv > 1.0f) { cv = 1.0f; }
            unsigned char* p = row + (unsigned long)x * 4UL;
            if (fastSolid && cv >= 0.999f) {
                p[0] = sr; p[1] = sg; p[2] = sb; p[3] = (unsigned char)255;
                x = x + 1;
                continue;
            }
            vec<float, 4> s;
            if (linear) { s = __gui_lut_at(pc, __gui_spread(lt + ldt * (float)(x - x0), pc.spread)); }
            else { s = __gui_paint_at(pc, (float)x + 0.5f, py); }
            vec<float, 4> ga = { galpha, galpha, galpha, galpha };
            s = s * ga;
            vec<float, 4> d = { (float)p[0], (float)p[1], (float)p[2], (float)p[3] };
            d = d * inv255;
            vec<float, 4> cvv = { cv, cv, cv, cv };
            vec<float, 4> r;
            if (mode == GUI_BLEND_SRC_OVER) {
                // d + (s + d(1-sa) - d) * cv  =  d * (1 - sa*cv) + s*cv
                float k = 1.0f - s[3] * cv;
                vec<float, 4> kv = { k, k, k, k };
                r = d * kv + s * cvv;
            } else {
                vec<float, 4> b = __gui_blend(mode, s, d);
                r = d + (b - d) * cvv;
            }
            r = r * k255;
            int i = 0;
            while (i < 4) {
                float f = r[i] + 0.5f;
                p[i] = f <= 0.0f ? (unsigned char)0 : (f >= 255.0f ? (unsigned char)255 : (unsigned char)(int)f);
                i = i + 1;
            }
            x = x + 1;
        }
    }
}

// ── scan conversion ───────────────────────────────────────────────────────

// Where coverage goes: composited into the canvas, multiplied into a new
// clip mask, or written to a plain coverage buffer.
#define GUI_SINK_COMPOSITE 0
#define GUI_SINK_MASK      1
#define GUI_SINK_COVERAGE  2

struct GuiRasterSink {
    int kind;
    int cx0; int cy0; int cx1; int cy1;     // device-space bounds to rasterize within
    unsigned char* out;                     // MASK: canvas-sized; COVERAGE: outW*outH
    int outW;
    int offX; int offY;                     // COVERAGE: device position of out[0]
    const struct GuiPaintCtx* pc;
};

static void __gui_sort_edges(&GuiCanvas c, int n) {
    int* idx = __gui_vi(&c.sorted, (unsigned long)n);
    unsafe {
        const struct GuiEdge* e = (const struct GuiEdge*)c.edges.data;
        int i = 0;
        while (i < n) { idx[i] = i; i = i + 1; }
        // Shell sort by y0 (gaps 1, 4, 13, 40, ...).
        int gap = 1;
        while (gap < n / 3) { gap = gap * 3 + 1; }
        while (gap >= 1) {
            i = gap;
            while (i < n) {
                int v = idx[i];
                float key = e[v].y0;
                int j = i;
                while (j >= gap && e[idx[j - gap]].y0 > key) {
                    idx[j] = idx[j - gap];
                    j = j - gap;
                }
                idx[j] = v;
                i = i + 1;
            }
            gap = gap / 3;
        }
    }
}

static void __gui_raster(&GuiCanvas c, int rule, &GuiRasterSink sink) {
    int n = (int)c.edges.len;
    if (n == 0) { return; }
    unsafe {
        const struct GuiEdge* e = (const struct GuiEdge*)c.edges.data;
        float minx = e[0].x0; float maxx = e[0].x0; float miny = e[0].y0; float maxy = e[0].y1;
        int i = 0;
        while (i < n) {
            if (e[i].x0 < minx) { minx = e[i].x0; }
            if (e[i].x1 < minx) { minx = e[i].x1; }
            if (e[i].x0 > maxx) { maxx = e[i].x0; }
            if (e[i].x1 > maxx) { maxx = e[i].x1; }
            if (e[i].y0 < miny) { miny = e[i].y0; }
            if (e[i].y1 > maxy) { maxy = e[i].y1; }
            i = i + 1;
        }
        int y0 = (int)floor_f(miny);
        int y1 = (int)ceil_f(maxy);
        if (y0 < sink.cy0) { y0 = sink.cy0; }
        if (y1 > sink.cy1) { y1 = sink.cy1; }
        int x0 = (int)floor_f(minx);
        int x1 = (int)ceil_f(maxx) + 1;
        if (x0 < sink.cx0) { x0 = sink.cx0; }
        if (x1 > sink.cx1) { x1 = sink.cx1; }
        if (y0 >= y1 || x0 >= x1) { return; }
        float fx0 = (float)x0;
        float fx1 = (float)x1;

        __gui_sort_edges(c, n);
        int* sorted = (int*)c.sorted.data;
        // acc[k] is pixel x0 + k; two slots of slack past x1.
        int span = x1 - x0;
        float* acc = __gui_vf(&c.acc, (unsigned long)(span + 3));
        safe_memset(acc[0..(unsigned long)(span + 3)], 0);
        float* cov = __gui_vf(&c.row, (unsigned long)(span + 1));
        int* active = __gui_vi(&c.active, (unsigned long)n);
        int nactive = 0;
        float* xs = __gui_vf(&c.xs, (unsigned long)(n * 2 + 2));
        int next = 0;
        float w = 1.0f / (float)GUI_AA_SAMPLES;

        int y = y0;
        // Skip the edges that end above the first row.
        while (y < y1) {
            float top = (float)y;
            float bot = top + 1.0f;
            // Edges starting before the bottom of this row join...
            while (next < n && e[sorted[next]].y0 < bot) {
                active[nactive] = sorted[next];
                nactive = nactive + 1;
                next = next + 1;
            }
            // ... and those ending above its top leave.
            int k = 0;
            while (k < nactive) {
                if (e[active[k]].y1 <= top) {
                    active[k] = active[nactive - 1];
                    nactive = nactive - 1;
                } else {
                    k = k + 1;
                }
            }
            if (nactive == 0) {
                if (next >= n) { break; }
                int jump = (int)floor_f(e[sorted[next]].y0);
                y = jump > y ? jump : y + 1;
                continue;
            }
            int lo = span + 2;
            int hi = -1;
            int s = 0;
            while (s < GUI_AA_SAMPLES) {
                float sy = top + ((float)s + 0.5f) * w;
                int nx = 0;
                k = 0;
                while (k < nactive) {
                    const struct GuiEdge* ed = &e[active[k]];
                    if (ed->y0 <= sy && sy < ed->y1) {
                        xs[nx * 2] = ed->x0 + (sy - ed->y0) * ed->dxdy;
                        xs[nx * 2 + 1] = (float)ed->dir;
                        nx = nx + 1;
                    }
                    k = k + 1;
                }
                s = s + 1;
                if (nx < 2) { continue; }
                // Insertion sort of the crossings by x.
                int a = 1;
                while (a < nx) {
                    float kx = xs[a * 2];
                    float kd = xs[a * 2 + 1];
                    int b = a - 1;
                    while (b >= 0 && xs[b * 2] > kx) {
                        xs[(b + 1) * 2] = xs[b * 2];
                        xs[(b + 1) * 2 + 1] = xs[b * 2 + 1];
                        b = b - 1;
                    }
                    xs[(b + 1) * 2] = kx;
                    xs[(b + 1) * 2 + 1] = kd;
                    a = a + 1;
                }
                int wind = 0;
                a = 0;
                while (a < nx - 1) {
                    wind = wind + (int)xs[a * 2 + 1];
                    int inside = rule == GUI_FILL_EVENODD ? ((wind & 1) != 0 ? 1 : 0) : (wind != 0 ? 1 : 0);
                    if (inside) {
                        float xa = xs[a * 2];
                        float xb = xs[(a + 1) * 2];
                        if (xa < fx0) { xa = fx0; }
                        if (xb > fx1) { xb = fx1; }
                        if (xb > xa) {
                            float ra = xa - fx0;
                            float rb = xb - fx0;
                            int ia = (int)ra;
                            int ib = (int)rb;
                            float fa = ra - (float)ia;
                            float fb = rb - (float)ib;
                            acc[ia] = acc[ia] + (1.0f - fa) * w;
                            acc[ia + 1] = acc[ia + 1] + fa * w;
                            acc[ib] = acc[ib] - (1.0f - fb) * w;
                            acc[ib + 1] = acc[ib + 1] - fb * w;
                            if (ia < lo) { lo = ia; }
                            if (ib + 1 > hi) { hi = ib + 1; }
                        }
                    }
                    a = a + 1;
                }
            }
            if (hi >= lo) {
                // Integrate the difference buffer over [lo, hi] (and clear it).
                float run = 0.0f;
                int q = lo;
                while (q <= hi) {
                    run = run + acc[q];
                    acc[q] = 0.0f;
                    float v = run;
                    if (v < 0.0f) { v = -v; }
                    if (v < 1e-5f) { v = 0.0f; }
                    if (v > 1.0f) { v = 1.0f; }
                    cov[q] = v;
                    q = q + 1;
                }
                int ex = hi;
                if (ex >= span) { ex = span - 1; }
                if (sink.kind == GUI_SINK_COMPOSITE) {
                    if (ex >= lo) { __gui_composite_row(c, sink.pc, y, x0 + lo, x0 + ex + 1, cov + lo); }
                } else if (sink.kind == GUI_SINK_MASK) {
                    unsigned char* mrow = sink.out + (unsigned long)y * (unsigned long)c.width;
                    int q2 = lo;
                    while (q2 <= ex) {
                        float v = cov[q2];
                        if (!c.st.antiAlias) { v = v >= 0.5f ? 1.0f : 0.0f; }
                        mrow[x0 + q2] = (unsigned char)(int)(v * 255.0f + 0.5f);
                        q2 = q2 + 1;
                    }
                } else {
                    unsigned char* orow = sink.out + (unsigned long)(y - sink.offY) * (unsigned long)sink.outW;
                    int q2 = lo;
                    while (q2 <= ex) {
                        int v = (int)(cov[q2] * 255.0f + 0.5f);
                        int ox = x0 + q2 - sink.offX;
                        int prev = (int)orow[ox] + v;
                        orow[ox] = (unsigned char)(prev > 255 ? 255 : prev);
                        q2 = q2 + 1;
                    }
                }
            }
            y = y + 1;
        }
    }
}

// ── entry points used by gui_canvas*.sc ───────────────────────────────────

static void __gui_sink_clip(&GuiCanvas c, &GuiRasterSink sink) {
    sink.cx0 = c.st.clipX0; sink.cy0 = c.st.clipY0;
    sink.cx1 = c.st.clipX1; sink.cy1 = c.st.clipY1;
}

// Fills c.edges with 'paint' under the canvas state.
void __gui_canvas_fill_edges(&GuiCanvas c, int rule, const &GuiPaint paint) {
    if (c.edges.len == 0UL) { return; }
    if (c.st.clipX0 >= c.st.clipX1 || c.st.clipY0 >= c.st.clipY1) { return; }
    struct GuiPaintCtx pc;
    if (!__gui_paint_ctx(c, paint, &pc)) { return; }
    if (pc.alpha <= 0.0f && c.st.blend == GUI_BLEND_SRC_OVER) { return; }
    struct GuiRasterSink sink;
    sink.kind = GUI_SINK_COMPOSITE;
    __gui_sink_clip(c, &sink);
    sink.out = (unsigned char*)0;
    sink.outW = 0; sink.offX = 0; sink.offY = 0;
    unsafe { sink.pc = (const struct GuiPaintCtx*)&pc; }
    __gui_raster(c, rule, &sink);
}

// A new clip mask: the coverage of c.edges times the current mask.
?&heap unsigned char __gui_canvas_mask_edges(&GuiCanvas c, int rule) {
    unsigned long n = (unsigned long)c.width * (unsigned long)c.height;
    unsigned char* m;
    unsafe { m = (unsigned char*)alloc_zeroed_raw_(n > 0UL ? n : 1UL); }
    struct GuiRasterSink sink;
    sink.kind = GUI_SINK_MASK;
    __gui_sink_clip(c, &sink);
    sink.out = m;
    sink.outW = c.width; sink.offX = 0; sink.offY = 0;
    sink.pc = (const struct GuiPaintCtx*)0;
    __gui_raster(c, rule, &sink);
    unsafe {
        if (!c.st.mask.is_null()) {
            const unsigned char* old = (const unsigned char*)c.st.mask;
            unsigned long i = 0UL;
            while (i < n) {
                m[i] = (unsigned char)(((unsigned int)m[i] * (unsigned int)old[i] + 127U) / 255U);
                i = i + 1UL;
            }
        }
        return (&heap unsigned char)m;
    }
}

// Rasterizes c.edges (device space) into a w x h coverage buffer whose
// pixel (0,0) is device (offX, offY); no clipping beyond the buffer.
void __gui_canvas_coverage_edges(&GuiCanvas c, int rule, &unsigned char out, int w, int h, int offX, int offY) {
    struct GuiRasterSink sink;
    sink.kind = GUI_SINK_COVERAGE;
    sink.cx0 = offX; sink.cy0 = offY;
    sink.cx1 = offX + w; sink.cy1 = offY + h;
    unsafe { sink.out = (unsigned char*)out; }
    sink.outW = w;
    sink.offX = offX; sink.offY = offY;
    sink.pc = (const struct GuiPaintCtx*)0;
    __gui_raster(c, rule, &sink);
}

// Composites a coverage bitmap (w x h bytes) with its top-left at device
// (dx, dy) using 'paint', through the clip.
void __gui_canvas_blit_coverage(&GuiCanvas c, const &unsigned char covBytes, int w, int h, int dx, int dy,
                                const &GuiPaint paint) {
    struct GuiPaintCtx pc;
    if (!__gui_paint_ctx(c, paint, &pc)) { return; }
    int x0 = dx; int x1 = dx + w;
    int y0 = dy; int y1 = dy + h;
    if (x0 < c.st.clipX0) { x0 = c.st.clipX0; }
    if (y0 < c.st.clipY0) { y0 = c.st.clipY0; }
    if (x1 > c.st.clipX1) { x1 = c.st.clipX1; }
    if (y1 > c.st.clipY1) { y1 = c.st.clipY1; }
    if (x0 >= x1 || y0 >= y1) { return; }
    float* cov = __gui_vf(&c.row, (unsigned long)(x1 - x0 + 1));
    unsafe {
        int y = y0;
        while (y < y1) {
            const unsigned char* src = (const unsigned char*)covBytes + (unsigned long)(y - dy) * (unsigned long)w + (unsigned long)(x0 - dx);
            int any = 0;
            int x = 0;
            while (x < x1 - x0) {
                cov[x] = (float)src[x] / 255.0f;
                if (src[x] != (unsigned char)0) { any = 1; }
                x = x + 1;
            }
            if (any) { __gui_composite_row(c, &pc, y, x0, x1, cov); }
            y = y + 1;
        }
    }
}

// Composites a full-coverage axis-aligned integer rectangle (fast path for
// fill_rect under a translation; clipped here).
void __gui_canvas_fill_irect(&GuiCanvas c, int x0, int y0, int x1, int y1, const &GuiPaint paint) {
    if (x0 < c.st.clipX0) { x0 = c.st.clipX0; }
    if (y0 < c.st.clipY0) { y0 = c.st.clipY0; }
    if (x1 > c.st.clipX1) { x1 = c.st.clipX1; }
    if (y1 > c.st.clipY1) { y1 = c.st.clipY1; }
    if (x0 >= x1 || y0 >= y1) { return; }
    struct GuiPaintCtx pc;
    if (!__gui_paint_ctx(c, paint, &pc)) { return; }
    float* cov = __gui_vf(&c.row, (unsigned long)(x1 - x0 + 1));
    unsafe {
        int x = 0;
        while (x < x1 - x0) { cov[x] = 1.0f; x = x + 1; }
        int y = y0;
        while (y < y1) {
            __gui_composite_row(c, &pc, y, x0, x1, cov);
            y = y + 1;
        }
    }
}

} // namespace std
