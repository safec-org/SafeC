// SafeC Standard Library — GUI: the vector canvas' state, paths, shapes and
// images (see gui_canvas.h). Rasterization is in gui_canvas_raster.sc,
// stroking in gui_canvas_stroke.sc, text in gui_canvas_text.sc.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_path.h>
#include <std/gui/gui_png.h>
#include <std/gui/gui_text.h>
#include <std/mem.h>
#include <std/math.h>
#include <std/collections/vec.h>
extern void* memset(void* ptr, int val, size_t n);

namespace std {

// gui_canvas_raster.sc / gui_canvas_stroke.sc
void __gui_canvas_flatten(const &GuiPath p, struct GuiMatrix m, float tol, &Vec<float> pts, &Vec<int> cont);
void __gui_canvas_edges(&GuiCanvas c, &Vec<float> pts, &Vec<int> cont, struct GuiMatrix m, int transform);
void __gui_canvas_fill_edges(&GuiCanvas c, int rule, const &GuiPaint paint);
?&heap unsigned char __gui_canvas_mask_edges(&GuiCanvas c, int rule);
void __gui_canvas_fill_irect(&GuiCanvas c, int x0, int y0, int x1, int y1, const &GuiPaint paint);
void __gui_canvas_stroke_polys(&Vec<float> pts, &Vec<int> cont, float width, int cap, int lineJoin,
                               float miterLimit, float tol, &Vec<float> out, &Vec<int> outCont);
void __gui_canvas_dash(&Vec<float> pts, &Vec<int> cont, const &float dash, int n, float offset,
                       &Vec<float> out, &Vec<int> outCont);

#define GUI_FLATTEN_TOL 0.05f

// ── paints ────────────────────────────────────────────────────────────────

static struct GuiPaint __gui_paint_base(int kind) {
    struct GuiPaint p;
    unsafe { memset((void*)&p, 0, sizeof(struct GuiPaint)); }
    p.kind = kind;
    p.color = GuiColor::rgba((unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)255);
    p.nstops = 0;
    p.spread = GUI_SPREAD_PAD;
    p.image = null;
    p.matrix = GuiMatrix::identity();
    p.filter = GUI_FILTER_BILINEAR;
    p.opacity = 1.0f;
    return p;
}

struct GuiPaint GuiPaint::solid(struct GuiColor c) {
    struct GuiPaint p = __gui_paint_base(GUI_PAINT_SOLID);
    p.color = c;
    return p;
}

struct GuiPaint GuiPaint::linear(float x0, float y0, float x1, float y1) {
    struct GuiPaint p = __gui_paint_base(GUI_PAINT_LINEAR);
    p.x0 = x0; p.y0 = y0; p.x1 = x1; p.y1 = y1;
    return p;
}

struct GuiPaint GuiPaint::radial(float cx, float cy, float r) {
    return GuiPaint::radial2(cx, cy, 0.0f, cx, cy, r);
}

struct GuiPaint GuiPaint::radial2(float x0, float y0, float r0, float x1, float y1, float r1) {
    struct GuiPaint p = __gui_paint_base(GUI_PAINT_RADIAL);
    p.x0 = x0; p.y0 = y0; p.r0 = r0 < 0.0f ? 0.0f : r0;
    p.x1 = x1; p.y1 = y1; p.r1 = r1 < 0.0f ? 0.0f : r1;
    return p;
}

struct GuiPaint GuiPaint::sweep(float cx, float cy, float startAngle) {
    struct GuiPaint p = __gui_paint_base(GUI_PAINT_SWEEP);
    p.x0 = cx; p.y0 = cy; p.r0 = startAngle;
    return p;
}

struct GuiPaint GuiPaint::pattern(const &GuiImage img) {
    struct GuiPaint p = __gui_paint_base(GUI_PAINT_IMAGE);
    p.image = img;
    p.spread = GUI_SPREAD_REPEAT;
    return p;
}

void GuiPaint::add_stop(float offset, struct GuiColor c) {
    if (self.nstops >= GUI_GRADIENT_MAX_STOPS) { return; }
    if (offset < 0.0f) { offset = 0.0f; }
    if (offset > 1.0f) { offset = 1.0f; }
    self.stops[self.nstops].offset = offset;
    self.stops[self.nstops].color = c;
    self.nstops = self.nstops + 1;
}

void GuiPaint::set_transform(struct GuiMatrix m) { self.matrix = m; }
void GuiPaint::set_spread(int spread) { self.spread = spread; }
void GuiPaint::set_filter(int filter) { self.filter = filter; }
void GuiPaint::set_opacity(float opacity) {
    self.opacity = opacity < 0.0f ? 0.0f : (opacity > 1.0f ? 1.0f : opacity);
}

// ── construction / state ──────────────────────────────────────────────────

static void __gui_canvas_default_state(&GuiCanvas c) {
    c.st.m = GuiMatrix::identity();
    c.st.fill = GuiPaint::solid(GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)0));
    c.st.stroke = GuiPaint::solid(GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)0));
    c.st.lineWidth = 1.0f;
    c.st.miterLimit = 10.0f;
    c.st.lineCap = GUI_CAP_BUTT;
    c.st.lineJoin = GUI_JOIN_MITER;
    c.st.ndash = 0;
    c.st.dashOffset = 0.0f;
    c.st.alpha = 1.0f;
    c.st.blend = GUI_BLEND_SRC_OVER;
    c.st.fillRule = GUI_FILL_NONZERO;
    c.st.filter = GUI_FILTER_BILINEAR;
    c.st.antiAlias = 1;
    c.st.clipX0 = 0; c.st.clipY0 = 0;
    c.st.clipX1 = c.width; c.st.clipY1 = c.height;
    c.st.mask = null;
    c.st.ownsMask = 0;
    c.st.font = null;
    c.st.fontSize = 12.0f;
    c.st.textAlign = GUI_TEXT_LEFT;
    c.st.textBaseline = GUI_BASELINE_ALPHABETIC;
}

struct GuiCanvas GuiCanvas::for_pixels(&unsigned char px, int width, int height, int stride) {
    struct GuiCanvas c;
    c.px = px;
    c.width = width;
    c.height = height;
    c.stride = stride;
    c.saved = Vec::new();
    c.path = GuiPath::new();
    c.edges = Vec::new();
    c.acc = Vec::new();
    c.xs = Vec::new();
    c.sorted = Vec::new();
    c.active = Vec::new();
    c.fpts = Vec::new();
    c.fcont = Vec::new();
    c.dpts = Vec::new();
    c.dcont = Vec::new();
    c.spts = Vec::new();
    c.scont = Vec::new();
    c.row = Vec::new();
    c.lut = Vec::new();
    c.scratchPath = GuiPath::new();
    __gui_canvas_default_state(&c);
    return c;
}

struct GuiCanvas GuiCanvas::for_window(&GuiWindow win) {
    unsafe { return GuiCanvas::for_pixels((&unsigned char)win.pixels, win.width, win.height, win.width * 4); }
}

struct GuiCanvas GuiCanvas::for_image(&GuiImage img) {
    img.premultiply();
    unsafe { return GuiCanvas::for_pixels((&unsigned char)img.pixels, img.width, img.height, img.width * 4); }
}

static void __gui_canvas_drop_masks(&GuiCanvas c) {
    unsafe {
        if (c.st.ownsMask && !c.st.mask.is_null()) { dealloc_raw_((void*)c.st.mask); }
        c.st.mask = null;
        c.st.ownsMask = 0;
        unsigned long i = 0UL;
        while (i < c.saved.len) {
            &GuiCanvasState s = c.saved.at(i);
            if (s.ownsMask && !s.mask.is_null()) { dealloc_raw_((void*)s.mask); }
            s.mask = null;
            s.ownsMask = 0;
            i = i + 1UL;
        }
    }
}

void GuiCanvas::retarget_window(&GuiWindow win) {
    int resized = (win.width != self.width || win.height != self.height) ? 1 : 0;
    unsafe { self.px = (&unsigned char)win.pixels; }
    self.stride = win.width * 4;
    if (resized) {
        __gui_canvas_drop_masks(self);
        self.width = win.width;
        self.height = win.height;
        self.st.clipX0 = 0; self.st.clipY0 = 0;
        self.st.clipX1 = self.width; self.st.clipY1 = self.height;
        unsigned long i = 0UL;
        while (i < self.saved.len) {
            &GuiCanvasState s = self.saved.at(i);
            s.clipX0 = 0; s.clipY0 = 0; s.clipX1 = self.width; s.clipY1 = self.height;
            i = i + 1UL;
        }
    }
}

void GuiCanvas::reset() {
    __gui_canvas_drop_masks(self);
    self.saved.clear();
    self.path.clear();
    __gui_canvas_default_state(self);
}

void GuiCanvas::free() { self.drop(); }

void GuiCanvas::drop() {
    __gui_canvas_drop_masks(self);
    self.saved.free();
    self.path.free();
    self.edges.free();
    self.acc.free();
    self.xs.free();
    self.sorted.free();
    self.active.free();
    self.fpts.free();
    self.fcont.free();
    self.dpts.free();
    self.dcont.free();
    self.spts.free();
    self.scont.free();
    self.row.free();
    self.lut.free();
    self.scratchPath.free();
    self.px = null;
}

void GuiCanvas::save() {
    self.saved.push(self.st);
    self.st.ownsMask = 0;    // the saved state keeps owning it
}

void GuiCanvas::restore() {
    if (self.saved.len == 0UL) { return; }
    unsafe {
        if (self.st.ownsMask && !self.st.mask.is_null()) { dealloc_raw_((void*)self.st.mask); }
    }
    ?struct GuiCanvasState top = self.saved.pop();
    match (top) {
        case some(s): { self.st = s; }
        case none: { }
    }
}

void GuiCanvas::translate(float tx, float ty) { self.st.m = self.st.m.translate(tx, ty); }
void GuiCanvas::scale(float sx, float sy) { self.st.m = self.st.m.scale(sx, sy); }
void GuiCanvas::rotate(float radians) { self.st.m = self.st.m.rotate(radians); }
void GuiCanvas::transform(struct GuiMatrix m) { self.st.m = self.st.m.mul(m); }
void GuiCanvas::set_transform(struct GuiMatrix m) { self.st.m = m; }
void GuiCanvas::reset_transform() { self.st.m = GuiMatrix::identity(); }
struct GuiMatrix GuiCanvas::get_transform() const { return self.st.m; }

void GuiCanvas::set_fill_color(struct GuiColor c) { self.st.fill = GuiPaint::solid(c); }
void GuiCanvas::set_fill_paint(const &GuiPaint p) { self.st.fill = *p; }
void GuiCanvas::set_stroke_color(struct GuiColor c) { self.st.stroke = GuiPaint::solid(c); }
void GuiCanvas::set_stroke_paint(const &GuiPaint p) { self.st.stroke = *p; }
void GuiCanvas::set_line_width(float w) { self.st.lineWidth = w < 0.0f ? 0.0f : w; }
void GuiCanvas::set_line_cap(int lineCap) { self.st.lineCap = lineCap; }
void GuiCanvas::set_line_join(int lineJoin) { self.st.lineJoin = lineJoin; }
void GuiCanvas::set_miter_limit(float limit) { self.st.miterLimit = limit < 1.0f ? 1.0f : limit; }

void GuiCanvas::set_dash(const &float lengths, int n, float offset) {
    if (n <= 0) { self.st.ndash = 0; return; }
    if (n > GUI_DASH_MAX) { n = GUI_DASH_MAX; }
    float total = 0.0f;
    unsafe {
        int i = 0;
        while (i < n) {
            float v = lengths[i] < 0.0f ? 0.0f : lengths[i];
            self.st.dash[i] = v;
            total = total + v;
            i = i + 1;
        }
    }
    if (total <= 0.0f) { self.st.ndash = 0; return; }
    // An odd list repeats once more (SVG), when it fits.
    if ((n % 2) == 1) {
        if (n * 2 <= GUI_DASH_MAX) {
            int i = 0;
            while (i < n) { self.st.dash[n + i] = self.st.dash[i]; i = i + 1; }
            n = n * 2;
        } else {
            n = n - 1;
        }
    }
    self.st.ndash = n;
    self.st.dashOffset = offset;
}

void GuiCanvas::set_global_alpha(float a) { self.st.alpha = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a); }
void GuiCanvas::set_blend_mode(int mode) {
    self.st.blend = (mode >= 0 && mode < GUI_BLEND_COUNT) ? mode : GUI_BLEND_SRC_OVER;
}
void GuiCanvas::set_fill_rule(int rule) { self.st.fillRule = rule; }
void GuiCanvas::set_image_filter(int filter) { self.st.filter = filter; }
void GuiCanvas::set_anti_alias(int on) { self.st.antiAlias = on ? 1 : 0; }

// ── clipping ──────────────────────────────────────────────────────────────

static int __gui_near_int(float v) {
    float r = floor_f(v + 0.5f);
    float d = v - r;
    return (d > -0.01f && d < 0.01f) ? 1 : 0;
}

static void __gui_clip_irect(&GuiCanvas c, int x0, int y0, int x1, int y1) {
    if (x0 > c.st.clipX0) { c.st.clipX0 = x0; }
    if (y0 > c.st.clipY0) { c.st.clipY0 = y0; }
    if (x1 < c.st.clipX1) { c.st.clipX1 = x1; }
    if (y1 < c.st.clipY1) { c.st.clipY1 = y1; }
    if (c.st.clipX1 < c.st.clipX0) { c.st.clipX1 = c.st.clipX0; }
    if (c.st.clipY1 < c.st.clipY0) { c.st.clipY1 = c.st.clipY0; }
}

void GuiCanvas::clip_path(const &GuiPath p) {
    __gui_canvas_flatten(p, self.st.m, GUI_FLATTEN_TOL, &self.fpts, &self.fcont);
    __gui_canvas_edges(self, &self.fpts, &self.fcont, GuiMatrix::identity(), 0);
    if (self.edges.len == 0UL) {
        // Nothing inside: an empty clip.
        __gui_clip_irect(self, 0, 0, 0, 0);
        return;
    }
    ?&heap unsigned char m = __gui_canvas_mask_edges(self, self.st.fillRule);
    unsafe {
        if (self.st.ownsMask && !self.st.mask.is_null()) { dealloc_raw_((void*)self.st.mask); }
    }
    self.st.mask = m;
    self.st.ownsMask = 1;
    // Nothing outside the path's bounds survives: tighten the rectangle.
    struct GuiRect b = p.bounds();
    struct GuiRect db = self.st.m.apply_rect(b);
    __gui_clip_irect(self, (int)floor_f(db.x), (int)floor_f(db.y),
                     (int)ceil_f(db.x + db.w), (int)ceil_f(db.y + db.h));
}

void GuiCanvas::clip_rect(float x, float y, float w, float h) {
    if (self.st.m.is_axis_aligned()) {
        struct GuiRect r = self.st.m.apply_rect(GuiRect::new(x, y, w, h));
        if (__gui_near_int(r.x) && __gui_near_int(r.y) && __gui_near_int(r.x + r.w) && __gui_near_int(r.y + r.h)) {
            __gui_clip_irect(self, (int)floor_f(r.x + 0.5f), (int)floor_f(r.y + 0.5f),
                             (int)floor_f(r.x + r.w + 0.5f), (int)floor_f(r.y + r.h + 0.5f));
            return;
        }
    }
    self.scratchPath.clear();
    self.scratchPath.rect(x, y, w, h);
    self.clip_path(&self.scratchPath);
}

void GuiCanvas::clip() { self.clip_path(&self.path); }

void GuiCanvas::reset_clip() {
    unsafe {
        if (self.st.ownsMask && !self.st.mask.is_null()) { dealloc_raw_((void*)self.st.mask); }
    }
    self.st.mask = null;
    self.st.ownsMask = 0;
    self.st.clipX0 = 0; self.st.clipY0 = 0;
    self.st.clipX1 = self.width; self.st.clipY1 = self.height;
}

struct GuiRect GuiCanvas::clip_bounds() const {
    return GuiRect::new((float)self.st.clipX0, (float)self.st.clipY0,
                        (float)(self.st.clipX1 - self.st.clipX0), (float)(self.st.clipY1 - self.st.clipY0));
}

// ── the current path ──────────────────────────────────────────────────────

void GuiCanvas::begin_path() { self.path.clear(); }
void GuiCanvas::move_to(float x, float y) { self.path.move_to(x, y); }
void GuiCanvas::line_to(float x, float y) { self.path.line_to(x, y); }
void GuiCanvas::quad_to(float cx, float cy, float x, float y) { self.path.quad_to(cx, cy, x, y); }
void GuiCanvas::cubic_to(float c1x, float c1y, float c2x, float c2y, float x, float y) {
    self.path.cubic_to(c1x, c1y, c2x, c2y, x, y);
}
void GuiCanvas::arc(float cx, float cy, float r, float a0, float a1, int ccw) { self.path.arc(cx, cy, r, a0, a1, ccw); }
void GuiCanvas::arc_to(float x1, float y1, float x2, float y2, float r) { self.path.arc_to(x1, y1, x2, y2, r); }
void GuiCanvas::rect(float x, float y, float w, float h) { self.path.rect(x, y, w, h); }
void GuiCanvas::rounded_rect(float x, float y, float w, float h, float r) { self.path.rounded_rect(x, y, w, h, r); }
void GuiCanvas::ellipse(float cx, float cy, float rx, float ry) { self.path.ellipse(cx, cy, rx, ry); }
void GuiCanvas::close_path() { self.path.close(); }
void GuiCanvas::fill() { self.fill_path(&self.path); }
void GuiCanvas::stroke() { self.stroke_path(&self.path); }

// ── filling and stroking ──────────────────────────────────────────────────

static void __gui_canvas_fill_with(&GuiCanvas c, const &GuiPath p, const &GuiPaint paint, int rule) {
    __gui_canvas_flatten(p, c.st.m, GUI_FLATTEN_TOL, &c.fpts, &c.fcont);
    __gui_canvas_edges(c, &c.fpts, &c.fcont, GuiMatrix::identity(), 0);
    __gui_canvas_fill_edges(c, rule, paint);
}

void GuiCanvas::fill_path(const &GuiPath p) {
    __gui_canvas_fill_with(self, p, &self.st.fill, self.st.fillRule);
}

// The stroke of 'p' as polygons in user space (in spts/scont).
static int __gui_canvas_stroke_outline(&GuiCanvas c, const &GuiPath p) {
    float s = c.st.m.scale_factor();
    if (s <= 0.0f) { return 0; }
    float tol = GUI_FLATTEN_TOL / s;
    __gui_canvas_flatten(p, GuiMatrix::identity(), tol, &c.fpts, &c.fcont);
    if (c.st.ndash > 0) {
        __gui_canvas_dash(&c.fpts, &c.fcont, &c.st.dash[0], c.st.ndash, c.st.dashOffset, &c.dpts, &c.dcont);
        __gui_canvas_stroke_polys(&c.dpts, &c.dcont, c.st.lineWidth, c.st.lineCap, c.st.lineJoin,
                                  c.st.miterLimit, tol, &c.spts, &c.scont);
    } else {
        __gui_canvas_stroke_polys(&c.fpts, &c.fcont, c.st.lineWidth, c.st.lineCap, c.st.lineJoin,
                                  c.st.miterLimit, tol, &c.spts, &c.scont);
    }
    return 1;
}

void GuiCanvas::stroke_path(const &GuiPath p) {
    if (self.st.lineWidth <= 0.0f) { return; }
    if (!__gui_canvas_stroke_outline(self, p)) { return; }
    __gui_canvas_edges(self, &self.spts, &self.scont, self.st.m, 1);
    __gui_canvas_fill_edges(self, GUI_FILL_NONZERO, &self.st.stroke);
}

void GuiCanvas::stroke_to_path(const &GuiPath p, &GuiPath out) {
    if (self.st.lineWidth <= 0.0f) { return; }
    if (!__gui_canvas_stroke_outline(self, p)) { return; }
    unsafe {
        const float* pts = (const float*)self.spts.data;
        const int* ct = (const int*)self.scont.data;
        unsigned long nc = self.scont.len / 3UL;
        unsigned long k = 0UL;
        while (k < nc) {
            int first = ct[k * 3UL];
            int count = ct[k * 3UL + 1UL];
            int i = 0;
            while (i < count) {
                float x = pts[(first + i) * 2];
                float y = pts[(first + i) * 2 + 1];
                if (i == 0) { out.move_to(x, y); } else { out.line_to(x, y); }
                i = i + 1;
            }
            out.close();
            k = k + 1UL;
        }
    }
}

// A device-space integer rectangle for (x,y,w,h) under a translation, if
// there is one (so fill_rect can skip the scan converter).
static int __gui_canvas_irect(&GuiCanvas c, float x, float y, float w, float h,
                              &int x0, &int y0, &int x1, &int y1) {
    if (!c.st.m.is_translation()) { return 0; }
    float ax = x + c.st.m.e;
    float ay = y + c.st.m.f;
    float bx = ax + w;
    float by = ay + h;
    if (!__gui_near_int(ax) || !__gui_near_int(ay) || !__gui_near_int(bx) || !__gui_near_int(by)) { return 0; }
    int ix0 = (int)floor_f(ax + 0.5f);
    int iy0 = (int)floor_f(ay + 0.5f);
    int ix1 = (int)floor_f(bx + 0.5f);
    int iy1 = (int)floor_f(by + 0.5f);
    *x0 = ix0 < ix1 ? ix0 : ix1;
    *x1 = ix0 < ix1 ? ix1 : ix0;
    *y0 = iy0 < iy1 ? iy0 : iy1;
    *y1 = iy0 < iy1 ? iy1 : iy0;
    return 1;
}

static void __gui_canvas_fill_rect_with(&GuiCanvas c, float x, float y, float w, float h, const &GuiPaint paint) {
    int x0 = 0; int y0 = 0; int x1 = 0; int y1 = 0;
    if (__gui_canvas_irect(c, x, y, w, h, &x0, &y0, &x1, &y1)) {
        __gui_canvas_fill_irect(c, x0, y0, x1, y1, paint);
        return;
    }
    c.scratchPath.clear();
    c.scratchPath.rect(x, y, w, h);
    __gui_canvas_fill_with(c, &c.scratchPath, paint, GUI_FILL_NONZERO);
}

void GuiCanvas::fill_rect(float x, float y, float w, float h) {
    __gui_canvas_fill_rect_with(self, x, y, w, h, &self.st.fill);
}

void GuiCanvas::stroke_rect(float x, float y, float w, float h) {
    self.scratchPath.clear();
    self.scratchPath.rect(x, y, w, h);
    self.stroke_path(&self.scratchPath);
}

void GuiCanvas::clear_rect(float x, float y, float w, float h) {
    int saveBlend = self.st.blend;
    float saveAlpha = self.st.alpha;
    self.st.blend = GUI_BLEND_CLEAR;
    self.st.alpha = 1.0f;
    struct GuiPaint p = GuiPaint::solid(GuiColor::rgba((unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)0));
    __gui_canvas_fill_rect_with(self, x, y, w, h, &p);
    self.st.blend = saveBlend;
    self.st.alpha = saveAlpha;
}

void GuiCanvas::clear(struct GuiColor c) {
    unsigned int a = (unsigned int)c.a;
    unsigned char b[4];
    b[0] = (unsigned char)(((unsigned int)c.r * a + 127U) / 255U);
    b[1] = (unsigned char)(((unsigned int)c.g * a + 127U) / 255U);
    b[2] = (unsigned char)(((unsigned int)c.b * a + 127U) / 255U);
    b[3] = (unsigned char)a;
    unsafe {
        unsigned int word = *(unsigned int*)&b[0];
        int y = 0;
        while (y < self.height) {
            unsigned int* row = (unsigned int*)((unsigned char*)self.px + (unsigned long)y * (unsigned long)self.stride);
            int x = 0;
            while (x < self.width) { row[x] = word; x = x + 1; }
            y = y + 1;
        }
    }
}

void GuiCanvas::fill_rounded_rect(float x, float y, float w, float h, float r) {
    if (r <= 0.0f) { self.fill_rect(x, y, w, h); return; }
    self.scratchPath.clear();
    self.scratchPath.rounded_rect(x, y, w, h, r);
    self.fill_path(&self.scratchPath);
}

void GuiCanvas::stroke_rounded_rect(float x, float y, float w, float h, float r) {
    self.scratchPath.clear();
    self.scratchPath.rounded_rect(x, y, w, h, r);
    self.stroke_path(&self.scratchPath);
}

void GuiCanvas::fill_circle(float cx, float cy, float r) {
    self.scratchPath.clear();
    self.scratchPath.circle(cx, cy, r);
    self.fill_path(&self.scratchPath);
}

void GuiCanvas::stroke_circle(float cx, float cy, float r) {
    self.scratchPath.clear();
    self.scratchPath.circle(cx, cy, r);
    self.stroke_path(&self.scratchPath);
}

void GuiCanvas::fill_ellipse(float cx, float cy, float rx, float ry) {
    self.scratchPath.clear();
    self.scratchPath.ellipse(cx, cy, rx, ry);
    self.fill_path(&self.scratchPath);
}

void GuiCanvas::draw_line(float x0, float y0, float x1, float y1) {
    self.scratchPath.clear();
    self.scratchPath.move_to(x0, y0);
    self.scratchPath.line_to(x1, y1);
    self.stroke_path(&self.scratchPath);
}

// ── images ────────────────────────────────────────────────────────────────

void GuiCanvas::draw_image_rect(const &GuiImage img, float sx, float sy, float sw, float sh,
                                float dx, float dy, float dw, float dh) {
    if (img.width < 1 || img.height < 1 || sw == 0.0f || sh == 0.0f || dw == 0.0f || dh == 0.0f) { return; }
    struct GuiPaint p = GuiPaint::pattern(img);
    p.spread = GUI_SPREAD_PAD;
    p.filter = self.st.filter;
    // image pixel (u, v) -> user space
    p.matrix = GuiMatrix::translation(dx, dy).scale(dw / sw, dh / sh).translate(-sx, -sy);
    // Unscaled at whole pixels: an exact copy (nearest sampling).
    if (self.st.m.is_translation() && dw == sw && dh == sh &&
        __gui_near_int(dx + self.st.m.e) && __gui_near_int(dy + self.st.m.f) &&
        __gui_near_int(sx) && __gui_near_int(sy)) {
        p.filter = GUI_FILTER_NEAREST;
    }
    // Only the source rectangle shows: clip to the destination rectangle.
    __gui_canvas_fill_rect_with(self, dx, dy, dw, dh, &p);
}

void GuiCanvas::draw_image(const &GuiImage img, float x, float y) {
    self.draw_image_rect(img, 0.0f, 0.0f, (float)img.width, (float)img.height,
                         x, y, (float)img.width, (float)img.height);
}

void GuiCanvas::draw_image_scaled(const &GuiImage img, float x, float y, float w, float h) {
    self.draw_image_rect(img, 0.0f, 0.0f, (float)img.width, (float)img.height, x, y, w, h);
}

// ── reading back ──────────────────────────────────────────────────────────

struct GuiColor GuiCanvas::get_pixel(int x, int y) const {
    if (x < 0 || y < 0 || x >= self.width || y >= self.height) {
        return GuiColor::rgba((unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)0);
    }
    unsafe {
        const unsigned char* p = (const unsigned char*)self.px + (unsigned long)y * (unsigned long)self.stride + (unsigned long)x * 4UL;
        unsigned int a = (unsigned int)p[3];
        if (a == 0U) { return GuiColor::rgba((unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)0); }
        if (a == 255U) { return GuiColor::rgba(p[0], p[1], p[2], p[3]); }
        unsigned int r = ((unsigned int)p[0] * 255U + a / 2U) / a;
        unsigned int g = ((unsigned int)p[1] * 255U + a / 2U) / a;
        unsigned int b = ((unsigned int)p[2] * 255U + a / 2U) / a;
        return GuiColor::rgba((unsigned char)(r > 255U ? 255U : r), (unsigned char)(g > 255U ? 255U : g),
                              (unsigned char)(b > 255U ? 255U : b), (unsigned char)a);
    }
}

} // namespace std
