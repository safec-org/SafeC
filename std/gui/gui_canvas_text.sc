// SafeC Standard Library — GUI: text on the vector canvas (see
// gui_canvas.h / gui_text.h).
//
// Under a transform that is a translation plus a uniform positive scale
// (the usual case — UI text), each glyph is rasterized once per (size,
// quarter-pixel horizontal offset) into an anti-aliased coverage bitmap
// cached on its typeface, and drawn by compositing that bitmap with the
// fill paint at a whole-pixel baseline. Under any other transform
// (rotation, skew, mirroring) glyph outlines are filled as paths.
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_text.h>
#include <std/mem.h>
#include <std/math.h>
#include <std/collections/vec.h>

namespace std {

void __gui_canvas_flatten(const &GuiPath p, struct GuiMatrix m, float tol, &Vec<float> pts, &Vec<int> cont);
void __gui_canvas_edges(&GuiCanvas c, &Vec<float> pts, &Vec<int> cont, struct GuiMatrix m, int transform);
void __gui_canvas_fill_edges(&GuiCanvas c, int rule, const &GuiPaint paint);
void __gui_canvas_coverage_edges(&GuiCanvas c, int rule, &unsigned char out, int w, int h, int offX, int offY);
void __gui_canvas_blit_coverage(&GuiCanvas c, const &unsigned char covBytes, int w, int h, int dx, int dy,
                                const &GuiPaint paint);

void GuiCanvas::set_font(const &GuiTypeface face, float sizePx) {
    self.st.font = face;
    self.st.fontSize = sizePx > 0.0f ? sizePx : 1.0f;
}

void GuiCanvas::set_font_size(float sizePx) { self.st.fontSize = sizePx > 0.0f ? sizePx : 1.0f; }
void GuiCanvas::set_text_align(int align) { self.st.textAlign = align; }
void GuiCanvas::set_text_baseline(int baseline) { self.st.textBaseline = baseline; }

static &GuiTypeface __gui_canvas_face(const &GuiCanvas c) {
    match (c.st.font) {
        case some(f): {
            unsafe { return (struct GuiTypeface*)f; }
        }
        case null: { }
    }
    return GuiTypeface::fallback();
}

// 1 if glyphs can come from the bitmap cache under the current transform
// (translation + uniform positive scale); *scale gets the scale.
static int __gui_canvas_text_cacheable(const &GuiCanvas c, &float scale) {
    struct GuiMatrix m = c.st.m;
    if (m.b != 0.0f || m.c != 0.0f || m.a <= 0.0f) { return 0; }
    float d = m.a - m.d;
    if (d < -1e-4f || d > 1e-4f) { return 0; }
    *scale = m.a;
    return 1;
}

// Draws one glyph with its pen at user-space (ux, uy).
static void __gui_canvas_glyph(&GuiCanvas c, &GuiTypeface face, int glyph, float size, float ux, float uy,
                               const &GuiPaint paint) {
    float scale = 1.0f;
    if (!__gui_canvas_text_cacheable(c, &scale)) {
        c.scratchPath.clear();
        if (!face.glyph_path(glyph, &c.scratchPath, size, ux, uy)) { return; }
        __gui_canvas_flatten(&c.scratchPath, c.st.m, 0.2f, &c.fpts, &c.fcont);
        __gui_canvas_edges(c, &c.fpts, &c.fcont, GuiMatrix::identity(), 0);
        __gui_canvas_fill_edges(c, GUI_FILL_NONZERO, paint);
        return;
    }
    float dsize = size * scale;
    float dx = c.st.m.apply_x(ux, uy);
    float dy = c.st.m.apply_y(ux, uy);
    float fx = floor_f(dx);
    int ix = (int)fx;
    int sub = (int)((dx - fx) * 4.0f);
    if (sub > 3) { sub = 3; }
    if (sub < 0) { sub = 0; }
    int iy = (int)floor_f(dy + 0.5f);
    int sizeKey = (int)(dsize * 64.0f + 0.5f);
    ?&GuiGlyphBitmap hit = face.cache_find(glyph, sizeKey, sub);
    &GuiGlyphBitmap bm;
    match (hit) {
        case some(b): { bm = b; }
        case null: {
            struct GuiGlyphBitmap nb;
            nb.glyph = glyph;
            nb.sizeKey = sizeKey;
            nb.subpixel = sub;
            nb.width = 0; nb.height = 0; nb.left = 0; nb.top = 0;
            nb.coverage = null;
            c.scratchPath.clear();
            if (face.glyph_path(glyph, &c.scratchPath, (float)sizeKey / 64.0f, (float)sub * 0.25f, 0.0f)) {
                __gui_canvas_flatten(&c.scratchPath, GuiMatrix::identity(), 0.15f, &c.fpts, &c.fcont);
                __gui_canvas_edges(c, &c.fpts, &c.fcont, GuiMatrix::identity(), 0);
                if (c.edges.len > 0UL) {
                    float x0 = 0.0f; float y0 = 0.0f; float x1 = 0.0f; float y1 = 0.0f;
                    unsafe {
                        const struct GuiEdge* e = (const struct GuiEdge*)c.edges.data;
                        x0 = e[0].x0; x1 = e[0].x0; y0 = e[0].y0; y1 = e[0].y1;
                        unsigned long k = 0UL;
                        while (k < c.edges.len) {
                            if (e[k].x0 < x0) { x0 = e[k].x0; }
                            if (e[k].x1 < x0) { x0 = e[k].x1; }
                            if (e[k].x0 > x1) { x1 = e[k].x0; }
                            if (e[k].x1 > x1) { x1 = e[k].x1; }
                            if (e[k].y0 < y0) { y0 = e[k].y0; }
                            if (e[k].y1 > y1) { y1 = e[k].y1; }
                            k = k + 1UL;
                        }
                    }
                    int bx0 = (int)floor_f(x0);
                    int by0 = (int)floor_f(y0);
                    int bx1 = (int)ceil_f(x1) + 1;
                    int by1 = (int)ceil_f(y1);
                    int w = bx1 - bx0;
                    int h = by1 - by0;
                    if (w > 0 && h > 0 && w < 4096 && h < 4096) {
                        unsafe {
                            unsigned char* cov = (unsigned char*)alloc_zeroed_raw_((unsigned long)w * (unsigned long)h);
                            __gui_canvas_coverage_edges(c, GUI_FILL_NONZERO, (&unsigned char)cov, w, h, bx0, by0);
                            nb.coverage = (&heap unsigned char)cov;
                        }
                        nb.width = w;
                        nb.height = h;
                        nb.left = bx0;
                        nb.top = by0;
                    }
                }
            }
            bm = face.cache_insert(nb);
        }
    }
    if (bm.width <= 0 || bm.height <= 0) { return; }
    unsafe {
        __gui_canvas_blit_coverage(c, (const unsigned char*)bm.coverage, bm.width, bm.height,
                                   ix + bm.left, iy + bm.top, paint);
    }
}

float GuiCanvas::measure_text(const &char text) const {
    &GuiTypeface face = __gui_canvas_face(self);
    return face.measure(text, self.st.fontSize);
}

float GuiCanvas::fill_text_n(const &char text, unsigned long len, float x, float y) {
    &GuiTypeface face = __gui_canvas_face(self);
    &GuiTypeface fb = GuiTypeface::fallback();
    float size = self.st.fontSize;
    float width = face.measure_n(text, len, size);
    if (self.st.textAlign == GUI_TEXT_CENTER) { x = x - width * 0.5f; }
    else if (self.st.textAlign == GUI_TEXT_RIGHT) { x = x - width; }
    float asc = face.ascent(size);
    float desc = face.descent(size);
    if (self.st.textBaseline == GUI_BASELINE_TOP) { y = y + asc; }
    else if (self.st.textBaseline == GUI_BASELINE_MIDDLE) { y = y + (asc - desc) * 0.5f; }
    else if (self.st.textBaseline == GUI_BASELINE_BOTTOM) { y = y - desc; }
    struct GuiPaint paint = self.st.fill;
    float pen = 0.0f;
    int prev = 0;
    int selfFb;
    unsafe { selfFb = ((const void*)face == (const void*)fb) ? 1 : 0; }
    unsigned long i = 0UL;
    while (i < len) {
        unsigned int cp = gui_utf8_next(text, len, &i);
        if (cp == 0U) { break; }
        int g = face.glyph_index(cp);
        if (g == 0 && !selfFb && fb.glyph_index(cp) != 0) {
            int g2 = fb.glyph_index(cp);
            __gui_canvas_glyph(self, fb, g2, size, x + pen, y, &paint);
            pen = pen + fb.advance(g2, size);
            prev = 0;
            continue;
        }
        if (prev != 0) { pen = pen + face.kerning(prev, g, size); }
        if (cp != 32U && cp != 9U) { __gui_canvas_glyph(self, face, g, size, x + pen, y, &paint); }
        pen = pen + face.advance(g, size);
        prev = g;
    }
    return width;
}

float GuiCanvas::fill_text(const &char text, float x, float y) {
    unsigned long n = 0UL;
    unsafe { while (text[n] != '\0') { n = n + 1UL; } }
    return self.fill_text_n(text, n, x, y);
}

void GuiCanvas::fill_text_layout(const &GuiTextLayout layout, float x, float y) {
    struct GuiPaint paint = self.st.fill;
    unsigned long i = 0UL;
    while (i < layout.glyphs.len) {
        &GuiTextGlyph g = layout.glyphs.at(i);
        i = i + 1UL;
        if (g.space || g.glyph <= 0) { continue; }
        match (g.face) {
            case some(f): {
                unsafe { __gui_canvas_glyph(self, (struct GuiTypeface*)f, g.glyph, layout.size, x + g.x, y + g.y, &paint); }
            }
            case null: { }
        }
    }
}

} // namespace std
