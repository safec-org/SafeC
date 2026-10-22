#pragma once
// SafeC Standard Library — GUI: the anti-aliased 2D vector canvas.
//
// A software rasterizer with the drawing model of the HTML canvas / Skia /
// Cairo family, drawing into a GuiWindow's pixel buffer or a GuiImage:
//
//   paths      move/line/quadratic/cubic/arc/close (gui_path.h), filled by
//              the nonzero or even-odd rule, stroked with any width, miter/
//              round/bevel joins, butt/round/square caps and dash patterns
//   paint      solid colors, linear / radial (two-circle) / sweep
//              gradients with up to 16 stops and pad/repeat/reflect spread,
//              and image patterns (any transform, clamp/repeat/none,
//              nearest or bilinear sampling)
//   state      an affine transform, global alpha, blend mode, rectangular
//              and path clipping, all saved/restored with save()/restore()
//   compositing premultiplied alpha; 12 Porter-Duff operators and 12
//              separable blend modes (multiply, screen, overlay, ...)
//   images     drawn scaled/rotated/cropped with filtering
//   text       TrueType glyphs from gui_text.h, anti-aliased and cached
//
// Anti-aliasing is exact in x and 16x supersampled in y: each pixel row
// is cut by 16 sample lines and every edge crossing contributes its exact
// horizontal coverage, accumulated in a difference buffer and integrated
// once per row — so a pixel's coverage is the true area inside the shape
// to within 1/16 of a pixel vertically, for either fill rule, with
// overlapping and self-intersecting paths. Compositing runs four channels
// at a time on SIMD vectors (vec<float, 4>, std/simd's f32x4); solid opaque
// spans are filled with plain 32-bit stores. Everything is deterministic:
// the same calls produce the same bytes on every platform.
//
//   struct GuiCanvas c = GuiCanvas::for_window(&win);
//   c.clear(GuiColor::hex(0xFFFFFFU));
//   c.set_fill_color(GuiColor::hex(0x3366CCU));
//   c.fill_rounded_rect(20.0f, 20.0f, 120.0f, 40.0f, 8.0f);
//   c.set_stroke_color(GuiColor::hex(0x000000U));
//   c.set_line_width(2.0f);
//   c.begin_path(); c.move_to(20.0f, 80.0f); c.line_to(140.0f, 120.0f); c.stroke();
//   c.set_font(GuiTypeface::fallback(), 16.0f);
//   c.fill_text("Hello", 20.0f, 150.0f);
//   c.free();
//   win.present();
//
// A canvas holds no copy of the pixels, only a pointer: after the window
// is resized (its buffer reallocated), point the canvas at it again with
// retarget_window().

#include <std/gui/gui.h>
#include <std/gui/gui_path.h>
#include <std/gui/gui_png.h>
#include <std/gui/gui_text.h>
#include <std/collections/vec.h>

// GuiPaint.kind
#define GUI_PAINT_SOLID  0
#define GUI_PAINT_LINEAR 1
#define GUI_PAINT_RADIAL 2
#define GUI_PAINT_SWEEP  3
#define GUI_PAINT_IMAGE  4

// GuiPaint.spread: what gradients do past their ends, and image patterns
// outside the image.
#define GUI_SPREAD_PAD     0   // extend the end colors / clamp to the edge pixels
#define GUI_SPREAD_REPEAT  1
#define GUI_SPREAD_REFLECT 2
#define GUI_SPREAD_NONE    3   // transparent (image patterns)

#define GUI_GRADIENT_MAX_STOPS 16
#define GUI_DASH_MAX 16

// Fill rules
#define GUI_FILL_NONZERO 0
#define GUI_FILL_EVENODD 1

// Line caps / joins
#define GUI_CAP_BUTT    0
#define GUI_CAP_ROUND   1
#define GUI_CAP_SQUARE  2
#define GUI_JOIN_MITER  0
#define GUI_JOIN_ROUND  1
#define GUI_JOIN_BEVEL  2

// Image filtering
#define GUI_FILTER_NEAREST  0
#define GUI_FILTER_BILINEAR 1

// Text baseline (GuiCanvas::set_text_baseline): what fill_text's y means.
#define GUI_BASELINE_ALPHABETIC 0
#define GUI_BASELINE_TOP        1
#define GUI_BASELINE_MIDDLE     2
#define GUI_BASELINE_BOTTOM     3

// Blend modes (GuiCanvas::set_blend_mode). Porter-Duff operators:
#define GUI_BLEND_SRC_OVER   0   // the default: source over destination
#define GUI_BLEND_SRC        1   // replace
#define GUI_BLEND_DST        2
#define GUI_BLEND_DST_OVER   3
#define GUI_BLEND_SRC_IN     4
#define GUI_BLEND_DST_IN     5
#define GUI_BLEND_SRC_OUT    6
#define GUI_BLEND_DST_OUT    7
#define GUI_BLEND_SRC_ATOP   8
#define GUI_BLEND_DST_ATOP   9
#define GUI_BLEND_XOR        10
#define GUI_BLEND_CLEAR      11
#define GUI_BLEND_PLUS       12  // additive, clamped
// Separable blend modes (W3C Compositing and Blending), composited source-over:
#define GUI_BLEND_MULTIPLY    13
#define GUI_BLEND_SCREEN      14
#define GUI_BLEND_OVERLAY     15
#define GUI_BLEND_DARKEN      16
#define GUI_BLEND_LIGHTEN     17
#define GUI_BLEND_COLOR_DODGE 18
#define GUI_BLEND_COLOR_BURN  19
#define GUI_BLEND_HARD_LIGHT  20
#define GUI_BLEND_SOFT_LIGHT  21
#define GUI_BLEND_DIFFERENCE  22
#define GUI_BLEND_EXCLUSION   23
#define GUI_BLEND_COUNT       24

namespace std {

struct GuiColorStop {
    float offset;            // 0..1
    struct GuiColor color;
};

// What to fill or stroke with. A value type: build one, add stops, hand
// it to set_fill_paint()/set_stroke_paint() (which copy it).
struct GuiPaint {
    int kind;                    // GUI_PAINT_*
    struct GuiColor color;       // SOLID
    // LINEAR: from (x0,y0) to (x1,y1).
    // RADIAL: from the circle (x0,y0,r0) to the circle (x1,y1,r1).
    // SWEEP:  around (x0,y0), starting at angle r0 (radians, clockwise).
    float x0; float y0; float r0;
    float x1; float y1; float r1;
    int nstops;
    struct GuiColorStop stops[GUI_GRADIENT_MAX_STOPS];
    int spread;                  // GUI_SPREAD_*
    const ?&GuiImage image;      // IMAGE: the pattern (must outlive the drawing)
    struct GuiMatrix matrix;     // paint space -> user space (pattern/gradient transform)
    int filter;                  // GUI_FILTER_* (IMAGE)
    float opacity;               // multiplies the paint's alpha (default 1)

    static struct GuiPaint solid(struct GuiColor c);
    static struct GuiPaint linear(float x0, float y0, float x1, float y1);
    static struct GuiPaint radial(float cx, float cy, float r);
    // Two-circle radial gradient, like the HTML canvas' createRadialGradient.
    static struct GuiPaint radial2(float x0, float y0, float r0, float x1, float y1, float r1);
    static struct GuiPaint sweep(float cx, float cy, float startAngle);
    // An image pattern, its pixel (0,0) at user-space (0,0), one image pixel
    // per user unit (change with set_transform()). Spread: GUI_SPREAD_REPEAT.
    static struct GuiPaint pattern(const &GuiImage img);

    // Adds a color stop (stops may be added in any order; beyond
    // GUI_GRADIENT_MAX_STOPS they're ignored).
    void add_stop(float offset, struct GuiColor c);
    void set_transform(struct GuiMatrix m);
    void set_spread(int spread);
    void set_filter(int filter);
    void set_opacity(float opacity);
};

// Everything save()/restore() saves.
struct GuiCanvasState {
    struct GuiMatrix m;
    struct GuiPaint fill;
    struct GuiPaint stroke;
    float lineWidth;
    float miterLimit;
    int lineCap;
    int lineJoin;
    float dash[GUI_DASH_MAX];
    int ndash;
    float dashOffset;
    float alpha;
    int blend;
    int fillRule;
    int filter;            // default image filter for draw_image*
    int antiAlias;
    // device-space clip: a rectangle, and optionally a coverage mask
    int clipX0; int clipY0; int clipX1; int clipY1;
    ?&heap unsigned char mask;     // width*height bytes, or null
    int ownsMask;
    const ?&GuiTypeface font;      // null: GuiTypeface::fallback()
    float fontSize;
    int textAlign;                 // GUI_TEXT_LEFT/CENTER/RIGHT
    int textBaseline;              // GUI_BASELINE_*
};

// A line segment of a flattened path, in device space (internal).
struct GuiEdge {
    float x0; float y0;            // y0 < y1
    float x1; float y1;
    float dxdy;
    int dir;                       // +1 downward in the original path, -1 upward
};

struct GuiCanvas {
    ?&unsigned char px;            // target pixels: premultiplied RGBA8888
    int width;
    int height;
    int stride;                    // bytes per row
    struct GuiCanvasState st;
    struct Vec<struct GuiCanvasState> saved;   // the save() stack
    struct GuiPath path;           // the current path (begin_path/move_to/...)
    // scratch buffers, kept between calls to avoid reallocating
    struct Vec<struct GuiEdge> edges;
    struct Vec<float> acc;
    struct Vec<float> xs;
    struct Vec<int> sorted;
    struct Vec<int> active;
    struct Vec<float> fpts;        // flattened polylines: x,y pairs ...
    struct Vec<int> fcont;         // ... and (first point, point count, closed) triples
    struct Vec<float> dpts;        // dashed polylines
    struct Vec<int> dcont;
    struct Vec<float> spts;        // stroke outline polygons
    struct Vec<int> scont;
    struct Vec<float> row;         // one row of coverage
    struct Vec<float> lut;         // gradient colors (256 premultiplied RGBA)
    struct GuiPath scratchPath;

    // ── targets ──────────────────────────────────────────────────────────
    static struct GuiCanvas for_window(&GuiWindow win);
    // Draws into an image (premultiplying its pixels first if they're
    // straight-alpha, see GuiImage.premultiplied).
    static struct GuiCanvas for_image(&GuiImage img);
    // Any premultiplied RGBA8888 buffer.
    static struct GuiCanvas for_pixels(&unsigned char px, int width, int height, int stride);
    // Point at the window's (possibly reallocated) buffer again. Resets
    // the clip if the size changed.
    void retarget_window(&GuiWindow win);
    // Back to the initial state (identity transform, black fill, no clip,
    // empty save stack and path).
    void reset();
    void free();

    // ── state ────────────────────────────────────────────────────────────
    void save();
    void restore();
    void translate(float tx, float ty);
    void scale(float sx, float sy);
    void rotate(float radians);
    // Pre-multiplies the current transform by m (m applies first).
    void transform(struct GuiMatrix m);
    void set_transform(struct GuiMatrix m);
    void reset_transform();
    struct GuiMatrix get_transform() const;

    void set_fill_color(struct GuiColor c);
    void set_fill_paint(const &GuiPaint p);
    void set_stroke_color(struct GuiColor c);
    void set_stroke_paint(const &GuiPaint p);
    void set_line_width(float w);
    void set_line_cap(int lineCap);
    void set_line_join(int lineJoin);
    void set_miter_limit(float limit);
    // Alternating on/off lengths (n <= GUI_DASH_MAX; an odd count repeats
    // once more, as in SVG); n = 0 turns dashing off.
    void set_dash(const &float lengths, int n, float offset);
    void set_global_alpha(float a);
    void set_blend_mode(int mode);
    void set_fill_rule(int rule);
    void set_image_filter(int filter);
    // 0 = aliased (hard pixel edges: coverage rounded to 0 or 1).
    void set_anti_alias(int on);

    // ── clipping (intersects with the current clip; restore() undoes) ────
    void clip_rect(float x, float y, float w, float h);
    void clip_path(const &GuiPath p);
    void clip();                   // with the current path
    void reset_clip();
    // The clip's device-space bounds.
    struct GuiRect clip_bounds() const;

    // ── the current path ─────────────────────────────────────────────────
    void begin_path();
    void move_to(float x, float y);
    void line_to(float x, float y);
    void quad_to(float cx, float cy, float x, float y);
    void cubic_to(float c1x, float c1y, float c2x, float c2y, float x, float y);
    void arc(float cx, float cy, float r, float a0, float a1, int ccw);
    void arc_to(float x1, float y1, float x2, float y2, float r);
    void rect(float x, float y, float w, float h);
    void rounded_rect(float x, float y, float w, float h, float r);
    void ellipse(float cx, float cy, float rx, float ry);
    void close_path();
    void fill();
    void stroke();

    // ── paths and shapes ─────────────────────────────────────────────────
    void fill_path(const &GuiPath p);
    void stroke_path(const &GuiPath p);
    void fill_rect(float x, float y, float w, float h);
    void stroke_rect(float x, float y, float w, float h);
    // Makes the rectangle transparent (within the clip).
    void clear_rect(float x, float y, float w, float h);
    // Fills the whole target with 'c', ignoring transform, clip and blend.
    void clear(struct GuiColor c);
    void fill_rounded_rect(float x, float y, float w, float h, float r);
    void stroke_rounded_rect(float x, float y, float w, float h, float r);
    void fill_circle(float cx, float cy, float r);
    void stroke_circle(float cx, float cy, float r);
    void fill_ellipse(float cx, float cy, float rx, float ry);
    void draw_line(float x0, float y0, float x1, float y1);
    // The filled outline of stroking 'p' with the current line settings,
    // in user space — e.g. to hit-test or clip to a stroke.
    void stroke_to_path(const &GuiPath p, &GuiPath out);

    // ── images ───────────────────────────────────────────────────────────
    void draw_image(const &GuiImage img, float x, float y);
    void draw_image_scaled(const &GuiImage img, float x, float y, float w, float h);
    // The (sx,sy,sw,sh) part of the image into (dx,dy,dw,dh).
    void draw_image_rect(const &GuiImage img, float sx, float sy, float sw, float sh,
                         float dx, float dy, float dw, float dh);

    // ── text ─────────────────────────────────────────────────────────────
    void set_font(const &GuiTypeface face, float sizePx);
    void set_font_size(float sizePx);
    void set_text_align(int align);
    void set_text_baseline(int baseline);
    // Draws one line of UTF-8 text with the fill paint at (x, y) (see
    // set_text_baseline/set_text_align). Returns its advance width.
    float fill_text(const &char text, float x, float y);
    float fill_text_n(const &char text, unsigned long len, float x, float y);
    float measure_text(const &char text) const;
    // Draws a built layout with its top-left at (x, y).
    void fill_text_layout(const &GuiTextLayout layout, float x, float y);

    // ── reading back ─────────────────────────────────────────────────────
    // The straight-alpha color at device pixel (x, y) (transparent if out
    // of bounds).
    struct GuiColor get_pixel(int x, int y) const;
};

} // namespace std
