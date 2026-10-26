#pragma once
// SafeC Standard Library — GUI: 2D geometry for the vector canvas.
//
// GuiMatrix (2x3 affine transforms), GuiRect, and GuiPath — a vector path
// of move/line/quadratic/cubic/close segments with the usual builders
// (arcs, rounded rectangles, ellipses, SVG path data). Paths are plain
// data in user coordinates: gui_canvas.h fills, strokes and clips with
// them, gui_text.h produces glyph outlines as them.
//
//   struct GuiPath p = GuiPath::new();
//   p.move_to(10.0f, 10.0f);
//   p.cubic_to(40.0f, 0.0f, 60.0f, 40.0f, 90.0f, 10.0f);
//   p.rounded_rect(10.0f, 50.0f, 80.0f, 30.0f, 6.0f);
//   canvas.fill_path(&p);
//   p.free();
//
// Angles are in radians, y points down (screen coordinates), so a positive
// angle turns clockwise on screen — the HTML canvas convention.

#include <std/collections/vec.h>

// GuiPathSeg.verb
#define GUI_PATH_MOVE  0
#define GUI_PATH_LINE  1
#define GUI_PATH_QUAD  2
#define GUI_PATH_CUBIC 3
#define GUI_PATH_CLOSE 4

namespace std {

struct GuiPoint {
    float x;
    float y;
};

struct GuiRect {
    float x;
    float y;
    float w;
    float h;

    static struct GuiRect new(float x, float y, float w, float h);
    int contains(float px, float py) const;
    // The overlap of two rects (w/h 0 if they don't overlap) / the
    // smallest rect holding both.
    struct GuiRect intersect(struct GuiRect o) const;
    struct GuiRect unite(struct GuiRect o) const;
    int is_empty() const;
};

// x' = a*x + c*y + e
// y' = b*x + d*y + f
struct GuiMatrix {
    float a; float b;
    float c; float d;
    float e; float f;

    static struct GuiMatrix identity();
    static struct GuiMatrix translation(float tx, float ty);
    static struct GuiMatrix scaling(float sx, float sy);
    static struct GuiMatrix rotation(float radians);
    static struct GuiMatrix new(float a, float b, float c, float d, float e, float f);

    // 'self' applied after 'inner': (self * inner)(p) = self(inner(p)).
    struct GuiMatrix mul(struct GuiMatrix inner) const;
    // self * T/S/R — i.e. the new transform first translates/scales/
    // rotates points, then applies self (how a canvas' translate() etc.
    // compose).
    struct GuiMatrix translate(float tx, float ty) const;
    struct GuiMatrix scale(float sx, float sy) const;
    struct GuiMatrix rotate(float radians) const;
    // 1 and the inverse in 'out', or 0 if the matrix is singular.
    int  invert(&GuiMatrix out) const;
    float apply_x(float x, float y) const;
    float apply_y(float x, float y) const;
    // The transformed bounding box of 'r'.
    struct GuiRect apply_rect(struct GuiRect r) const;
    // sqrt(|det|): how much the matrix scales areas, per axis on average.
    float scale_factor() const;
    int  is_identity() const;
    int  is_translation() const;     // only e/f differ from identity
    int  is_axis_aligned() const;    // no rotation/skew (b == c == 0)
};

struct GuiPathSeg {
    int verb;          // GUI_PATH_*
    float x1; float y1;   // LINE/MOVE: the point; QUAD: control; CUBIC: control 1
    float x2; float y2;   // QUAD: end point; CUBIC: control 2
    float x3; float y3;   // CUBIC: end point
};

struct GuiPath {
    struct Vec<struct GuiPathSeg> segs;
    float curX; float curY;       // current point
    float startX; float startY;   // start of the current subpath
    int hasCurrent;

    static struct GuiPath new();

    void move_to(float x, float y);
    void line_to(float x, float y);
    void quad_to(float cx, float cy, float x, float y);
    void cubic_to(float c1x, float c1y, float c2x, float c2y, float x, float y);
    // Closes the current subpath (a straight line back to its start).
    void close();

    // Circular arc around (cx,cy) from angle a0 to a1, clockwise on screen
    // unless 'ccw'; joined to the current point with a line (HTML canvas'
    // arc()). Approximated with cubics (error < 0.01% of the radius).
    void arc(float cx, float cy, float r, float a0, float a1, int ccw);
    // A rounded corner from the current point toward (x1,y1), then toward
    // (x2,y2), with radius r (HTML canvas' arcTo()).
    void arc_to(float x1, float y1, float x2, float y2, float r);
    // An elliptic arc (rotated by 'rotation') — HTML canvas' ellipse().
    void ellipse_arc(float cx, float cy, float rx, float ry, float rotation, float a0, float a1, int ccw);
    // SVG's elliptical arc command from the current point to (x,y).
    void svg_arc_to(float rx, float ry, float xAxisRotation, int largeArc, int sweep, float x, float y);

    // Closed shapes, each its own subpath, clockwise on screen.
    void rect(float x, float y, float w, float h);
    void rounded_rect(float x, float y, float w, float h, float r);
    // Per-corner radii: top-left, top-right, bottom-right, bottom-left.
    void rounded_rect4(float x, float y, float w, float h, float tl, float tr, float br, float bl);
    void ellipse(float cx, float cy, float rx, float ry);
    void circle(float cx, float cy, float r);
    // A closed polygon through n points (xy pairs: pts[0..2n)).
    void polygon(const &float pts, int n);

    // Parses SVG path data ("M10 10 h 20 q 5 5 0 10 z ..." — every command
    // of SVG 1.1, absolute and relative) and appends it. Returns 1, or 0 at
    // the first malformed command (what came before it stays).
    int  add_svg(const &char d);

    // Appends 'other' transformed by 'm'.
    void append(const &GuiPath other, struct GuiMatrix m);
    // Transforms every point of this path in place.
    void transform(struct GuiMatrix m);

    // Bounding box of the points (control points included: may be a bit
    // larger than the curve).
    struct GuiRect bounds() const;
    int  is_empty() const;
    unsigned long length() const;   // number of segments
    void clear();
    void drop();
    void free() consume;   // = drop()
};

} // namespace std
