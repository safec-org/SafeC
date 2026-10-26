// SafeC Standard Library — GUI: paths and affine transforms (see gui_path.h).
#pragma once
#include <std/gui/gui_path.h>
#include <std/math.h>
#include <std/collections/vec.h>

namespace std {

#define GUI_PI_F 3.14159265358979f
#define GUI_KAPPA_F 0.5522847498f

// ── rects ─────────────────────────────────────────────────────────────────

struct GuiRect GuiRect::new(float x, float y, float w, float h) {
    struct GuiRect r;
    r.x = x; r.y = y; r.w = w; r.h = h;
    return r;
}

int GuiRect::contains(float px, float py) const {
    return (px >= self.x && py >= self.y && px < self.x + self.w && py < self.y + self.h) ? 1 : 0;
}

struct GuiRect GuiRect::intersect(struct GuiRect o) const {
    float x0 = self.x > o.x ? self.x : o.x;
    float y0 = self.y > o.y ? self.y : o.y;
    float x1 = self.x + self.w < o.x + o.w ? self.x + self.w : o.x + o.w;
    float y1 = self.y + self.h < o.y + o.h ? self.y + self.h : o.y + o.h;
    if (x1 < x0) { x1 = x0; }
    if (y1 < y0) { y1 = y0; }
    return GuiRect::new(x0, y0, x1 - x0, y1 - y0);
}

struct GuiRect GuiRect::unite(struct GuiRect o) const {
    if (self.w <= 0.0f || self.h <= 0.0f) { return o; }
    if (o.w <= 0.0f || o.h <= 0.0f) { return GuiRect::new(self.x, self.y, self.w, self.h); }
    float x0 = self.x < o.x ? self.x : o.x;
    float y0 = self.y < o.y ? self.y : o.y;
    float x1 = self.x + self.w > o.x + o.w ? self.x + self.w : o.x + o.w;
    float y1 = self.y + self.h > o.y + o.h ? self.y + self.h : o.y + o.h;
    return GuiRect::new(x0, y0, x1 - x0, y1 - y0);
}

int GuiRect::is_empty() const {
    return (self.w <= 0.0f || self.h <= 0.0f) ? 1 : 0;
}

// ── matrices ──────────────────────────────────────────────────────────────

struct GuiMatrix GuiMatrix::new(float a, float b, float c, float d, float e, float f) {
    struct GuiMatrix m;
    m.a = a; m.b = b; m.c = c; m.d = d; m.e = e; m.f = f;
    return m;
}

struct GuiMatrix GuiMatrix::identity() {
    return GuiMatrix::new(1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

struct GuiMatrix GuiMatrix::translation(float tx, float ty) {
    return GuiMatrix::new(1.0f, 0.0f, 0.0f, 1.0f, tx, ty);
}

struct GuiMatrix GuiMatrix::scaling(float sx, float sy) {
    return GuiMatrix::new(sx, 0.0f, 0.0f, sy, 0.0f, 0.0f);
}

struct GuiMatrix GuiMatrix::rotation(float radians) {
    float s = sin_f(radians);
    float c = cos_f(radians);
    return GuiMatrix::new(c, s, -s, c, 0.0f, 0.0f);
}

struct GuiMatrix GuiMatrix::mul(struct GuiMatrix n) const {
    // self * n
    return GuiMatrix::new(self.a * n.a + self.c * n.b,
                          self.b * n.a + self.d * n.b,
                          self.a * n.c + self.c * n.d,
                          self.b * n.c + self.d * n.d,
                          self.a * n.e + self.c * n.f + self.e,
                          self.b * n.e + self.d * n.f + self.f);
}

struct GuiMatrix GuiMatrix::translate(float tx, float ty) const {
    return self.mul(GuiMatrix::translation(tx, ty));
}

struct GuiMatrix GuiMatrix::scale(float sx, float sy) const {
    return self.mul(GuiMatrix::scaling(sx, sy));
}

struct GuiMatrix GuiMatrix::rotate(float radians) const {
    return self.mul(GuiMatrix::rotation(radians));
}

int GuiMatrix::invert(&GuiMatrix out) const {
    double det = (double)self.a * (double)self.d - (double)self.b * (double)self.c;
    if (det == 0.0 || det != det) { return 0; }
    double id = 1.0 / det;
    double a = (double)self.d * id;
    double b = -(double)self.b * id;
    double c = -(double)self.c * id;
    double d = (double)self.a * id;
    double e = -((double)self.e * a + (double)self.f * c);
    double f = -((double)self.e * b + (double)self.f * d);
    *out = GuiMatrix::new((float)a, (float)b, (float)c, (float)d, (float)e, (float)f);
    return 1;
}

float GuiMatrix::apply_x(float x, float y) const { return self.a * x + self.c * y + self.e; }
float GuiMatrix::apply_y(float x, float y) const { return self.b * x + self.d * y + self.f; }

struct GuiRect GuiMatrix::apply_rect(struct GuiRect r) const {
    float xs[4]; float ys[4];
    xs[0] = self.apply_x(r.x, r.y);             ys[0] = self.apply_y(r.x, r.y);
    xs[1] = self.apply_x(r.x + r.w, r.y);       ys[1] = self.apply_y(r.x + r.w, r.y);
    xs[2] = self.apply_x(r.x, r.y + r.h);       ys[2] = self.apply_y(r.x, r.y + r.h);
    xs[3] = self.apply_x(r.x + r.w, r.y + r.h); ys[3] = self.apply_y(r.x + r.w, r.y + r.h);
    float x0 = xs[0]; float x1 = xs[0]; float y0 = ys[0]; float y1 = ys[0];
    int i = 1;
    while (i < 4) {
        if (xs[i] < x0) { x0 = xs[i]; }
        if (xs[i] > x1) { x1 = xs[i]; }
        if (ys[i] < y0) { y0 = ys[i]; }
        if (ys[i] > y1) { y1 = ys[i]; }
        i = i + 1;
    }
    return GuiRect::new(x0, y0, x1 - x0, y1 - y0);
}

float GuiMatrix::scale_factor() const {
    float det = self.a * self.d - self.b * self.c;
    if (det < 0.0f) { det = -det; }
    return sqrt_f(det);
}

int GuiMatrix::is_identity() const {
    return (self.a == 1.0f && self.b == 0.0f && self.c == 0.0f && self.d == 1.0f &&
            self.e == 0.0f && self.f == 0.0f) ? 1 : 0;
}

int GuiMatrix::is_translation() const {
    return (self.a == 1.0f && self.b == 0.0f && self.c == 0.0f && self.d == 1.0f) ? 1 : 0;
}

int GuiMatrix::is_axis_aligned() const {
    return (self.b == 0.0f && self.c == 0.0f) ? 1 : 0;
}

// ── paths ─────────────────────────────────────────────────────────────────

struct GuiPath GuiPath::new() {
    struct GuiPath p;
    p.segs = Vec::new();
    p.curX = 0.0f; p.curY = 0.0f;
    p.startX = 0.0f; p.startY = 0.0f;
    p.hasCurrent = 0;
    return p;
}

static void __gui_path_push(&GuiPath p, int verb, float x1, float y1, float x2, float y2, float x3, float y3) {
    struct GuiPathSeg s;
    s.verb = verb;
    s.x1 = x1; s.y1 = y1; s.x2 = x2; s.y2 = y2; s.x3 = x3; s.y3 = y3;
    p.segs.push(s);
}

void GuiPath::move_to(float x, float y) {
    __gui_path_push(self, GUI_PATH_MOVE, x, y, 0.0f, 0.0f, 0.0f, 0.0f);
    self.curX = x; self.curY = y;
    self.startX = x; self.startY = y;
    self.hasCurrent = 1;
}

void GuiPath::line_to(float x, float y) {
    if (!self.hasCurrent) { self.move_to(x, y); return; }
    __gui_path_push(self, GUI_PATH_LINE, x, y, 0.0f, 0.0f, 0.0f, 0.0f);
    self.curX = x; self.curY = y;
}

void GuiPath::quad_to(float cx, float cy, float x, float y) {
    if (!self.hasCurrent) { self.move_to(cx, cy); }
    __gui_path_push(self, GUI_PATH_QUAD, cx, cy, x, y, 0.0f, 0.0f);
    self.curX = x; self.curY = y;
}

void GuiPath::cubic_to(float c1x, float c1y, float c2x, float c2y, float x, float y) {
    if (!self.hasCurrent) { self.move_to(c1x, c1y); }
    __gui_path_push(self, GUI_PATH_CUBIC, c1x, c1y, c2x, c2y, x, y);
    self.curX = x; self.curY = y;
}

void GuiPath::close() {
    if (!self.hasCurrent) { return; }
    __gui_path_push(self, GUI_PATH_CLOSE, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    self.curX = self.startX;
    self.curY = self.startY;
}

// Emits the arc of the ellipse (center cx,cy, radii rx,ry, rotated by
// 'rot') from angle a0 through the signed angle 'sweep' as cubics, starting
// with a line (or move) to its first point.
static void __gui_path_arc_sweep(&GuiPath p, float cx, float cy, float rx, float ry, float rot,
                                 float a0, float sweep, int connect) {
    float cr = cos_f(rot);
    float sr = sin_f(rot);
    float x0 = cos_f(a0);
    float y0 = sin_f(a0);
    float sx = cx + cr * rx * x0 - sr * ry * y0;
    float sy = cy + sr * rx * x0 + cr * ry * y0;
    if (connect && p.hasCurrent) { p.line_to(sx, sy); } else { p.move_to(sx, sy); }
    float total = sweep < 0.0f ? -sweep : sweep;
    if (total < 1e-6f) { return; }
    int n = (int)ceil_f(total / (GUI_PI_F * 0.5f) - 1e-4f);
    if (n < 1) { n = 1; }
    float step = sweep / (float)n;
    float k = 4.0f / 3.0f * tan_f(step / 4.0f);
    float a = a0;
    int i = 0;
    while (i < n) {
        float b = a + step;
        float ca = cos_f(a); float sa = sin_f(a);
        float cb = cos_f(b); float sb = sin_f(b);
        // unit-circle control points
        float p1x = ca - k * sa; float p1y = sa + k * ca;
        float p2x = cb + k * sb; float p2y = sb - k * cb;
        float c1x = cx + cr * rx * p1x - sr * ry * p1y;
        float c1y = cy + sr * rx * p1x + cr * ry * p1y;
        float c2x = cx + cr * rx * p2x - sr * ry * p2y;
        float c2y = cy + sr * rx * p2x + cr * ry * p2y;
        float ex = cx + cr * rx * cb - sr * ry * sb;
        float ey = cy + sr * rx * cb + cr * ry * sb;
        p.cubic_to(c1x, c1y, c2x, c2y, ex, ey);
        a = b;
        i = i + 1;
    }
}

// The signed sweep from a0 to a1 in the HTML canvas' rules.
static float __gui_arc_sweep(float a0, float a1, int ccw) {
    float twoPi = 2.0f * GUI_PI_F;
    if (!ccw) {
        float d = a1 - a0;
        if (d >= twoPi) { return twoPi; }
        while (d < 0.0f) { d = d + twoPi; }
        while (d >= twoPi) { d = d - twoPi; }
        return d;
    }
    float d = a0 - a1;
    if (d >= twoPi) { return -twoPi; }
    while (d < 0.0f) { d = d + twoPi; }
    while (d >= twoPi) { d = d - twoPi; }
    return -d;
}

void GuiPath::arc(float cx, float cy, float r, float a0, float a1, int ccw) {
    if (r < 0.0f) { r = -r; }
    __gui_path_arc_sweep(self, cx, cy, r, r, 0.0f, a0, __gui_arc_sweep(a0, a1, ccw), 1);
}

void GuiPath::ellipse_arc(float cx, float cy, float rx, float ry, float rotation, float a0, float a1, int ccw) {
    if (rx < 0.0f) { rx = -rx; }
    if (ry < 0.0f) { ry = -ry; }
    __gui_path_arc_sweep(self, cx, cy, rx, ry, rotation, a0, __gui_arc_sweep(a0, a1, ccw), 1);
}

void GuiPath::arc_to(float x1, float y1, float x2, float y2, float r) {
    if (!self.hasCurrent) { self.move_to(x1, y1); }
    float x0 = self.curX;
    float y0 = self.curY;
    float v1x = x0 - x1; float v1y = y0 - y1;
    float v2x = x2 - x1; float v2y = y2 - y1;
    float l1 = sqrt_f(v1x * v1x + v1y * v1y);
    float l2 = sqrt_f(v2x * v2x + v2y * v2y);
    if (r <= 0.0f || l1 < 1e-6f || l2 < 1e-6f) { self.line_to(x1, y1); return; }
    v1x = v1x / l1; v1y = v1y / l1;
    v2x = v2x / l2; v2y = v2y / l2;
    float cross = v1x * v2y - v1y * v2x;
    float dot = v1x * v2x + v1y * v2y;
    if (cross > -1e-6f && cross < 1e-6f) { self.line_to(x1, y1); return; }   // collinear
    float theta = acos_f(dot < -1.0f ? -1.0f : (dot > 1.0f ? 1.0f : dot));  // angle between the legs
    float d = r / tan_f(theta * 0.5f);
    float t1x = x1 + v1x * d; float t1y = y1 + v1y * d;
    float t2x = x1 + v2x * d; float t2y = y1 + v2y * d;
    float bx = v1x + v2x; float by = v1y + v2y;
    float bl = sqrt_f(bx * bx + by * by);
    float h = r / sin_f(theta * 0.5f);
    float cx = x1 + bx / bl * h;
    float cy = y1 + by / bl * h;
    float a0 = atan2_f(t1y - cy, t1x - cx);
    float a1 = atan2_f(t2y - cy, t2x - cx);
    float sw = a1 - a0;
    while (sw > GUI_PI_F) { sw = sw - 2.0f * GUI_PI_F; }
    while (sw < -GUI_PI_F) { sw = sw + 2.0f * GUI_PI_F; }
    self.line_to(t1x, t1y);
    __gui_path_arc_sweep(self, cx, cy, r, r, 0.0f, a0, sw, 1);
}

void GuiPath::svg_arc_to(float rx, float ry, float phiDeg, int largeArc, int sweepFlag, float x, float y) {
    // SVG 1.1 implementation notes F.6.5/F.6.6: endpoint to center form.
    if (!self.hasCurrent) { self.move_to(x, y); return; }
    float x1 = self.curX;
    float y1 = self.curY;
    if (x1 == x && y1 == y) { return; }
    if (rx < 0.0f) { rx = -rx; }
    if (ry < 0.0f) { ry = -ry; }
    if (rx < 1e-6f || ry < 1e-6f) { self.line_to(x, y); return; }
    float phi = phiDeg * GUI_PI_F / 180.0f;
    float cp = cos_f(phi);
    float sp = sin_f(phi);
    float dx2 = (x1 - x) * 0.5f;
    float dy2 = (y1 - y) * 0.5f;
    float x1p = cp * dx2 + sp * dy2;
    float y1p = -sp * dx2 + cp * dy2;
    float lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if (lambda > 1.0f) {
        float s = sqrt_f(lambda);
        rx = rx * s;
        ry = ry * s;
    }
    float num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
    float den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
    float coef = 0.0f;
    if (den > 0.0f && num > 0.0f) { coef = sqrt_f(num / den); }
    if (largeArc == sweepFlag) { coef = -coef; }
    float cxp = coef * rx * y1p / ry;
    float cyp = -coef * ry * x1p / rx;
    float cx = cp * cxp - sp * cyp + (x1 + x) * 0.5f;
    float cy = sp * cxp + cp * cyp + (y1 + y) * 0.5f;
    float ux = (x1p - cxp) / rx; float uy = (y1p - cyp) / ry;
    float vx = (-x1p - cxp) / rx; float vy = (-y1p - cyp) / ry;
    float a0 = atan2_f(uy, ux);
    float dtheta = atan2_f(ux * vy - uy * vx, ux * vx + uy * vy);
    if (!sweepFlag && dtheta > 0.0f) { dtheta = dtheta - 2.0f * GUI_PI_F; }
    if (sweepFlag && dtheta < 0.0f) { dtheta = dtheta + 2.0f * GUI_PI_F; }
    __gui_path_arc_sweep(self, cx, cy, rx, ry, phi, a0, dtheta, 1);
    // Land exactly on the end point.
    self.curX = x;
    self.curY = y;
    unsigned long n = self.segs.len;
    if (n > 0UL) {
        &GuiPathSeg last = self.segs.at(n - 1UL);
        if (last.verb == GUI_PATH_CUBIC) { last.x3 = x; last.y3 = y; }
    }
}

void GuiPath::rect(float x, float y, float w, float h) {
    self.move_to(x, y);
    self.line_to(x + w, y);
    self.line_to(x + w, y + h);
    self.line_to(x, y + h);
    self.close();
}

void GuiPath::rounded_rect4(float x, float y, float w, float h, float tl, float tr, float br, float bl) {
    if (w < 0.0f) { x = x + w; w = -w; }
    if (h < 0.0f) { y = y + h; h = -h; }
    float lim = (w < h ? w : h) * 0.5f;
    if (tl > lim) { tl = lim; }
    if (tr > lim) { tr = lim; }
    if (br > lim) { br = lim; }
    if (bl > lim) { bl = lim; }
    if (tl < 0.0f) { tl = 0.0f; }
    if (tr < 0.0f) { tr = 0.0f; }
    if (br < 0.0f) { br = 0.0f; }
    if (bl < 0.0f) { bl = 0.0f; }
    float k = 1.0f - GUI_KAPPA_F;
    self.move_to(x + tl, y);
    self.line_to(x + w - tr, y);
    if (tr > 0.0f) { self.cubic_to(x + w - tr * k, y, x + w, y + tr * k, x + w, y + tr); }
    self.line_to(x + w, y + h - br);
    if (br > 0.0f) { self.cubic_to(x + w, y + h - br * k, x + w - br * k, y + h, x + w - br, y + h); }
    self.line_to(x + bl, y + h);
    if (bl > 0.0f) { self.cubic_to(x + bl * k, y + h, x, y + h - bl * k, x, y + h - bl); }
    self.line_to(x, y + tl);
    if (tl > 0.0f) { self.cubic_to(x, y + tl * k, x + tl * k, y, x + tl, y); }
    self.close();
}

void GuiPath::rounded_rect(float x, float y, float w, float h, float r) {
    self.rounded_rect4(x, y, w, h, r, r, r, r);
}

void GuiPath::ellipse(float cx, float cy, float rx, float ry) {
    float kx = rx * GUI_KAPPA_F;
    float ky = ry * GUI_KAPPA_F;
    self.move_to(cx + rx, cy);
    self.cubic_to(cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry);
    self.cubic_to(cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy);
    self.cubic_to(cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry);
    self.cubic_to(cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy);
    self.close();
}

void GuiPath::circle(float cx, float cy, float r) {
    self.ellipse(cx, cy, r, r);
}

void GuiPath::polygon_n_(const &float pts, int n) {
    if (n <= 0) { return; }
    unsafe {
        self.move_to(pts[0], pts[1]);
        int i = 1;
        while (i < n) {
            self.line_to(pts[i * 2], pts[i * 2 + 1]);
            i = i + 1;
        }
    }
    self.close();
}

void GuiPath::polygon([]const float pts_s) {
    unsigned long np = (unsigned long)pts_s.len;
    float pts_d_[1];
    const &float pts = &pts_d_[0];
    if (pts_s.len > 0UL) { unsafe { pts = (const &float)pts_s.ptr; } }
    int n = (int)(np / 2UL);
    self.polygon_n_(pts, n);
}

void GuiPath::append(const &GuiPath other, struct GuiMatrix m) {
    unsigned long i = 0UL;
    while (i < other.segs.len) {
        struct GuiPathSeg s = *other.segs.at(i);
        float x1 = m.apply_x(s.x1, s.y1); float y1 = m.apply_y(s.x1, s.y1);
        float x2 = m.apply_x(s.x2, s.y2); float y2 = m.apply_y(s.x2, s.y2);
        float x3 = m.apply_x(s.x3, s.y3); float y3 = m.apply_y(s.x3, s.y3);
        if (s.verb == GUI_PATH_MOVE) { self.move_to(x1, y1); }
        else if (s.verb == GUI_PATH_LINE) { self.line_to(x1, y1); }
        else if (s.verb == GUI_PATH_QUAD) { self.quad_to(x1, y1, x2, y2); }
        else if (s.verb == GUI_PATH_CUBIC) { self.cubic_to(x1, y1, x2, y2, x3, y3); }
        else { self.close(); }
        i = i + 1UL;
    }
}

void GuiPath::transform(struct GuiMatrix m) {
    unsigned long i = 0UL;
    while (i < self.segs.len) {
        &GuiPathSeg s = self.segs.at(i);
        float x1 = m.apply_x(s.x1, s.y1); float y1 = m.apply_y(s.x1, s.y1);
        float x2 = m.apply_x(s.x2, s.y2); float y2 = m.apply_y(s.x2, s.y2);
        float x3 = m.apply_x(s.x3, s.y3); float y3 = m.apply_y(s.x3, s.y3);
        s.x1 = x1; s.y1 = y1; s.x2 = x2; s.y2 = y2; s.x3 = x3; s.y3 = y3;
        i = i + 1UL;
    }
    float cx = m.apply_x(self.curX, self.curY);
    float cy = m.apply_y(self.curX, self.curY);
    float sx = m.apply_x(self.startX, self.startY);
    float sy = m.apply_y(self.startX, self.startY);
    self.curX = cx; self.curY = cy;
    self.startX = sx; self.startY = sy;
}

struct GuiRect GuiPath::bounds() const {
    int any = 0;
    float x0 = 0.0f; float y0 = 0.0f; float x1 = 0.0f; float y1 = 0.0f;
    unsigned long i = 0UL;
    while (i < self.segs.len) {
        struct GuiPathSeg s = *self.segs.at(i);
        int np = 0;
        if (s.verb == GUI_PATH_MOVE || s.verb == GUI_PATH_LINE) { np = 1; }
        else if (s.verb == GUI_PATH_QUAD) { np = 2; }
        else if (s.verb == GUI_PATH_CUBIC) { np = 3; }
        int k = 0;
        while (k < np) {
            float px = k == 0 ? s.x1 : (k == 1 ? s.x2 : s.x3);
            float py = k == 0 ? s.y1 : (k == 1 ? s.y2 : s.y3);
            if (!any) { x0 = px; x1 = px; y0 = py; y1 = py; any = 1; }
            if (px < x0) { x0 = px; }
            if (px > x1) { x1 = px; }
            if (py < y0) { y0 = py; }
            if (py > y1) { y1 = py; }
            k = k + 1;
        }
        i = i + 1UL;
    }
    return GuiRect::new(x0, y0, x1 - x0, y1 - y0);
}

int GuiPath::is_empty() const {
    unsigned long i = 0UL;
    while (i < self.segs.len) {
        int v = self.segs.at(i).verb;
        if (v != GUI_PATH_MOVE && v != GUI_PATH_CLOSE) { return 0; }
        i = i + 1UL;
    }
    return 1;
}

unsigned long GuiPath::length() const { return self.segs.len; }

void GuiPath::clear() {
    self.segs.clear();
    self.hasCurrent = 0;
    self.curX = 0.0f; self.curY = 0.0f;
    self.startX = 0.0f; self.startY = 0.0f;
}

void GuiPath::free() { self.drop(); }

void GuiPath::drop() {
    self.segs.free();
    self.hasCurrent = 0;
}

// ── SVG path data ─────────────────────────────────────────────────────────

static int __gui_svg_ws(char c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',') ? 1 : 0;
}

static void __gui_svg_skip(const &char d, &unsigned long i) {
    unsafe { while (d[*i] != '\0' && __gui_svg_ws(d[*i])) { *i = *i + 1UL; } }
}

// A number at d[*i] (after separators); 1 and the value, or 0.
static int __gui_svg_num(const &char d, &unsigned long i, &float out) {
    __gui_svg_skip(d, i);
    unsafe {
        unsigned long p = *i;
        double sign = 1.0;
        if (d[p] == '+' || d[p] == '-') { if (d[p] == '-') { sign = -1.0; } p = p + 1UL; }
        double v = 0.0;
        int digits = 0;
        while (d[p] >= '0' && d[p] <= '9') { v = v * 10.0 + (double)(d[p] - '0'); p = p + 1UL; digits = digits + 1; }
        if (d[p] == '.') {
            p = p + 1UL;
            double scale = 0.1;
            while (d[p] >= '0' && d[p] <= '9') { v = v + (double)(d[p] - '0') * scale; scale = scale * 0.1; p = p + 1UL; digits = digits + 1; }
        }
        if (digits == 0) { return 0; }
        if (d[p] == 'e' || d[p] == 'E') {
            unsigned long q = p + 1UL;
            double esign = 1.0;
            if (d[q] == '+' || d[q] == '-') { if (d[q] == '-') { esign = -1.0; } q = q + 1UL; }
            if (d[q] >= '0' && d[q] <= '9') {
                int ex = 0;
                while (d[q] >= '0' && d[q] <= '9') { ex = ex * 10 + (int)(d[q] - '0'); q = q + 1UL; }
                double m = 1.0;
                int k = 0;
                while (k < ex && k < 64) { m = m * 10.0; k = k + 1; }
                v = esign > 0.0 ? v * m : v / m;
                p = q;
            }
        }
        *i = p;
        *out = (float)(v * sign);
        return 1;
    }
}

// An arc flag: a single '0' or '1' (may be followed directly by the next number).
static int __gui_svg_flag(const &char d, &unsigned long i, &int out) {
    __gui_svg_skip(d, i);
    unsafe {
        char c = d[*i];
        if (c != '0' && c != '1') { return 0; }
        *out = c == '1' ? 1 : 0;
        *i = *i + 1UL;
        return 1;
    }
}

static int __gui_svg_has_num(const &char d, &unsigned long i) {
    __gui_svg_skip(d, i);
    unsafe {
        char c = d[*i];
        return ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.') ? 1 : 0;
    }
}

int GuiPath::add_svg(const &char d) {
    unsigned long i = 0UL;
    char cmd = ' ';
    float lastCx = 0.0f; float lastCy = 0.0f;   // last control point for S/T
    char lastCmd = ' ';
    while (1) {
        __gui_svg_skip(d, &i);
        char c;
        unsafe { c = d[i]; }
        if (c == '\0') { break; }
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            cmd = c;
            i = i + 1UL;
        } else if (cmd == ' ') {
            return 0;
        }
        int rel = (cmd >= 'a' && cmd <= 'z') ? 1 : 0;
        char up = rel ? (char)(cmd - 32) : cmd;
        float ox = rel ? self.curX : 0.0f;
        float oy = rel ? self.curY : 0.0f;
        if (up == 'Z') {
            self.close();
            lastCmd = 'Z';
            cmd = ' ';
            continue;
        }
        float a = 0.0f; float b = 0.0f; float c2 = 0.0f; float d2 = 0.0f; float e = 0.0f; float f = 0.0f;
        if (up == 'M') {
            if (!__gui_svg_num(d, &i, &a) || !__gui_svg_num(d, &i, &b)) { return 0; }
            self.move_to(ox + a, oy + b);
            cmd = rel ? 'l' : 'L';   // further pairs are line-tos
        } else if (up == 'L') {
            if (!__gui_svg_num(d, &i, &a) || !__gui_svg_num(d, &i, &b)) { return 0; }
            self.line_to(ox + a, oy + b);
        } else if (up == 'H') {
            if (!__gui_svg_num(d, &i, &a)) { return 0; }
            self.line_to(ox + a, self.curY);
        } else if (up == 'V') {
            if (!__gui_svg_num(d, &i, &a)) { return 0; }
            self.line_to(self.curX, oy + a);
        } else if (up == 'C') {
            if (!__gui_svg_num(d, &i, &a) || !__gui_svg_num(d, &i, &b) || !__gui_svg_num(d, &i, &c2) ||
                !__gui_svg_num(d, &i, &d2) || !__gui_svg_num(d, &i, &e) || !__gui_svg_num(d, &i, &f)) { return 0; }
            self.cubic_to(ox + a, oy + b, ox + c2, oy + d2, ox + e, oy + f);
            lastCx = ox + c2; lastCy = oy + d2;
        } else if (up == 'S') {
            if (!__gui_svg_num(d, &i, &c2) || !__gui_svg_num(d, &i, &d2) ||
                !__gui_svg_num(d, &i, &e) || !__gui_svg_num(d, &i, &f)) { return 0; }
            float r1x = self.curX; float r1y = self.curY;
            if (lastCmd == 'C' || lastCmd == 'S') { r1x = 2.0f * self.curX - lastCx; r1y = 2.0f * self.curY - lastCy; }
            self.cubic_to(r1x, r1y, ox + c2, oy + d2, ox + e, oy + f);
            lastCx = ox + c2; lastCy = oy + d2;
        } else if (up == 'Q') {
            if (!__gui_svg_num(d, &i, &a) || !__gui_svg_num(d, &i, &b) ||
                !__gui_svg_num(d, &i, &e) || !__gui_svg_num(d, &i, &f)) { return 0; }
            self.quad_to(ox + a, oy + b, ox + e, oy + f);
            lastCx = ox + a; lastCy = oy + b;
        } else if (up == 'T') {
            if (!__gui_svg_num(d, &i, &e) || !__gui_svg_num(d, &i, &f)) { return 0; }
            float qx = self.curX; float qy = self.curY;
            if (lastCmd == 'Q' || lastCmd == 'T') { qx = 2.0f * self.curX - lastCx; qy = 2.0f * self.curY - lastCy; }
            self.quad_to(qx, qy, ox + e, oy + f);
            lastCx = qx; lastCy = qy;
        } else if (up == 'A') {
            int large = 0; int sweep = 0;
            if (!__gui_svg_num(d, &i, &a) || !__gui_svg_num(d, &i, &b) || !__gui_svg_num(d, &i, &c2) ||
                !__gui_svg_flag(d, &i, &large) || !__gui_svg_flag(d, &i, &sweep) ||
                !__gui_svg_num(d, &i, &e) || !__gui_svg_num(d, &i, &f)) { return 0; }
            self.svg_arc_to(a, b, c2, large, sweep, ox + e, oy + f);
        } else {
            return 0;
        }
        lastCmd = up;
        if (!__gui_svg_has_num(d, &i)) {
            // The next thing must be a command letter (or the end).
            unsafe {
                char n = d[i];
                if (n != '\0' && !((n >= 'A' && n <= 'Z') || (n >= 'a' && n <= 'z'))) { return 0; }
            }
        }
    }
    return 1;
}

} // namespace std
