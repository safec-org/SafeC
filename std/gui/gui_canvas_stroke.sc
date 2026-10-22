// SafeC Standard Library — GUI: the canvas' stroker and dasher (see
// gui_canvas.h).
//
// A stroke is turned into filled polygons in user space — one quad per
// segment, a wedge per join (miter, round or bevel), a cap shape per open
// end — every polygon wound the same way, so filling them all with the
// nonzero rule gives their exact union (no seams where they overlap, no
// double-blended pixels). The polygons are then transformed and filled
// like any path, which also makes strokes transform correctly
// (non-uniform scale gives a non-uniform line width, as it should).
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/math.h>
#include <std/collections/vec.h>

namespace std {

#define GUI_STROKE_PI 3.14159265358979f

static float __gui_poly_area(const float* xy, int n) {
    float a = 0.0f;
    unsafe {
        int i = 0;
        while (i < n) {
            int j = i + 1 < n ? i + 1 : 0;
            a = a + xy[i * 2] * xy[j * 2 + 1] - xy[j * 2] * xy[i * 2 + 1];
            i = i + 1;
        }
    }
    return a;
}

// Appends a closed polygon, wound so its signed area is positive.
static void __gui_emit_poly(&Vec<float> out, &Vec<int> cont, const float* xy, int n) {
    if (n < 3) { return; }
    float area = __gui_poly_area(xy, n);
    if (area > -1e-9f && area < 1e-9f) { return; }
    cont.push((int)(out.len / 2UL));
    cont.push(n);
    cont.push(1);
    unsafe {
        if (area > 0.0f) {
            int i = 0;
            while (i < n) { out.push(xy[i * 2]); out.push(xy[i * 2 + 1]); i = i + 1; }
        } else {
            int i = n - 1;
            while (i >= 0) { out.push(xy[i * 2]); out.push(xy[i * 2 + 1]); i = i - 1; }
        }
    }
}

// Segments to approximate an arc of 'angle' radians at radius r within tol.
static int __gui_arc_steps(float r, float angle, float tol) {
    if (angle < 0.0f) { angle = -angle; }
    if (r <= tol) { return angle > 0.5f ? 4 : 1; }
    float c = 1.0f - tol / r;
    if (c < -1.0f) { c = -1.0f; }
    float step = 2.0f * acos_f(c);
    if (step < 0.01f) { step = 0.01f; }
    int n = (int)ceil_f(angle / step);
    if (n < 1) { n = 1; }
    if (n > 256) { n = 256; }
    return n;
}

static void __gui_emit_circle(&Vec<float> out, &Vec<int> cont, float cx, float cy, float r, float tol) {
    int n = __gui_arc_steps(r, 2.0f * GUI_STROKE_PI, tol);
    if (n < 8) { n = 8; }
    float buf[520];
    if (n > 256) { n = 256; }
    int i = 0;
    while (i < n) {
        float a = 2.0f * GUI_STROKE_PI * (float)i / (float)n;
        buf[i * 2] = cx + r * cos_f(a);
        buf[i * 2 + 1] = cy + r * sin_f(a);
        i = i + 1;
    }
    unsafe { __gui_emit_poly(out, cont, &buf[0], n); }
}

// A wedge around v from offset direction (ax,ay) to (bx,by) (unit
// vectors), the short way: the round join / round cap piece.
static void __gui_emit_fan(&Vec<float> out, &Vec<int> cont, float vx, float vy, float ax, float ay,
                           float bx, float by, float r, float tol) {
    float a0 = atan2_f(ay, ax);
    float a1 = atan2_f(by, bx);
    float sw = a1 - a0;
    while (sw > GUI_STROKE_PI) { sw = sw - 2.0f * GUI_STROKE_PI; }
    while (sw < -GUI_STROKE_PI) { sw = sw + 2.0f * GUI_STROKE_PI; }
    int n = __gui_arc_steps(r, sw, tol);
    if (n > 250) { n = 250; }
    float buf[520];
    buf[0] = vx;
    buf[1] = vy;
    int i = 0;
    while (i <= n) {
        float a = a0 + sw * (float)i / (float)n;
        buf[(i + 1) * 2] = vx + r * cos_f(a);
        buf[(i + 1) * 2 + 1] = vy + r * sin_f(a);
        i = i + 1;
    }
    unsafe { __gui_emit_poly(out, cont, &buf[0], n + 2); }
}

static void __gui_emit_join(&Vec<float> out, &Vec<int> cont, float vx, float vy,
                            float d0x, float d0y, float d1x, float d1y,
                            float hw, int lineJoin, float miterLimit, float tol) {
    float cross = d0x * d1y - d0y * d1x;
    float dot = d0x * d1x + d0y * d1y;
    if (cross > -1e-6f && cross < 1e-6f && dot > 0.0f) { return; }   // straight on
    // Unit normals (-dy, dx); the join goes on the outer side of the turn.
    float n0x = -d0y; float n0y = d0x;
    float n1x = -d1y; float n1y = d1x;
    float side = cross > 0.0f ? -1.0f : 1.0f;
    float ax = n0x * side; float ay = n0y * side;
    float bx = n1x * side; float by = n1y * side;
    if (lineJoin == GUI_JOIN_ROUND) {
        __gui_emit_fan(out, cont, vx, vy, ax, ay, bx, by, hw, tol);
        return;
    }
    float buf[8];
    buf[0] = vx; buf[1] = vy;
    buf[2] = vx + ax * hw; buf[3] = vy + ay * hw;
    if (lineJoin == GUI_JOIN_MITER) {
        float mx = ax + bx;
        float my = ay + by;
        float ml = sqrt_f(mx * mx + my * my);           // 2 cos(phi/2)
        if (ml > 1e-6f) {
            float ratio = 2.0f / ml;                     // miter length / line width
            if (ratio <= miterLimit) {
                float k = hw * ratio / ml;               // hw / cos(phi/2), along the unit bisector
                buf[4] = vx + mx * k; buf[5] = vy + my * k;
                buf[6] = vx + bx * hw; buf[7] = vy + by * hw;
                unsafe { __gui_emit_poly(out, cont, &buf[0], 4); }
                return;
            }
        }
    }
    // bevel
    buf[4] = vx + bx * hw; buf[5] = vy + by * hw;
    unsafe { __gui_emit_poly(out, cont, &buf[0], 3); }
}

static void __gui_emit_cap(&Vec<float> out, &Vec<int> cont, float px, float py, float dx, float dy,
                           float hw, int cap, float tol) {
    // (dx,dy): unit direction pointing out of the line at this end.
    if (cap == GUI_CAP_BUTT) { return; }
    float nx = -dy; float ny = dx;
    if (cap == GUI_CAP_SQUARE) {
        float buf[8];
        buf[0] = px + nx * hw; buf[1] = py + ny * hw;
        buf[2] = px + nx * hw + dx * hw; buf[3] = py + ny * hw + dy * hw;
        buf[4] = px - nx * hw + dx * hw; buf[5] = py - ny * hw + dy * hw;
        buf[6] = px - nx * hw; buf[7] = py - ny * hw;
        unsafe { __gui_emit_poly(out, cont, &buf[0], 4); }
        return;
    }
    // round: the outer half disc, as two quarter fans
    __gui_emit_fan(out, cont, px, py, nx, ny, dx, dy, hw, tol);
    __gui_emit_fan(out, cont, px, py, dx, dy, -nx, -ny, hw, tol);
}

// Strokes the polylines (x,y pairs + (first, count, closed) triples) into
// polygons appended to out/outCont.
void __gui_canvas_stroke_polys(&Vec<float> pts, &Vec<int> cont, float width, int cap, int lineJoin,
                               float miterLimit, float tol, &Vec<float> out, &Vec<int> outCont) {
    out.clear();
    outCont.clear();
    float hw = width * 0.5f;
    if (hw <= 0.0f) { return; }
    struct Vec<float> clean = Vec::new();
    unsafe {
        const float* p = (const float*)pts.data;
        const int* ct = (const int*)cont.data;
        unsigned long nc = cont.len / 3UL;
        unsigned long k = 0UL;
        while (k < nc) {
            int first = ct[k * 3UL];
            int count = ct[k * 3UL + 1UL];
            int closed = ct[k * 3UL + 2UL];
            k = k + 1UL;
            // Drop repeated points.
            clean.clear();
            int i = 0;
            while (i < count) {
                float x = p[(first + i) * 2];
                float y = p[(first + i) * 2 + 1];
                unsigned long m = clean.len;
                if (m >= 2UL) {
                    const float* cp = (const float*)clean.data;
                    float ddx = x - cp[m - 2UL];
                    float ddy = y - cp[m - 1UL];
                    if (ddx * ddx + ddy * ddy < 1e-10f) { i = i + 1; continue; }
                }
                clean.push(x);
                clean.push(y);
                i = i + 1;
            }
            int n = (int)(clean.len / 2UL);
            const float* q = (const float*)clean.data;
            if (closed && n >= 2) {
                float ddx = q[0] - q[(n - 1) * 2];
                float ddy = q[1] - q[(n - 1) * 2 + 1];
                if (ddx * ddx + ddy * ddy < 1e-10f) { n = n - 1; }
            }
            if (n == 0) { continue; }
            if (n == 1) {
                // A dot: drawn by round and square caps.
                if (cap == GUI_CAP_ROUND) { __gui_emit_circle(out, outCont, q[0], q[1], hw, tol); }
                else if (cap == GUI_CAP_SQUARE) {
                    float sq[8];
                    sq[0] = q[0] - hw; sq[1] = q[1] - hw; sq[2] = q[0] + hw; sq[3] = q[1] - hw;
                    sq[4] = q[0] + hw; sq[5] = q[1] + hw; sq[6] = q[0] - hw; sq[7] = q[1] + hw;
                    __gui_emit_poly(out, outCont, &sq[0], 4);
                }
                continue;
            }
            if (n == 2) { closed = 0; }
            int segs = closed ? n : n - 1;
            int s = 0;
            while (s < segs) {
                int a = s;
                int b = s + 1 < n ? s + 1 : 0;
                float ax = q[a * 2]; float ay = q[a * 2 + 1];
                float bx = q[b * 2]; float by = q[b * 2 + 1];
                float dx = bx - ax; float dy = by - ay;
                float len = sqrt_f(dx * dx + dy * dy);
                dx = dx / len; dy = dy / len;
                float nx = -dy * hw; float ny = dx * hw;
                float quad[8];
                quad[0] = ax + nx; quad[1] = ay + ny;
                quad[2] = bx + nx; quad[3] = by + ny;
                quad[4] = bx - nx; quad[5] = by - ny;
                quad[6] = ax - nx; quad[7] = ay - ny;
                __gui_emit_poly(out, outCont, &quad[0], 4);
                // The join at b with the next segment.
                if (closed || s + 1 < segs) {
                    int c2 = b + 1 < n ? b + 1 : 0;
                    float cx = q[c2 * 2]; float cy = q[c2 * 2 + 1];
                    float ex = cx - bx; float ey = cy - by;
                    float el = sqrt_f(ex * ex + ey * ey);
                    if (el > 0.0f) {
                        __gui_emit_join(out, outCont, bx, by, dx, dy, ex / el, ey / el, hw, lineJoin, miterLimit, tol);
                    }
                }
                s = s + 1;
            }
            if (!closed) {
                float sx = q[0]; float sy = q[1];
                float tx = q[2] - sx; float ty = q[3] - sy;
                float tl = sqrt_f(tx * tx + ty * ty);
                __gui_emit_cap(out, outCont, sx, sy, -tx / tl, -ty / tl, hw, cap, tol);
                float ex = q[(n - 1) * 2]; float ey = q[(n - 1) * 2 + 1];
                float ux = ex - q[(n - 2) * 2]; float uy = ey - q[(n - 2) * 2 + 1];
                float ul = sqrt_f(ux * ux + uy * uy);
                __gui_emit_cap(out, outCont, ex, ey, ux / ul, uy / ul, hw, cap, tol);
            }
        }
    }
    clean.free();
}

// Cuts the polylines into dashes: alternating on/off lengths dash[0..n),
// starting 'offset' into the pattern. The dashes are open polylines.
void __gui_canvas_dash(&Vec<float> pts, &Vec<int> cont, const &float dash, int n, float offset,
                       &Vec<float> out, &Vec<int> outCont) {
    out.clear();
    outCont.clear();
    float total = 0.0f;
    unsafe {
        int i = 0;
        while (i < n) { total = total + (dash[i] > 0.0f ? dash[i] : 0.0f); i = i + 1; }
    }
    if (total <= 0.0f) { return; }
    unsafe {
        const float* p = (const float*)pts.data;
        const int* ct = (const int*)cont.data;
        unsigned long nc = cont.len / 3UL;
        unsigned long k = 0UL;
        while (k < nc) {
            int first = ct[k * 3UL];
            int count = ct[k * 3UL + 1UL];
            int closed = ct[k * 3UL + 2UL];
            k = k + 1UL;
            // Position in the pattern at the start of this subpath.
            float phase = offset - total * floor_f(offset / total);
            int di = 0;
            while (phase >= dash[di] && di < n) { phase = phase - dash[di]; di = (di + 1) % n; }
            float left = dash[di] - phase;     // of the current dash/gap
            int on = (di % 2) == 0 ? 1 : 0;
            int drawing = 0;
            int segs = closed ? count : count - 1;
            int s = 0;
            while (s < segs) {
                int a = s;
                int b = s + 1 < count ? s + 1 : 0;
                float ax = p[(first + a) * 2]; float ay = p[(first + a) * 2 + 1];
                float bx = p[(first + b) * 2]; float by = p[(first + b) * 2 + 1];
                float dx = bx - ax; float dy = by - ay;
                float len = sqrt_f(dx * dx + dy * dy);
                float pos = 0.0f;
                if (on && !drawing) {
                    outCont.push((int)(out.len / 2UL)); outCont.push(1); outCont.push(0);
                    out.push(ax); out.push(ay);
                    drawing = 1;
                }
                while (len - pos > left) {
                    pos = pos + left;
                    float t = len > 0.0f ? pos / len : 0.0f;
                    float x = ax + dx * t;
                    float y = ay + dy * t;
                    if (on) {
                        out.push(x); out.push(y);
                        int* oc = (int*)outCont.data;
                        oc[outCont.len - 2UL] = oc[outCont.len - 2UL] + 1;
                        drawing = 0;
                    } else {
                        outCont.push((int)(out.len / 2UL)); outCont.push(1); outCont.push(0);
                        out.push(x); out.push(y);
                        drawing = 1;
                    }
                    on = on ? 0 : 1;
                    di = (di + 1) % n;
                    left = dash[di];
                    if (left <= 0.0f && !on) { left = 0.0f; }
                }
                left = left - (len - pos);
                if (on && drawing) {
                    out.push(bx); out.push(by);
                    int* oc = (int*)outCont.data;
                    oc[outCont.len - 2UL] = oc[outCont.len - 2UL] + 1;
                }
                s = s + 1;
            }
        }
    }
}

} // namespace std
