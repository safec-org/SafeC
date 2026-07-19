// SafeC Standard Library — GUI: minimal SVG rendering implementation (see
// gui_svg.h for exact scope). A direct single-pass text scan: find every
// shape tag anywhere in the document (ignoring nesting — <g> grouping is
// simply not special-cased, which is exactly equivalent to "supported,
// with no transform applied," since transforms aren't implemented anyway)
// and draw it immediately; no retained SVG DOM.
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_svg.h>
#include <std/mem.sc>
#include <std/collections/string.h>
#include <std/collections/string.sc>

namespace std {

static int __svg_is_ws(char c) { return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',') ? 1 : 0; }

static int __svg_streq_n(const char* a, const char* b, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            if (a[i] != b[i]) { return 0; }
            if (a[i] == '\0') { return 1; }
            i = i + 1UL;
        }
        return 1;
    }
}

static int __svg_tag_is(const char* text, unsigned long nameStart, unsigned long nameLen,
                         const char* tagName, unsigned long tagLen) {
    if (nameLen != tagLen) { return 0; }
    unsafe { return __svg_streq_n(text + nameStart, tagName, nameLen); }
}

// Scans forward from 'pos' for `name="value"` within [pos, tagEnd), returns
// the value (empty String if absent).
static struct String __svg_attr(const char* text, unsigned long pos, unsigned long tagEnd, const char* name) {
  unsafe {
    unsigned long nameLen = 0UL;
    while (name[nameLen] != '\0') { nameLen = nameLen + 1UL; }
    unsigned long i = pos;
    while (i + nameLen + 2UL < tagEnd) {
        if (__svg_streq_n(text + i, name, nameLen) && text[i + nameLen] == '=') {
            unsigned long q = i + nameLen + 1UL;
            if (text[q] == '"') {
                unsigned long vstart = q + 1UL;
                unsigned long vend = vstart;
                while (vend < tagEnd && text[vend] != '"') { vend = vend + 1UL; }
                return string_from_n(text + vstart, vend - vstart);
            }
        }
        i = i + 1UL;
    }
    return string_new();
  }
}

static double __svg_attr_num(const char* text, unsigned long pos, unsigned long tagEnd, const char* name, double def) {
    struct String v = __svg_attr(text, pos, tagEnd, name);
    unsigned long len; unsafe { len = v.len; }
    if (len == 0UL) { unsafe { v.free(); } return def; }
    int ok = 0;
    double r;
    unsafe { r = v.parse_float(&ok); }
    unsafe { v.free(); }
    return ok ? r : def;
}

static struct GuiColor __svg_color(struct String v, int* has) {
    unsafe { *has = 1; }
    unsigned long len; unsafe { len = v.len; }
    if (len == 0UL) { unsafe { *has = 0; } return gui_rgba(0U, 0U, 0U, 255U); }
    int eqNone; unsafe { eqNone = v.eq_cstr("none"); }
    if (eqNone) { unsafe { *has = 0; } return gui_rgba(0U, 0U, 0U, 0U); }
    unsafe {
        if (v.data[0] == '#' && v.len >= 7UL) {
            unsigned char comp[3];
            int ci = 0;
            while (ci < 3) {
                char c0 = v.data[1UL + (unsigned long)ci * 2UL];
                char c1 = v.data[2UL + (unsigned long)ci * 2UL];
                // Nested ternaries here previously tripped a real codegen
                // bug (LLVM "PHI node operands are not the same type as
                // the result" — a genuine safec compiler issue, not a
                // logic error) — plain if/else avoids it and is the
                // established workaround pattern elsewhere in this
                // codebase for the same class of bug.
                int h0;
                if (c0 >= '0' && c0 <= '9') { h0 = (int)c0 - (int)'0'; }
                else if (c0 >= 'a' && c0 <= 'f') { h0 = (int)c0 - (int)'a' + 10; }
                else if (c0 >= 'A' && c0 <= 'F') { h0 = (int)c0 - (int)'A' + 10; }
                else { h0 = 0; }
                int h1;
                if (c1 >= '0' && c1 <= '9') { h1 = (int)c1 - (int)'0'; }
                else if (c1 >= 'a' && c1 <= 'f') { h1 = (int)c1 - (int)'a' + 10; }
                else if (c1 >= 'A' && c1 <= 'F') { h1 = (int)c1 - (int)'A' + 10; }
                else { h1 = 0; }
                comp[ci] = (unsigned char)(h0 * 16 + h1);
                ci = ci + 1;
            }
            return gui_rgba(comp[0], comp[1], comp[2], 255U);
        }
    }
    int eqBlack; unsafe { eqBlack = v.eq_cstr("black"); }
    if (eqBlack) { return gui_rgba(0U, 0U, 0U, 255U); }
    int eqWhite; unsafe { eqWhite = v.eq_cstr("white"); }
    if (eqWhite) { return gui_rgba(255U, 255U, 255U, 255U); }
    int eqRed; unsafe { eqRed = v.eq_cstr("red"); }
    if (eqRed) { return gui_rgba(255U, 0U, 0U, 255U); }
    int eqGreen; unsafe { eqGreen = v.eq_cstr("green"); }
    if (eqGreen) { return gui_rgba(0U, 128U, 0U, 255U); }
    int eqBlue; unsafe { eqBlue = v.eq_cstr("blue"); }
    if (eqBlue) { return gui_rgba(0U, 0U, 255U, 255U); }
    int eqYellow; unsafe { eqYellow = v.eq_cstr("yellow"); }
    if (eqYellow) { return gui_rgba(255U, 255U, 0U, 255U); }
    int eqCyan; unsafe { eqCyan = v.eq_cstr("cyan"); }
    if (eqCyan) { return gui_rgba(0U, 255U, 255U, 255U); }
    int eqMagenta; unsafe { eqMagenta = v.eq_cstr("magenta"); }
    if (eqMagenta) { return gui_rgba(255U, 0U, 255U, 255U); }
    int eqGray; unsafe { eqGray = v.eq_cstr("gray"); }
    if (eqGray) { return gui_rgba(128U, 128U, 128U, 255U); }
    int eqTransparent; unsafe { eqTransparent = v.eq_cstr("transparent"); }
    if (eqTransparent) { unsafe { *has = 0; } return gui_rgba(0U, 0U, 0U, 0U); }
    unsafe { *has = 0; }
    return gui_rgba(0U, 0U, 0U, 255U);
}

#define GUI_SVG_MAX_POINTS 512

// Even-odd-rule scanline fill.
static void __svg_fill_polygon(struct GuiWindow* win, int ox, int oy,
                                const double* xs, const double* ys, int n, struct GuiColor color) {
    if (n < 3) { return; }
    unsafe {
        double minY = ys[0]; double maxY = ys[0];
        int i = 1;
        while (i < n) {
            if (ys[i] < minY) { minY = ys[i]; }
            if (ys[i] > maxY) { maxY = ys[i]; }
            i = i + 1;
        }
        int y0 = (int)minY; int y1 = (int)maxY;
        int y = y0;
        while (y <= y1) {
            double xints[GUI_SVG_MAX_POINTS];
            int count = 0;
            int j = 0;
            while (j < n) {
                int k = (j + 1) % n;
                double ya = ys[j]; double yb = ys[k];
                double xa = xs[j]; double xb = xs[k];
                double yc = (double)y + 0.5;
                if ((ya <= yc && yb > yc) || (yb <= yc && ya > yc)) {
                    double t = (yc - ya) / (yb - ya);
                    if (count < GUI_SVG_MAX_POINTS) {
                        xints[count] = xa + t * (xb - xa);
                        count = count + 1;
                    }
                }
                j = j + 1;
            }
            // Insertion sort (count is small for typical shapes).
            int a = 1;
            while (a < count) {
                double key = xints[a];
                int b = a - 1;
                while (b >= 0 && xints[b] > key) { xints[b + 1] = xints[b]; b = b - 1; }
                xints[b + 1] = key;
                a = a + 1;
            }
            int p = 0;
            while (p + 1 < count) {
                int xStart = (int)xints[p];
                int xEnd = (int)xints[p + 1];
                gui_fill_rect(win, ox + xStart, oy + y, xEnd - xStart + 1, 1, color);
                p = p + 2;
            }
            y = y + 1;
        }
    }
}

static void __svg_stroke_polyline(struct GuiWindow* win, int ox, int oy,
                                   const double* xs, const double* ys, int n, int closed,
                                   struct GuiColor color) {
    if (n < 2) { return; }
    unsafe {
        int i = 0;
        while (i < n - 1) {
            gui_draw_line(win, ox + (int)xs[i], oy + (int)ys[i], ox + (int)xs[i + 1], oy + (int)ys[i + 1], color);
            i = i + 1;
        }
        if (closed) {
            gui_draw_line(win, ox + (int)xs[n - 1], oy + (int)ys[n - 1], ox + (int)xs[0], oy + (int)ys[0], color);
        }
    }
}

static void __svg_draw_shape(struct GuiWindow* win, int ox, int oy,
                              const double* xs, const double* ys, int n,
                              struct GuiColor fill, int hasFill, struct GuiColor stroke, int hasStroke) {
    if (hasFill) { __svg_fill_polygon(win, ox, oy, xs, ys, n, fill); }
    if (hasStroke) { __svg_stroke_polyline(win, ox, oy, xs, ys, n, 1, stroke); }
}

// Parses a "points" attribute ("x1,y1 x2,y2 ...") into xs/ys arrays.
// Returns the point count.
static int __svg_parse_points(struct String pts, double* xs, double* ys, int maxPoints) {
    int n = 0;
    unsigned long i = 0UL;
    unsigned long len; unsafe { len = pts.len; }
    unsafe {
        while (i < len && n < maxPoints) {
            while (i < len && __svg_is_ws(pts.data[i])) { i = i + 1UL; }
            if (i >= len) { break; }
            unsigned long start = i;
            while (i < len && pts.data[i] != ',' && !__svg_is_ws(pts.data[i])) { i = i + 1UL; }
            struct String xsStr = string_from_n((const char*)pts.data + start, i - start);
            int ok1; double xv; xv = xsStr.parse_float(&ok1); xsStr.free();
            while (i < len && (pts.data[i] == ',' || __svg_is_ws(pts.data[i]))) { i = i + 1UL; }
            start = i;
            while (i < len && !__svg_is_ws(pts.data[i]) && pts.data[i] != ',') { i = i + 1UL; }
            struct String ysStr = string_from_n((const char*)pts.data + start, i - start);
            int ok2; double yv; yv = ysStr.parse_float(&ok2); ysStr.free();
            if (ok1 && ok2) { xs[n] = xv; ys[n] = yv; n = n + 1; }
        }
    }
    return n;
}

static void __bezier_flatten(double x0, double y0, double x1, double y1, double x2, double y2,
                              double x3, double y3, double* xs, double* ys, int* n, int maxPoints) {
  unsafe {
    int steps = 16;
    int i = 1;
    while (i <= steps && *n < maxPoints) {
        double t = (double)i / (double)steps;
        double mt = 1.0 - t;
        double xx = mt*mt*mt*x0 + 3.0*mt*mt*t*x1 + 3.0*mt*t*t*x2 + t*t*t*x3;
        double yy = mt*mt*mt*y0 + 3.0*mt*mt*t*y1 + 3.0*mt*t*t*y2 + t*t*t*y3;
        xs[*n] = xx; ys[*n] = yy;
        *n = *n + 1;
        i = i + 1;
    }
  }
}

static void __svg_draw_path(struct GuiWindow* win, int ox, int oy, struct String d,
                             struct GuiColor fill, int hasFill, struct GuiColor stroke, int hasStroke) {
    unsigned long len; unsafe { len = d.len; }
    unsigned long i = 0UL;
    double xs[GUI_SVG_MAX_POINTS];
    double ys[GUI_SVG_MAX_POINTS];
    int n = 0;
    double curX = 0.0; double curY = 0.0;
    double startX = 0.0; double startY = 0.0;
    char cmd = '\0';

    // One generic numeric-token scanner drives every command (M/L/H/V/C/Z).
    while (1) {
        unsafe { while (i < len && __svg_is_ws(d.data[i])) { i = i + 1UL; } }
        int atEnd2; unsafe { atEnd2 = (i >= len) ? 1 : 0; }
        if (atEnd2) { break; }
        char cc; unsafe { cc = d.data[i]; }
        if ((cc >= 'A' && cc <= 'Z') || (cc >= 'a' && cc <= 'z')) {
            cmd = cc;
            i = i + 1UL;
            unsafe { while (i < len && __svg_is_ws(d.data[i])) { i = i + 1UL; } }
        }
        if (cmd == 'Z' || cmd == 'z') {
            curX = startX; curY = startY;
            if (n < GUI_SVG_MAX_POINTS) { xs[n] = curX; ys[n] = curY; n = n + 1; }
            continue;
        }

        int numCount = 0;
        if (cmd == 'H' || cmd == 'h') { numCount = 1; }
        else if (cmd == 'V' || cmd == 'v') { numCount = 1; }
        else if (cmd == 'C' || cmd == 'c') { numCount = 6; }
        else { numCount = 2; } // M/L default

        double nums[6];
        int got = 0;
        while (got < numCount) {
            unsafe { while (i < len && __svg_is_ws(d.data[i])) { i = i + 1UL; } }
            unsigned long start2; unsafe { start2 = i; }
            unsafe {
                if (i < len && (d.data[i] == '-' || d.data[i] == '+')) { i = i + 1UL; }
                while (i < len && ((d.data[i] >= '0' && d.data[i] <= '9') || d.data[i] == '.')) { i = i + 1UL; }
            }
            unsigned long numLen; unsafe { numLen = i - start2; }
            if (numLen == 0UL) { break; }
            struct String numStr;
            unsafe { numStr = string_from_n((const char*)d.data + start2, numLen); }
            int ok; double val;
            unsafe { val = numStr.parse_float(&ok); numStr.free(); }
            nums[got] = ok ? val : 0.0;
            got = got + 1;
        }
        if (got < numCount) { break; } // malformed / end of tokens

        if (cmd == 'M') { curX = nums[0]; curY = nums[1]; startX = curX; startY = curY; n = 0; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } cmd = 'L'; }
        else if (cmd == 'm') { curX = curX + nums[0]; curY = curY + nums[1]; startX = curX; startY = curY; n = 0; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } cmd = 'l'; }
        else if (cmd == 'L') { curX = nums[0]; curY = nums[1]; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } }
        else if (cmd == 'l') { curX = curX + nums[0]; curY = curY + nums[1]; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } }
        else if (cmd == 'H') { curX = nums[0]; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } }
        else if (cmd == 'h') { curX = curX + nums[0]; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } }
        else if (cmd == 'V') { curY = nums[0]; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } }
        else if (cmd == 'v') { curY = curY + nums[0]; if (n < GUI_SVG_MAX_POINTS) { xs[n]=curX; ys[n]=curY; n=n+1; } }
        else if (cmd == 'C') {
            __bezier_flatten(curX, curY, nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], xs, ys, &n, GUI_SVG_MAX_POINTS);
            curX = nums[4]; curY = nums[5];
        }
        else if (cmd == 'c') {
            double x1 = curX + nums[0]; double y1 = curY + nums[1];
            double x2 = curX + nums[2]; double y2 = curY + nums[3];
            double x3 = curX + nums[4]; double y3 = curY + nums[5];
            __bezier_flatten(curX, curY, x1, y1, x2, y2, x3, y3, xs, ys, &n, GUI_SVG_MAX_POINTS);
            curX = x3; curY = y3;
        }
    }

    if (n > 0) { __svg_draw_shape(win, ox, oy, xs, ys, n, fill, hasFill, stroke, hasStroke); }
}

static unsigned long __svg_find_tag_end(const char* text, unsigned long len, unsigned long start) {
    unsigned long i = start;
    unsafe {
        int inQuote = 0;
        while (i < len) {
            char c = text[i];
            if (c == '"') { inQuote = !inQuote; }
            if (c == '>' && !inQuote) { return i; }
            i = i + 1UL;
        }
    }
    return len;
}

int gui_draw_svg(struct GuiWindow* win, int ox, int oy, const char* svgText) {
    unsigned long len = 0UL;
    unsafe { while (svgText[len] != '\0') { len = len + 1UL; } }

    int sawSvg = 0;
    unsigned long i = 0UL;
    while (i < len) {
        char c; unsafe { c = svgText[i]; }
        if (c != '<') { i = i + 1UL; continue; }
        unsafe { if (svgText[i + 1UL] == '/') { i = __svg_find_tag_end(svgText, len, i) + 1UL; continue; } }

        unsigned long nameStart = i + 1UL;
        unsigned long nameEnd = nameStart;
        unsafe {
            while (nameEnd < len && ((svgText[nameEnd] >= 'a' && svgText[nameEnd] <= 'z') ||
                                      (svgText[nameEnd] >= 'A' && svgText[nameEnd] <= 'Z'))) {
                nameEnd = nameEnd + 1UL;
            }
        }
        unsigned long tagEnd = __svg_find_tag_end(svgText, len, i);
        unsigned long nameLen = nameEnd - nameStart;

        if (__svg_tag_is(svgText, nameStart, nameLen, "svg", 3UL)) { sawSvg = 1; }
        else if (__svg_tag_is(svgText, nameStart, nameLen, "rect", 4UL)) {
            double x = __svg_attr_num(svgText, nameEnd, tagEnd, "x", 0.0);
            double y = __svg_attr_num(svgText, nameEnd, tagEnd, "y", 0.0);
            double w = __svg_attr_num(svgText, nameEnd, tagEnd, "width", 0.0);
            double h = __svg_attr_num(svgText, nameEnd, tagEnd, "height", 0.0);
            struct String fillStr = __svg_attr(svgText, nameEnd, tagEnd, "fill");
            int hasFill; struct GuiColor fill = __svg_color(fillStr, &hasFill);
            unsafe { fillStr.free(); }
            if (!hasFill && fillStr.len == 0UL) { hasFill = 1; fill = gui_rgba(0U,0U,0U,255U); }
            if (hasFill) { gui_fill_rect(win, ox + (int)x, oy + (int)y, (int)w, (int)h, fill); }
        }
        else if (__svg_tag_is(svgText, nameStart, nameLen, "circle", 6UL)) {
            double cx = __svg_attr_num(svgText, nameEnd, tagEnd, "cx", 0.0);
            double cy = __svg_attr_num(svgText, nameEnd, tagEnd, "cy", 0.0);
            double r = __svg_attr_num(svgText, nameEnd, tagEnd, "r", 0.0);
            struct String fillStr = __svg_attr(svgText, nameEnd, tagEnd, "fill");
            int hasFill; struct GuiColor fill = __svg_color(fillStr, &hasFill);
            unsafe { hasFill = (fillStr.len == 0UL) ? 1 : hasFill; fillStr.free(); }
            if (hasFill) {
                int yy = -(int)r;
                while (yy <= (int)r) {
                    int xx = -(int)r;
                    while (xx <= (int)r) {
                        if ((double)(xx*xx + yy*yy) <= r*r) {
                            gui_set_pixel(win, ox + (int)cx + xx, oy + (int)cy + yy,
                                          ((unsigned int)fill.r << 24) | ((unsigned int)fill.g << 16) |
                                          ((unsigned int)fill.b << 8) | (unsigned int)fill.a);
                        }
                        xx = xx + 1;
                    }
                    yy = yy + 1;
                }
            }
        }
        else if (__svg_tag_is(svgText, nameStart, nameLen, "line", 4UL)) {
            double x1 = __svg_attr_num(svgText, nameEnd, tagEnd, "x1", 0.0);
            double y1 = __svg_attr_num(svgText, nameEnd, tagEnd, "y1", 0.0);
            double x2 = __svg_attr_num(svgText, nameEnd, tagEnd, "x2", 0.0);
            double y2 = __svg_attr_num(svgText, nameEnd, tagEnd, "y2", 0.0);
            struct String strokeStr = __svg_attr(svgText, nameEnd, tagEnd, "stroke");
            int hasStroke; struct GuiColor stroke = __svg_color(strokeStr, &hasStroke);
            unsafe { hasStroke = (strokeStr.len == 0UL) ? 1 : hasStroke; strokeStr.free(); }
            if (hasStroke) { gui_draw_line(win, ox + (int)x1, oy + (int)y1, ox + (int)x2, oy + (int)y2, stroke); }
        }
        else if ((__svg_tag_is(svgText, nameStart, nameLen, "polygon", 7UL)) ||
                 (__svg_tag_is(svgText, nameStart, nameLen, "polyline", 8UL))) {
            int isPolygon = (nameLen == 7UL) ? 1 : 0;
            struct String ptsStr = __svg_attr(svgText, nameEnd, tagEnd, "points");
            double xs[GUI_SVG_MAX_POINTS]; double ys[GUI_SVG_MAX_POINTS];
            int n = __svg_parse_points(ptsStr, xs, ys, GUI_SVG_MAX_POINTS);
            unsafe { ptsStr.free(); }
            struct String fillStr = __svg_attr(svgText, nameEnd, tagEnd, "fill");
            int hasFill; struct GuiColor fill = __svg_color(fillStr, &hasFill);
            unsafe { if (isPolygon && fillStr.len == 0UL) { hasFill = 1; fill = gui_rgba(0U,0U,0U,255U); } fillStr.free(); }
            struct String strokeStr = __svg_attr(svgText, nameEnd, tagEnd, "stroke");
            int hasStroke; struct GuiColor stroke = __svg_color(strokeStr, &hasStroke);
            unsafe { strokeStr.free(); }
            if (isPolygon) { __svg_draw_shape(win, ox, oy, xs, ys, n, fill, hasFill, stroke, hasStroke); }
            else { __svg_stroke_polyline(win, ox, oy, xs, ys, n, 0, hasStroke ? stroke : gui_rgba(0U,0U,0U,255U)); }
        }
        else if (__svg_tag_is(svgText, nameStart, nameLen, "path", 4UL)) {
            struct String dStr = __svg_attr(svgText, nameEnd, tagEnd, "d");
            struct String fillStr = __svg_attr(svgText, nameEnd, tagEnd, "fill");
            int hasFill; struct GuiColor fill = __svg_color(fillStr, &hasFill);
            unsafe { if (fillStr.len == 0UL) { hasFill = 1; fill = gui_rgba(0U,0U,0U,255U); } fillStr.free(); }
            struct String strokeStr = __svg_attr(svgText, nameEnd, tagEnd, "stroke");
            int hasStroke; struct GuiColor stroke = __svg_color(strokeStr, &hasStroke);
            unsafe { strokeStr.free(); }
            __svg_draw_path(win, ox, oy, dStr, fill, hasFill, stroke, hasStroke);
            unsafe { dStr.free(); }
        }

        i = tagEnd + 1UL;
    }
    return sawSvg;
}

} // namespace std
