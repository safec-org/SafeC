// SafeC Standard Library — GUI widgets: containers (see gui_widget.h) —
// vertical/horizontal boxes, grids, stacks, panels, group boxes, status
// bars, dialogs, scroll views, splitters and tabs — and the scrollbars the
// scrolling widgets share.
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_widget_internal.h>
#include <std/math.h>
#include <std/collections/vec.h>

namespace std {

#define GUI_DRAG_VBAR     100
#define GUI_DRAG_HBAR     101
#define GUI_DRAG_SPLIT    102
#define GUI_DRAG_DIALOG   103

static int __wl_vertical(const &Widget w) {
    int k = w.kind;
    if (k == WIDGET_VSTACK || k == WIDGET_GROUP || k == WIDGET_PANEL || k == WIDGET_DIALOG) { return 1; }
    return 0;
}

static int __wl_is_box(const &Widget w) {
    int k = w.kind;
    return (k == WIDGET_VSTACK || k == WIDGET_HSTACK || k == WIDGET_GROUP || k == WIDGET_PANEL ||
            k == WIDGET_DIALOG || k == WIDGET_STATUSBAR) ? 1 : 0;
}

static int __wl_title_h(&GuiUi ui, const &Widget w) {
    if (w.kind == WIDGET_DIALOG) { return __w_ipx(ui, ui.theme.rowHeight + 10); }
    if (w.kind == WIDGET_GROUP) { return (int)ceil_f(__w_line_h(ui, w)); }
    if (w.kind == WIDGET_TABS) { return __w_ipx(ui, ui.theme.rowHeight + 6); }
    return 0;
}

static int __wl_margin_w(&GuiUi ui, const &Widget c) { return __w_ipx(ui, c.marginL) + __w_ipx(ui, c.marginR); }
static int __wl_margin_h(&GuiUi ui, const &Widget c) { return __w_ipx(ui, c.marginT) + __w_ipx(ui, c.marginB); }

static int __wl_max_px(&GuiUi ui, int logical) { return logical > 0 ? __w_ipx(ui, logical) : 1000000000; }
static int __wl_min_px(&GuiUi ui, int logical) { return logical > 0 ? __w_ipx(ui, logical) : 0; }

// ── boxes ─────────────────────────────────────────────────────────────────

static void __wl_box_measure(&GuiUi ui, &Widget w, int vertical, &int cw, &int ch) {
    int main = 0;
    int cross = 0;
    int n = 0;
    unsigned long i = 0UL;
    while (i < w.children.len) {
        &Widget c = *w.children.at(i);
        i = i + 1UL;
        if (!c.visible) { continue; }
        int mw = 0; int mh = 0;
        __w_measure(ui, c, &mw, &mh);
        mw = mw + __wl_margin_w(ui, c);
        mh = mh + __wl_margin_h(ui, c);
        if (vertical) { main = main + mh; if (mw > cross) { cross = mw; } }
        else { main = main + mw; if (mh > cross) { cross = mh; } }
        n = n + 1;
    }
    if (n > 1) { main = main + __w_spacing(ui, w) * (n - 1); }
    if (vertical) { *cw = cross; *ch = main; } else { *cw = main; *ch = cross; }
}

static void __wl_box_arrange(&GuiUi ui, &Widget w, int vertical, int cx, int cy, int cw, int ch) {
    unsigned long nc = w.children.len;
    if (nc == 0UL) { return; }
    int spacing = __w_spacing(ui, w);
    struct Vec<int> mains = Vec::new();
    struct Vec<int> crosses = Vec::new();
    int total = 0;
    int weights = 0;
    int n = 0;
    unsigned long i = 0UL;
    while (i < nc) {
        &Widget c = *w.children.at(i);
        int mw = 0; int mh = 0;
        int crossSize = 0;
        int mainSize = 0;
        if (c.visible) {
            __w_measure(ui, c, &mw, &mh);
            int availCross = vertical ? cw - __wl_margin_w(ui, c) : ch - __wl_margin_h(ui, c);
            if (availCross < 0) { availCross = 0; }
            if (c.alignment == WIDGET_ALIGN_STRETCH) {
                crossSize = availCross;
                int mx = vertical ? __wl_max_px(ui, c.maxW) : __wl_max_px(ui, c.maxH);
                if (vertical && c.prefW > 0) { mx = __w_ipx(ui, c.prefW); }
                if (!vertical && c.prefH > 0) { mx = __w_ipx(ui, c.prefH); }
                if (crossSize > mx) { crossSize = mx; }
            } else {
                crossSize = vertical ? mw : mh;
                if (crossSize > availCross) { crossSize = availCross; }
            }
            mainSize = vertical ? __w_height_for(ui, c, crossSize) : mw;
            total = total + mainSize + (vertical ? __wl_margin_h(ui, c) : __wl_margin_w(ui, c));
            if (c.stretch > 0) { weights = weights + c.stretch; }
            n = n + 1;
        }
        mains.push(mainSize);
        crosses.push(crossSize);
        i = i + 1UL;
    }
    if (n > 1) { total = total + spacing * (n - 1); }
    int avail = vertical ? ch : cw;
    int slack = avail - total;
    unsafe {
        int* m = (int*)mains.data;
        if (slack > 0 && weights > 0) {
            int given = 0;
            int lastStretch = -1;
            i = 0UL;
            while (i < nc) {
                &Widget c = *w.children.at(i);
                if (c.visible && c.stretch > 0) {
                    int extra = slack * c.stretch / weights;
                    int mx = vertical ? __wl_max_px(ui, c.maxH) : __wl_max_px(ui, c.maxW);
                    if (m[i] + extra > mx) { extra = mx - m[i]; if (extra < 0) { extra = 0; } }
                    m[i] = m[i] + extra;
                    given = given + extra;
                    lastStretch = (int)i;
                }
                i = i + 1UL;
            }
            // rounding leftovers go to the last stretching child
            if (lastStretch >= 0 && given < slack) {
                &Widget c = *w.children.at((unsigned long)lastStretch);
                int mx = vertical ? __wl_max_px(ui, c.maxH) : __wl_max_px(ui, c.maxW);
                int add = slack - given;
                if (m[lastStretch] + add > mx) { add = mx - m[lastStretch]; }
                if (add > 0) { m[lastStretch] = m[lastStretch] + add; }
            }
        } else if (slack < 0) {
            // Too little room: stretching children give up space first
            // (down to their minimum), then everyone else.
            int pass = 0;
            while (pass < 2 && slack < 0) {
                int room = 0;
                i = 0UL;
                while (i < nc) {
                    &Widget c = *w.children.at(i);
                    if (c.visible && (pass == 1 || c.stretch > 0)) {
                        int mn = vertical ? __wl_min_px(ui, c.minH) : __wl_min_px(ui, c.minW);
                        if (m[i] > mn) { room = room + (m[i] - mn); }
                    }
                    i = i + 1UL;
                }
                if (room > 0) {
                    int need = -slack;
                    if (need > room) { need = room; }
                    int taken = 0;
                    i = 0UL;
                    while (i < nc) {
                        &Widget c = *w.children.at(i);
                        if (c.visible && (pass == 1 || c.stretch > 0)) {
                            int mn = vertical ? __wl_min_px(ui, c.minH) : __wl_min_px(ui, c.minW);
                            if (m[i] > mn) {
                                int give = (int)((long)need * (long)(m[i] - mn) / (long)room);
                                m[i] = m[i] - give;
                                taken = taken + give;
                            }
                        }
                        i = i + 1UL;
                    }
                    slack = slack + taken;
                }
                pass = pass + 1;
            }
        }
        int pos = vertical ? cy : cx;
        int first = 1;
        i = 0UL;
        while (i < nc) {
            &Widget c = *w.children.at(i);
            if (!c.visible) { __w_arrange(ui, c, 0, 0, 0, 0); i = i + 1UL; continue; }
            if (!first) { pos = pos + spacing; }
            first = 0;
            int cr = ((int*)crosses.data)[i];
            if (vertical) {
                int availCross = cw - __wl_margin_w(ui, c);
                int x = cx + __w_ipx(ui, c.marginL);
                if (c.alignment == WIDGET_ALIGN_CENTER) { x = x + (availCross - cr) / 2; }
                else if (c.alignment == WIDGET_ALIGN_END) { x = x + availCross - cr; }
                pos = pos + __w_ipx(ui, c.marginT);
                __w_arrange(ui, c, x, pos, cr, m[i]);
                pos = pos + m[i] + __w_ipx(ui, c.marginB);
            } else {
                int availCross = ch - __wl_margin_h(ui, c);
                int y = cy + __w_ipx(ui, c.marginT);
                if (c.alignment == WIDGET_ALIGN_CENTER) { y = y + (availCross - cr) / 2; }
                else if (c.alignment == WIDGET_ALIGN_END) { y = y + availCross - cr; }
                pos = pos + __w_ipx(ui, c.marginL);
                __w_arrange(ui, c, pos, y, m[i], cr);
                pos = pos + m[i] + __w_ipx(ui, c.marginR);
            }
            i = i + 1UL;
        }
    }
    mains.free();
    crosses.free();
}

// ── grids ─────────────────────────────────────────────────────────────────

// Cell positions: explicit (gridRow/gridCol) or flowing row by row into
// the next free cell. rc[2i], rc[2i+1] = row, col of child i.
static int __wl_grid_place(&Widget w, &Vec<int> rc) {
    int cols = w.columns < 1 ? 1 : w.columns;
    struct Vec<int> occupied = Vec::new();   // rows*cols flags
    int rows = 0;
    int cursor = 0;
    unsigned long i = 0UL;
    while (i < w.children.len) {
        &Widget c = *w.children.at(i);
        int r = c.gridRow;
        int col = c.gridCol;
        int rs = c.gridRowSpan < 1 ? 1 : c.gridRowSpan;
        int cs = c.gridColSpan < 1 ? 1 : c.gridColSpan;
        if (cs > cols) { cs = cols; }
        if (!c.visible) { rc.push(-1); rc.push(-1); i = i + 1UL; continue; }
        if (r < 0 || col < 0) {
            // next free run of cs cells
            int found = 0;
            while (!found) {
                int rr = cursor / cols;
                int cc = cursor % cols;
                if (cc + cs <= cols) {
                    int ok = 1;
                    int a = 0;
                    while (a < rs && ok) {
                        int b = 0;
                        while (b < cs && ok) {
                            unsigned long idx = (unsigned long)((rr + a) * cols + cc + b);
                            if (idx < occupied.len && *occupied.at(idx) != 0) { ok = 0; }
                            b = b + 1;
                        }
                        a = a + 1;
                    }
                    if (ok) { r = rr; col = cc; found = 1; }
                }
                if (!found) { cursor = cursor + 1; }
            }
        }
        if (col + cs > cols) { cs = cols - col; if (cs < 1) { cs = 1; col = cols - 1; } }
        int a = 0;
        while (a < rs) {
            int b = 0;
            while (b < cs) {
                unsigned long idx = (unsigned long)((r + a) * cols + col + b);
                while (occupied.len <= idx) { occupied.push(0); }
                occupied.set(idx, 1);
                b = b + 1;
            }
            a = a + 1;
        }
        if (r + rs > rows) { rows = r + rs; }
        cursor = r * cols + col + cs;
        rc.push(r);
        rc.push(col);
        i = i + 1UL;
    }
    occupied.free();
    return rows;
}

static int __wl_weight(const &Vec<int> v, int i) {
    if (i < 0 || (unsigned long)i >= v.len) { return 0; }
    return *v.at((unsigned long)i);
}

// Column widths / row heights for the grid; returns the row count.
static int __wl_grid_sizes(&GuiUi ui, &Widget w, &Vec<int> rc, &Vec<int> colW, &Vec<int> rowH) {
    int cols = w.columns < 1 ? 1 : w.columns;
    int rows = __wl_grid_place(w, rc);
    int sp = __w_spacing(ui, w);
    int k = 0;
    while (k < cols) { colW.push(0); k = k + 1; }
    k = 0;
    while (k < rows) { rowH.push(0); k = k + 1; }
    // single-cell children first, then spans take what's missing
    int pass = 0;
    while (pass < 2) {
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget c = *w.children.at(i);
            int r = *rc.at(i * 2UL);
            int col = *rc.at(i * 2UL + 1UL);
            i = i + 1UL;
            if (r < 0) { continue; }
            int rs = c.gridRowSpan < 1 ? 1 : c.gridRowSpan;
            int cs = c.gridColSpan < 1 ? 1 : c.gridColSpan;
            if (col + cs > cols) { cs = cols - col; }
            int spans = (rs > 1 || cs > 1) ? 1 : 0;
            if (spans != pass) { continue; }
            int mw = 0; int mh = 0;
            __w_measure(ui, c, &mw, &mh);
            mw = mw + __wl_margin_w(ui, c);
            mh = mh + __wl_margin_h(ui, c);
            int have = sp * (cs - 1);
            int b = 0;
            while (b < cs) { have = have + *colW.at((unsigned long)(col + b)); b = b + 1; }
            if (mw > have) {
                int add = (mw - have + cs - 1) / cs;
                b = 0;
                while (b < cs) { colW.set((unsigned long)(col + b), *colW.at((unsigned long)(col + b)) + add); b = b + 1; }
            }
            have = sp * (rs - 1);
            int a = 0;
            while (a < rs) { have = have + *rowH.at((unsigned long)(r + a)); a = a + 1; }
            if (mh > have) {
                int add = (mh - have + rs - 1) / rs;
                a = 0;
                while (a < rs) { rowH.set((unsigned long)(r + a), *rowH.at((unsigned long)(r + a)) + add); a = a + 1; }
            }
        }
        pass = pass + 1;
    }
    return rows;
}

static void __wl_grid_measure(&GuiUi ui, &Widget w, &int cw, &int ch) {
    struct Vec<int> rc = Vec::new();
    struct Vec<int> colW = Vec::new();
    struct Vec<int> rowH = Vec::new();
    int rows = __wl_grid_sizes(ui, w, &rc, &colW, &rowH);
    int sp = __w_spacing(ui, w);
    int tw = 0;
    unsigned long i = 0UL;
    while (i < colW.len) { tw = tw + *colW.at(i); i = i + 1UL; }
    if (colW.len > 1UL) { tw = tw + sp * ((int)colW.len - 1); }
    int th = 0;
    i = 0UL;
    while (i < rowH.len) { th = th + *rowH.at(i); i = i + 1UL; }
    if (rows > 1) { th = th + sp * (rows - 1); }
    *cw = tw;
    *ch = th;
    rc.free(); colW.free(); rowH.free();
}

static void __wl_distribute(&Vec<int> sizes, const &Vec<int> weights, int slack) {
    if (slack <= 0) { return; }
    int total = 0;
    unsigned long i = 0UL;
    while (i < sizes.len) { total = total + __wl_weight(weights, (int)i); i = i + 1UL; }
    if (total <= 0) { return; }
    int given = 0;
    int last = -1;
    i = 0UL;
    while (i < sizes.len) {
        int wt = __wl_weight(weights, (int)i);
        if (wt > 0) {
            int add = slack * wt / total;
            sizes.set(i, *sizes.at(i) + add);
            given = given + add;
            last = (int)i;
        }
        i = i + 1UL;
    }
    if (last >= 0 && given < slack) { sizes.set((unsigned long)last, *sizes.at((unsigned long)last) + slack - given); }
}

static void __wl_place(&GuiUi ui, &Widget c, int x, int y, int cw, int ch, int defaultV) {
    // A child in a cell (x, y, cw, ch): aligned on both axes.
    x = x + __w_ipx(ui, c.marginL);
    y = y + __w_ipx(ui, c.marginT);
    cw = cw - __wl_margin_w(ui, c);
    ch = ch - __wl_margin_h(ui, c);
    if (cw < 0) { cw = 0; }
    if (ch < 0) { ch = 0; }
    int mw = 0; int mh = 0;
    __w_measure(ui, c, &mw, &mh);
    int ha = c.alignment;
    int va = c.valign >= 0 ? c.valign : defaultV;
    int ww = ha == WIDGET_ALIGN_STRETCH ? cw : (mw < cw ? mw : cw);
    if (ha == WIDGET_ALIGN_STRETCH && ww > __wl_max_px(ui, c.maxW)) { ww = __wl_max_px(ui, c.maxW); }
    if (va != WIDGET_ALIGN_STRETCH) { mh = __w_height_for(ui, c, ww); }
    int hh = va == WIDGET_ALIGN_STRETCH ? ch : (mh < ch ? mh : ch);
    if (va == WIDGET_ALIGN_STRETCH && hh > __wl_max_px(ui, c.maxH)) { hh = __wl_max_px(ui, c.maxH); }
    int px = x;
    int py = y;
    if (ha == WIDGET_ALIGN_CENTER) { px = x + (cw - ww) / 2; }
    else if (ha == WIDGET_ALIGN_END) { px = x + cw - ww; }
    if (va == WIDGET_ALIGN_CENTER) { py = y + (ch - hh) / 2; }
    else if (va == WIDGET_ALIGN_END) { py = y + ch - hh; }
    __w_arrange(ui, c, px, py, ww, hh);
}

static void __wl_grid_arrange(&GuiUi ui, &Widget w, int cx, int cy, int cw, int ch) {
    struct Vec<int> rc = Vec::new();
    struct Vec<int> colW = Vec::new();
    struct Vec<int> rowH = Vec::new();
    int rows = __wl_grid_sizes(ui, w, &rc, &colW, &rowH);
    int sp = __w_spacing(ui, w);
    int tw = 0;
    unsigned long i = 0UL;
    while (i < colW.len) { tw = tw + *colW.at(i); i = i + 1UL; }
    if (colW.len > 1UL) { tw = tw + sp * ((int)colW.len - 1); }
    int th = 0;
    i = 0UL;
    while (i < rowH.len) { th = th + *rowH.at(i); i = i + 1UL; }
    if (rows > 1) { th = th + sp * (rows - 1); }
    __wl_distribute(&colW, &w.colStretch, cw - tw);
    __wl_distribute(&rowH, &w.rowStretch, ch - th);
    // offsets
    struct Vec<int> colX = Vec::new();
    struct Vec<int> rowY = Vec::new();
    int p = cx;
    i = 0UL;
    while (i < colW.len) { colX.push(p); p = p + *colW.at(i) + sp; i = i + 1UL; }
    p = cy;
    i = 0UL;
    while (i < rowH.len) { rowY.push(p); p = p + *rowH.at(i) + sp; i = i + 1UL; }
    int cols = (int)colW.len;
    i = 0UL;
    while (i < w.children.len) {
        &Widget c = *w.children.at(i);
        int r = *rc.at(i * 2UL);
        int col = *rc.at(i * 2UL + 1UL);
        i = i + 1UL;
        if (r < 0) { __w_arrange(ui, c, 0, 0, 0, 0); continue; }
        int rs = c.gridRowSpan < 1 ? 1 : c.gridRowSpan;
        int cs = c.gridColSpan < 1 ? 1 : c.gridColSpan;
        if (col + cs > cols) { cs = cols - col; }
        int x = *colX.at((unsigned long)col);
        int y = *rowY.at((unsigned long)r);
        int ww = sp * (cs - 1);
        int b = 0;
        while (b < cs) { ww = ww + *colW.at((unsigned long)(col + b)); b = b + 1; }
        int hh = sp * (rs - 1);
        int a = 0;
        while (a < rs) { hh = hh + *rowH.at((unsigned long)(r + a)); a = a + 1; }
        __wl_place(ui, c, x, y, ww, hh, WIDGET_ALIGN_CENTER);
    }
    rc.free(); colW.free(); rowH.free(); colX.free(); rowY.free();
}

// ── scroll views ──────────────────────────────────────────────────────────

static ?&Widget __wl_first_visible(&Widget w) {
    unsigned long i = 0UL;
    while (i < w.children.len) {
        &Widget c = *w.children.at(i);
        if (c.visible) { return c; }
        i = i + 1UL;
    }
    return null;
}

// The viewport of a scroll view (its content box minus the scrollbars it
// needs for its current content size).
static void __wl_scroll_view(&GuiUi ui, const &Widget w, &int vx, &int vy, &int vw, &int vh) {
    __w_content(ui, w, vx, vy, vw, vh);
    int sb = __w_ipx(ui, ui.theme.scrollbarSize);
    int needV = w.contentH > (float)*vh ? 1 : 0;
    int needH = w.contentW > (float)(*vw - (needV ? sb : 0)) ? 1 : 0;
    if (needH && !needV && w.contentH > (float)(*vh - sb)) { needV = 1; }
    if (needV) { *vw = *vw - sb; }
    if (needH) { *vh = *vh - sb; }
    if (*vw < 0) { *vw = 0; }
    if (*vh < 0) { *vh = 0; }
}

static void __wl_scroll_arrange(&GuiUi ui, &Widget w) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    ?&Widget oc = __wl_first_visible(w);
    match (oc) {
        case null: { w.contentW = 0.0f; w.contentH = 0.0f; return; }
        case some(c): {
            int sb = __w_ipx(ui, ui.theme.scrollbarSize);
            int mw = 0; int mh = 0;
            __w_measure(ui, c, &mw, &mh);
            mw = mw + __wl_margin_w(ui, c);
            // Width first (assume a vertical bar if the content is taller).
            int vw = cw;
            int childW = c.alignment == WIDGET_ALIGN_STRETCH ? (mw > vw ? mw : vw) : mw;
            int childH = __w_height_for(ui, c, childW - __wl_margin_w(ui, c)) + __wl_margin_h(ui, c);
            if (childH > ch) {
                vw = cw - sb;
                childW = c.alignment == WIDGET_ALIGN_STRETCH ? (mw > vw ? mw : vw) : mw;
                childH = __w_height_for(ui, c, childW - __wl_margin_w(ui, c)) + __wl_margin_h(ui, c);
            }
            int vh = ch - (childW > vw ? sb : 0);
            int vstretch = c.valign == WIDGET_ALIGN_STRETCH || (c.valign < 0 && c.alignment == WIDGET_ALIGN_STRETCH);
            if (vstretch && childH < vh) { childH = vh; }
            w.contentW = (float)childW;
            w.contentH = (float)childH;
            int vx = 0; int vy = 0; int vw2 = 0; int vh2 = 0;
            __wl_scroll_view(ui, w, &vx, &vy, &vw2, &vh2);
            __w_scroll_clamp(w, (float)vw2, (float)vh2);
            __w_arrange(ui, c, cx + __w_ipx(ui, c.marginL) - (int)w.scrollX, cy + __w_ipx(ui, c.marginT) - (int)w.scrollY,
                        childW - __wl_margin_w(ui, c), childH - __wl_margin_h(ui, c));
            // the other (hidden) children don't take part
            unsigned long i = 0UL;
            while (i < w.children.len) {
                &Widget o = *w.children.at(i);
                if (!o.visible) { __w_arrange(ui, o, 0, 0, 0, 0); }
                i = i + 1UL;
            }
        }
    }
}

void __w_scroll_clamp(&Widget w, float vw, float vh) {
    float mx = w.contentW - vw;
    float my = w.contentH - vh;
    if (mx < 0.0f) { mx = 0.0f; }
    if (my < 0.0f) { my = 0.0f; }
    if (w.scrollX > mx) { w.scrollX = mx; }
    if (w.scrollY > my) { w.scrollY = my; }
    if (w.scrollX < 0.0f) { w.scrollX = 0.0f; }
    if (w.scrollY < 0.0f) { w.scrollY = 0.0f; }
}

void __w_scroll_into_view(&Widget w, float top, float bottom, float vh) {
    if (top < w.scrollY) { w.scrollY = top; }
    else if (bottom > w.scrollY + vh) { w.scrollY = bottom - vh; }
}

// Thumb geometry along one axis: track start/length, content and view
// sizes, offset -> (thumb start, thumb length).
static void __wl_thumb(float t0, float tlen, float content, float view, float off, &float ts, &float tl) {
    float len = content > 0.0f ? tlen * view / content : tlen;
    if (len < 18.0f) { len = 18.0f; }
    if (len > tlen) { len = tlen; }
    float range = content - view;
    float pos = range > 0.0f ? (tlen - len) * off / range : 0.0f;
    *ts = t0 + pos;
    *tl = len;
}

void __w_scrollbars_paint(const &GuiUi ui, &GuiCanvas c, &Widget w, float x, float y, float vw, float vh) {
    float sb = __w_px(ui, (float)ui.theme.scrollbarSize);
    int needV = w.contentH > vh + 0.5f ? 1 : 0;
    int needH = w.contentW > vw + 0.5f ? 1 : 0;
    float inset = __w_px(ui, 2.0f);
    if (needV) {
        float ts = 0.0f; float tl = 0.0f;
        __wl_thumb(y + inset, vh - 2.0f * inset, w.contentH, vh, w.scrollY, &ts, &tl);
        int hot = (w.dragPart == GUI_DRAG_VBAR) || (w.hover && ui.mouseX >= x + vw && ui.mouseX < x + vw + sb) ? 1 : 0;
        c.set_fill_color(hot ? ui.theme.scrollbarHover : ui.theme.scrollbar);
        float bw = sb - 2.0f * inset;
        c.fill_rounded_rect(x + vw + inset, ts, bw, tl, bw * 0.5f);
    }
    if (needH) {
        float ts = 0.0f; float tl = 0.0f;
        __wl_thumb(x + inset, vw - 2.0f * inset, w.contentW, vw, w.scrollX, &ts, &tl);
        int hot = (w.dragPart == GUI_DRAG_HBAR) || (w.hover && ui.mouseY >= y + vh && ui.mouseY < y + vh + sb) ? 1 : 0;
        c.set_fill_color(hot ? ui.theme.scrollbarHover : ui.theme.scrollbar);
        float bh = sb - 2.0f * inset;
        c.fill_rounded_rect(ts, y + vh + inset, tl, bh, bh * 0.5f);
    }
}

int __w_scrollbars_event(&GuiUi ui, &Widget w, &GuiEvent ev, float x, float y, float vw, float vh) {
    float sb = __w_px(ui, (float)ui.theme.scrollbarSize);
    float inset = __w_px(ui, 2.0f);
    int needV = w.contentH > vh + 0.5f ? 1 : 0;
    int needH = w.contentW > vw + 0.5f ? 1 : 0;
    float ex = (float)ev.x;
    float ey = (float)ev.y;
    if (ev.kind == GUI_EVENT_SCROLL) {
        float ox = w.scrollX;
        float oy = w.scrollY;
        float step = __w_px(ui, (float)ui.theme.rowHeight) * 2.0f;
        float dy = (float)ev.scrollDy;
        float dx = (float)ev.scrollDx;
        if (ev.has_mod(GUI_MOD_SHIFT) && dx == 0.0f) { dx = -dy; dy = 0.0f; }
        if (needV) { w.scrollY = w.scrollY - dy * step; }
        if (needH) { w.scrollX = w.scrollX + dx * step; }
        __w_scroll_clamp(w, vw, vh);
        return (ox != w.scrollX || oy != w.scrollY) ? 1 : 0;
    }
    if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT) {
        if (needV && ex >= x + vw && ex < x + vw + sb && ey >= y && ey < y + vh) {
            float ts = 0.0f; float tl = 0.0f;
            __wl_thumb(y + inset, vh - 2.0f * inset, w.contentH, vh, w.scrollY, &ts, &tl);
            if (ey < ts || ey >= ts + tl) {
                // page toward the click
                w.scrollY = w.scrollY + (ey < ts ? -vh : vh) * 0.9f;
                __w_scroll_clamp(w, vw, vh);
                __wl_thumb(y + inset, vh - 2.0f * inset, w.contentH, vh, w.scrollY, &ts, &tl);
            }
            w.dragPart = GUI_DRAG_VBAR;
            w.dragY = ey - ts;
            return 1;
        }
        if (needH && ey >= y + vh && ey < y + vh + sb && ex >= x && ex < x + vw) {
            float ts = 0.0f; float tl = 0.0f;
            __wl_thumb(x + inset, vw - 2.0f * inset, w.contentW, vw, w.scrollX, &ts, &tl);
            if (ex < ts || ex >= ts + tl) {
                w.scrollX = w.scrollX + (ex < ts ? -vw : vw) * 0.9f;
                __w_scroll_clamp(w, vw, vh);
                __wl_thumb(x + inset, vw - 2.0f * inset, w.contentW, vw, w.scrollX, &ts, &tl);
            }
            w.dragPart = GUI_DRAG_HBAR;
            w.dragX = ex - ts;
            return 1;
        }
        return 0;
    }
    if (ev.kind == GUI_EVENT_MOUSE_MOVE && (w.dragPart == GUI_DRAG_VBAR || w.dragPart == GUI_DRAG_HBAR)) {
        if (w.dragPart == GUI_DRAG_VBAR) {
            float ts = 0.0f; float tl = 0.0f;
            __wl_thumb(y + inset, vh - 2.0f * inset, w.contentH, vh, 0.0f, &ts, &tl);
            float track = vh - 2.0f * inset - tl;
            float pos = ey - w.dragY - (y + inset);
            w.scrollY = track > 0.0f ? pos / track * (w.contentH - vh) : 0.0f;
        } else {
            float ts = 0.0f; float tl = 0.0f;
            __wl_thumb(x + inset, vw - 2.0f * inset, w.contentW, vw, 0.0f, &ts, &tl);
            float track = vw - 2.0f * inset - tl;
            float pos = ex - w.dragX - (x + inset);
            w.scrollX = track > 0.0f ? pos / track * (w.contentW - vw) : 0.0f;
        }
        __w_scroll_clamp(w, vw, vh);
        return 1;
    }
    if (ev.kind == GUI_EVENT_MOUSE_UP && (w.dragPart == GUI_DRAG_VBAR || w.dragPart == GUI_DRAG_HBAR)) {
        w.dragPart = 0;
        return 1;
    }
    return 0;
}

// ── splitters ─────────────────────────────────────────────────────────────

static int __wl_divider(&GuiUi ui) { return __w_ipx(ui, 6); }

static void __wl_split_parts(&GuiUi ui, &Widget w, &int firstSize, &int avail) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    int total = (w.orientation == WIDGET_HORIZONTAL ? cw : ch) - __wl_divider(ui);
    if (total < 0) { total = 0; }
    double v = w.sliderValue;
    if (v < 0.0) { v = 0.0; }
    if (v > 1.0) { v = 1.0; }
    int a = (int)((double)total * v + 0.5);
    // children's minimum sizes win over the ratio
    if (w.children.len >= 2UL) {
        &Widget c0 = *w.children.at(0UL);
        &Widget c1 = *w.children.at(1UL);
        int min0 = w.orientation == WIDGET_HORIZONTAL ? __wl_min_px(ui, c0.minW) : __wl_min_px(ui, c0.minH);
        int min1 = w.orientation == WIDGET_HORIZONTAL ? __wl_min_px(ui, c1.minW) : __wl_min_px(ui, c1.minH);
        if (a > total - min1) { a = total - min1; }
        if (a < min0) { a = min0; }
    }
    if (a < 0) { a = 0; }
    if (a > total) { a = total; }
    *firstSize = a;
    *avail = total;
}

static void __wl_split_arrange(&GuiUi ui, &Widget w) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    int a = 0; int total = 0;
    __wl_split_parts(ui, w, &a, &total);
    int d = __wl_divider(ui);
    unsigned long i = 0UL;
    while (i < w.children.len) {
        &Widget c = *w.children.at(i);
        if (i == 0UL) {
            if (w.orientation == WIDGET_HORIZONTAL) { __w_arrange(ui, c, cx, cy, a, ch); }
            else { __w_arrange(ui, c, cx, cy, cw, a); }
        } else if (i == 1UL) {
            if (w.orientation == WIDGET_HORIZONTAL) { __w_arrange(ui, c, cx + a + d, cy, total - a, ch); }
            else { __w_arrange(ui, c, cx, cy + a + d, cw, total - a); }
        } else {
            __w_arrange(ui, c, 0, 0, 0, 0);
        }
        i = i + 1UL;
    }
}

static int __wl_on_divider(&GuiUi ui, &Widget w, float x, float y) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    int a = 0; int total = 0;
    __wl_split_parts(ui, w, &a, &total);
    int d = __wl_divider(ui);
    if (w.orientation == WIDGET_HORIZONTAL) {
        return (x >= (float)(cx + a) && x < (float)(cx + a + d) && y >= (float)cy && y < (float)(cy + ch)) ? 1 : 0;
    }
    return (y >= (float)(cy + a) && y < (float)(cy + a + d) && x >= (float)cx && x < (float)(cx + cw)) ? 1 : 0;
}

// ── tabs ──────────────────────────────────────────────────────────────────

static float __wl_tab_w(&GuiUi ui, &Widget w, unsigned long i) {
    unsafe { return __w_text_w(ui, w, (const char*)w.items.at(i).data) + __w_px(ui, 24.0f); }
}

static int __wl_tab_at(&GuiUi ui, &Widget w, float x, float y) {
    int bar = __wl_title_h(ui, w);
    if (y < (float)w.y || y >= (float)(w.y + bar)) { return -1; }
    float tx = (float)w.x + __w_px(ui, 4.0f);
    unsigned long i = 0UL;
    while (i < w.items.len && i < w.children.len) {
        float tw = __wl_tab_w(ui, w, i);
        if (x >= tx && x < tx + tw) { return (int)i; }
        tx = tx + tw + __w_px(ui, 2.0f);
        i = i + 1UL;
    }
    return -1;
}

static void __wl_tabs_arrange(&GuiUi ui, &Widget w) {
    int bar = __wl_title_h(ui, w);
    int p = __w_ipx(ui, ui.theme.padding + 2);
    unsigned long i = 0UL;
    while (i < w.children.len) {
        &Widget c = *w.children.at(i);
        if ((int)i == w.selected) {
            __w_arrange(ui, c, w.x + p, w.y + bar + p, w.w - 2 * p, w.h - bar - 2 * p);
        } else {
            __w_arrange(ui, c, 0, 0, 0, 0);
        }
        i = i + 1UL;
    }
}

static void __wl_set_tab(&GuiUi ui, &Widget w, int t) {
    if (t < 0 || (unsigned long)t >= w.children.len || t == w.selected) { return; }
    w.selected = t;
    ui.needsLayout = 1;
    __w_fire_change(w);
}

// ── the family hooks ──────────────────────────────────────────────────────

void __wk_layout_measure(&GuiUi ui, &Widget w, &int cw, &int ch) {
    *cw = 0;
    *ch = 0;
    int k = w.kind;
    if (__wl_is_box(w)) {
        __wl_box_measure(ui, w, __wl_vertical(w), cw, ch);
        int t = __wl_title_h(ui, w);
        *ch = *ch + t;
        if (k == WIDGET_GROUP || k == WIDGET_DIALOG) {
            float tw = 0.0f;
            unsafe { tw = __w_text_w(ui, w, (const char*)w.text.data) + __w_px(ui, k == WIDGET_DIALOG ? 40.0f : 16.0f); }
            if ((int)tw > *cw) { *cw = (int)tw; }
        }
        if (k == WIDGET_STATUSBAR) {
            int h = __w_ipx(ui, ui.theme.rowHeight);
            if (*ch < h) { *ch = h; }
        }
        return;
    }
    if (k == WIDGET_GRID) { __wl_grid_measure(ui, w, cw, ch); return; }
    if (k == WIDGET_STACK) {
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget c = *w.children.at(i);
            i = i + 1UL;
            if (!c.visible) { continue; }
            int mw = 0; int mh = 0;
            __w_measure(ui, c, &mw, &mh);
            mw = mw + __wl_margin_w(ui, c);
            mh = mh + __wl_margin_h(ui, c);
            if (mw > *cw) { *cw = mw; }
            if (mh > *ch) { *ch = mh; }
        }
        return;
    }
    if (k == WIDGET_SCROLL) {
        ?&Widget oc = __wl_first_visible(w);
        match (oc) {
            case some(c): {
                int mw = 0; int mh = 0;
                __w_measure(ui, c, &mw, &mh);
                int cap = __w_ipx(ui, 200);
                *cw = mw + __wl_margin_w(ui, c) + __w_ipx(ui, ui.theme.scrollbarSize);
                *ch = mh + __wl_margin_h(ui, c);
                if (*ch > cap) { *ch = cap; }
            }
            case null: { }
        }
        return;
    }
    if (k == WIDGET_SPLITTER) {
        int a = 0; int b = 0;
        unsigned long i = 0UL;
        while (i < w.children.len && i < 2UL) {
            &Widget c = *w.children.at(i);
            int mw = 0; int mh = 0;
            __w_measure(ui, c, &mw, &mh);
            if (w.orientation == WIDGET_HORIZONTAL) { a = a + mw; if (mh > b) { b = mh; } }
            else { a = a + mh; if (mw > b) { b = mw; } }
            i = i + 1UL;
        }
        a = a + __wl_divider(ui);
        if (w.orientation == WIDGET_HORIZONTAL) { *cw = a; *ch = b; } else { *cw = b; *ch = a; }
        return;
    }
    if (k == WIDGET_TABS) {
        int p = __w_ipx(ui, ui.theme.padding + 2);
        float bar = 0.0f;
        unsigned long i = 0UL;
        while (i < w.items.len) { bar = bar + __wl_tab_w(ui, w, i) + __w_px(ui, 2.0f); i = i + 1UL; }
        int mxw = (int)bar + __w_ipx(ui, 8);
        int mxh = 0;
        i = 0UL;
        while (i < w.children.len) {
            &Widget c = *w.children.at(i);
            int mw = 0; int mh = 0;
            int vis = c.visible;
            c.visible = 1;
            __w_measure(ui, c, &mw, &mh);
            c.visible = vis;
            if (mw + 2 * p > mxw) { mxw = mw + 2 * p; }
            if (mh > mxh) { mxh = mh; }
            i = i + 1UL;
        }
        *cw = mxw;
        *ch = mxh + __wl_title_h(ui, w) + 2 * p;
        return;
    }
}

void __wk_layout_arrange(&GuiUi ui, &Widget w) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    int k = w.kind;
    if (__wl_is_box(w)) {
        int t = __wl_title_h(ui, w);
        __wl_box_arrange(ui, w, __wl_vertical(w), cx, cy + t, cw, ch - t);
        return;
    }
    if (k == WIDGET_GRID) { __wl_grid_arrange(ui, w, cx, cy, cw, ch); return; }
    if (k == WIDGET_STACK) {
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget c = *w.children.at(i);
            if (c.visible) { __wl_place(ui, c, cx, cy, cw, ch, c.alignment); } else { __w_arrange(ui, c, 0, 0, 0, 0); }
            i = i + 1UL;
        }
        return;
    }
    if (k == WIDGET_SCROLL) { __wl_scroll_arrange(ui, w); return; }
    if (k == WIDGET_SPLITTER) { __wl_split_arrange(ui, w); return; }
    if (k == WIDGET_TABS) { __wl_tabs_arrange(ui, w); return; }
}

int __wk_layout_child_area(&GuiUi ui, &Widget w, float x, float y) {
    int k = w.kind;
    if (k == WIDGET_SCROLL) {
        int vx = 0; int vy = 0; int vw = 0; int vh = 0;
        __wl_scroll_view(ui, w, &vx, &vy, &vw, &vh);
        return (x >= (float)vx && y >= (float)vy && x < (float)(vx + vw) && y < (float)(vy + vh)) ? 1 : 0;
    }
    if (k == WIDGET_SPLITTER) { return __wl_on_divider(ui, w, x, y) ? 0 : 1; }
    if (k == WIDGET_TABS) { return y >= (float)(w.y + __wl_title_h(ui, w)) ? 1 : 0; }
    if (k == WIDGET_DIALOG) { return y >= (float)(w.y + __wl_title_h(ui, w)) ? 1 : 0; }
    return 1;
}

static void __wl_paint_clipped_children(&GuiUi ui, &GuiCanvas c, &Widget w, int x, int y, int ww, int hh) {
    c.save();
    c.clip_rect((float)x, (float)y, (float)ww, (float)hh);
    __w_paint_children(ui, c, w);
    c.restore();
}

void __wk_layout_paint(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    float r = __w_px(ui, ui.theme.radius);
    float bw = __w_px(ui, ui.theme.borderWidth);
    int k = w.kind;
    if (k == WIDGET_VSTACK || k == WIDGET_HSTACK || k == WIDGET_GRID || k == WIDGET_STACK) {
        // only what the widget's own style asks for (the legacy look)
        if ((w.style.setMask & WIDGET_STYLE_BG) != 0 && w.style.bg.a != (unsigned char)0) {
            c.set_fill_color(w.style.bg);
            c.fill_rect(x, y, ww, hh);
        }
        if (w.style.borderWidth > 0) {
            __w_ring(c, x, y, ww, hh, 0.0f, __w_px(ui, (float)w.style.borderWidth), w.style.border);
        }
        __w_paint_children(ui, c, w);
        return;
    }
    if (k == WIDGET_PANEL) {
        __w_box(c, x, y, ww, hh, r, __w_bg_or(w, ui.theme.panel), __w_border_or(w, ui.theme.border), bw);
        __w_paint_children(ui, c, w);
        return;
    }
    if (k == WIDGET_GROUP) {
        float th = (float)__wl_title_h(ui, w);
        float top = y + th * 0.5f;
        struct GuiColor bg = __w_bg_or(w, ui.theme.panel);
        __w_box(c, x, top, ww, hh - th * 0.5f, r, bg, __w_border_or(w, ui.theme.border), bw);
        // the title sits on the frame's top edge
        float tx = x + __w_px(ui, 10.0f);
        float tw = 0.0f;
        unsafe { tw = __w_text_w(ui, w, (const char*)w.text.data); }
        if (tw > 0.0f) {
            c.set_fill_color(ui.theme.window);
            c.fill_rect(tx - __w_px(ui, 4.0f), top - bw, tw + __w_px(ui, 8.0f), bw * 2.0f + 1.0f);
            c.set_fill_color(bg);
            c.fill_rect(tx - __w_px(ui, 4.0f), top + bw, tw + __w_px(ui, 8.0f), th * 0.5f);
            unsafe { __w_text(ui, c, w, (const char*)w.text.data, tx, y, __w_fg(ui, w)); }
        }
        __w_paint_children(ui, c, w);
        return;
    }
    if (k == WIDGET_STATUSBAR) {
        c.set_fill_color(__w_bg_or(w, ui.theme.header));
        c.fill_rect(x, y, ww, hh);
        c.set_fill_color(ui.theme.border);
        c.fill_rect(x, y, ww, bw);
        __w_paint_children(ui, c, w);
        return;
    }
    if (k == WIDGET_DIALOG) {
        // shadow, frame, title bar with a close button
        float sh = __w_px(ui, 6.0f);
        c.set_fill_color(ui.theme.shadow);
        c.fill_rounded_rect(x - sh * 0.5f, y + sh * 0.5f, ww + sh, hh + sh, r + sh);
        __w_box(c, x, y, ww, hh, r * 1.5f, __w_bg_or(w, ui.theme.panel), ui.theme.borderStrong, bw);
        float th = (float)__wl_title_h(ui, w);
        c.save();
        c.clip_rect(x, y, ww, th);
        __w_box(c, x, y, ww, th + r * 2.0f, r * 1.5f, ui.theme.header, ui.theme.borderStrong, bw);
        c.restore();
        c.set_fill_color(ui.theme.border);
        c.fill_rect(x, y + th - bw, ww, bw);
        int bold = w.style.bold;
        w.style.bold = 1;
        unsafe { __w_text_in(ui, c, w, (const char*)w.text.data, x + __w_px(ui, 12.0f), y, ww, th, GUI_TEXT_LEFT, ui.theme.text); }
        w.style.bold = bold;
        // the close cross
        float cs = __w_px(ui, 8.0f);
        float ccx = x + ww - th * 0.5f;
        float ccy = y + th * 0.5f;
        int hot = w.hotIndex == 1 ? 1 : 0;
        if (hot) {
            c.set_fill_color(ui.theme.hover);
            c.fill_circle(ccx, ccy, th * 0.35f);
        }
        c.save();
        c.set_stroke_color(ui.theme.textDim);
        c.set_line_width(__w_px(ui, 1.5f));
        c.set_line_cap(GUI_CAP_ROUND);
        c.draw_line(ccx - cs * 0.5f, ccy - cs * 0.5f, ccx + cs * 0.5f, ccy + cs * 0.5f);
        c.draw_line(ccx + cs * 0.5f, ccy - cs * 0.5f, ccx - cs * 0.5f, ccy + cs * 0.5f);
        c.restore();
        __w_paint_children(ui, c, w);
        return;
    }
    if (k == WIDGET_SCROLL) {
        if ((w.style.setMask & WIDGET_STYLE_BG) != 0) { c.set_fill_color(w.style.bg); c.fill_rect(x, y, ww, hh); }
        if (w.style.borderWidth > 0) { __w_ring(c, x, y, ww, hh, 0.0f, __w_px(ui, (float)w.style.borderWidth), w.style.border); }
        int vx = 0; int vy = 0; int vw = 0; int vh = 0;
        __wl_scroll_view(ui, w, &vx, &vy, &vw, &vh);
        __wl_paint_clipped_children(ui, c, w, vx, vy, vw, vh);
        __w_scrollbars_paint(ui, c, w, (float)vx, (float)vy, (float)vw, (float)vh);
        return;
    }
    if (k == WIDGET_SPLITTER) {
        unsigned long i = 0UL;
        while (i < w.children.len && i < 2UL) {
            &Widget ch = *w.children.at(i);
            c.save();
            c.clip_rect((float)ch.x, (float)ch.y, (float)ch.w, (float)ch.h);
            __w_paint(ui, c, ch);
            c.restore();
            i = i + 1UL;
        }
        int cx = 0; int cy = 0; int cw = 0; int chh = 0;
        __w_content(ui, w, &cx, &cy, &cw, &chh);
        int a = 0; int total = 0;
        __wl_split_parts(ui, w, &a, &total);
        float d = (float)__wl_divider(ui);
        struct GuiColor gc = (w.dragPart == GUI_DRAG_SPLIT || w.hotIndex == 1) ? ui.theme.accent : ui.theme.border;
        c.set_fill_color(gc);
        if (w.orientation == WIDGET_HORIZONTAL) {
            float gx = (float)(cx + a) + d * 0.5f - bw * 0.5f;
            c.fill_rect(gx, (float)cy, bw, (float)chh);
            float my = (float)cy + (float)chh * 0.5f;
            int dot = -1;
            while (dot <= 1) { c.fill_circle(gx + bw * 0.5f, my + (float)dot * __w_px(ui, 5.0f), __w_px(ui, 1.5f)); dot = dot + 1; }
        } else {
            float gy = (float)(cy + a) + d * 0.5f - bw * 0.5f;
            c.fill_rect((float)cx, gy, (float)cw, bw);
            float mx = (float)cx + (float)cw * 0.5f;
            int dot = -1;
            while (dot <= 1) { c.fill_circle(mx + (float)dot * __w_px(ui, 5.0f), gy + bw * 0.5f, __w_px(ui, 1.5f)); dot = dot + 1; }
        }
        return;
    }
    if (k == WIDGET_TABS) {
        float bar = (float)__wl_title_h(ui, w);
        // page frame
        __w_box(c, x, y + bar - bw, ww, hh - bar + bw, r, __w_bg_or(w, ui.theme.panel), ui.theme.border, bw);
        float tx = x + __w_px(ui, 4.0f);
        unsigned long i = 0UL;
        while (i < w.items.len && i < w.children.len) {
            float tw = __wl_tab_w(ui, w, i);
            int sel = (int)i == w.selected ? 1 : 0;
            float top = y + (sel ? 0.0f : __w_px(ui, 3.0f));
            if (sel) {
                c.save();
                c.clip_rect(tx - bw, y, tw + 2.0f * bw, bar);
                __w_box(c, tx, top, tw, bar + r, r, __w_bg_or(w, ui.theme.panel), ui.theme.border, bw);
                c.restore();
                c.set_fill_color(ui.theme.accent);
                c.fill_rounded_rect(tx + __w_px(ui, 6.0f), top + bw, tw - __w_px(ui, 12.0f), __w_px(ui, 2.5f), __w_px(ui, 1.0f));
            } else if (w.hotIndex == (int)i) {
                c.set_fill_color(ui.theme.hover);
                c.fill_rounded_rect(tx, top, tw, bar - top + y - __w_px(ui, 2.0f), r);
            }
            unsafe {
                __w_text_in(ui, c, w, (const char*)w.items.at(i).data, tx, top, tw, bar - (top - y),
                            GUI_TEXT_CENTER, sel ? ui.theme.text : ui.theme.textDim);
            }
            if (sel && w.focused) { __w_focus_ring(ui, c, w, tx + bw, top + bw, tw - 2.0f * bw, bar - (top - y) - 2.0f * bw, r); }
            tx = tx + tw + __w_px(ui, 2.0f);
            i = i + 1UL;
        }
        if (w.selected >= 0 && (unsigned long)w.selected < w.children.len) {
            &Widget page = *w.children.at((unsigned long)w.selected);
            c.save();
            c.clip_rect(x + bw, y + bar, ww - 2.0f * bw, hh - bar - bw);
            __w_paint(ui, c, page);
            c.restore();
        }
        return;
    }
    __w_paint_children(ui, c, w);
}

void __wk_layout_paint_over(&GuiUi ui, &GuiCanvas c, &Widget w) { }

int __wk_layout_event(&GuiUi ui, &Widget w, &GuiEvent ev) {
    int k = w.kind;
    float ex = (float)ev.x;
    float ey = (float)ev.y;
    if (k == WIDGET_SCROLL) {
        int vx = 0; int vy = 0; int vw = 0; int vh = 0;
        __wl_scroll_view(ui, w, &vx, &vy, &vw, &vh);
        float ox = w.scrollX;
        float oy = w.scrollY;
        int r = __w_scrollbars_event(ui, w, ev, (float)vx, (float)vy, (float)vw, (float)vh);
        if (ox != w.scrollX || oy != w.scrollY) { ui.needsLayout = 1; }
        return r;
    }
    if (k == WIDGET_SPLITTER) {
        if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT && __wl_on_divider(ui, w, ex, ey)) {
            w.dragPart = GUI_DRAG_SPLIT;
            return 1;
        }
        if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
            int on = __wl_on_divider(ui, w, ex, ey);
            int hot = (on || w.dragPart == GUI_DRAG_SPLIT) ? 1 : 0;
            if (hot != (w.hotIndex == 1 ? 1 : 0)) { w.hotIndex = hot ? 1 : -1; ui.dirty = 1; }
            if (hot) { __w_set_cursor(ui, w.orientation == WIDGET_HORIZONTAL ? GUI_CURSOR_RESIZE_EW : GUI_CURSOR_RESIZE_NS); }
            if (w.dragPart == GUI_DRAG_SPLIT) {
                int cx = 0; int cy = 0; int cw = 0; int ch = 0;
                __w_content(ui, w, &cx, &cy, &cw, &ch);
                int total = (w.orientation == WIDGET_HORIZONTAL ? cw : ch) - __wl_divider(ui);
                if (total > 0) {
                    float pos = w.orientation == WIDGET_HORIZONTAL ? ex - (float)cx : ey - (float)cy;
                    pos = pos - (float)__wl_divider(ui) * 0.5f;
                    double v = (double)pos / (double)total;
                    if (v < 0.0) { v = 0.0; }
                    if (v > 1.0) { v = 1.0; }
                    if (v != w.sliderValue) {
                        w.sliderValue = v;
                        ui.needsLayout = 1;
                        __w_fire_change(w);
                    }
                }
                return 1;
            }
            return hot;
        }
        if (ev.kind == GUI_EVENT_MOUSE_UP && w.dragPart == GUI_DRAG_SPLIT) {
            w.dragPart = 0;
            return 1;
        }
        return 0;
    }
    if (k == WIDGET_TABS) {
        if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT) {
            int t = __wl_tab_at(ui, w, ex, ey);
            if (t >= 0) { __wl_set_tab(ui, w, t); return 1; }
            return 0;
        }
        if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
            int t = __wl_tab_at(ui, w, ex, ey);
            if (t != w.hotIndex) { w.hotIndex = t; ui.dirty = 1; }
            return 0;
        }
        if (ev.kind == GUI_EVENT_KEY_DOWN && w.focused) {
            int n = (int)w.children.len;
            if (n > 0 && (ev.keycode == GUI_KEY_LEFT || ev.keycode == GUI_KEY_RIGHT)) {
                int t = w.selected + (ev.keycode == GUI_KEY_RIGHT ? 1 : -1);
                if (t < 0) { t = n - 1; }
                if (t >= n) { t = 0; }
                __wl_set_tab(ui, w, t);
                return 1;
            }
            if (n > 0 && (ev.keycode == GUI_KEY_HOME || ev.keycode == GUI_KEY_END)) {
                __wl_set_tab(ui, w, ev.keycode == GUI_KEY_HOME ? 0 : n - 1);
                return 1;
            }
        }
        // Ctrl+Tab / Ctrl+Shift+Tab from anywhere inside
        if (ev.kind == GUI_EVENT_KEY_DOWN && ev.keycode == GUI_KEY_TAB && ev.has_mod(GUI_MOD_CTRL)) {
            int n = (int)w.children.len;
            if (n > 0) {
                int t = w.selected + (ev.has_mod(GUI_MOD_SHIFT) ? -1 : 1);
                if (t < 0) { t = n - 1; }
                if (t >= n) { t = 0; }
                __wl_set_tab(ui, w, t);
                ev.prevent_default();
                return 1;
            }
        }
        return 0;
    }
    if (k == WIDGET_DIALOG) {
        float th = (float)__wl_title_h(ui, w);
        float ccx = (float)(w.x + w.w) - th * 0.5f;
        float ccy = (float)w.y + th * 0.5f;
        float dx = ex - ccx;
        float dy = ey - ccy;
        int onClose = (dx * dx + dy * dy <= th * th * 0.16f) ? 1 : 0;
        if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
            int hot = onClose ? 1 : -1;
            if (hot != w.hotIndex) { w.hotIndex = hot; ui.dirty = 1; }
            if (w.dragPart == GUI_DRAG_DIALOG) {
                int nx = (int)(ex - w.dragX);
                int ny = (int)(ey - w.dragY);
                __w_arrange(ui, w, nx, ny, w.w, w.h);
                return 1;
            }
            return 0;
        }
        if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT && ey < (float)w.y + th) {
            if (onClose) { w.dragPart = 0; w.close_dialog(GUI_RESULT_CANCEL); return 1; }
            w.dragPart = GUI_DRAG_DIALOG;
            w.dragX = ex - (float)w.x;
            w.dragY = ey - (float)w.y;
            return 1;
        }
        if (ev.kind == GUI_EVENT_MOUSE_UP && w.dragPart == GUI_DRAG_DIALOG) {
            w.dragPart = 0;
            return 1;
        }
        // clicks on the dialog's background are used up (it's modal)
        if (ev.kind == GUI_EVENT_MOUSE_DOWN || ev.kind == GUI_EVENT_MOUSE_UP) { return 1; }
        return 0;
    }
    return 0;
}

} // namespace std
