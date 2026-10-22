// SafeC Standard Library — GUI widgets: combo boxes (and their drop-down
// lists), list views, tree views and table views (see gui_widget.h).
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_text.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_widget_internal.h>
#include <std/math.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

#define GUI_DRAG_COLRESIZE 40   // + column index

static float __wli_row(&GuiUi ui) { return __w_px(ui, (float)ui.theme.rowHeight); }

static int __wli_inside(const &Widget w, float x, float y) {
    return (x >= (float)w.x && y >= (float)w.y && x < (float)(w.x + w.w) && y < (float)(w.y + w.h)) ? 1 : 0;
}

// ── tree order ────────────────────────────────────────────────────────────

// The visible nodes of a tree in display order (expanded ancestors only).
static void __wli_tree_rows(const &Widget w, &Vec<int> out) {
    out.clear();
    int n = (int)w.nodes.len;
    if (n == 0) { return; }
    // first-child / next-sibling links, in insertion order
    struct Vec<int> first = Vec::new();
    struct Vec<int> last = Vec::new();
    struct Vec<int> next = Vec::new();
    int i = 0;
    while (i < n) { first.push(-1); last.push(-1); next.push(-1); i = i + 1; }
    int topFirst = -1;
    int topLast = -1;
    i = 0;
    while (i < n) {
        int p = w.nodes.at((unsigned long)i).parent;
        if (p < 0 || p >= n || p >= i) {
            if (topLast < 0) { topFirst = i; } else { next.set((unsigned long)topLast, i); }
            topLast = i;
        } else {
            int l = *last.at((unsigned long)p);
            if (l < 0) { first.set((unsigned long)p, i); } else { next.set((unsigned long)l, i); }
            last.set((unsigned long)p, i);
        }
        i = i + 1;
    }
    // depth-first walk with an explicit stack
    struct Vec<int> pending = Vec::new();
    int cur = topFirst;
    int guard = 0;
    while (guard < n * 4 + 8) {
        if (cur >= 0) {
            out.push(cur);
            int fc = *first.at((unsigned long)cur);
            if (fc >= 0 && w.nodes.at((unsigned long)cur).expanded) {
                pending.push(*next.at((unsigned long)cur));
                cur = fc;
            } else {
                cur = *next.at((unsigned long)cur);
            }
        } else {
            ?int top = pending.pop();
            match (top) {
                case some(t): { cur = t; }
                case none: { break; }
            }
        }
        guard = guard + 1;
    }
    pending.free();
    first.free();
    last.free();
    next.free();
}

static int __wli_tree_has_children(const &Widget w, int node) {
    unsigned long i = (unsigned long)node + 1UL;
    while (i < w.nodes.len) {
        if (w.nodes.at(i).parent == node) { return 1; }
        i = i + 1UL;
    }
    return 0;
}

// ── row counts and geometry ───────────────────────────────────────────────

static int __wli_header_h(&GuiUi ui, const &Widget w) {
    return w.kind == WIDGET_TABLE ? (int)__wli_row(ui) : 0;
}

// Rows in display order: list items, visible tree nodes, table rows.
static int __wli_count(const &Widget w, &Vec<int> treeRows) {
    if (w.kind == WIDGET_TREE) { __wli_tree_rows(w, treeRows); return (int)treeRows.len; }
    if (w.kind == WIDGET_TABLE) { return w.rowCount; }
    if (w.kind == WIDGET_POPUPLIST) {
        match (w.owner) { case some(o): { return (int)o.items.len; } case null: { return 0; } }
    }
    return (int)w.items.len;
}

// The viewport for rows (inside the border, below a table header).
static void __wli_view(&GuiUi ui, const &Widget w, &int vx, &int vy, &int vw, &int vh) {
    int b = __w_ipx(ui, 1);
    *vx = w.x + b;
    *vy = w.y + b + __wli_header_h(ui, w);
    *vw = w.w - 2 * b;
    *vh = w.h - 2 * b - __wli_header_h(ui, w);
    int sb = __w_ipx(ui, ui.theme.scrollbarSize);
    if (w.contentH > (float)*vh) { *vw = *vw - sb; }
    if (w.contentW > (float)*vw) { *vh = *vh - sb; }
    if (*vw < 0) { *vw = 0; }
    if (*vh < 0) { *vh = 0; }
}

// Table columns' pixel widths (fixed ones as given, the rest share).
static void __wli_col_widths(&GuiUi ui, const &Widget w, int avail, &Vec<int> out) {
    out.clear();
    int fixed = 0;
    int flex = 0;
    unsigned long i = 0UL;
    while (i < w.tableCols.len) {
        int cw = w.tableCols.at(i).width;
        if (cw > 0) { fixed = fixed + __w_ipx(ui, cw); } else { flex = flex + 1; }
        i = i + 1UL;
    }
    int share = flex > 0 ? (avail - fixed) / flex : 0;
    int minw = __w_ipx(ui, 60);
    if (share < minw) { share = minw; }
    i = 0UL;
    while (i < w.tableCols.len) {
        int cw = w.tableCols.at(i).width;
        out.push(cw > 0 ? __w_ipx(ui, cw) : share);
        i = i + 1UL;
    }
}

static void __wli_update_content(&GuiUi ui, &Widget w, int count) {
    w.contentH = __wli_row(ui) * (float)count;
    w.contentW = 0.0f;
    if (w.kind == WIDGET_TABLE) {
        struct Vec<int> cols = Vec::new();
        __wli_col_widths(ui, w, w.w - __w_ipx(ui, 2), &cols);
        int total = 0;
        unsigned long i = 0UL;
        while (i < cols.len) { total = total + *cols.at(i); i = i + 1UL; }
        cols.free();
        w.contentW = (float)total;
    }
}

// ── measuring ─────────────────────────────────────────────────────────────

void __wk_list_measure(&GuiUi ui, &Widget w, &int cw, &int ch) {
    int k = w.kind;
    float row = __wli_row(ui);
    if (k == WIDGET_COMBO) {
        float widest = 0.0f;
        unsigned long i = 0UL;
        while (i < w.items.len) {
            float tw = 0.0f;
            unsafe { tw = __w_text_w(ui, w, (const char*)w.items.at(i).data); }
            if (tw > widest) { widest = tw; }
            i = i + 1UL;
        }
        float pw = 0.0f;
        unsafe { pw = __w_text_w(ui, w, (const char*)w.placeholder.data); }
        if (pw > widest) { widest = pw; }
        float minw = __w_px(ui, 80.0f);
        if (widest < minw) { widest = minw; }
        *cw = (int)ceil_f(widest + __w_px(ui, 22.0f));
        int ctl = __w_ipx(ui, ui.theme.controlHeight) - __w_pad(ui, w, 1) - __w_pad(ui, w, 3);
        float lh = __w_line_h(ui, w);
        *ch = (int)ceil_f(lh) > ctl ? (int)ceil_f(lh) : ctl;
        return;
    }
    if (k == WIDGET_POPUPLIST) {
        int n = 0;
        int ow = __w_ipx(ui, 120);
        float widest = 0.0f;
        match (w.owner) {
            case some(o): {
                n = (int)o.items.len;
                ow = o.w;
                unsigned long i = 0UL;
                while (i < o.items.len) {
                    float tw = 0.0f;
                    unsafe { tw = __w_text_w(ui, o, (const char*)o.items.at(i).data); }
                    if (tw > widest) { widest = tw; }
                    i = i + 1UL;
                }
            }
            case null: { }
        }
        int want = (int)ceil_f(widest + __w_px(ui, 28.0f));
        *cw = ow > want ? ow : want;
        int shown = n < 10 ? n : 10;
        if (shown < 1) { shown = 1; }
        *ch = (int)ceil_f(row * (float)shown) + __w_ipx(ui, 2);
        return;
    }
    if (k == WIDGET_TABLE) {
        int total = 0;
        unsigned long i = 0UL;
        while (i < w.tableCols.len) {
            int cw0 = w.tableCols.at(i).width;
            total = total + (cw0 > 0 ? __w_ipx(ui, cw0) : __w_ipx(ui, 100));
            i = i + 1UL;
        }
        *cw = total > __w_ipx(ui, 200) ? total : __w_ipx(ui, 200);
        *ch = __w_ipx(ui, 160);
        return;
    }
    *cw = __w_ipx(ui, k == WIDGET_TREE ? 180 : 160);
    *ch = __w_ipx(ui, k == WIDGET_TREE ? 160 : 120);
}

// ── painting ──────────────────────────────────────────────────────────────

static void __wli_paint_combo(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    float r = __w_px(ui, ui.theme.radius);
    float bw = __w_px(ui, ui.theme.borderWidth);
    int en = __w_effectively_enabled(w);
    struct GuiColor fill = __w_bg_or(w, ui.theme.button);
    if (en && (w.hover || w.open)) { fill = ui.theme.buttonHover; }
    if (!en) { fill = fill.mix(ui.theme.window, 0.5f); }
    struct GuiColor border = (w.open || w.focused) ? ui.theme.accent : (w.hover && en ? ui.theme.borderStrong : ui.theme.border);
    __w_box(c, x, y, ww, hh, r, fill, border, bw);
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    c.save();
    c.clip_rect((float)cx, y, (float)cw - __w_px(ui, 16.0f), hh);
    if (w.selected >= 0 && (unsigned long)w.selected < w.items.len) {
        unsafe { __w_text_in(ui, c, w, (const char*)w.items.at((unsigned long)w.selected).data, (float)cx, y, (float)cw, hh, GUI_TEXT_LEFT, __w_fg(ui, w)); }
    } else {
        unsafe { __w_text_in(ui, c, w, (const char*)w.placeholder.data, (float)cx, y, (float)cw, hh, GUI_TEXT_LEFT, ui.theme.textDim); }
    }
    c.restore();
    __w_chevron(c, x + ww - __w_px(ui, 14.0f), y + hh * 0.5f, __w_px(ui, 9.0f), w.open ? 2 : 0,
                en ? ui.theme.textDim : ui.theme.textDisabled, __w_px(ui, 1.5f));
    if (w.focused && !w.open) { __w_focus_ring(ui, c, w, x, y, ww, hh, r); }
}

static void __wli_row_text(&GuiUi ui, &GuiCanvas c, &Widget w, const &char s, float x, float y, float cw, float rh,
                           int align, struct GuiColor col) {
    c.save();
    c.clip_rect(x, y, cw, rh);
    __w_text_in(ui, c, w, s, x, y, cw, rh, align, col);
    c.restore();
}

void __wk_list_paint(&GuiUi ui, &GuiCanvas c, &Widget w) {
    int k = w.kind;
    if (k == WIDGET_COMBO) { __wli_paint_combo(ui, c, w); return; }
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    float r = __w_px(ui, ui.theme.radius);
    float bw = __w_px(ui, ui.theme.borderWidth);
    float row = __wli_row(ui);
    int en = __w_effectively_enabled(w);
    ?&Widget src = null;   // where the items come from (a popup shows its combo's)
    if (k == WIDGET_POPUPLIST) {
        src = w.owner;
        float sh = __w_px(ui, 4.0f);
        c.set_fill_color(ui.theme.shadow);
        c.fill_rounded_rect(x + sh * 0.5f, y + sh, ww, hh, r);
        __w_box(c, x, y, ww, hh, r, ui.theme.panel, ui.theme.borderStrong, bw);
    } else {
        src = w;
        struct GuiColor border = w.focused ? ui.theme.accent : ui.theme.border;
        __w_box(c, x, y, ww, hh, r, __w_bg_or(w, ui.theme.input), border, bw);
    }
    struct Vec<int> treeRows = Vec::new();
    int count = __wli_count(w, &treeRows);
    __wli_update_content(ui, w, count);
    int vx = 0; int vy = 0; int vw = 0; int vh = 0;
    __wli_view(ui, w, &vx, &vy, &vw, &vh);
    __w_scroll_clamp(w, (float)vw, (float)vh);
    struct Vec<int> cols = Vec::new();
    if (k == WIDGET_TABLE) {
        __wli_col_widths(ui, w, w.w - __w_ipx(ui, 2), &cols);
        // header
        float hy = y + bw;
        float hh2 = (float)__wli_header_h(ui, w);
        c.save();
        c.clip_rect(x + bw, hy, ww - 2.0f * bw, hh2);
        c.set_fill_color(ui.theme.header);
        c.fill_rect(x + bw, hy, ww - 2.0f * bw, hh2);
        float hx = (float)vx - w.scrollX;
        unsigned long ci = 0UL;
        int bold = w.style.bold;
        w.style.bold = 1;
        while (ci < w.tableCols.len) {
            float colw = (float)*cols.at(ci);
            unsafe {
                __wli_row_text(ui, c, w, (const char*)w.tableCols.at(ci).title.data, hx + __w_px(ui, 8.0f), hy,
                               colw - __w_px(ui, 12.0f), hh2, w.tableCols.at(ci).align, ui.theme.text);
            }
            c.set_fill_color(ui.theme.border);
            c.fill_rect(hx + colw - bw, hy + __w_px(ui, 4.0f), bw, hh2 - __w_px(ui, 8.0f));
            hx = hx + colw;
            ci = ci + 1UL;
        }
        w.style.bold = bold;
        c.set_fill_color(ui.theme.border);
        c.fill_rect(x + bw, hy + hh2 - bw, ww - 2.0f * bw, bw);
        c.restore();
    }
    c.save();
    c.clip_rect((float)vx, (float)vy, (float)vw, (float)vh);
    int first = (int)(w.scrollY / row);
    if (first < 0) { first = 0; }
    int i = first;
    while (i < count) {
        float ry = (float)vy + (float)i * row - w.scrollY;
        if (ry >= (float)(vy + vh)) { break; }
        int sel = 0;
        int hot = w.hotIndex == i ? 1 : 0;
        if (k == WIDGET_POPUPLIST) {
            match (src) { case some(o): { sel = o.selected == i ? 1 : 0; } case null: { } }
        } else if (k == WIDGET_TREE) {
            sel = *treeRows.at((unsigned long)i) == w.selected ? 1 : 0;
        } else {
            sel = w.selected == i ? 1 : 0;
        }
        if (k == WIDGET_TABLE && (i % 2) == 1) {
            c.set_fill_color(ui.theme.altRow);
            c.fill_rect((float)vx, ry, (float)vw, row);
        }
        struct GuiColor tc = en ? __w_fg(ui, w) : ui.theme.textDisabled;
        if (k == WIDGET_POPUPLIST) {
            if (hot) {
                c.set_fill_color(ui.theme.selection);
                c.fill_rounded_rect((float)vx + __w_px(ui, 3.0f), ry + __w_px(ui, 1.0f), (float)vw - __w_px(ui, 6.0f), row - __w_px(ui, 2.0f), r);
                tc = ui.theme.selectionText;
            }
        } else if (sel) {
            struct GuiColor sc = w.focused ? ui.theme.selection : ui.theme.selection.with_alpha(0.25f);
            c.set_fill_color(sc);
            c.fill_rect((float)vx, ry, (float)vw, row);
            if (w.focused) { tc = ui.theme.selectionText; }
        } else if (hot && en) {
            c.set_fill_color(ui.theme.hover);
            c.fill_rect((float)vx, ry, (float)vw, row);
        }
        float pad = __w_px(ui, 8.0f);
        if (k == WIDGET_LIST) {
            unsafe { __wli_row_text(ui, c, w, (const char*)w.items.at((unsigned long)i).data, (float)vx + pad, ry, (float)vw - 2.0f * pad, row, GUI_TEXT_LEFT, tc); }
        } else if (k == WIDGET_POPUPLIST) {
            match (src) {
                case some(o): {
                    if (sel) { __w_check_mark(c, (float)vx + __w_px(ui, 4.0f), ry + (row - __w_px(ui, 14.0f)) * 0.5f, __w_px(ui, 14.0f), tc, __w_px(ui, 1.6f)); }
                    unsafe { __wli_row_text(ui, c, o, (const char*)o.items.at((unsigned long)i).data, (float)vx + __w_px(ui, 22.0f), ry, (float)vw - __w_px(ui, 26.0f), row, GUI_TEXT_LEFT, tc); }
                }
                case null: { }
            }
        } else if (k == WIDGET_TREE) {
            int node = *treeRows.at((unsigned long)i);
            &WidgetTreeNode nd = w.nodes.at((unsigned long)node);
            float ind = (float)nd.depth * __w_px(ui, 16.0f);
            float tx = (float)vx + __w_px(ui, 4.0f) + ind;
            if (__wli_tree_has_children(w, node)) {
                __w_chevron(c, tx + __w_px(ui, 8.0f), ry + row * 0.5f, __w_px(ui, 8.0f), nd.expanded ? 0 : 1, sel && w.focused ? tc : ui.theme.textDim, __w_px(ui, 1.5f));
            }
            unsafe { __wli_row_text(ui, c, w, (const char*)nd.text.data, tx + __w_px(ui, 18.0f), ry, (float)vw - (tx - (float)vx) - __w_px(ui, 22.0f), row, GUI_TEXT_LEFT, tc); }
        } else if (k == WIDGET_TABLE) {
            float cx2 = (float)vx - w.scrollX;
            unsigned long ci = 0UL;
            while (ci < w.tableCols.len) {
                float colw = (float)*cols.at(ci);
                unsafe {
                    __wli_row_text(ui, c, w, (const char*)w.cells.at((unsigned long)i * w.tableCols.len + ci).data,
                                   cx2 + pad, ry, colw - pad - __w_px(ui, 4.0f), row, w.tableCols.at(ci).align, tc);
                }
                cx2 = cx2 + colw;
                ci = ci + 1UL;
            }
        }
        i = i + 1;
    }
    c.restore();
    __w_scrollbars_paint(ui, c, w, (float)vx, (float)vy, (float)vw, (float)vh);
    if (w.focused && k != WIDGET_POPUPLIST) { __w_focus_ring(ui, c, w, x, y, ww, hh, r); }
    cols.free();
    treeRows.free();
}

// ── events ────────────────────────────────────────────────────────────────

static int __wli_row_at(&GuiUi ui, &Widget w, float x, float y, int count) {
    int vx = 0; int vy = 0; int vw = 0; int vh = 0;
    __wli_view(ui, w, &vx, &vy, &vw, &vh);
    if (x < (float)vx || x >= (float)(vx + vw) || y < (float)vy || y >= (float)(vy + vh)) { return -1; }
    int r = (int)((y - (float)vy + w.scrollY) / __wli_row(ui));
    if (r < 0 || r >= count) { return -1; }
    return r;
}

static void __wli_show_row(&GuiUi ui, &Widget w, int r) {
    int vx = 0; int vy = 0; int vw = 0; int vh = 0;
    __wli_view(ui, w, &vx, &vy, &vw, &vh);
    float row = __wli_row(ui);
    __w_scroll_into_view(w, (float)r * row, (float)(r + 1) * row, (float)vh);
    __w_scroll_clamp(w, (float)vw, (float)vh);
}

static void __wli_open_combo(&GuiUi ui, &Widget w) {
    ?&Widget op = w.popup;
    &Widget p;
    match (op) {
        case some(x): { p = x; }
        case null: {
            p = Widget::list();
            p.kind = WIDGET_POPUPLIST;
            p.owner = w;
            w.popup = p;
        }
    }
    p.hotIndex = w.selected;
    p.scrollY = 0.0f;
    w.open = 1;
    ui.popup(p, (float)w.x, (float)(w.y + w.h + __w_ipx(ui, 2)));
    // keep the selected row in view
    struct Vec<int> none = Vec::new();
    int count = __wli_count(p, &none);
    none.free();
    __wli_update_content(ui, p, count);
    if (w.selected >= 0) { __wli_show_row(ui, p, w.selected); }
}

static void __wli_pick(&GuiUi ui, &Widget popup, int index) {
    match (popup.owner) {
        case some(o): {
            __w_popup_close_from(ui, popup);
            o.open = 0;
            if (index >= 0 && index != o.selected) {
                o.selected = index;
                __w_fire_change(o);
            }
            ui.set_focus(o);
        }
        case null: { }
    }
}

// Moves the selection to display row r (tree: node of that row).
static void __wli_select_row(&GuiUi ui, &Widget w, int r, &Vec<int> treeRows, int count) {
    if (count <= 0) { return; }
    if (r < 0) { r = 0; }
    if (r >= count) { r = count - 1; }
    int value = w.kind == WIDGET_TREE ? *treeRows.at((unsigned long)r) : r;
    __wli_show_row(ui, w, r);
    if (value != w.selected) {
        w.selected = value;
        __w_fire_change(w);
    }
}

static int __wli_display_row(const &Widget w, &Vec<int> treeRows) {
    if (w.kind != WIDGET_TREE) { return w.selected; }
    unsigned long i = 0UL;
    while (i < treeRows.len) {
        if (*treeRows.at(i) == w.selected) { return (int)i; }
        i = i + 1UL;
    }
    return -1;
}

int __wk_list_event(&GuiUi ui, &Widget w, &GuiEvent ev) {
    int k = w.kind;
    float ex = (float)ev.x;
    float ey = (float)ev.y;
    int inside = __wli_inside(w, ex, ey);
    if (k == WIDGET_COMBO) {
        if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT && inside) {
            if (w.open) {
                match (w.popup) { case some(p): { __w_popup_close_from(ui, p); } case null: { } }
                w.open = 0;
            } else if (w.items.len > 0UL) {
                __wli_open_combo(ui, w);
            }
            return 1;
        }
        if (ev.kind == GUI_EVENT_MOUSE_UP && inside) { return 1; }
        if (ev.kind == GUI_EVENT_KEY_DOWN && w.focused && !w.open) {
            int kc = ev.keycode;
            int n = (int)w.items.len;
            if ((kc == GUI_KEY_ENTER || kc == GUI_KEY_SPACE || (kc == GUI_KEY_DOWN && ev.has_mod(GUI_MOD_ALT))) && n > 0) {
                __wli_open_combo(ui, w);
                return 1;
            }
            if (n > 0 && (kc == GUI_KEY_UP || kc == GUI_KEY_DOWN || kc == GUI_KEY_HOME || kc == GUI_KEY_END)) {
                int s = w.selected;
                if (kc == GUI_KEY_UP) { s = s - 1; }
                else if (kc == GUI_KEY_DOWN) { s = s + 1; }
                else if (kc == GUI_KEY_HOME) { s = 0; }
                else { s = n - 1; }
                if (s < 0) { s = 0; }
                if (s >= n) { s = n - 1; }
                if (s != w.selected) { w.selected = s; __w_fire_change(w); }
                return 1;
            }
        }
        if (ev.kind == GUI_EVENT_SCROLL && inside && w.focused) {
            int n = (int)w.items.len;
            int s = w.selected + (ev.scrollDy > 0.0 ? -1 : 1);
            if (s >= 0 && s < n && s != w.selected) { w.selected = s; __w_fire_change(w); }
            return 1;
        }
        return 0;
    }
    struct Vec<int> treeRows = Vec::new();
    int count = __wli_count(w, &treeRows);
    __wli_update_content(ui, w, count);
    int vx = 0; int vy = 0; int vw = 0; int vh = 0;
    __wli_view(ui, w, &vx, &vy, &vw, &vh);
    int r = 0;
    // scrollbars first
    if (__w_scrollbars_event(ui, w, ev, (float)vx, (float)vy, (float)vw, (float)vh)) {
        treeRows.free();
        ui.dirty = 1;
        return 1;
    }
    if (ev.kind == GUI_EVENT_SCROLL) { treeRows.free(); return 0; }
    if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
        // table column resizing
        if (w.dragPart >= GUI_DRAG_COLRESIZE && k == WIDGET_TABLE) {
            int col = w.dragPart - GUI_DRAG_COLRESIZE;
            int nw = (int)((ex - w.dragX) / ui.scale);
            if (nw < 30) { nw = 30; }
            &WidgetColumn cdef = w.tableCols.at((unsigned long)col);
            cdef.width = nw;
            ui.dirty = 1;
            treeRows.free();
            return 1;
        }
        int h = __wli_row_at(ui, w, ex, ey, count);
        if (h != w.hotIndex) { w.hotIndex = h; ui.dirty = 1; }
        if (k == WIDGET_TABLE) {
            // near a header divider: the resize cursor
            int hy = w.y + __w_ipx(ui, 1);
            if (ey >= (float)hy && ey < (float)(hy + __wli_header_h(ui, w))) {
                struct Vec<int> cols = Vec::new();
                __wli_col_widths(ui, w, w.w - __w_ipx(ui, 2), &cols);
                float cx2 = (float)vx - w.scrollX;
                unsigned long ci = 0UL;
                int near = 0;
                while (ci < cols.len) {
                    cx2 = cx2 + (float)*cols.at(ci);
                    float d = ex - cx2;
                    if (d > -4.0f * ui.scale && d < 4.0f * ui.scale) { near = 1; }
                    ci = ci + 1UL;
                }
                cols.free();
                if (near) { __w_set_cursor(ui, GUI_CURSOR_RESIZE_EW); }
            }
        }
        if (k == WIDGET_POPUPLIST) { treeRows.free(); return 1; }
        treeRows.free();
        return 0;
    }
    if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT) {
        if (k == WIDGET_TABLE) {
            int hy = w.y + __w_ipx(ui, 1);
            if (ey >= (float)hy && ey < (float)(hy + __wli_header_h(ui, w))) {
                struct Vec<int> cols = Vec::new();
                __wli_col_widths(ui, w, w.w - __w_ipx(ui, 2), &cols);
                float cx2 = (float)vx - w.scrollX;
                unsigned long ci = 0UL;
                while (ci < cols.len) {
                    float left = cx2;
                    cx2 = cx2 + (float)*cols.at(ci);
                    float d = ex - cx2;
                    if (d > -4.0f * ui.scale && d < 4.0f * ui.scale) {
                        // freeze every column's width, then drag this one
                        unsigned long cj = 0UL;
                        while (cj < cols.len) {
                            &WidgetColumn cd = w.tableCols.at(cj);
                            cd.width = (int)((float)*cols.at(cj) / ui.scale);
                            cj = cj + 1UL;
                        }
                        w.dragPart = GUI_DRAG_COLRESIZE + (int)ci;
                        w.dragX = left;
                        break;
                    }
                    ci = ci + 1UL;
                }
                cols.free();
                treeRows.free();
                return 1;
            }
        }
        r = __wli_row_at(ui, w, ex, ey, count);
        if (k == WIDGET_POPUPLIST) {
            if (r >= 0) { w.hotIndex = r; w.pressed = 1; }
            treeRows.free();
            return 1;
        }
        if (r >= 0) {
            if (k == WIDGET_TREE) {
                int node = *treeRows.at((unsigned long)r);
                &WidgetTreeNode nd = w.nodes.at((unsigned long)node);
                float ind = (float)nd.depth * __w_px(ui, 16.0f);
                float tx = (float)vx + __w_px(ui, 4.0f) + ind;
                int onArrow = (ex >= tx && ex < tx + __w_px(ui, 18.0f)) ? 1 : 0;
                if (__wli_tree_has_children(w, node) && (onArrow || ev.clicks == 2)) {
                    nd.expanded = nd.expanded ? 0 : 1;
                    ui.dirty = 1;
                    if (onArrow) { treeRows.free(); return 1; }
                }
            }
            __wli_select_row(ui, w, r, &treeRows, count);
            if (ev.clicks == 2) { __w_fire_activate(w); }
        }
        treeRows.free();
        return 1;
    }
    if (ev.kind == GUI_EVENT_MOUSE_UP) {
        if (w.dragPart >= GUI_DRAG_COLRESIZE) { w.dragPart = 0; treeRows.free(); return 1; }
        if (k == WIDGET_POPUPLIST && w.pressed) {
            w.pressed = 0;
            r = __wli_row_at(ui, w, ex, ey, count);
            if (r >= 0) { __wli_pick(ui, w, r); }
            treeRows.free();
            return 1;
        }
        treeRows.free();
        return inside;
    }
    if (ev.kind == GUI_EVENT_KEY_DOWN && (w.focused || k == WIDGET_POPUPLIST)) {
        int kc = ev.keycode;
        float row = __wli_row(ui);
        int page = (int)((float)vh / row);
        if (page < 1) { page = 1; }
        if (k == WIDGET_POPUPLIST) {
            int h = w.hotIndex;
            if (kc == GUI_KEY_UP) { h = h - 1; }
            else if (kc == GUI_KEY_DOWN) { h = h + 1; }
            else if (kc == GUI_KEY_HOME) { h = 0; }
            else if (kc == GUI_KEY_END) { h = count - 1; }
            else if (kc == GUI_KEY_PAGE_UP) { h = h - page; }
            else if (kc == GUI_KEY_PAGE_DOWN) { h = h + page; }
            else if (kc == GUI_KEY_ENTER || kc == GUI_KEY_SPACE) { __wli_pick(ui, w, w.hotIndex); treeRows.free(); return 1; }
            else { treeRows.free(); return kc == GUI_KEY_TAB ? 0 : (kc == GUI_KEY_ESCAPE ? 0 : 1); }
            if (h < 0) { h = 0; }
            if (h >= count) { h = count - 1; }
            w.hotIndex = h;
            __wli_show_row(ui, w, h);
            treeRows.free();
            return 1;
        }
        int cur = __wli_display_row(w, &treeRows);
        int nr = cur;
        if (kc == GUI_KEY_UP) { nr = cur < 0 ? 0 : cur - 1; }
        else if (kc == GUI_KEY_DOWN) { nr = cur + 1; }
        else if (kc == GUI_KEY_HOME) { nr = 0; }
        else if (kc == GUI_KEY_END) { nr = count - 1; }
        else if (kc == GUI_KEY_PAGE_UP) { nr = cur - page; }
        else if (kc == GUI_KEY_PAGE_DOWN) { nr = cur + page; }
        else if (kc == GUI_KEY_ENTER) {
            if (w.selected >= 0) { __w_fire_activate(w); }
            treeRows.free();
            return 1;
        } else if (k == WIDGET_TREE && (kc == GUI_KEY_LEFT || kc == GUI_KEY_RIGHT) && w.selected >= 0) {
            &WidgetTreeNode nd = w.nodes.at((unsigned long)w.selected);
            int has = __wli_tree_has_children(w, w.selected);
            if (kc == GUI_KEY_RIGHT) {
                if (has && !nd.expanded) { nd.expanded = 1; ui.dirty = 1; }
                else if (has) { nr = cur + 1; }
            } else {
                if (has && nd.expanded) { nd.expanded = 0; ui.dirty = 1; }
                else if (nd.parent >= 0) {
                    w.selected = nd.parent;
                    __w_fire_change(w);
                    struct Vec<int> rows2 = Vec::new();
                    __wli_tree_rows(w, &rows2);
                    int pr = __wli_display_row(w, &rows2);
                    if (pr >= 0) { __wli_show_row(ui, w, pr); }
                    rows2.free();
                }
            }
            if (nr == cur) { treeRows.free(); return 1; }
            struct Vec<int> rows3 = Vec::new();
            int c3 = __wli_count(w, &rows3);
            __wli_select_row(ui, w, nr, &rows3, c3);
            rows3.free();
            treeRows.free();
            return 1;
        } else {
            treeRows.free();
            return 0;
        }
        __wli_select_row(ui, w, nr, &treeRows, count);
        treeRows.free();
        return 1;
    }
    treeRows.free();
    return 0;
}

} // namespace std
