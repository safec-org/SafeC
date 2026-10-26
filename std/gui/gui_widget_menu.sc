// SafeC Standard Library — GUI widgets: menu bars, menus (drop-down,
// submenus, context menus), menu items with check marks and keyboard
// shortcuts, and tooltips (see gui_widget.h).
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

static int __wm_is_sep(const &Widget w) {
    return (w.kind == WIDGET_MENUITEM && w.orientation == WIDGET_VERTICAL) ? 1 : 0;
}

static float __wm_row_h(&GuiUi ui, const &Widget item) {
    if (__wm_is_sep(item)) { return __w_px(ui, 9.0f); }
    return __w_px(ui, (float)ui.theme.rowHeight + 2.0f);
}

static float __wm_bar_title_w(&GuiUi ui, &Widget bar, &Widget menu) {
    float tw = 0.0f;
    unsafe { tw = __w_text_w(ui, bar, (const char*)menu.text.data); }
    return tw + __w_px(ui, 20.0f);
}

// The menubar entry under (x, y), or -1.
static int __wm_bar_at(&GuiUi ui, &Widget bar, float x, float y) {
    if (y < (float)bar.y || y >= (float)(bar.y + bar.h)) { return -1; }
    float px = (float)bar.x + __w_px(ui, 4.0f);
    unsigned long i = 0UL;
    while (i < bar.children.len) {
        &Widget m = *bar.children.at(i);
        if (m.visible) {
            float tw = __wm_bar_title_w(ui, bar, m);
            if (x >= px && x < px + tw) { return (int)i; }
            px = px + tw;
        }
        i = i + 1UL;
    }
    return -1;
}

static float __wm_bar_x(&GuiUi ui, &Widget bar, unsigned long index) {
    float px = (float)bar.x + __w_px(ui, 4.0f);
    unsigned long i = 0UL;
    while (i < index && i < bar.children.len) {
        &Widget m = *bar.children.at(i);
        if (m.visible) { px = px + __wm_bar_title_w(ui, bar, m); }
        i = i + 1UL;
    }
    return px;
}

// ── shortcuts ─────────────────────────────────────────────────────────────

static int __wm_word_eq(const &char s, unsigned long a, unsigned long b, const &char word) {
    unsafe {
        unsigned long n = 0UL;
        while (word[n] != '\0') { n = n + 1UL; }
        if (b - a != n) { return 0; }
        unsigned long i = 0UL;
        while (i < n) {
            char c1 = s[a + i];
            char c2 = word[i];
            if (c1 >= 'A' && c1 <= 'Z') { c1 = (char)(c1 + 32); }
            if (c2 >= 'A' && c2 <= 'Z') { c2 = (char)(c2 + 32); }
            if (c1 != c2) { return 0; }
            i = i + 1UL;
        }
        return 1;
    }
}

// Parses "Ctrl+Shift+S" into modifiers and a GUI_KEY_*; 0 if malformed.
static int __wm_parse_shortcut(const &String sc, &int mods, &int key) {
    *mods = 0;
    *key = GUI_KEY_UNKNOWN;
    if (sc.len == 0UL) { return 0; }
    unsafe {
        const char* s = (const char*)sc.data;
        unsigned long a = 0UL;
        while (a < sc.len) {
            unsigned long b = a;
            // a token ends at '+', unless the token is the '+' key itself
            while (b < sc.len && (s[b] != '+' || b == a)) { b = b + 1UL; }
            if (__wm_word_eq(s, a, b, "ctrl") || __wm_word_eq(s, a, b, "control") || __wm_word_eq(s, a, b, "cmd") ||
                __wm_word_eq(s, a, b, "command") || __wm_word_eq(s, a, b, "primary")) {
                *mods = *mods | gui_primary_mod();
            } else if (__wm_word_eq(s, a, b, "shift")) { *mods = *mods | GUI_MOD_SHIFT; }
            else if (__wm_word_eq(s, a, b, "alt") || __wm_word_eq(s, a, b, "option")) { *mods = *mods | GUI_MOD_ALT; }
            else if (__wm_word_eq(s, a, b, "meta") || __wm_word_eq(s, a, b, "super") || __wm_word_eq(s, a, b, "win")) { *mods = *mods | GUI_MOD_META; }
            else if (b - a == 1UL) {
                char c = s[a];
                if (c >= 'a' && c <= 'z') { c = (char)(c - 32); }
                if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) { *key = (int)c; }
                else { return 0; }
            }
            else if ((b - a == 2UL || b - a == 3UL) && (s[a] == 'F' || s[a] == 'f')) {
                int n = 0;
                unsigned long i = a + 1UL;
                while (i < b) { if (s[i] < '0' || s[i] > '9') { return 0; } n = n * 10 + (int)(s[i] - '0'); i = i + 1UL; }
                if (n < 1 || n > 12) { return 0; }
                *key = GUI_KEY_F1 + n - 1;
            }
            else if (__wm_word_eq(s, a, b, "delete") || __wm_word_eq(s, a, b, "del")) { *key = GUI_KEY_DELETE; }
            else if (__wm_word_eq(s, a, b, "backspace")) { *key = GUI_KEY_BACKSPACE; }
            else if (__wm_word_eq(s, a, b, "enter") || __wm_word_eq(s, a, b, "return")) { *key = GUI_KEY_ENTER; }
            else if (__wm_word_eq(s, a, b, "escape") || __wm_word_eq(s, a, b, "esc")) { *key = GUI_KEY_ESCAPE; }
            else if (__wm_word_eq(s, a, b, "tab")) { *key = GUI_KEY_TAB; }
            else if (__wm_word_eq(s, a, b, "space")) { *key = GUI_KEY_SPACE; }
            else if (__wm_word_eq(s, a, b, "up")) { *key = GUI_KEY_UP; }
            else if (__wm_word_eq(s, a, b, "down")) { *key = GUI_KEY_DOWN; }
            else if (__wm_word_eq(s, a, b, "left")) { *key = GUI_KEY_LEFT; }
            else if (__wm_word_eq(s, a, b, "right")) { *key = GUI_KEY_RIGHT; }
            else if (__wm_word_eq(s, a, b, "home")) { *key = GUI_KEY_HOME; }
            else if (__wm_word_eq(s, a, b, "end")) { *key = GUI_KEY_END; }
            else if (__wm_word_eq(s, a, b, "pageup") || __wm_word_eq(s, a, b, "pgup")) { *key = GUI_KEY_PAGE_UP; }
            else if (__wm_word_eq(s, a, b, "pagedown") || __wm_word_eq(s, a, b, "pgdn")) { *key = GUI_KEY_PAGE_DOWN; }
            else if (__wm_word_eq(s, a, b, "insert") || __wm_word_eq(s, a, b, "ins")) { *key = GUI_KEY_INSERT; }
            else { return 0; }
            a = b + 1UL;
        }
    }
    return *key != GUI_KEY_UNKNOWN ? 1 : 0;
}

// Activates a menu item: closes the menus, toggles a checkable item,
// fires on_change (checkable) and on_click.
static void __wm_activate(&GuiUi ui, &Widget item) {
    ui.close_popups();
    if (item.wrap) {
        item.checked = item.checked ? 0 : 1;
        __w_fire_change(item);
    }
    __w_fire_click(item);
    ui.dirty = 1;
}

static int __wm_holds_focus(&GuiUi ui, &Widget w) {
    ?&Widget cur = ui.focus;
    int guard = 0;
    while (guard < 256) {
        match (cur) {
            case some(f): {
                unsafe { if ((const unsigned char*)f == (const unsigned char*)w) { return 1; } }
                cur = f.parent;
            }
            case null: { return 0; }
        }
        guard = guard + 1;
    }
    return 0;
}

static int __wm_shortcut_in(&GuiUi ui, &Widget w, &GuiEvent ev, int mods, int depth) {
    if (depth > 64 || !w.visible) { return 0; }
    if (w.kind == WIDGET_MENUITEM && w.shortcut.len > 0UL && __w_effectively_enabled(w)) {
        int m = 0; int k = 0;
        if (__wm_parse_shortcut(&w.shortcut, &m, &k) && k == ev.keycode && m == mods) {
            __wm_activate(ui, w);
            return 1;
        }
    }
    unsigned long i = 0UL;
    while (i < w.children.len) {
        if (__wm_shortcut_in(ui, *w.children.at(i), ev, mods, depth + 1)) { return 1; }
        i = i + 1UL;
    }
    // a context menu's shortcuts apply while its widget (or something in
    // it) has the keyboard focus
    match (w.contextMenu) {
        case some(m): {
            if (__wm_holds_focus(ui, w) && __wm_shortcut_in(ui, m, ev, mods, depth + 1)) { return 1; }
        }
        case null: { }
    }
    return 0;
}

int __wk_menu_shortcut(&GuiUi ui, &Widget root, &GuiEvent ev) {
    if (ev.kind != GUI_EVENT_KEY_DOWN) { return 0; }
    int mods = ev.modifiers & (GUI_MOD_CTRL | GUI_MOD_SHIFT | GUI_MOD_ALT | GUI_MOD_META);
    // a plain key (no modifier) is only a shortcut for F-keys, Delete and
    // Insert (the focused widget saw the key first)
    if (mods == 0 && !(ev.keycode >= GUI_KEY_F1 && ev.keycode <= GUI_KEY_F12) &&
        ev.keycode != GUI_KEY_DELETE && ev.keycode != GUI_KEY_INSERT) { return 0; }
    return __wm_shortcut_in(ui, root, ev, mods, 0);
}

// ── measuring and layout ──────────────────────────────────────────────────

void __wk_menu_measure(&GuiUi ui, &Widget w, &int cw, &int ch) {
    int k = w.kind;
    if (k == WIDGET_MENUBAR) {
        float total = __w_px(ui, 8.0f);
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget m = *w.children.at(i);
            if (m.visible) { total = total + __wm_bar_title_w(ui, w, m); }
            i = i + 1UL;
        }
        *cw = (int)ceil_f(total);
        *ch = __w_ipx(ui, ui.theme.rowHeight + 4);
        return;
    }
    if (k == WIDGET_MENU) {
        float widest = 0.0f;
        float total = __w_px(ui, 8.0f);
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget it = *w.children.at(i);
            i = i + 1UL;
            if (!it.visible) { continue; }
            total = total + __wm_row_h(ui, it);
            if (__wm_is_sep(it)) { continue; }
            float tw = 0.0f;
            unsafe { tw = __w_text_w(ui, it, (const char*)it.text.data); }
            float sw = 0.0f;
            unsafe { if (it.shortcut.len > 0UL) { sw = __w_text_w(ui, it, (const char*)it.shortcut.data) + __w_px(ui, 28.0f); } }
            float arrow = it.kind == WIDGET_MENU ? __w_px(ui, 18.0f) : 0.0f;
            float iw = __w_px(ui, 28.0f) + tw + sw + arrow + __w_px(ui, 14.0f);
            if (iw > widest) { widest = iw; }
        }
        float minw = __w_px(ui, 140.0f);
        if (widest < minw) { widest = minw; }
        *cw = (int)ceil_f(widest);
        *ch = (int)ceil_f(total);
        return;
    }
    if (k == WIDGET_TOOLTIP) {
        float sz = ui.theme.smallFontSize;
        float saveFs = w.style.fontSize;
        w.style.fontSize = sz;
        struct GuiTextLayout l = GuiTextLayout::new();
        unsafe { l.build_n_(__w_face(ui, w), __w_font_size(ui, w), (const char*)w.text.data, w.text.len, __w_px(ui, 320.0f), GUI_TEXT_LEFT); }
        w.style.fontSize = saveFs;
        *cw = (int)ceil_f(l.width);
        *ch = (int)ceil_f(l.height);
        l.free();
        return;
    }
    // a menu item on its own (not in a menu): like a label
    float tw = 0.0f;
    unsafe { tw = __w_text_w(ui, w, (const char*)w.text.data); }
    *cw = (int)ceil_f(tw);
    *ch = (int)ceil_f(__w_line_h(ui, w));
}

void __wk_menu_arrange(&GuiUi ui, &Widget w) {
    if (w.kind == WIDGET_MENU) {
        float y = (float)w.y + __w_px(ui, 4.0f);
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget it = *w.children.at(i);
            if (it.visible) {
                float h = __wm_row_h(ui, it);
                it.x = w.x; it.y = (int)y; it.w = w.w; it.h = (int)h;
                y = y + h;
            }
            i = i + 1UL;
        }
    }
}

// ── painting ──────────────────────────────────────────────────────────────

void __wk_menu_paint(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    float r = __w_px(ui, ui.theme.radius);
    float bw = __w_px(ui, ui.theme.borderWidth);
    int k = w.kind;
    if (k == WIDGET_MENUBAR) {
        c.set_fill_color(__w_bg_or(w, ui.theme.header));
        c.fill_rect(x, y, ww, hh);
        c.set_fill_color(ui.theme.border);
        c.fill_rect(x, y + hh - bw, ww, bw);
        float px = x + __w_px(ui, 4.0f);
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget m = *w.children.at(i);
            i = i + 1UL;
            if (!m.visible) { continue; }
            float tw = __wm_bar_title_w(ui, w, m);
            int hot = (m.open || w.hotIndex == (int)(i - 1UL)) && __w_effectively_enabled(m) ? 1 : 0;
            struct GuiColor tc = __w_effectively_enabled(m) ? ui.theme.text : ui.theme.textDisabled;
            if (m.open) {
                c.set_fill_color(ui.theme.selection);
                c.fill_rounded_rect(px, y + __w_px(ui, 3.0f), tw, hh - __w_px(ui, 6.0f), r);
                tc = ui.theme.selectionText;
            } else if (hot) {
                c.set_fill_color(ui.theme.hover);
                c.fill_rounded_rect(px, y + __w_px(ui, 3.0f), tw, hh - __w_px(ui, 6.0f), r);
            }
            unsafe { __w_text_in(ui, c, w, (const char*)m.text.data, px, y, tw, hh, GUI_TEXT_CENTER, tc); }
            px = px + tw;
        }
        return;
    }
    if (k == WIDGET_MENU) {
        if (!w.open) { return; }   // a menu shows only as an open popup
        float sh = __w_px(ui, 4.0f);
        c.set_fill_color(ui.theme.shadow);
        c.fill_rounded_rect(x + sh * 0.5f, y + sh, ww, hh, r);
        __w_box(c, x, y, ww, hh, r, ui.theme.panel, ui.theme.borderStrong, bw);
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget it = *w.children.at(i);
            i = i + 1UL;
            if (!it.visible) { continue; }
            float iy = (float)it.y;
            float ih = (float)it.h;
            if (__wm_is_sep(it)) {
                c.set_fill_color(ui.theme.border);
                c.fill_rect(x + __w_px(ui, 8.0f), floor_f(iy + ih * 0.5f), ww - __w_px(ui, 16.0f), bw);
                continue;
            }
            int en = __w_effectively_enabled(it);
            int hot = (w.hotIndex == (int)(i - 1UL) && en) || (it.kind == WIDGET_MENU && it.open) ? 1 : 0;
            struct GuiColor tc = en ? ui.theme.text : ui.theme.textDisabled;
            struct GuiColor dim = en ? ui.theme.textDim : ui.theme.textDisabled;
            if (hot) {
                c.set_fill_color(ui.theme.selection);
                c.fill_rounded_rect(x + __w_px(ui, 4.0f), iy + __w_px(ui, 1.0f), ww - __w_px(ui, 8.0f), ih - __w_px(ui, 2.0f), r);
                tc = ui.theme.selectionText;
                dim = ui.theme.selectionText;
            }
            if (it.kind == WIDGET_MENUITEM && it.checked) {
                __w_check_mark(c, x + __w_px(ui, 8.0f), iy + (ih - __w_px(ui, 14.0f)) * 0.5f, __w_px(ui, 14.0f), tc, __w_px(ui, 1.6f));
            }
            unsafe { __w_text_in(ui, c, it, (const char*)it.text.data, x + __w_px(ui, 28.0f), iy, ww, ih, GUI_TEXT_LEFT, tc); }
            if (it.shortcut.len > 0UL) {
                unsafe { __w_text_in(ui, c, it, (const char*)it.shortcut.data, x, iy, ww - __w_px(ui, 14.0f), ih, GUI_TEXT_RIGHT, dim); }
            }
            if (it.kind == WIDGET_MENU) {
                __w_chevron(c, x + ww - __w_px(ui, 14.0f), iy + ih * 0.5f, __w_px(ui, 8.0f), 1, tc, __w_px(ui, 1.5f));
            }
        }
        return;
    }
    if (k == WIDGET_TOOLTIP) {
        c.set_fill_color(ui.theme.shadow);
        c.fill_rounded_rect(x + __w_px(ui, 1.0f), y + __w_px(ui, 2.0f), ww, hh, r);
        __w_box(c, x, y, ww, hh, r, ui.theme.tooltip, ui.theme.tooltip, 0.0f);
        int cx = 0; int cy = 0; int cw = 0; int ch = 0;
        __w_content(ui, w, &cx, &cy, &cw, &ch);
        float saveFs = w.style.fontSize;
        w.style.fontSize = ui.theme.smallFontSize;
        struct GuiTextLayout l = GuiTextLayout::new();
        unsafe { l.build_n_(__w_face(ui, w), __w_font_size(ui, w), (const char*)w.text.data, w.text.len, __w_px(ui, 320.0f), GUI_TEXT_LEFT); }
        c.set_fill_color(ui.theme.tooltipText);
        c.fill_text_layout(&l, (float)cx, (float)cy);
        l.free();
        w.style.fontSize = saveFs;
        return;
    }
    // a stray menu item: its text
    unsafe { __w_text(ui, c, w, (const char*)w.text.data, x, y, __w_fg(ui, w)); }
}

// ── opening menus ─────────────────────────────────────────────────────────

void __wk_menu_open(&GuiUi ui, &Widget menu) {
    match (menu.parent) {
        case some(p): {
            if (p.kind == WIDGET_MENUBAR) {
                unsigned long idx = 0UL;
                while (idx < p.children.len) {
                    unsafe { if ((const unsigned char*)*p.children.at(idx) == (const unsigned char*)menu) { break; } }
                    idx = idx + 1UL;
                }
                float px = __wm_bar_x(ui, p, idx);
                // the popup keeps the menu's parent link (for keyboard
                // navigation across the bar); popup() clears it, so restore
                ?&Widget keep = menu.parent;
                ui.popup(menu, px, (float)(p.y + p.h));
                menu.parent = keep;
                menu.owner = p;
                return;
            }
            if (p.kind == WIDGET_MENU) {
                ?&Widget keep = menu.parent;
                int mw = 0; int mh = 0;
                __w_measure(ui, menu, &mw, &mh);
                float sx = (float)(p.x + p.w) - __w_px(ui, 4.0f);
                int ww = 0;
                match (ui.win) { case some(win): { ww = win.width; } case null: { ww = ui.root.w; } }
                if (sx + (float)mw > (float)ww) { sx = (float)p.x - (float)mw + __w_px(ui, 4.0f); }
                ui.popup(menu, sx, (float)menu.y - __w_px(ui, 4.0f));
                menu.parent = keep;
                menu.owner = p;
                return;
            }
        }
        case null: { }
    }
    ui.popup(menu, (float)menu.x, (float)menu.y);
}

static int __wm_item_at(&Widget menu, float x, float y) {
    if (x < (float)menu.x || x >= (float)(menu.x + menu.w)) { return -1; }
    unsigned long i = 0UL;
    while (i < menu.children.len) {
        &Widget it = *menu.children.at(i);
        if (it.visible && y >= (float)it.y && y < (float)(it.y + it.h)) {
            return __wm_is_sep(it) ? -1 : (int)i;
        }
        i = i + 1UL;
    }
    return -1;
}

// Closes the submenus open above 'menu'.
static void __wm_close_above(&GuiUi ui, &Widget menu) {
    unsigned long i = 0UL;
    while (i < menu.children.len) {
        &Widget it = *menu.children.at(i);
        if (it.kind == WIDGET_MENU && it.open) { __w_popup_close_from(ui, it); }
        i = i + 1UL;
    }
}

static void __wm_hover_item(&GuiUi ui, &Widget menu, int idx) {
    if (idx == menu.hotIndex) { return; }
    menu.hotIndex = idx;
    __wm_close_above(ui, menu);
    if (idx >= 0) {
        &Widget it = *menu.children.at((unsigned long)idx);
        if (it.kind == WIDGET_MENU && __w_effectively_enabled(it) && it.children.len > 0UL) {
            it.hotIndex = -1;
            __wk_menu_open(ui, it);
        }
    }
    ui.dirty = 1;
}

static void __wm_step(&Widget menu, int dir) {
    int n = (int)menu.children.len;
    if (n == 0) { return; }
    int i = menu.hotIndex;
    int tries = 0;
    while (tries < n) {
        i = i + dir;
        if (i < 0) { i = n - 1; }
        if (i >= n) { i = 0; }
        &Widget it = *menu.children.at((unsigned long)i);
        if (it.visible && !__wm_is_sep(it) && __w_effectively_enabled(it)) { menu.hotIndex = i; return; }
        tries = tries + 1;
    }
}

// The menubar a (menubar) menu belongs to, if any.
static ?&Widget __wm_bar_of(&Widget menu) {
    match (menu.parent) {
        case some(p): { if (p.kind == WIDGET_MENUBAR) { return p; } }
        case null: { }
    }
    return null;
}

static void __wm_bar_switch(&GuiUi ui, &Widget bar, &Widget from, int dir) {
    unsigned long n = bar.children.len;
    unsigned long idx = 0UL;
    while (idx < n) {
        unsafe { if ((const unsigned char*)*bar.children.at(idx) == (const unsigned char*)from) { break; } }
        idx = idx + 1UL;
    }
    if (idx >= n) { return; }
    unsigned long tries = 0UL;
    unsigned long j = idx;
    while (tries < n) {
        j = dir > 0 ? (j + 1UL) % n : (j + n - 1UL) % n;
        &Widget m = *bar.children.at(j);
        if (m.visible && __w_effectively_enabled(m)) {
            ui.close_popups();
            m.hotIndex = -1;
            __wk_menu_open(ui, m);
            __wm_step(m, 1);
            return;
        }
        tries = tries + 1UL;
    }
}

// ── events ────────────────────────────────────────────────────────────────

int __wk_menu_event(&GuiUi ui, &Widget w, &GuiEvent ev) {
    int k = w.kind;
    float ex = (float)ev.x;
    float ey = (float)ev.y;
    if (k == WIDGET_MENUBAR) {
        int at = __wm_bar_at(ui, w, ex, ey);
        // which entry is open now
        int openIdx = -1;
        unsigned long i = 0UL;
        while (i < w.children.len) { if ((*w.children.at(i)).open) { openIdx = (int)i; } i = i + 1UL; }
        if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
            if (at != w.hotIndex) { w.hotIndex = at; ui.dirty = 1; }
            if (openIdx >= 0 && at >= 0 && at != openIdx) {
                &Widget m = *w.children.at((unsigned long)at);
                if (__w_effectively_enabled(m)) {
                    ui.close_popups();
                    m.hotIndex = -1;
                    __wk_menu_open(ui, m);
                }
            }
            // dragging from the bar into the open menu
            if (openIdx >= 0 && w.pressed) {
                &Widget m = *w.children.at((unsigned long)openIdx);
                if (m.open && ex >= (float)m.x && ey >= (float)m.y && ex < (float)(m.x + m.w) && ey < (float)(m.y + m.h)) {
                    __wm_hover_item(ui, m, __wm_item_at(m, ex, ey));
                }
            }
            return at >= 0 ? 1 : 0;
        }
        if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT) {
            if (at < 0) { return 0; }
            &Widget m = *w.children.at((unsigned long)at);
            if (at == openIdx) { ui.close_popups(); w.pressed = 0; return 1; }
            ui.close_popups();
            if (__w_effectively_enabled(m) && m.children.len > 0UL) {
                m.hotIndex = -1;
                __wk_menu_open(ui, m);
            } else if (__w_effectively_enabled(m)) {
                __w_fire_click(m);    // a bar entry with no items: a plain button
            }
            w.pressed = 1;
            return 1;
        }
        if (ev.kind == GUI_EVENT_MOUSE_UP) {
            int was = w.pressed;
            w.pressed = 0;
            if (was && openIdx >= 0) {
                // released over an item of the open menu: activate it
                &Widget m = *w.children.at((unsigned long)openIdx);
                int idx = __wm_item_at(m, ex, ey);
                if (idx >= 0) {
                    &Widget it = *m.children.at((unsigned long)idx);
                    if (it.kind == WIDGET_MENUITEM && __w_effectively_enabled(it)) { __wm_activate(ui, it); }
                }
            }
            return was;
        }
        return 0;
    }
    if (k == WIDGET_MENU) {
        if (!w.open) { return 0; }
        if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
            int inside = (ex >= (float)w.x && ey >= (float)w.y && ex < (float)(w.x + w.w) && ey < (float)(w.y + w.h)) ? 1 : 0;
            if (inside) { __wm_hover_item(ui, w, __wm_item_at(w, ex, ey)); }
            return 1;
        }
        if (ev.kind == GUI_EVENT_MOUSE_DOWN) { w.pressed = 1; return 1; }
        if (ev.kind == GUI_EVENT_MOUSE_UP) {
            int idx = __wm_item_at(w, ex, ey);
            w.pressed = 0;
            if (idx >= 0) {
                &Widget it = *w.children.at((unsigned long)idx);
                if (!__w_effectively_enabled(it)) { return 1; }
                if (it.kind == WIDGET_MENU) {
                    if (!it.open && it.children.len > 0UL) { __wk_menu_open(ui, it); }
                    return 1;
                }
                __wm_activate(ui, it);
            }
            return 1;
        }
        if (ev.kind == GUI_EVENT_KEY_DOWN) {
            int kc = ev.keycode;
            if (kc == GUI_KEY_UP || kc == GUI_KEY_DOWN) {
                __wm_step(w, kc == GUI_KEY_UP ? -1 : 1);
                __wm_close_above(ui, w);
                ui.dirty = 1;
                return 1;
            }
            if (kc == GUI_KEY_HOME || kc == GUI_KEY_END) {
                w.hotIndex = kc == GUI_KEY_HOME ? -1 : (int)w.children.len;
                __wm_step(w, kc == GUI_KEY_HOME ? 1 : -1);
                ui.dirty = 1;
                return 1;
            }
            if (kc == GUI_KEY_ENTER || kc == GUI_KEY_SPACE || kc == GUI_KEY_RIGHT) {
                if (w.hotIndex >= 0 && (unsigned long)w.hotIndex < w.children.len) {
                    &Widget it = *w.children.at((unsigned long)w.hotIndex);
                    if (it.kind == WIDGET_MENU && __w_effectively_enabled(it) && it.children.len > 0UL) {
                        it.hotIndex = -1;
                        __wk_menu_open(ui, it);
                        __wm_step(it, 1);
                        return 1;
                    }
                    if (kc != GUI_KEY_RIGHT && __w_effectively_enabled(it)) { __wm_activate(ui, it); return 1; }
                }
                if (kc == GUI_KEY_RIGHT) {
                    // from a menubar menu: the next one
                    ?&Widget top = w;
                    match (w.parent) { case some(p): { if (p.kind == WIDGET_MENU) { top = null; } } case null: { } }
                    match (top) {
                        case some(t): {
                            ?&Widget bar = __wm_bar_of(t);
                            match (bar) { case some(b): { __wm_bar_switch(ui, b, t, 1); } case null: { } }
                        }
                        case null: { }
                    }
                }
                return 1;
            }
            if (kc == GUI_KEY_LEFT) {
                match (w.parent) {
                    case some(p): {
                        if (p.kind == WIDGET_MENU) { __w_popup_close_from(ui, w); return 1; }
                        if (p.kind == WIDGET_MENUBAR) { __wm_bar_switch(ui, p, w, -1); return 1; }
                    }
                    case null: { }
                }
                return 1;
            }
            if (kc == GUI_KEY_TAB) { return 1; }
            return kc == GUI_KEY_ESCAPE ? 0 : 1;
        }
        return 1;
    }
    return 0;
}

} // namespace std
