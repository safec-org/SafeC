// SafeC Standard Library — GUI widgets: single-line text inputs and
// multi-line text areas (and the editing field of number spinners): caret
// and selection, keyboard and mouse editing, word moves, the clipboard,
// password masking, horizontal / vertical scrolling (see gui_widget.h).
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

#define GUI_DRAG_SELECT 30

// ── text helpers ──────────────────────────────────────────────────────────

static int __wt_multiline(const &Widget w) { return w.kind == WIDGET_TEXTAREA ? 1 : 0; }

static unsigned long __wt_next(const &Widget w, unsigned long i) {
    if (i >= w.text.len) { return w.text.len; }
    unsigned long p = i;
    unsafe { gui_utf8_next((const char*)w.text.data, w.text.len, &p); }
    return p;
}

static unsigned long __wt_prev(const &Widget w, unsigned long i) {
    if (i == 0UL) { return 0UL; }
    unsafe { return gui_utf8_prev((const char*)w.text.data, i); }
}

static int __wt_is_word(unsigned int cp) {
    if (cp >= 128U) { return 1; }
    return ((cp >= 48U && cp <= 57U) || (cp >= 65U && cp <= 90U) || (cp >= 97U && cp <= 122U) || cp == 95U) ? 1 : 0;
}

static unsigned int __wt_cp_at(const &Widget w, unsigned long i) {
    if (i >= w.text.len) { return 0U; }
    unsigned long p = i;
    unsafe { return gui_utf8_next((const char*)w.text.data, w.text.len, &p); }
}

static unsigned long __wt_word_left(const &Widget w, unsigned long i) {
    // skip separators, then the word
    while (i > 0UL && !__wt_is_word(__wt_cp_at(w, __wt_prev(w, i)))) { i = __wt_prev(w, i); }
    while (i > 0UL && __wt_is_word(__wt_cp_at(w, __wt_prev(w, i)))) { i = __wt_prev(w, i); }
    return i;
}

static unsigned long __wt_word_right(const &Widget w, unsigned long i) {
    unsigned long n = w.text.len;
    while (i < n && !__wt_is_word(__wt_cp_at(w, i))) { i = __wt_next(w, i); }
    while (i < n && __wt_is_word(__wt_cp_at(w, i))) { i = __wt_next(w, i); }
    return i;
}

static int __wt_has_sel(const &Widget w) { return w.anchor != w.caret ? 1 : 0; }

static void __wt_delete_range(&Widget w, unsigned long a, unsigned long b) {
    if (b <= a) { return; }
    w.text.remove_range(a, b);
    w.caret = a;
    w.anchor = a;
}

static int __wt_delete_sel(&Widget w) {
    if (!__wt_has_sel(w)) { return 0; }
    __wt_delete_range(w, w.selection_start(), w.selection_end());
    return 1;
}

void __w_edit_insert(&Widget w, const &char s, unsigned long n, int multiline) {
    __wt_delete_sel(w);
    // single-line fields drop line breaks (pasted text)
    struct String clean = String::new();
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            char ch = s[i];
            if (!multiline && (ch == '\n' || ch == '\r')) { clean.push_char(' '); }
            else if (ch == '\r') { }
            else { clean.push_char(ch); }
            i = i + 1UL;
        }
        if (w.caret >= w.text.len) { w.text.push_n((const char*)clean.data, clean.len); }
        else { w.text.insert(w.caret, (const char*)clean.data); }
    }
    w.caret = w.caret + clean.len;
    w.anchor = w.caret;
    clean.free();
}

// The text as displayed: bullets for a password field.
static void __wt_display(const &Widget w, &String out) {
    out.clear();
    if (!w.password) {
        unsafe { out.push_n((const char*)w.text.data, w.text.len); }
        return;
    }
    unsigned long i = 0UL;
    while (i < w.text.len) {
        i = __wt_next(w, i);
        out.push("\xE2\x80\xA2");   // U+2022
    }
}

// Byte offsets between the text and its display form.
static unsigned long __wt_to_display(const &Widget w, unsigned long pos) {
    if (!w.password) { return pos; }
    unsigned long i = 0UL;
    unsigned long n = 0UL;
    while (i < pos && i < w.text.len) { i = __wt_next(w, i); n = n + 1UL; }
    return n * 3UL;
}

static unsigned long __wt_from_display(const &Widget w, unsigned long dpos) {
    if (!w.password) { return dpos; }
    unsigned long count = dpos / 3UL;
    unsigned long i = 0UL;
    while (count > 0UL && i < w.text.len) { i = __wt_next(w, i); count = count - 1UL; }
    return i;
}

// The field's text box (content box; spinners leave room for their buttons).
static void __wt_box(&GuiUi ui, const &Widget w, &int cx, &int cy, &int cw, &int ch) {
    __w_content(ui, w, cx, cy, cw, ch);
    if (w.kind == WIDGET_SPINNER) { *cw = *cw - __w_ipx(ui, 20); if (*cw < 0) { *cw = 0; } }
    if (w.kind == WIDGET_TEXTAREA && w.contentH > (float)*ch) { *cw = *cw - __w_ipx(ui, ui.theme.scrollbarSize); }
}

static void __wt_layout(&GuiUi ui, &Widget w, &GuiTextLayout l, &String disp) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __wt_box(ui, w, &cx, &cy, &cw, &ch);
    __wt_display(w, disp);
    float maxW = (__wt_multiline(w) && w.wrap) ? (float)cw : 0.0f;
    unsafe { l.build(__w_face(ui, w), __w_font_size(ui, w), (const char*)disp.data, disp.len, maxW, GUI_TEXT_LEFT); }
}

// Keeps the caret in view (horizontal scroll for single-line fields,
// vertical for text areas).
static void __wt_scroll_to_caret(&GuiUi ui, &Widget w) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    struct GuiTextLayout l = GuiTextLayout::new();
    struct String disp = String::new();
    __wt_layout(ui, w, &l, &disp);
    __wt_box(ui, w, &cx, &cy, &cw, &ch);
    float x = 0.0f; float top = 0.0f; float h = 0.0f;
    l.caret(__wt_to_display(w, w.caret), &x, &top, &h);
    if (__wt_multiline(w)) {
        w.contentH = l.height;
        w.contentW = (float)cw;
        __w_scroll_into_view(w, top, top + h, (float)ch);
        __w_scroll_clamp(w, (float)cw, (float)ch);
    } else {
        float pad = __w_px(ui, 2.0f);
        if (x - w.textScroll > (float)cw - pad) { w.textScroll = x - (float)cw + pad; }
        if (x - w.textScroll < 0.0f) { w.textScroll = x; }
        float maxs = l.width - (float)cw + pad;
        if (maxs < 0.0f) { maxs = 0.0f; }
        if (w.textScroll > maxs) { w.textScroll = maxs; }
        if (w.textScroll < 0.0f) { w.textScroll = 0.0f; }
    }
    l.free();
    disp.free();
}

// The byte offset under window position (x, y).
static unsigned long __wt_hit(&GuiUi ui, &Widget w, float x, float y) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __wt_box(ui, w, &cx, &cy, &cw, &ch);
    struct GuiTextLayout l = GuiTextLayout::new();
    struct String disp = String::new();
    __wt_layout(ui, w, &l, &disp);
    float lx = x - (float)cx + w.textScroll;
    float ly = 0.0f;
    if (__wt_multiline(w)) { ly = y - (float)cy + w.scrollY; }
    else { ly = l.lineHeight * 0.5f; }
    unsigned long pos = __wt_from_display(w, l.hit_test(lx, ly));
    l.free();
    disp.free();
    return pos;
}

// ── measure / paint ───────────────────────────────────────────────────────

void __wk_text_measure(&GuiUi ui, &Widget w, &int cw, &int ch) {
    float lh = __w_line_h(ui, w);
    if (w.kind == WIDGET_TEXTAREA) {
        *cw = __w_ipx(ui, 240);
        *ch = __w_ipx(ui, 120);
        return;
    }
    *cw = __w_ipx(ui, 160);
    int ctl = __w_ipx(ui, ui.theme.controlHeight) - __w_pad(ui, w, 1) - __w_pad(ui, w, 3);
    *ch = (int)ceil_f(lh) > ctl ? (int)ceil_f(lh) : ctl;
}

void __wk_text_paint(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    float r = __w_px(ui, ui.theme.radius);
    float bw = __w_px(ui, ui.theme.borderWidth);
    int en = __w_effectively_enabled(w);
    int multi = __wt_multiline(w);
    if (w.kind != WIDGET_SPINNER) {
        struct GuiColor border = w.focused ? ui.theme.accent : (w.hover && en ? ui.theme.borderStrong : ui.theme.border);
        struct GuiColor bg = __w_bg_or(w, ui.theme.input);
        if (!en || w.readOnly) { bg = bg.mix(ui.theme.window, 0.5f); }
        __w_box(c, x, y, ww, hh, r, bg, __w_border_or(w, border), w.focused ? bw * 1.5f : bw);
    }
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __wt_box(ui, w, &cx, &cy, &cw, &ch);
    struct GuiTextLayout l = GuiTextLayout::new();
    struct String disp = String::new();
    __wt_layout(ui, w, &l, &disp);
    if (multi) {
        w.contentH = l.height;
        w.contentW = (float)cw;
        __w_scroll_clamp(w, (float)cw, (float)ch);
    }
    float ox = (float)cx - (multi ? 0.0f : w.textScroll);
    float oy = (float)cy - (multi ? w.scrollY : 0.0f);
    if (!multi) { oy = (float)cy + ((float)ch - l.lineHeight) * 0.5f; }
    c.save();
    c.clip_rect((float)cx - bw, (float)cy, (float)cw + 2.0f * bw, (float)ch);
    if (w.text.len == 0UL && w.placeholder.len > 0UL && !w.focused) {
        unsafe { __w_text(ui, c, w, (const char*)w.placeholder.data, (float)cx, multi ? (float)cy : oy, ui.theme.textDim); }
    }
    // selection
    if (__wt_has_sel(w) && (w.focused || !multi)) {
        unsigned long a = __wt_to_display(w, w.selection_start());
        unsigned long b = __wt_to_display(w, w.selection_end());
        struct GuiColor selc = w.focused ? ui.theme.selection.with_alpha(ui.theme.dark ? 0.6f : 0.35f) : ui.theme.hover;
        c.set_fill_color(selc);
        unsigned long li = l.line_of(a);
        unsigned long lb = l.line_of(b);
        while (li <= lb && li < l.lines.len) {
            &GuiTextLine ln = l.lines.at(li);
            float x0 = 0.0f; float x1 = 0.0f; float t = 0.0f; float h = 0.0f;
            unsigned long s0 = a > ln.byteStart ? a : ln.byteStart;
            unsigned long s1 = b < ln.byteEnd ? b : ln.byteEnd;
            l.caret(s0, &x0, &t, &h);
            l.caret(s1, &x1, &t, &h);
            if (li < lb) { x1 = x1 + __w_px(ui, 5.0f); }   // the line break
            c.fill_rect(ox + x0, oy + ln.top, x1 - x0, l.lineHeight);
            li = li + 1UL;
        }
    }
    c.set_fill_color(en ? __w_fg(ui, w) : ui.theme.textDisabled);
    c.fill_text_layout(&l, floor_f(ox + 0.5f), floor_f(oy + 0.5f));
    // caret (blinking)
    if (w.focused && !w.readOnly) {
        long long phase = (long long)(ui.nowMs / 530.0);
        if ((phase % 2LL) == 0LL || w.pressed) {
            float px = 0.0f; float top = 0.0f; float h = 0.0f;
            l.caret(__wt_to_display(w, w.caret), &px, &top, &h);
            c.set_fill_color(ui.theme.text);
            c.fill_rect(floor_f(ox + px), oy + top + __w_px(ui, 1.0f), __w_px(ui, 1.5f), h - __w_px(ui, 2.0f));
        }
    }
    c.restore();
    if (multi) {
        int bx = 0; int by = 0; int bww = 0; int bhh = 0;
        __w_content(ui, w, &bx, &by, &bww, &bhh);
        __w_scrollbars_paint(ui, c, w, (float)cx, (float)cy, (float)cw, (float)ch);
    }
    if (w.focused && w.kind != WIDGET_SPINNER && ui.theme.dark == 0) { }   // the accent border is the focus mark
    l.free();
    disp.free();
}

// ── editing ───────────────────────────────────────────────────────────────

static void __wt_move(&Widget w, unsigned long to, int extend) {
    w.caret = to;
    if (!extend) { w.anchor = to; }
}

// Line-wise caret movement in a text area: dy lines down (negative: up).
static unsigned long __wt_vertical(&GuiUi ui, &Widget w, int dy) {
    struct GuiTextLayout l = GuiTextLayout::new();
    struct String disp = String::new();
    __wt_layout(ui, w, &l, &disp);
    float x = 0.0f; float top = 0.0f; float h = 0.0f;
    l.caret(__wt_to_display(w, w.caret), &x, &top, &h);
    float ny = top + h * 0.5f + (float)dy * l.lineHeight;
    unsigned long pos = w.caret;
    if (ny < 0.0f) { pos = 0UL; }
    else if (ny >= l.height) { pos = w.text.len; }
    else { pos = __wt_from_display(w, l.hit_test(x, ny)); }
    l.free();
    disp.free();
    return pos;
}

static void __wt_line_bounds(&GuiUi ui, &Widget w, &unsigned long a, &unsigned long b) {
    struct GuiTextLayout l = GuiTextLayout::new();
    struct String disp = String::new();
    __wt_layout(ui, w, &l, &disp);
    unsigned long li = l.line_of(__wt_to_display(w, w.caret));
    *a = __wt_from_display(w, l.lines.at(li).byteStart);
    *b = __wt_from_display(w, l.lines.at(li).byteEnd);
    l.free();
    disp.free();
}

static void __wt_copy(&Widget w) {
    if (!__wt_has_sel(w) || w.password) { return; }
    struct String s = w.text.substr(w.selection_start(), w.selection_end());
    unsafe { gui_clipboard_set_text((const char*)s.data); }
    s.free();
}

int __w_edit_key(&GuiUi ui, &Widget w, &GuiEvent ev, int multiline) {
    if (ev.kind == GUI_EVENT_CHAR) {
        if (w.readOnly || ev.codepoint < 32U || ev.codepoint == 127U) { return 0; }
        char enc[4];
        int n = gui_utf8_encode(ev.codepoint, &enc[0]);
        __w_edit_insert(w, &enc[0], (unsigned long)n, multiline);
        __w_fire_change(w);
        __wt_scroll_to_caret(ui, w);
        return 1;
    }
    if (ev.kind != GUI_EVENT_KEY_DOWN) { return 0; }
    int k = ev.keycode;
    int shift = ev.has_mod(GUI_MOD_SHIFT);
    int primary = ev.primary_mod();
    // word moves: Ctrl elsewhere, Option on macOS
    int word = gui_primary_mod() == GUI_MOD_META ? ev.has_mod(GUI_MOD_ALT) : ev.has_mod(GUI_MOD_CTRL);
    int changed = 0;
    int moved = 1;
    if (k == GUI_KEY_LEFT) {
        if (__wt_has_sel(w) && !shift) { __wt_move(w, w.selection_start(), 0); }
        else { __wt_move(w, word ? __wt_word_left(w, w.caret) : __wt_prev(w, w.caret), shift); }
    } else if (k == GUI_KEY_RIGHT) {
        if (__wt_has_sel(w) && !shift) { __wt_move(w, w.selection_end(), 0); }
        else { __wt_move(w, word ? __wt_word_right(w, w.caret) : __wt_next(w, w.caret), shift); }
    } else if (k == GUI_KEY_HOME || k == GUI_KEY_END) {
        if (multiline && !primary) {
            unsigned long a = 0UL; unsigned long b = 0UL;
            __wt_line_bounds(ui, w, &a, &b);
            __wt_move(w, k == GUI_KEY_HOME ? a : b, shift);
        } else {
            __wt_move(w, k == GUI_KEY_HOME ? 0UL : w.text.len, shift);
        }
    } else if (multiline && (k == GUI_KEY_UP || k == GUI_KEY_DOWN)) {
        __wt_move(w, __wt_vertical(ui, w, k == GUI_KEY_UP ? -1 : 1), shift);
    } else if (multiline && (k == GUI_KEY_PAGE_UP || k == GUI_KEY_PAGE_DOWN)) {
        int cx = 0; int cy = 0; int cw = 0; int ch = 0;
        __wt_box(ui, w, &cx, &cy, &cw, &ch);
        int lines = (int)((float)ch / __w_line_h(ui, w));
        if (lines < 1) { lines = 1; }
        __wt_move(w, __wt_vertical(ui, w, k == GUI_KEY_PAGE_UP ? -lines : lines), shift);
    } else if (!multiline && (k == GUI_KEY_UP || k == GUI_KEY_DOWN)) {
        return 0;
    } else {
        moved = 0;
    }
    if (moved) {
        __wt_scroll_to_caret(ui, w);
        ui.dirty = 1;
        return 1;
    }
    if (primary && k == (int)'A') { w.select_all(); ui.dirty = 1; return 1; }
    if (primary && k == (int)'C') { __wt_copy(w); return 1; }
    if (primary && k == (int)'X') {
        __wt_copy(w);
        if (!w.readOnly && !w.password && __wt_delete_sel(w)) { changed = 1; }
        if (changed) { __w_fire_change(w); __wt_scroll_to_caret(ui, w); }
        return 1;
    }
    if (primary && k == (int)'V') {
        if (w.readOnly) { return 1; }
        struct String clip = String::new();
        if (gui_clipboard_get_text(&clip) && clip.len > 0UL) {
            unsafe { __w_edit_insert(w, (const char*)clip.data, clip.len, multiline); }
            __w_fire_change(w);
            __wt_scroll_to_caret(ui, w);
        }
        clip.free();
        return 1;
    }
    if (k == GUI_KEY_BACKSPACE || k == GUI_KEY_DELETE) {
        if (w.readOnly) { return 1; }
        if (!__wt_delete_sel(w)) {
            if (k == GUI_KEY_BACKSPACE && w.caret > 0UL) {
                unsigned long from = word ? __wt_word_left(w, w.caret) : __wt_prev(w, w.caret);
                __wt_delete_range(w, from, w.caret);
                changed = 1;
            } else if (k == GUI_KEY_DELETE && w.caret < w.text.len) {
                unsigned long to = word ? __wt_word_right(w, w.caret) : __wt_next(w, w.caret);
                __wt_delete_range(w, w.caret, to);
                changed = 1;
            }
        } else {
            changed = 1;
        }
        if (changed) { __w_fire_change(w); __wt_scroll_to_caret(ui, w); }
        return 1;
    }
    if (k == GUI_KEY_ENTER) {
        if (multiline && !w.readOnly) {
            __w_edit_insert(w, "\n", 1UL, 1);
            __w_fire_change(w);
            __wt_scroll_to_caret(ui, w);
            return 1;
        }
        if (!multiline && w.kind == WIDGET_TEXTINPUT) { __w_fire_activate(w); return 1; }
        return 0;
    }
    if (primary || ev.has_mod(GUI_MOD_CTRL)) { return 0; }   // other shortcuts: not ours
    if (ev.codepoint >= 32U && ev.codepoint != 127U) {
        if (w.readOnly) { return 1; }
        char enc[4];
        int n = gui_utf8_encode(ev.codepoint, &enc[0]);
        __w_edit_insert(w, &enc[0], (unsigned long)n, multiline);
        __w_fire_change(w);
        __wt_scroll_to_caret(ui, w);
        return 1;
    }
    return 0;
}

// ── events ────────────────────────────────────────────────────────────────

int __wk_text_event(&GuiUi ui, &Widget w, &GuiEvent ev) {
    int multi = __wt_multiline(w);
    float ex = (float)ev.x;
    float ey = (float)ev.y;
    if (multi) {
        int cx = 0; int cy = 0; int cw = 0; int ch = 0;
        __wt_box(ui, w, &cx, &cy, &cw, &ch);
        if (w.dragPart != GUI_DRAG_SELECT && __w_scrollbars_event(ui, w, ev, (float)cx, (float)cy, (float)cw, (float)ch)) {
            ui.dirty = 1;
            return 1;
        }
        if (ev.kind == GUI_EVENT_SCROLL) { return 0; }
    }
    if (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT) {
        unsigned long pos = __wt_hit(ui, w, ex, ey);
        if (ev.clicks >= 3) {
            if (multi) {
                w.caret = pos;
                unsigned long a = 0UL; unsigned long b = 0UL;
                __wt_line_bounds(ui, w, &a, &b);
                w.anchor = a;
                w.caret = b;
            } else {
                w.select_all();
            }
        } else if (ev.clicks == 2) {
            unsigned long a = pos;
            unsigned long b = pos;
            if (__wt_is_word(__wt_cp_at(w, pos)) || (pos > 0UL && __wt_is_word(__wt_cp_at(w, __wt_prev(w, pos))))) {
                while (a > 0UL && __wt_is_word(__wt_cp_at(w, __wt_prev(w, a)))) { a = __wt_prev(w, a); }
                while (b < w.text.len && __wt_is_word(__wt_cp_at(w, b))) { b = __wt_next(w, b); }
            } else {
                b = __wt_next(w, pos);
            }
            w.anchor = a;
            w.caret = b;
        } else {
            w.caret = pos;
            if (!ev.has_mod(GUI_MOD_SHIFT)) { w.anchor = pos; }
        }
        w.pressed = 1;
        w.dragPart = GUI_DRAG_SELECT;
        ui.dirty = 1;
        return 1;
    }
    if (ev.kind == GUI_EVENT_MOUSE_MOVE && w.dragPart == GUI_DRAG_SELECT && w.pressed) {
        w.caret = __wt_hit(ui, w, ex, ey);
        __wt_scroll_to_caret(ui, w);
        ui.dirty = 1;
        return 1;
    }
    if (ev.kind == GUI_EVENT_MOUSE_UP && w.dragPart == GUI_DRAG_SELECT) {
        w.pressed = 0;
        w.dragPart = 0;
        return 1;
    }
    if ((ev.kind == GUI_EVENT_KEY_DOWN || ev.kind == GUI_EVENT_CHAR) && w.focused) {
        return __w_edit_key(ui, w, ev, multi);
    }
    return 0;
}

} // namespace std
