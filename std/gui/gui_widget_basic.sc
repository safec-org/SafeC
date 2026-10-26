// SafeC Standard Library — GUI widgets: labels, buttons, check boxes,
// toggles, radio groups, sliders, progress bars, number spinners,
// separators, image views, canvases and color pickers (see gui_widget.h).
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_text.h>
#include <std/gui/gui_png.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_widget_internal.h>
#include <std/math.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

#define GUI_DRAG_SLIDER  1
#define GUI_DRAG_SV      2
#define GUI_DRAG_HUE     3
#define GUI_DRAG_UP      4
#define GUI_DRAG_DOWN    5

static int __wb_inside(const &Widget w, float x, float y) {
    return (x >= (float)w.x && y >= (float)w.y && x < (float)(w.x + w.w) && y < (float)(w.y + w.h)) ? 1 : 0;
}

static float __wb_box(&GuiUi ui) { return __w_px(ui, 16.0f); }

static int __wb_vertical_sep(const &Widget w) {
    match (w.parent) {
        case some(p): { return (p.kind == WIDGET_HSTACK || p.kind == WIDGET_STATUSBAR) ? 1 : 0; }
        case null: { }
    }
    return w.orientation == WIDGET_VERTICAL ? 1 : 0;
}

// ── measuring ─────────────────────────────────────────────────────────────

static void __wb_text_block(&GuiUi ui, &Widget w, &int cw, &int ch) {
    // a label's text, newlines included
    struct GuiTextLayout l = GuiTextLayout::new();
    match (w.style.font) {
        case some(bf): {
            *cw = 0;
            unsafe { *cw = (int)ceil_f(__w_text_wn(ui, w, (const char*)w.text.data, w.text.len)); }
            *ch = (int)ceil_f(__w_line_h(ui, w));
            return;
        }
        case null: { }
    }
    unsafe { l.build_n_(__w_face(ui, w), __w_font_size(ui, w), (const char*)w.text.data, w.text.len, 0.0f, w.textAlign); }
    *cw = (int)ceil_f(l.width);
    *ch = (int)ceil_f(l.height);
    l.free();
}

void __wk_basic_measure(&GuiUi ui, &Widget w, &int cw, &int ch) {
    int k = w.kind;
    float lh = __w_line_h(ui, w);
    int ctl = __w_ipx(ui, ui.theme.controlHeight);
    if (k == WIDGET_LABEL) { __wb_text_block(ui, w, cw, ch); return; }
    if (k == WIDGET_BUTTON) {
        float tw = 0.0f;
        unsafe { tw = __w_text_w(ui, w, (const char*)w.text.data); }
        *cw = (int)ceil_f(tw);
        int inner = ctl - __w_pad(ui, w, 1) - __w_pad(ui, w, 3);
        *ch = (int)ceil_f(lh) > inner ? (int)ceil_f(lh) : inner;
        if ((w.style.setMask & WIDGET_STYLE_PADDING) != 0) { *ch = (int)ceil_f(lh); }
        return;
    }
    if (k == WIDGET_CHECKBOX || k == WIDGET_TOGGLE) {
        float tw = 0.0f;
        unsafe { tw = __w_text_w(ui, w, (const char*)w.text.data); }
        float mark = k == WIDGET_TOGGLE ? __w_px(ui, 34.0f) : __wb_box(ui);
        float gap = tw > 0.0f ? __w_px(ui, 8.0f) : 0.0f;
        *cw = (int)ceil_f(mark + gap + tw);
        float h = lh > __w_px(ui, 20.0f) ? lh : __w_px(ui, 20.0f);
        *ch = (int)ceil_f(h);
        return;
    }
    if (k == WIDGET_RADIO) {
        float row = __w_px(ui, (float)ui.theme.rowHeight);
        float gap = __w_px(ui, 8.0f);
        float total = 0.0f;
        float widest = 0.0f;
        unsigned long i = 0UL;
        while (i < w.items.len) {
            float tw = 0.0f;
            unsafe { tw = __w_text_w(ui, w, (const char*)w.items.at(i).data); }
            float iw = __wb_box(ui) + gap + tw;
            if (iw > widest) { widest = iw; }
            total = total + iw + (i > 0UL ? __w_px(ui, 16.0f) : 0.0f);
            i = i + 1UL;
        }
        if (w.orientation == WIDGET_HORIZONTAL) { *cw = (int)ceil_f(total); *ch = (int)ceil_f(row); }
        else { *cw = (int)ceil_f(widest); *ch = (int)ceil_f(row * (float)w.items.len); }
        return;
    }
    if (k == WIDGET_SLIDER) {
        if (w.orientation == WIDGET_VERTICAL) { *cw = __w_ipx(ui, 22); *ch = __w_ipx(ui, 160); }
        else { *cw = __w_ipx(ui, 160); *ch = __w_ipx(ui, 22); }
        return;
    }
    if (k == WIDGET_PROGRESS) { *cw = __w_ipx(ui, 160); *ch = __w_ipx(ui, 8); return; }
    if (k == WIDGET_SPINNER) {
        struct String s = String::new();
        __w_format_value(w, w.sliderMax, &s);
        float a = 0.0f;
        unsafe { a = __w_text_w(ui, w, (const char*)s.data); }
        __w_format_value(w, w.sliderMin, &s);
        float b = 0.0f;
        unsafe { b = __w_text_w(ui, w, (const char*)s.data); }
        s.free();
        float tw = a > b ? a : b;
        float minw = __w_px(ui, 48.0f);
        if (tw < minw) { tw = minw; }
        *cw = (int)ceil_f(tw + __w_px(ui, 22.0f));
        *ch = ctl - __w_pad(ui, w, 1) - __w_pad(ui, w, 3);
        return;
    }
    if (k == WIDGET_SEPARATOR) {
        int t = __w_ipx(ui, 9);
        if (__wb_vertical_sep(w)) { *cw = t; *ch = 1; } else { *cw = 1; *ch = t; }
        return;
    }
    if (k == WIDGET_IMAGE) {
        match (w.image) {
            case some(img): { *cw = __w_ipx(ui, img.width); *ch = __w_ipx(ui, img.height); }
            case null: { *cw = __w_ipx(ui, 64); *ch = __w_ipx(ui, 64); }
        }
        return;
    }
    if (k == WIDGET_CANVAS) { *cw = __w_ipx(ui, 100); *ch = __w_ipx(ui, 100); return; }
    if (k == WIDGET_COLOR) { *cw = __w_ipx(ui, 220); *ch = __w_ipx(ui, 170); return; }
    *cw = 0;
    *ch = 0;
}

void __wk_basic_arrange(&GuiUi ui, &Widget w) {
    // children of basic widgets (rare) share the content box
    if (w.children.len == 0UL) { return; }
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    unsigned long i = 0UL;
    while (i < w.children.len) { __w_arrange(ui, *w.children.at(i), cx, cy, cw, ch); i = i + 1UL; }
}

// ── geometry shared by paint and events ───────────────────────────────────

// Slider track: start, length along the axis, and the thumb radius.
static void __wb_slider_track(&GuiUi ui, const &Widget w, &float t0, &float tlen, &float rad) {
    float r = __w_px(ui, 8.0f);
    *rad = r;
    if (w.orientation == WIDGET_VERTICAL) {
        *t0 = (float)w.y + r;
        *tlen = (float)w.h - 2.0f * r;
    } else {
        *t0 = (float)w.x + r;
        *tlen = (float)w.w - 2.0f * r;
    }
    if (*tlen < 1.0f) { *tlen = 1.0f; }
}

static double __wb_frac(const &Widget w) {
    double range = w.sliderMax - w.sliderMin;
    if (range == 0.0) { return 0.0; }
    double f = (w.sliderValue - w.sliderMin) / range;
    if (f < 0.0) { f = 0.0; }
    if (f > 1.0) { f = 1.0; }
    return f;
}

// The radio item under (x, y), or -1.
static int __wb_radio_at(&GuiUi ui, &Widget w, float x, float y) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    float row = __w_px(ui, (float)ui.theme.rowHeight);
    if (w.orientation == WIDGET_HORIZONTAL) {
        float px = (float)cx;
        float gap = __w_px(ui, 8.0f);
        unsigned long i = 0UL;
        while (i < w.items.len) {
            float tw = 0.0f;
            unsafe { tw = __w_text_w(ui, w, (const char*)w.items.at(i).data); }
            float iw = __wb_box(ui) + gap + tw;
            if (x >= px && x < px + iw && y >= (float)cy && y < (float)cy + row) { return (int)i; }
            px = px + iw + __w_px(ui, 16.0f);
            i = i + 1UL;
        }
        return -1;
    }
    if (y < (float)cy || x < (float)cx || x >= (float)(cx + cw)) { return -1; }
    int idx = (int)((y - (float)cy) / row);
    if (idx < 0 || (unsigned long)idx >= w.items.len) { return -1; }
    return idx;
}

// Color picker parts: the saturation/value square and the hue bar.
static void __wb_color_parts(&GuiUi ui, const &Widget w, &float sx, &float sy, &float ss, &float hx, &float hw,
                             &float sw) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    float swatch = __w_px(ui, 26.0f);
    float side = (float)ch - swatch - __w_px(ui, 8.0f);
    float bar = __w_px(ui, 18.0f);
    float gap = __w_px(ui, 10.0f);
    if (side > (float)cw - bar - gap) { side = (float)cw - bar - gap; }
    if (side < 10.0f) { side = 10.0f; }
    *sx = (float)cx;
    *sy = (float)cy;
    *ss = side;
    *hx = (float)cx + side + gap;
    *hw = bar;
    *sw = swatch;
}

// ── spinner value / text ──────────────────────────────────────────────────

static void __wb_spinner_set(&Widget w, double v, int fire) {
    double nv = __w_clamp_value(w, v);
    int changed = nv != w.sliderValue ? 1 : 0;
    w.sliderValue = nv;
    __w_format_value(w, nv, &w.text);
    w.caret = w.text.len;
    w.anchor = w.caret;
    if (changed && fire) { __w_fire_change(w); }
}

static double __wb_parse(const &String s, &int ok) {
    // a decimal number: [-+]digits[.digits]
    double v = 0.0;
    double sign = 1.0;
    unsigned long i = 0UL;
    int digits = 0;
    unsafe {
        while (i < s.len && s.data[i] == ' ') { i = i + 1UL; }
        if (i < s.len && (s.data[i] == '-' || s.data[i] == '+')) { if (s.data[i] == '-') { sign = -1.0; } i = i + 1UL; }
        while (i < s.len && s.data[i] >= '0' && s.data[i] <= '9') { v = v * 10.0 + (double)(s.data[i] - '0'); i = i + 1UL; digits = digits + 1; }
        if (i < s.len && (s.data[i] == '.' || s.data[i] == ',')) {
            i = i + 1UL;
            double f = 0.1;
            while (i < s.len && s.data[i] >= '0' && s.data[i] <= '9') { v = v + (double)(s.data[i] - '0') * f; f = f * 0.1; i = i + 1UL; digits = digits + 1; }
        }
        while (i < s.len && s.data[i] == ' ') { i = i + 1UL; }
    }
    *ok = (digits > 0 && i == s.len) ? 1 : 0;
    return v * sign;
}

void __w_spinner_commit(&Widget w) {
    int ok = 0;
    double v = __wb_parse(&w.text, &ok);
    if (ok) { __wb_spinner_set(w, v, 1); }
    else { __wb_spinner_set(w, w.sliderValue, 0); }
}

// ── painting ──────────────────────────────────────────────────────────────

static struct GuiColor __wb_button_fill(&GuiUi ui, &Widget w) {
    struct GuiColor base = __w_bg_or(w, ui.theme.button);
    if (!__w_effectively_enabled(w)) { return base.mix(ui.theme.window, 0.5f); }
    int custom = (w.style.setMask & WIDGET_STYLE_BG) != 0 ? 1 : 0;
    if (w.pressed && w.hover) { return custom ? base.shade(-0.15f) : ui.theme.buttonPressed; }
    if (w.hover) { return custom ? base.shade(0.08f) : ui.theme.buttonHover; }
    return base;
}

static void __wb_paint_check(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float x = (float)w.x;
    float y = (float)w.y;
    float r = __w_px(ui, ui.theme.radius);
    float bw = __w_px(ui, ui.theme.borderWidth);
    int on = w.checked;
    int en = __w_effectively_enabled(w);
    float markW = w.kind == WIDGET_TOGGLE ? __w_px(ui, 34.0f) : __wb_box(ui);
    float markH = w.kind == WIDGET_TOGGLE ? __w_px(ui, 18.0f) : __wb_box(ui);
    float my = y + ((float)w.h - markH) * 0.5f;
    struct GuiColor accent = en ? ui.theme.accent : ui.theme.textDisabled;
    if (w.kind == WIDGET_TOGGLE) {
        struct GuiColor track = on ? accent : (ui.theme.dark ? ui.theme.border : ui.theme.borderStrong.mix(ui.theme.window, 0.35f));
        if (w.hover && en) { track = track.shade(ui.theme.dark ? 0.1f : -0.08f); }
        __w_box(c, x, my, markW, markH, markH * 0.5f, track, track, 0.0f);
        float kr = markH * 0.5f - __w_px(ui, 2.5f);
        float kx = on ? x + markW - markH * 0.5f : x + markH * 0.5f;
        c.set_fill_color(ui.theme.shadow);
        c.fill_circle(kx, my + markH * 0.5f + __w_px(ui, 0.8f), kr);
        c.set_fill_color(GuiColor::hex(0xFFFFFFU));
        c.fill_circle(kx, my + markH * 0.5f, kr);
        if (w.focused) { __w_focus_ring(ui, c, w, x, my, markW, markH, markH * 0.5f); }
    } else {
        if (on) {
            __w_box(c, x, my, markW, markH, r, accent, accent, 0.0f);
            __w_check_mark(c, x, my, markW, ui.theme.accentText, __w_px(ui, 2.0f));
        } else {
            struct GuiColor border = w.hover && en ? ui.theme.borderStrong : ui.theme.border;
            __w_box(c, x, my, markW, markH, r, ui.theme.input, border, bw);
        }
        if (w.focused) { __w_focus_ring(ui, c, w, x, my, markW, markH, r); }
    }
    if (w.text.len > 0UL) {
        float tx = x + markW + __w_px(ui, 8.0f);
        unsafe { __w_text_in(ui, c, w, (const char*)w.text.data, tx, y, (float)w.w - (tx - x), (float)w.h, GUI_TEXT_LEFT, __w_fg(ui, w)); }
    }
}

static void __wb_paint_radio(&GuiUi ui, &GuiCanvas c, &Widget w) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    float row = __w_px(ui, (float)ui.theme.rowHeight);
    float box = __wb_box(ui);
    float gap = __w_px(ui, 8.0f);
    float bw = __w_px(ui, ui.theme.borderWidth);
    int en = __w_effectively_enabled(w);
    float px = (float)cx;
    float py = (float)cy;
    unsigned long i = 0UL;
    while (i < w.items.len) {
        float tw = 0.0f;
        unsafe { tw = __w_text_w(ui, w, (const char*)w.items.at(i).data); }
        float ccx = px + box * 0.5f;
        float ccy = py + row * 0.5f;
        int sel = (int)i == w.selected ? 1 : 0;
        if (sel) {
            struct GuiColor a = en ? ui.theme.accent : ui.theme.textDisabled;
            c.set_fill_color(a);
            c.fill_circle(ccx, ccy, box * 0.5f);
            c.set_fill_color(ui.theme.accentText);
            c.fill_circle(ccx, ccy, box * 0.2f);
        } else {
            struct GuiColor border = (w.hotIndex == (int)i && en) ? ui.theme.borderStrong : ui.theme.border;
            c.set_fill_color(border);
            c.fill_circle(ccx, ccy, box * 0.5f);
            c.set_fill_color(ui.theme.input);
            c.fill_circle(ccx, ccy, box * 0.5f - bw);
        }
        if (w.focused && (sel || (w.selected < 0 && i == 0UL))) {
            __w_ring(c, ccx - box * 0.5f - __w_px(ui, 2.0f), ccy - box * 0.5f - __w_px(ui, 2.0f),
                     box + __w_px(ui, 4.0f), box + __w_px(ui, 4.0f), box * 0.5f + __w_px(ui, 2.0f),
                     __w_px(ui, 2.0f), ui.theme.accent.with_alpha(0.75f));
        }
        unsafe { __w_text_in(ui, c, w, (const char*)w.items.at(i).data, px + box + gap, py, tw + 2.0f, row, GUI_TEXT_LEFT, __w_fg(ui, w)); }
        if (w.orientation == WIDGET_HORIZONTAL) { px = px + box + gap + tw + __w_px(ui, 16.0f); }
        else { py = py + row; }
        i = i + 1UL;
    }
}

static void __wb_paint_slider(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float t0 = 0.0f; float tlen = 0.0f; float rad = 0.0f;
    __wb_slider_track(ui, w, &t0, &tlen, &rad);
    float f = (float)__wb_frac(w);
    float th = __w_px(ui, 4.0f);
    int en = __w_effectively_enabled(w);
    struct GuiColor fill = en ? ui.theme.accent : ui.theme.textDisabled;
    struct GuiColor track = ui.theme.dark ? ui.theme.border : ui.theme.border.mix(ui.theme.window, 0.2f);
    float tx = 0.0f; float ty = 0.0f;
    if (w.orientation == WIDGET_VERTICAL) {
        float cx = (float)w.x + (float)w.w * 0.5f;
        __w_box(c, cx - th * 0.5f, t0, th, tlen, th * 0.5f, track, track, 0.0f);
        float pos = t0 + tlen * (1.0f - f);
        __w_box(c, cx - th * 0.5f, pos, th, t0 + tlen - pos, th * 0.5f, fill, fill, 0.0f);
        tx = cx; ty = pos;
    } else {
        float cy = (float)w.y + (float)w.h * 0.5f;
        __w_box(c, t0, cy - th * 0.5f, tlen, th, th * 0.5f, track, track, 0.0f);
        float pos = t0 + tlen * f;
        __w_box(c, t0, cy - th * 0.5f, pos - t0, th, th * 0.5f, fill, fill, 0.0f);
        tx = pos; ty = cy;
    }
    c.set_fill_color(ui.theme.shadow);
    c.fill_circle(tx, ty + __w_px(ui, 1.0f), rad);
    c.set_fill_color(fill);
    c.fill_circle(tx, ty, rad);
    c.set_fill_color(GuiColor::hex(0xFFFFFFU));
    float inner = (w.hover || w.dragPart == GUI_DRAG_SLIDER) && en ? rad * 0.45f : rad * 0.6f;
    c.fill_circle(tx, ty, inner);
    if (w.focused) {
        __w_ring(c, tx - rad - __w_px(ui, 2.0f), ty - rad - __w_px(ui, 2.0f), 2.0f * rad + __w_px(ui, 4.0f),
                 2.0f * rad + __w_px(ui, 4.0f), rad + __w_px(ui, 2.0f), __w_px(ui, 2.0f), ui.theme.accent.with_alpha(0.6f));
    }
}

static void __wb_paint_progress(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    float r = hh * 0.5f;
    struct GuiColor track = ui.theme.dark ? ui.theme.border : ui.theme.border.mix(ui.theme.window, 0.3f);
    __w_box(c, x, y, ww, hh, r, track, track, 0.0f);
    struct GuiColor fill = __w_effectively_enabled(w) ? ui.theme.accent : ui.theme.textDisabled;
    if ((w.style.setMask & WIDGET_STYLE_FG) != 0) { fill = w.style.fg; }
    if (w.indeterminate) {
        // a segment sweeping across, one pass per 1.4 s
        float phase = (float)(ui.nowMs / 1400.0);
        phase = phase - floor_f(phase);
        float seg = ww * 0.3f;
        float sx = x - seg + (ww + seg) * phase;
        c.save();
        c.clip_rect(x, y, ww, hh);
        c.set_fill_color(fill);
        c.fill_rounded_rect(sx, y, seg, hh, r);
        c.restore();
        return;
    }
    float f = (float)__wb_frac(w);
    if (f > 0.0f) {
        float fw = ww * f;
        if (fw < hh) {
            c.save();
            c.clip_rect(x, y, fw, hh);
            c.set_fill_color(fill);
            c.fill_rounded_rect(x, y, hh, hh, r);
            c.restore();
        } else {
            c.set_fill_color(fill);
            c.fill_rounded_rect(x, y, fw, hh, r);
        }
    }
}

static void __wb_paint_spinner(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    float r = __w_px(ui, ui.theme.radius);
    float bw = __w_px(ui, ui.theme.borderWidth);
    int en = __w_effectively_enabled(w);
    struct GuiColor border = w.focused ? ui.theme.accent : (w.hover && en ? ui.theme.borderStrong : ui.theme.border);
    __w_box(c, x, y, ww, hh, r, en ? ui.theme.input : ui.theme.input.mix(ui.theme.window, 0.5f), border, bw);
    float bwid = __w_px(ui, 20.0f);
    float bx = x + ww - bwid;
    // the text field part
    c.save();
    c.clip_rect(x + bw, y + bw, ww - bwid - bw, hh - 2.0f * bw);
    __wk_text_paint(ui, c, w);
    c.restore();
    // up / down buttons
    c.set_fill_color(ui.theme.border);
    c.fill_rect(bx, y + bw, bw, hh - 2.0f * bw);
    c.fill_rect(bx, y + hh * 0.5f - bw * 0.5f, bwid - bw, bw);
    struct GuiColor arrow = en ? ui.theme.textDim : ui.theme.textDisabled;
    if (w.hotIndex == 1 || w.dragPart == GUI_DRAG_UP) {
        c.set_fill_color(ui.theme.hover);
        c.fill_rect(bx + bw, y + bw, bwid - 2.0f * bw, hh * 0.5f - bw * 1.5f);
    }
    if (w.hotIndex == 2 || w.dragPart == GUI_DRAG_DOWN) {
        c.set_fill_color(ui.theme.hover);
        c.fill_rect(bx + bw, y + hh * 0.5f + bw * 0.5f, bwid - 2.0f * bw, hh * 0.5f - bw * 1.5f);
    }
    __w_chevron(c, bx + bwid * 0.5f, y + hh * 0.27f, __w_px(ui, 8.0f), 2, arrow, __w_px(ui, 1.5f));
    __w_chevron(c, bx + bwid * 0.5f, y + hh * 0.73f, __w_px(ui, 8.0f), 0, arrow, __w_px(ui, 1.5f));
    if (w.focused) { __w_focus_ring(ui, c, w, x, y, ww, hh, r); }
}

static void __wb_paint_image(&GuiUi ui, &GuiCanvas c, &Widget w) {
    int cx = 0; int cy = 0; int cw = 0; int ch = 0;
    __w_content(ui, w, &cx, &cy, &cw, &ch);
    if ((w.style.setMask & WIDGET_STYLE_BG) != 0) {
        c.set_fill_color(w.style.bg);
        c.fill_rect((float)w.x, (float)w.y, (float)w.w, (float)w.h);
    }
    match (w.image) {
        case null: {
            // a placeholder: a crossed box
            c.set_stroke_color(ui.theme.border);
            c.set_line_width(__w_px(ui, 1.0f));
            c.stroke_rect((float)cx + 0.5f, (float)cy + 0.5f, (float)cw - 1.0f, (float)ch - 1.0f);
            c.draw_line((float)cx, (float)cy, (float)(cx + cw), (float)(cy + ch));
            c.draw_line((float)(cx + cw), (float)cy, (float)cx, (float)(cy + ch));
        }
        case some(img): {
            if (img.width < 1 || img.height < 1 || cw < 1 || ch < 1) { return; }
            float iw = (float)img.width * ui.scale;
            float ih = (float)img.height * ui.scale;
            float dw = (float)cw;
            float dh = (float)ch;
            float sx = dw / iw;
            float sy = dh / ih;
            if (w.fit == WIDGET_FIT_CONTAIN) { float s = sx < sy ? sx : sy; dw = iw * s; dh = ih * s; }
            else if (w.fit == WIDGET_FIT_COVER) { float s = sx > sy ? sx : sy; dw = iw * s; dh = ih * s; }
            else if (w.fit == WIDGET_FIT_NONE) { dw = iw; dh = ih; }
            float dx = (float)cx + ((float)cw - dw) * 0.5f;
            float dy = (float)cy + ((float)ch - dh) * 0.5f;
            c.save();
            c.clip_rect((float)cx, (float)cy, (float)cw, (float)ch);
            c.draw_image_scaled(img, dx, dy, dw, dh);
            c.restore();
        }
    }
}

static void __wb_paint_color(&GuiUi ui, &GuiCanvas c, &Widget w) {
    float sx = 0.0f; float sy = 0.0f; float ss = 0.0f; float hx = 0.0f; float hw = 0.0f; float swh = 0.0f;
    __wb_color_parts(ui, w, &sx, &sy, &ss, &hx, &hw, &swh);
    float r = __w_px(ui, ui.theme.radius);
    // saturation (left -> right) and value (top -> bottom) over the hue
    c.save();
    c.scratchPath.clear();
    c.scratchPath.rounded_rect(sx, sy, ss, ss, r);
    c.clip_path(&c.scratchPath);
    c.set_fill_color(GuiColor::hsv(w.hue, 1.0f, 1.0f, 1.0f));
    c.fill_rect(sx, sy, ss, ss);
    struct GuiPaint white = GuiPaint::linear(sx, 0.0f, sx + ss, 0.0f);
    white.add_stop(0.0f, GuiColor::hexa(0xFFFFFFFFU));
    white.add_stop(1.0f, GuiColor::hexa(0xFFFFFF00U));
    c.set_fill_paint(&white);
    c.fill_rect(sx, sy, ss, ss);
    struct GuiPaint black = GuiPaint::linear(0.0f, sy, 0.0f, sy + ss);
    black.add_stop(0.0f, GuiColor::hexa(0x00000000U));
    black.add_stop(1.0f, GuiColor::hexa(0x000000FFU));
    c.set_fill_paint(&black);
    c.fill_rect(sx, sy, ss, ss);
    c.restore();
    __w_ring(c, sx, sy, ss, ss, r, __w_px(ui, 1.0f), ui.theme.border);
    float mx = sx + w.sat * ss;
    float my = sy + (1.0f - w.val) * ss;
    float mr = __w_px(ui, 6.0f);
    c.set_fill_color(GuiColor::hex(0xFFFFFFU));
    c.fill_circle(mx, my, mr);
    c.set_fill_color(GuiColor::rgb(w.color.r, w.color.g, w.color.b));
    c.fill_circle(mx, my, mr - __w_px(ui, 2.0f));
    // hue bar
    struct GuiPaint hue = GuiPaint::linear(0.0f, sy, 0.0f, sy + ss);
    int k = 0;
    while (k <= 6) {
        hue.add_stop((float)k / 6.0f, GuiColor::hsv((float)k * 60.0f, 1.0f, 1.0f, 1.0f));
        k = k + 1;
    }
    c.set_fill_paint(&hue);
    c.fill_rounded_rect(hx, sy, hw, ss, r);
    __w_ring(c, hx, sy, hw, ss, r, __w_px(ui, 1.0f), ui.theme.border);
    float hy = sy + w.hue / 360.0f * ss;
    __w_box(c, hx - __w_px(ui, 2.0f), hy - __w_px(ui, 3.0f), hw + __w_px(ui, 4.0f), __w_px(ui, 6.0f), __w_px(ui, 2.0f),
            GuiColor::hex(0xFFFFFFU), ui.theme.borderStrong, __w_px(ui, 1.0f));
    // swatch + hex
    float swy = sy + ss + __w_px(ui, 8.0f);
    __w_box(c, sx, swy, swh * 1.6f, swh, r, w.color, ui.theme.border, __w_px(ui, 1.0f));
    struct String hex = String::from("#");
    unsigned int v = ((unsigned int)w.color.r << 16) | ((unsigned int)w.color.g << 8) | (unsigned int)w.color.b;
    int sh = 20;
    while (sh >= 0) {
        unsigned int d = (v >> (unsigned int)sh) & 0xFU;
        hex.push_char((char)(d < 10U ? (unsigned int)'0' + d : (unsigned int)'A' + d - 10U));
        sh = sh - 4;
    }
    unsafe { __w_text_in(ui, c, w, (const char*)hex.data, sx + swh * 1.6f + __w_px(ui, 10.0f), swy, ss, swh, GUI_TEXT_LEFT, __w_fg(ui, w)); }
    hex.free();
    if (w.focused) { __w_focus_ring(ui, c, w, sx, sy, ss, ss, r); }
}

void __wk_basic_paint(&GuiUi ui, &GuiCanvas c, &Widget w) {
    int k = w.kind;
    float x = (float)w.x;
    float y = (float)w.y;
    float ww = (float)w.w;
    float hh = (float)w.h;
    if (k == WIDGET_LABEL) {
        if ((w.style.setMask & WIDGET_STYLE_BG) != 0) { c.set_fill_color(w.style.bg); c.fill_rect(x, y, ww, hh); }
        int cx = 0; int cy = 0; int cw = 0; int ch = 0;
        __w_content(ui, w, &cx, &cy, &cw, &ch);
        match (w.style.font) {
            case some(bf): {
                unsafe { __w_text(ui, c, w, (const char*)w.text.data, (float)cx, (float)cy, __w_fg(ui, w)); }
                return;
            }
            case null: { }
        }
        struct GuiTextLayout l = GuiTextLayout::new();
        unsafe {
            l.build_n_(__w_face(ui, w), __w_font_size(ui, w), (const char*)w.text.data, w.text.len,
                    w.wrap ? (float)cw : 0.0f, w.textAlign);
        }
        float tx = (float)cx;
        if (!w.wrap) {
            if (w.textAlign == GUI_TEXT_CENTER) { tx = (float)cx + ((float)cw - l.width) * 0.5f; }
            else if (w.textAlign == GUI_TEXT_RIGHT) { tx = (float)(cx + cw) - l.width; }
        }
        float ty = (float)cy + ((float)ch - l.height) * 0.5f;
        if (ty < (float)cy) { ty = (float)cy; }
        c.set_fill_color(__w_fg(ui, w));
        c.fill_text_layout(&l, floor_f(tx + 0.5f), floor_f(ty + 0.5f));
        l.free();
        return;
    }
    if (k == WIDGET_BUTTON) {
        float r = __w_px(ui, ui.theme.radius);
        float bw = (w.style.setMask & WIDGET_STYLE_BORDER) != 0 ? __w_px(ui, (float)w.style.borderWidth) : __w_px(ui, ui.theme.borderWidth);
        struct GuiColor border = __w_border_or(w, w.hover && __w_effectively_enabled(w) ? ui.theme.borderStrong : ui.theme.border);
        __w_box(c, x, y, ww, hh, r, __wb_button_fill(ui, w), border, bw);
        int cx = 0; int cy = 0; int cw = 0; int ch = 0;
        __w_content(ui, w, &cx, &cy, &cw, &ch);
        unsafe { __w_text_in(ui, c, w, (const char*)w.text.data, x, y, ww, hh, GUI_TEXT_CENTER, __w_fg(ui, w)); }
        if (w.focused) { __w_focus_ring(ui, c, w, x, y, ww, hh, r); }
        return;
    }
    if (k == WIDGET_CHECKBOX || k == WIDGET_TOGGLE) { __wb_paint_check(ui, c, w); return; }
    if (k == WIDGET_RADIO) { __wb_paint_radio(ui, c, w); return; }
    if (k == WIDGET_SLIDER) { __wb_paint_slider(ui, c, w); return; }
    if (k == WIDGET_PROGRESS) { __wb_paint_progress(ui, c, w); return; }
    if (k == WIDGET_SPINNER) { __wb_paint_spinner(ui, c, w); return; }
    if (k == WIDGET_SEPARATOR) {
        float t = __w_px(ui, 1.0f);
        c.set_fill_color(__w_border_or(w, ui.theme.border));
        if (__wb_vertical_sep(w)) { c.fill_rect(floor_f(x + ww * 0.5f), y, t, hh); }
        else { c.fill_rect(x, floor_f(y + hh * 0.5f), ww, t); }
        return;
    }
    if (k == WIDGET_IMAGE) { __wb_paint_image(ui, c, w); return; }
    if (k == WIDGET_CANVAS) {
        if ((w.style.setMask & WIDGET_STYLE_BG) != 0) { c.set_fill_color(w.style.bg); c.fill_rect(x, y, ww, hh); }
        unsafe {
            if (!w.onDraw.is_null()) {
                c.save();
                c.clip_rect(x, y, ww, hh);
                c.translate(x, y);
                c.scale(ui.scale, ui.scale);
                WidgetCanvasFn d = w.onDraw!;
                d(w, c, w.onDrawData);
                c.restore();
            }
        }
        return;
    }
    if (k == WIDGET_COLOR) { __wb_paint_color(ui, c, w); return; }
}

// ── events ────────────────────────────────────────────────────────────────

static void __wb_slider_to(&GuiUi ui, &Widget w, float px, float py) {
    float t0 = 0.0f; float tlen = 0.0f; float rad = 0.0f;
    __wb_slider_track(ui, w, &t0, &tlen, &rad);
    double f = 0.0;
    if (w.orientation == WIDGET_VERTICAL) { f = 1.0 - (double)((py - t0) / tlen); }
    else { f = (double)((px - t0) / tlen); }
    if (f < 0.0) { f = 0.0; }
    if (f > 1.0) { f = 1.0; }
    double v = __w_clamp_value(w, w.sliderMin + f * (w.sliderMax - w.sliderMin));
    if (v != w.sliderValue) {
        w.sliderValue = v;
        __w_fire_change(w);
    }
}

static void __wb_color_from(&GuiUi ui, &Widget w, float px, float py) {
    float sx = 0.0f; float sy = 0.0f; float ss = 0.0f; float hx = 0.0f; float hw = 0.0f; float swh = 0.0f;
    __wb_color_parts(ui, w, &sx, &sy, &ss, &hx, &hw, &swh);
    if (w.dragPart == GUI_DRAG_SV) {
        float s = (px - sx) / ss;
        float v = 1.0f - (py - sy) / ss;
        w.sat = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
        w.val = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    } else if (w.dragPart == GUI_DRAG_HUE) {
        float h = (py - sy) / ss * 360.0f;
        w.hue = h < 0.0f ? 0.0f : (h > 359.9f ? 359.9f : h);
    }
    struct GuiColor nc = GuiColor::hsv(w.hue, w.sat, w.val, (float)w.color.a / 255.0f);
    if (nc.r != w.color.r || nc.g != w.color.g || nc.b != w.color.b) {
        w.color = nc;
        __w_fire_change(w);
    } else {
        w.color = nc;
        ui.dirty = 1;
    }
}

int __wk_basic_event(&GuiUi ui, &Widget w, &GuiEvent ev) {
    int k = w.kind;
    float ex = (float)ev.x;
    float ey = (float)ev.y;
    int inside = __wb_inside(w, ex, ey);
    int down = (ev.kind == GUI_EVENT_MOUSE_DOWN && ev.button == GUI_BUTTON_LEFT) ? 1 : 0;
    int up = ev.kind == GUI_EVENT_MOUSE_UP ? 1 : 0;
    if (k == WIDGET_LABEL || k == WIDGET_SEPARATOR || k == WIDGET_PROGRESS) { return 0; }
    if (k == WIDGET_IMAGE || k == WIDGET_CANVAS) {
        int clickable = 0;
        clickable = !w.onClick.is_null() ? 1 : 0;
        if (!clickable) { return 0; }
    }
    if (k == WIDGET_BUTTON || k == WIDGET_CHECKBOX || k == WIDGET_TOGGLE || k == WIDGET_IMAGE || k == WIDGET_CANVAS) {
        if (down && inside) { w.pressed = 1; return 1; }
        if (up && w.pressed) {
            w.pressed = 0;
            if (inside) {
                if (k == WIDGET_CHECKBOX || k == WIDGET_TOGGLE) {
                    w.checked = w.checked ? 0 : 1;
                    __w_fire_change(w);
                }
                __w_fire_click(w);
            }
            return 1;
        }
        return 0;
    }
    if (k == WIDGET_RADIO) {
        if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
            int h = __wb_radio_at(ui, w, ex, ey);
            if (h != w.hotIndex) { w.hotIndex = h; ui.dirty = 1; }
            return 0;
        }
        if (down) {
            int h = __wb_radio_at(ui, w, ex, ey);
            if (h >= 0) {
                if (h != w.selected) { w.selected = h; __w_fire_change(w); }
                return 1;
            }
            return 0;
        }
        if (ev.kind == GUI_EVENT_KEY_DOWN && w.focused) {
            int n = (int)w.items.len;
            int kc = ev.keycode;
            if (n > 0 && (kc == GUI_KEY_UP || kc == GUI_KEY_DOWN || kc == GUI_KEY_LEFT || kc == GUI_KEY_RIGHT)) {
                int s = w.selected + ((kc == GUI_KEY_DOWN || kc == GUI_KEY_RIGHT) ? 1 : -1);
                if (w.selected < 0) { s = 0; }
                if (s < 0) { s = n - 1; }
                if (s >= n) { s = 0; }
                w.selected = s;
                __w_fire_change(w);
                return 1;
            }
            if (n > 0 && kc == GUI_KEY_SPACE && w.selected < 0) { w.selected = 0; __w_fire_change(w); return 1; }
        }
        return 0;
    }
    if (k == WIDGET_SLIDER) {
        if (down && inside) {
            w.dragPart = GUI_DRAG_SLIDER;
            w.pressed = 1;
            __wb_slider_to(ui, w, ex, ey);
            return 1;
        }
        if (ev.kind == GUI_EVENT_MOUSE_MOVE && w.dragPart == GUI_DRAG_SLIDER) { __wb_slider_to(ui, w, ex, ey); return 1; }
        if (up && w.dragPart == GUI_DRAG_SLIDER) {
            w.dragPart = 0;
            w.pressed = 0;
            __w_fire_click(w);
            return 1;
        }
        if (ev.kind == GUI_EVENT_SCROLL) {
            double st = w.step > 0.0 ? w.step : (w.sliderMax - w.sliderMin) * 0.05;
            double v = __w_clamp_value(w, w.sliderValue + (ev.scrollDy + ev.scrollDx) * st);
            if (v != w.sliderValue) { w.sliderValue = v; __w_fire_change(w); }
            return 1;
        }
        if (ev.kind == GUI_EVENT_KEY_DOWN && w.focused) {
            int kc = ev.keycode;
            double big = (w.sliderMax - w.sliderMin) * 0.2;
            double v = w.sliderValue;
            if (kc == GUI_KEY_HOME) { v = w.sliderMin; }
            else if (kc == GUI_KEY_END) { v = w.sliderMax; }
            else if (kc == GUI_KEY_PAGE_UP) { v = v + big; }
            else if (kc == GUI_KEY_PAGE_DOWN) { v = v - big; }
            else { return 0; }   // arrows: the default key action
            v = __w_clamp_value(w, v);
            if (v != w.sliderValue) { w.sliderValue = v; __w_fire_change(w); }
            return 1;
        }
        return 0;
    }
    if (k == WIDGET_SPINNER) {
        float bwid = __w_px(ui, 20.0f);
        float bx = (float)(w.x + w.w) - bwid;
        int onUp = (inside && ex >= bx && ey < (float)w.y + (float)w.h * 0.5f) ? 1 : 0;
        int onDown = (inside && ex >= bx && ey >= (float)w.y + (float)w.h * 0.5f) ? 1 : 0;
        if (ev.kind == GUI_EVENT_MOUSE_MOVE) {
            int hot = onUp ? 1 : (onDown ? 2 : -1);
            if (hot != w.hotIndex) { w.hotIndex = hot; ui.dirty = 1; }
            if (hot > 0) { __w_set_cursor(ui, GUI_CURSOR_ARROW); }
            if (w.dragPart == 0 && w.pressed) { return __wk_text_event(ui, w, ev); }
            return 0;
        }
        if (down && (onUp || onDown)) {
            __w_spinner_commit(w);
            w.dragPart = onUp ? GUI_DRAG_UP : GUI_DRAG_DOWN;
            __wb_spinner_set(w, w.sliderValue + (onUp ? w.step : -w.step), 1);
            return 1;
        }
        if (up && (w.dragPart == GUI_DRAG_UP || w.dragPart == GUI_DRAG_DOWN)) { w.dragPart = 0; return 1; }
        if (ev.kind == GUI_EVENT_SCROLL && (w.focused || w.hover)) {
            __w_spinner_commit(w);
            __wb_spinner_set(w, w.sliderValue + (ev.scrollDy > 0.0 ? w.step : -w.step), 1);
            return 1;
        }
        if (ev.kind == GUI_EVENT_KEY_DOWN && w.focused) {
            int kc = ev.keycode;
            if (kc == GUI_KEY_UP || kc == GUI_KEY_DOWN || kc == GUI_KEY_PAGE_UP || kc == GUI_KEY_PAGE_DOWN) {
                __w_spinner_commit(w);
                double st = (kc == GUI_KEY_PAGE_UP || kc == GUI_KEY_PAGE_DOWN) ? w.step * 10.0 : w.step;
                __wb_spinner_set(w, w.sliderValue + ((kc == GUI_KEY_UP || kc == GUI_KEY_PAGE_UP) ? st : -st), 1);
                return 1;
            }
            if (kc == GUI_KEY_ENTER) { __w_spinner_commit(w); w.select_all(); return 1; }
            if (kc == GUI_KEY_ESCAPE) { __wb_spinner_set(w, w.sliderValue, 0); return 0; }
        }
        if (ev.kind == GUI_EVENT_KEY_DOWN || ev.kind == GUI_EVENT_CHAR) {
            // only number characters reach the field
            unsigned int cp = ev.codepoint;
            if (cp != 0U && !((cp >= 48U && cp <= 57U) || cp == 45U || cp == 43U || cp == 46U || cp == 44U)) {
                if (ev.kind == GUI_EVENT_CHAR) { return 1; }
                ev.codepoint = 0U;
            }
        }
        return __wk_text_event(ui, w, ev);
    }
    if (k == WIDGET_COLOR) {
        float sx = 0.0f; float sy = 0.0f; float ss = 0.0f; float hx = 0.0f; float hw = 0.0f; float swh = 0.0f;
        __wb_color_parts(ui, w, &sx, &sy, &ss, &hx, &hw, &swh);
        if (down) {
            if (ex >= sx && ex < sx + ss && ey >= sy && ey < sy + ss) { w.dragPart = GUI_DRAG_SV; }
            else if (ex >= hx && ex < hx + hw && ey >= sy && ey < sy + ss) { w.dragPart = GUI_DRAG_HUE; }
            else { return inside; }
            __wb_color_from(ui, w, ex, ey);
            return 1;
        }
        if (ev.kind == GUI_EVENT_MOUSE_MOVE && (w.dragPart == GUI_DRAG_SV || w.dragPart == GUI_DRAG_HUE)) {
            __wb_color_from(ui, w, ex, ey);
            return 1;
        }
        if (up && (w.dragPart == GUI_DRAG_SV || w.dragPart == GUI_DRAG_HUE)) { w.dragPart = 0; return 1; }
        if (ev.kind == GUI_EVENT_KEY_DOWN && w.focused) {
            int kc = ev.keycode;
            float d = 0.02f;
            if (kc != GUI_KEY_LEFT && kc != GUI_KEY_RIGHT && kc != GUI_KEY_UP && kc != GUI_KEY_DOWN) { return 0; }
            if (ev.has_mod(GUI_MOD_SHIFT)) {
                w.hue = w.hue + ((kc == GUI_KEY_DOWN || kc == GUI_KEY_RIGHT) ? 5.0f : -5.0f);
                if (w.hue < 0.0f) { w.hue = w.hue + 360.0f; }
                if (w.hue >= 360.0f) { w.hue = w.hue - 360.0f; }
            } else if (kc == GUI_KEY_LEFT || kc == GUI_KEY_RIGHT) {
                w.sat = w.sat + (kc == GUI_KEY_RIGHT ? d : -d);
                if (w.sat < 0.0f) { w.sat = 0.0f; }
                if (w.sat > 1.0f) { w.sat = 1.0f; }
            } else {
                w.val = w.val + (kc == GUI_KEY_UP ? d : -d);
                if (w.val < 0.0f) { w.val = 0.0f; }
                if (w.val > 1.0f) { w.val = 1.0f; }
            }
            w.color = GuiColor::hsv(w.hue, w.sat, w.val, (float)w.color.a / 255.0f);
            __w_fire_change(w);
            return 1;
        }
        return 0;
    }
    return 0;
}

} // namespace std
