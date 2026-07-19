// SafeC Standard Library — GUI: retained-mode widget tree implementation
// (see gui_widget.h).
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_draw.sc>
#include <std/gui/gui_font.h>
#include <std/gui/gui_widget.h>
#include <std/mem.sc>
#include <std/collections/vec.sc>
#include <std/collections/string.sc>

namespace std {

static struct GuiColor __widget_default_bg(int kind) {
    if (kind == WIDGET_BUTTON) { return gui_rgba(70U, 70U, 78U, 255U); }
    if (kind == WIDGET_TEXTINPUT) { return gui_rgba(30U, 30U, 34U, 255U); }
    if (kind == WIDGET_CHECKBOX) { return gui_rgba(0U, 0U, 0U, 0U); }
    if (kind == WIDGET_SLIDER) { return gui_rgba(50U, 50U, 56U, 255U); }
    return gui_rgba(0U, 0U, 0U, 0U);
}

static struct Widget* __widget_new(int kind) {
    struct Widget* w;
    unsafe { w = (struct Widget*)alloc((unsigned long)sizeof(struct Widget)); }

    unsafe {
        w->kind = kind;
        w->x = 0; w->y = 0; w->w = 0; w->h = 0;
        w->prefW = 0; w->prefH = 0;
        w->stretch = 0;
        w->alignment = WIDGET_ALIGN_START;

        w->style.bg = __widget_default_bg(kind);
        w->style.fg = gui_rgba(230U, 230U, 235U, 255U);
        w->style.border = gui_rgba(110U, 110U, 120U, 255U);
        w->style.borderWidth = 0;
        w->style.padding = 6;
        w->style.spacing = 6;
        w->style.fontScale = 1;
        w->style.font = (const struct GuiFont*)0;

        w->text = string_new();
        w->visible = 1;
        w->enabled = 1;
        w->checked = 0;
        w->sliderValue = 0.0;
        w->sliderMin = 0.0;
        w->sliderMax = 1.0;
        w->caret = 0UL;

        w->hover = 0;
        w->pressed = 0;
        w->focused = 0;

        w->onClick = (void*)0;
        w->onClickData = (void*)0;
        w->onChange = (void*)0;
        w->onChangeData = (void*)0;

        w->customDraw = (void*)0;
        w->customLayout = (void*)0;
        w->customEvent = (void*)0;
        w->userData = (void*)0;

        w->children = vec_new(8UL); // sizeof(struct Widget*)
        w->parent = (struct Widget*)0;
    }
    return w;
}

struct Widget* widget_vstack() {
    return __widget_new(WIDGET_VSTACK);
}
struct Widget* widget_hstack() {
    return __widget_new(WIDGET_HSTACK);
}
struct Widget* widget_spacer() {
    struct Widget* w = __widget_new(WIDGET_SPACER);
    unsafe { w->stretch = 1; return w; }
}
struct Widget* widget_button(const char* text) {
    struct Widget* w = __widget_new(WIDGET_BUTTON);
    unsafe {
        w->text.free();
        w->text = string_from(text);
        return w;
    }
}
struct Widget* widget_label(const char* text) {
    struct Widget* w = __widget_new(WIDGET_LABEL);
    unsafe {
        w->text.free();
        w->text = string_from(text);
        return w;
    }
}
struct Widget* widget_checkbox(const char* text, int checked) {
    struct Widget* w = __widget_new(WIDGET_CHECKBOX);
    unsafe {
        w->text.free();
        w->text = string_from(text);
        w->checked = checked;
        return w;
    }
}
struct Widget* widget_textinput(const char* placeholder) {
    struct Widget* w = __widget_new(WIDGET_TEXTINPUT);
    unsafe {
        w->text.free();
        w->text = string_from(placeholder);
        return w;
    }
}
struct Widget* widget_slider(double minValue, double maxValue, double value) {
    struct Widget* w = __widget_new(WIDGET_SLIDER);
    unsafe {
        w->sliderMin = minValue;
        w->sliderMax = maxValue;
        w->sliderValue = value;
        return w;
    }
}
struct Widget* widget_custom(WidgetDrawFn draw, WidgetLayoutFn layout,
                              WidgetCustomEventFn eventFn, void* userData) {
    struct Widget* w = __widget_new(WIDGET_CUSTOM);
    unsafe {
        w->customDraw = (void*)draw;
        w->customLayout = (void*)layout;
        w->customEvent = (void*)eventFn;
        w->userData = userData;
        return w;
    }
}

void widget_add_child(struct Widget* parent, struct Widget* child) {
    unsafe {
        child->parent = parent;
        parent->children.push((const void*)&child);
    }
}

// ── customization ────────────────────────────────────────────────────────────

struct Widget* widget_set_bg(struct Widget* w, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    unsafe { w->style.bg = gui_rgba(r, g, b, a); }
    return w;
}
struct Widget* widget_set_fg(struct Widget* w, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    unsafe { w->style.fg = gui_rgba(r, g, b, a); }
    return w;
}
struct Widget* widget_set_border(struct Widget* w, unsigned char r, unsigned char g, unsigned char b,
                                  unsigned char a, int width) {
    unsafe { w->style.border = gui_rgba(r, g, b, a); w->style.borderWidth = width; }
    return w;
}
struct Widget* widget_set_padding(struct Widget* w, int padding) {
    unsafe { w->style.padding = padding; }
    return w;
}
struct Widget* widget_set_spacing(struct Widget* w, int spacing) {
    unsafe { w->style.spacing = spacing; }
    return w;
}
struct Widget* widget_set_pref_size(struct Widget* w, int prefW, int prefH) {
    unsafe { w->prefW = prefW; w->prefH = prefH; }
    return w;
}
struct Widget* widget_set_stretch(struct Widget* w, int stretch) {
    unsafe { w->stretch = stretch; }
    return w;
}
struct Widget* widget_set_alignment(struct Widget* w, int alignment) {
    unsafe { w->alignment = alignment; }
    return w;
}
struct Widget* widget_set_font_scale(struct Widget* w, int scale) {
    unsafe { w->style.fontScale = scale; }
    return w;
}
struct Widget* widget_set_font(struct Widget* w, const struct GuiFont* font) {
    unsafe { w->style.font = font; }
    return w;
}
struct Widget* widget_set_visible(struct Widget* w, int visible) {
    unsafe { w->visible = visible; }
    return w;
}
struct Widget* widget_set_enabled(struct Widget* w, int enabled) {
    unsafe { w->enabled = enabled; }
    return w;
}
struct Widget* widget_set_text(struct Widget* w, const char* text) {
    unsafe {
        w->text.free();
        w->text = string_from(text);
    }
    return w;
}
struct Widget* widget_on_click(struct Widget* w, WidgetCallback cb, void* userData) {
    unsafe { w->onClick = (void*)cb; w->onClickData = userData; }
    return w;
}
struct Widget* widget_on_change(struct Widget* w, WidgetCallback cb, void* userData) {
    unsafe { w->onChange = (void*)cb; w->onChangeData = userData; }
    return w;
}

// ── measurement ───────────────────────────────────────────────────────────────
// See gui_widget.h's WidgetLayoutFn doc comment on the single-pass
// simplification this implements: a container's own natural size is 0
// unless explicitly given via widget_set_pref_size() or a stretch weight
// — auto-sizing-to-content only applies to leaf (text-bearing) widgets.

static int __widget_text_w(struct Widget* w) {
    int scale; const struct GuiFont* font; unsigned long len = 0UL;
    unsafe {
        scale = w->style.fontScale;
        font = w->style.font;
        while (w->text.data[len] != '\0') { len = len + 1UL; }
    }
    if (font != (const struct GuiFont*)0) {
        unsafe { return gui_text_width_ex((const char*)w->text.data, scale, font); }
    }
    unsafe { return gui_text_width((const char*)w->text.data, scale); }
}

static int __widget_text_h(struct Widget* w) {
    int scale; const struct GuiFont* font;
    unsafe { scale = w->style.fontScale; font = w->style.font; }
    if (font != (const struct GuiFont*)0) {
        unsafe { return gui_text_height_ex(scale, font); }
    }
    return gui_text_height(scale);
}

static int __widget_natural_w(struct Widget* w) {
    int kind; int prefW; int padding;
    unsafe { kind = w->kind; prefW = w->prefW; padding = w->style.padding; }
    if (prefW > 0) { return prefW; }
    if (kind == WIDGET_LABEL) { return __widget_text_w(w); }
    if (kind == WIDGET_BUTTON || kind == WIDGET_TEXTINPUT) { return __widget_text_w(w) + 2 * padding; }
    if (kind == WIDGET_CHECKBOX) { return __widget_text_w(w) + 2 * padding + 20; }
    if (kind == WIDGET_SLIDER) { return 120; }
    return 0;
}

static int __widget_natural_h(struct Widget* w) {
    int kind; int prefH; int padding;
    unsafe { kind = w->kind; prefH = w->prefH; padding = w->style.padding; }
    if (prefH > 0) { return prefH; }
    if (kind == WIDGET_LABEL) { return __widget_text_h(w); }
    if (kind == WIDGET_BUTTON || kind == WIDGET_TEXTINPUT || kind == WIDGET_CHECKBOX) {
        return __widget_text_h(w) + 2 * padding;
    }
    if (kind == WIDGET_SLIDER) { return 24; }
    return 0;
}

// ── layout ───────────────────────────────────────────────────────────────────

void gui_layout(struct Widget* root, int x, int y, int w, int h) {
    int kind;
    unsafe { kind = root->kind; }

    if (kind == WIDGET_CUSTOM) {
        WidgetLayoutFn layoutFn;
        unsafe { layoutFn = (WidgetLayoutFn)root->customLayout; }
        if (layoutFn != (WidgetLayoutFn)0) {
            unsafe { layoutFn(root, x, y, w, h); }
            return;
        }
    }

    unsafe { root->x = x; root->y = y; root->w = w; root->h = h; }

    if (kind != WIDGET_VSTACK && kind != WIDGET_HSTACK) { return; }

    int padding; int spacing;
    unsafe { padding = root->style.padding; spacing = root->style.spacing; }
    int innerX = x + padding;
    int innerY = y + padding;
    int innerW = w - 2 * padding; if (innerW < 0) { innerW = 0; }
    int innerH = h - 2 * padding; if (innerH < 0) { innerH = 0; }
    int mainSize = (kind == WIDGET_VSTACK) ? innerH : innerW;

    unsigned long n;
    unsafe { n = root->children.length(); }

    // Pass 1: natural/preferred main-axis size per child, total stretch weight.
    int usedFixed = 0;
    int totalStretch = 0;
    unsigned long i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe {
            struct Widget** slot = (struct Widget**)root->children.get_raw(i);
            child = *slot;
        }
        int stretch; unsafe { stretch = child->stretch; }
        if (stretch > 0) {
            totalStretch = totalStretch + stretch;
        } else {
            int natural = (kind == WIDGET_VSTACK) ? __widget_natural_h(child) : __widget_natural_w(child);
            usedFixed = usedFixed + natural;
        }
        if (i > 0UL) { usedFixed = usedFixed + spacing; }
        i = i + 1UL;
    }
    int remaining = mainSize - usedFixed; if (remaining < 0) { remaining = 0; }

    // Pass 2: position each child.
    int cursor = (kind == WIDGET_VSTACK) ? innerY : innerX;
    i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe {
            struct Widget** slot = (struct Widget**)root->children.get_raw(i);
            child = *slot;
        }
        int stretch; int alignment;
        unsafe { stretch = child->stretch; alignment = child->alignment; }

        int mainLen;
        if (stretch > 0 && totalStretch > 0) {
            mainLen = (remaining * stretch) / totalStretch;
        } else if (stretch > 0) {
            mainLen = 0;
        } else {
            mainLen = (kind == WIDGET_VSTACK) ? __widget_natural_h(child) : __widget_natural_w(child);
        }

        int crossBoxSize = (kind == WIDGET_VSTACK) ? innerW : innerH;
        int crossLen = (kind == WIDGET_VSTACK) ? __widget_natural_w(child) : __widget_natural_h(child);
        if (alignment == WIDGET_ALIGN_STRETCH || crossLen > crossBoxSize) { crossLen = crossBoxSize; }
        int crossPos;
        if (alignment == WIDGET_ALIGN_CENTER) { crossPos = (crossBoxSize - crossLen) / 2; }
        else if (alignment == WIDGET_ALIGN_END) { crossPos = crossBoxSize - crossLen; }
        else { crossPos = 0; }
        if (crossPos < 0) { crossPos = 0; }

        if (kind == WIDGET_VSTACK) {
            gui_layout(child, innerX + crossPos, cursor, crossLen, mainLen);
        } else {
            gui_layout(child, cursor, innerY + crossPos, mainLen, crossLen);
        }

        cursor = cursor + mainLen + spacing;
        i = i + 1UL;
    }
}

// ── render ───────────────────────────────────────────────────────────────────

static void __widget_draw_text_centered(struct GuiWindow* win, struct Widget* w, int boxX, int boxY, int boxW, int boxH) {
    int tw = __widget_text_w(w);
    int th = __widget_text_h(w);
    int tx = boxX + (boxW - tw) / 2; if (tx < boxX) { tx = boxX; }
    int ty = boxY + (boxH - th) / 2; if (ty < boxY) { ty = boxY; }
    int scale; struct GuiColor fg; const struct GuiFont* font;
    unsafe { scale = w->style.fontScale; fg = w->style.fg; font = w->style.font; }
    unsafe {
        if (font != (const struct GuiFont*)0) {
            gui_draw_text_ex(win, tx, ty, (const char*)w->text.data, fg, scale, font);
        } else {
            gui_draw_text(win, tx, ty, (const char*)w->text.data, fg, scale);
        }
    }
}

void gui_render(struct GuiWindow* win, struct Widget* root) {
    int visible; unsafe { visible = root->visible; }
    if (!visible) { return; }

    int kind; int x; int y; int w; int h;
    unsafe { kind = root->kind; x = root->x; y = root->y; w = root->w; h = root->h; }

    if (kind == WIDGET_CUSTOM) {
        WidgetDrawFn drawFn; unsafe { drawFn = (WidgetDrawFn)root->customDraw; }
        if (drawFn != (WidgetDrawFn)0) { unsafe { drawFn(win, root); } }
        unsigned long n; unsafe { n = root->children.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            struct Widget* child;
            unsafe { struct Widget** slot = (struct Widget**)root->children.get_raw(i); child = *slot; }
            gui_render(win, child);
            i = i + 1UL;
        }
        return;
    }

    struct GuiColor bg; int borderWidth; struct GuiColor border; int enabled; int pressed; int hover;
    unsafe {
        bg = root->style.bg; borderWidth = root->style.borderWidth; border = root->style.border;
        enabled = root->enabled; pressed = root->pressed; hover = root->hover;
    }

    if (kind == WIDGET_VSTACK || kind == WIDGET_HSTACK) {
        if (bg.a > 0U) { gui_fill_rect(win, x, y, w, h, bg); }
        if (borderWidth > 0) { gui_draw_rect_border(win, x, y, w, h, border, borderWidth); }
    } else if (kind == WIDGET_BUTTON) {
        struct GuiColor fill = bg;
        if (!enabled) { fill = gui_rgba(50U, 50U, 54U, 255U); }
        else if (pressed) {
            unsigned char pr = (unsigned char)(bg.r > 30U ? bg.r - 30U : 0U);
            unsigned char pg = (unsigned char)(bg.g > 30U ? bg.g - 30U : 0U);
            unsigned char pb = (unsigned char)(bg.b > 30U ? bg.b - 30U : 0U);
            fill = gui_rgba(pr, pg, pb, bg.a);
        }
        else if (hover) {
            unsigned char hr = (unsigned char)(bg.r + 15U > 255U ? 255U : bg.r + 15U);
            unsigned char hg = (unsigned char)(bg.g + 15U > 255U ? 255U : bg.g + 15U);
            unsigned char hb = (unsigned char)(bg.b + 15U > 255U ? 255U : bg.b + 15U);
            fill = gui_rgba(hr, hg, hb, bg.a);
        }
        gui_fill_rect(win, x, y, w, h, fill);
        gui_draw_rect_border(win, x, y, w, h, border, borderWidth > 0 ? borderWidth : 1);
        __widget_draw_text_centered(win, root, x, y, w, h);
    } else if (kind == WIDGET_LABEL) {
        struct GuiColor fg; int scale; const struct GuiFont* font;
        unsafe { fg = root->style.fg; scale = root->style.fontScale; font = root->style.font; }
        unsafe {
            if (font != (const struct GuiFont*)0) {
                gui_draw_text_ex(win, x, y, (const char*)root->text.data, fg, scale, font);
            } else {
                gui_draw_text(win, x, y, (const char*)root->text.data, fg, scale);
            }
        }
    } else if (kind == WIDGET_CHECKBOX) {
        int boxSize = 16;
        int boxY = y + (h - boxSize) / 2;
        struct GuiColor boxFill = enabled ? gui_rgba(30U, 30U, 34U, 255U) : gui_rgba(45U, 45U, 48U, 255U);
        gui_fill_rect(win, x, boxY, boxSize, boxSize, boxFill);
        gui_draw_rect_border(win, x, boxY, boxSize, boxSize, border, 1);
        int checked; unsafe { checked = root->checked; }
        if (checked) {
            gui_fill_rect(win, x + 3, boxY + 3, boxSize - 6, boxSize - 6, gui_rgba(90U, 170U, 255U, 255U));
        }
        struct GuiColor fg; int scale; const struct GuiFont* font;
        unsafe { fg = root->style.fg; scale = root->style.fontScale; font = root->style.font; }
        int textY = y + (h - __widget_text_h(root)) / 2;
        unsafe {
            if (font != (const struct GuiFont*)0) {
                gui_draw_text_ex(win, x + boxSize + 6, textY, (const char*)root->text.data, fg, scale, font);
            } else {
                gui_draw_text(win, x + boxSize + 6, textY, (const char*)root->text.data, fg, scale);
            }
        }
    } else if (kind == WIDGET_TEXTINPUT) {
        int focused; unsafe { focused = root->focused; }
        gui_fill_rect(win, x, y, w, h, bg);
        gui_draw_rect_border(win, x, y, w, h, focused ? gui_rgba(90U, 170U, 255U, 255U) : border, borderWidth > 0 ? borderWidth : 1);
        int padding; unsafe { padding = root->style.padding; }
        struct GuiColor fg; int scale; const struct GuiFont* font;
        unsafe { fg = root->style.fg; scale = root->style.fontScale; font = root->style.font; }
        int textY = y + (h - __widget_text_h(root)) / 2;
        unsafe {
            if (font != (const struct GuiFont*)0) {
                gui_draw_text_ex(win, x + padding, textY, (const char*)root->text.data, fg, scale, font);
            } else {
                gui_draw_text(win, x + padding, textY, (const char*)root->text.data, fg, scale);
            }
        }
    } else if (kind == WIDGET_SLIDER) {
        gui_fill_rect(win, x, y + h / 2 - 3, w, 6, bg);
        double val; double minV; double maxV;
        unsafe { val = root->sliderValue; minV = root->sliderMin; maxV = root->sliderMax; }
        double frac = (maxV > minV) ? (val - minV) / (maxV - minV) : 0.0;
        if (frac < 0.0) { frac = 0.0; } if (frac > 1.0) { frac = 1.0; }
        int handleX = x + (int)(frac * (double)(w - 12));
        gui_fill_rect(win, handleX, y, 12, h, gui_rgba(90U, 170U, 255U, 255U));
    }

    if (kind == WIDGET_VSTACK || kind == WIDGET_HSTACK) {
        unsigned long n; unsafe { n = root->children.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            struct Widget* child;
            unsafe { struct Widget** slot = (struct Widget**)root->children.get_raw(i); child = *slot; }
            gui_render(win, child);
            i = i + 1UL;
        }
    }
}

// ── event dispatch ───────────────────────────────────────────────────────────

static int __widget_point_in_box(struct Widget* w, double px, double py) {
    int x; int y; int ww; int hh;
    unsafe { x = w->x; y = w->y; ww = w->w; hh = w->h; }
    return (px >= (double)x && px < (double)(x + ww) && py >= (double)y && py < (double)(y + hh)) ? 1 : 0;
}

static void __widget_update_hover(struct Widget* w, struct GuiEvent* ev) {
    double evx; double evy;
    unsafe { evx = ev->x; evy = ev->y; }
    int inBox = __widget_point_in_box(w, evx, evy);
    unsafe { w->hover = inBox; }
    unsigned long n; unsafe { n = w->children.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe { struct Widget** slot = (struct Widget**)w->children.get_raw(i); child = *slot; }
        __widget_update_hover(child, ev);
        i = i + 1UL;
    }
}

static int __widget_is_interactive(struct Widget* w) {
    int kind; WidgetCallback onClick;
    unsafe { kind = w->kind; onClick = (WidgetCallback)w->onClick; }
    if (kind == WIDGET_BUTTON || kind == WIDGET_CHECKBOX || kind == WIDGET_TEXTINPUT ||
        kind == WIDGET_SLIDER || kind == WIDGET_CUSTOM) {
        return 1;
    }
    return (onClick != (WidgetCallback)0) ? 1 : 0;
}

static int __widget_dispatch_one(struct Widget* w, struct GuiEvent* ev) {
    int evKind; double evX; double evY; int evKeycode; unsigned int evCodepoint;
    unsafe {
        evKind = ev->kind; evX = ev->x; evY = ev->y;
        evKeycode = ev->keycode; evCodepoint = ev->codepoint;
    }

    // Children first (deepest/topmost interactive element wins).
    unsigned long n; unsafe { n = w->children.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe { struct Widget** slot = (struct Widget**)w->children.get_raw(i); child = *slot; }
        if (__widget_dispatch_one(child, ev)) { return 1; }
        i = i + 1UL;
    }

    int kind; int enabled; int visible;
    unsafe { kind = w->kind; enabled = w->enabled; visible = w->visible; }
    if (!visible) { return 0; }

    if (kind == WIDGET_CUSTOM) {
        WidgetCustomEventFn eventFn; unsafe { eventFn = (WidgetCustomEventFn)w->customEvent; }
        if (eventFn != (WidgetCustomEventFn)0) {
            int r; unsafe { r = eventFn(w, ev); }
            if (r) { return 1; }
        }
    }

    if (!enabled) { return 0; }
    if (!__widget_is_interactive(w)) { return 0; }

    int inBox = __widget_point_in_box(w, evX, evY);

    if (evKind == GUI_EVENT_MOUSE_DOWN) {
        if (!inBox) { return 0; }
        unsafe { w->pressed = 1; if (kind == WIDGET_TEXTINPUT) { w->focused = 1; } }
        return 1;
    }
    if (evKind == GUI_EVENT_MOUSE_UP) {
        int wasPressed; unsafe { wasPressed = w->pressed; }
        if (!wasPressed) { return 0; }
        unsafe { w->pressed = 0; }
        if (!inBox) { return 1; }
        if (kind == WIDGET_CHECKBOX) {
            int cur; unsafe { cur = w->checked; w->checked = cur ? 0 : 1; }
            WidgetCallback cb; void* ud;
            unsafe { cb = (WidgetCallback)w->onChange; ud = w->onChangeData; }
            if (cb != (WidgetCallback)0) { unsafe { cb(w, ud); } }
        } else if (kind == WIDGET_SLIDER) {
            int x; int ww; unsafe { x = w->x; ww = w->w; }
            double frac = (evX - (double)x) / (double)(ww - 12);
            if (frac < 0.0) { frac = 0.0; } if (frac > 1.0) { frac = 1.0; }
            double minV; double maxV;
            unsafe { minV = w->sliderMin; maxV = w->sliderMax; w->sliderValue = minV + frac * (maxV - minV); }
            WidgetCallback cb; void* ud;
            unsafe { cb = (WidgetCallback)w->onChange; ud = w->onChangeData; }
            if (cb != (WidgetCallback)0) { unsafe { cb(w, ud); } }
        }
        WidgetCallback click; void* clickUd;
        unsafe { click = (WidgetCallback)w->onClick; clickUd = w->onClickData; }
        if (click != (WidgetCallback)0) { unsafe { click(w, clickUd); } }
        return 1;
    }
    if (kind == WIDGET_SLIDER && evKind == GUI_EVENT_MOUSE_MOVE) {
        int pressed; unsafe { pressed = w->pressed; }
        if (pressed) {
            int x; int ww; unsafe { x = w->x; ww = w->w; }
            double frac = (evX - (double)x) / (double)(ww - 12);
            if (frac < 0.0) { frac = 0.0; } if (frac > 1.0) { frac = 1.0; }
            double minV; double maxV;
            unsafe { minV = w->sliderMin; maxV = w->sliderMax; w->sliderValue = minV + frac * (maxV - minV); }
            WidgetCallback cb; void* ud;
            unsafe { cb = (WidgetCallback)w->onChange; ud = w->onChangeData; }
            if (cb != (WidgetCallback)0) { unsafe { cb(w, ud); } }
            return 1;
        }
    }
    if (kind == WIDGET_TEXTINPUT && evKind == GUI_EVENT_KEY_DOWN) {
        int focused; unsafe { focused = w->focused; }
        if (!focused) { return 0; }
        if (evKeycode == GUI_KEY_BACKSPACE) {
            unsafe {
                if (w->text.len > 0UL) { w->text.truncate(w->text.len - 1UL); }
            }
            WidgetCallback cb; void* ud;
            unsafe { cb = (WidgetCallback)w->onChange; ud = w->onChangeData; }
            if (cb != (WidgetCallback)0) { unsafe { cb(w, ud); } }
            return 1;
        }
        if (evCodepoint >= 32U && evCodepoint < 127U) {
            unsafe {
                char c = (char)evCodepoint;
                w->text.push_char(c);
            }
            WidgetCallback cb; void* ud;
            unsafe { cb = (WidgetCallback)w->onChange; ud = w->onChangeData; }
            if (cb != (WidgetCallback)0) { unsafe { cb(w, ud); } }
            return 1;
        }
    }
    return 0;
}

// ── default key actions + prevent_default ───────────────────────────────────

void gui_event_prevent_default(struct GuiEvent* ev) {
    unsafe { ev->defaultPrevented = 1; }
}

static int __widget_is_focusable(struct Widget* w) {
    int kind; unsafe { kind = w->kind; }
    return (kind == WIDGET_BUTTON || kind == WIDGET_CHECKBOX ||
            kind == WIDGET_TEXTINPUT || kind == WIDGET_SLIDER) ? 1 : 0;
}

static void __widget_collect_focusable(struct Widget* w, struct Vec* out) {
    if (__widget_is_focusable(w)) {
        unsafe { out->push((const void*)&w); }
    }
    unsigned long n; unsafe { n = w->children.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe { struct Widget** slot = (struct Widget**)w->children.get_raw(i); child = *slot; }
        __widget_collect_focusable(child, out);
        i = i + 1UL;
    }
}

static void __widget_clear_focus(struct Widget* w) {
    unsafe { w->focused = 0; }
    unsigned long n; unsafe { n = w->children.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe { struct Widget** slot = (struct Widget**)w->children.get_raw(i); child = *slot; }
        __widget_clear_focus(child);
        i = i + 1UL;
    }
}

static struct Widget* __widget_find_focused(struct Widget* w) {
    int f; unsafe { f = w->focused; }
    if (f) { return w; }
    unsigned long n; unsafe { n = w->children.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe { struct Widget** slot = (struct Widget**)w->children.get_raw(i); child = *slot; }
        struct Widget* found = __widget_find_focused(child);
        if (found != (struct Widget*)0) { return found; }
        i = i + 1UL;
    }
    return (struct Widget*)0;
}

static void __widget_apply_default_key_action(struct Widget* root, struct GuiEvent* ev) {
    int keycode; unsafe { keycode = ev->keycode; }

    if (keycode == GUI_KEY_TAB) {
        struct Vec list = vec_new(8UL);
        __widget_collect_focusable(root, &list);
        unsigned long n; unsafe { n = list.length(); }
        if (n > 0UL) {
            unsigned long foundAt = n; // sentinel: none focused yet
            unsigned long i = 0UL;
            while (i < n) {
                struct Widget* w;
                unsafe { struct Widget** slot = (struct Widget**)list.get_raw(i); w = *slot; }
                int f; unsafe { f = w->focused; }
                if (f) { foundAt = i; }
                i = i + 1UL;
            }
            unsigned long nextIdx = (foundAt == n) ? 0UL : ((foundAt + 1UL) % n);
            __widget_clear_focus(root);
            struct Widget* nextW;
            unsafe { struct Widget** slot = (struct Widget**)list.get_raw(nextIdx); nextW = *slot; }
            unsafe { nextW->focused = 1; }
        }
        unsafe { list.free(); }
        return;
    }

    if (keycode == GUI_KEY_ESCAPE) {
        __widget_clear_focus(root);
        return;
    }

    struct Widget* focused = __widget_find_focused(root);
    if (focused == (struct Widget*)0) { return; }
    int fkind; unsafe { fkind = focused->kind; }

    if (keycode == GUI_KEY_ENTER) {
        if (fkind == WIDGET_BUTTON) {
            WidgetCallback cb; void* ud;
            unsafe { cb = (WidgetCallback)focused->onClick; ud = focused->onClickData; }
            if (cb != (WidgetCallback)0) { unsafe { cb(focused, ud); } }
        } else if (fkind == WIDGET_CHECKBOX) {
            int cur; unsafe { cur = focused->checked; focused->checked = cur ? 0 : 1; }
            WidgetCallback cb; void* ud;
            unsafe { cb = (WidgetCallback)focused->onChange; ud = focused->onChangeData; }
            if (cb != (WidgetCallback)0) { unsafe { cb(focused, ud); } }
        }
        return;
    }

    if (fkind == WIDGET_SLIDER && (keycode == GUI_KEY_LEFT || keycode == GUI_KEY_RIGHT)) {
        double minV; double maxV; double val;
        unsafe { minV = focused->sliderMin; maxV = focused->sliderMax; val = focused->sliderValue; }
        double step = (maxV - minV) * 0.05;
        val = (keycode == GUI_KEY_LEFT) ? (val - step) : (val + step);
        if (val < minV) { val = minV; }
        if (val > maxV) { val = maxV; }
        unsafe { focused->sliderValue = val; }
        WidgetCallback cb; void* ud;
        unsafe { cb = (WidgetCallback)focused->onChange; ud = focused->onChangeData; }
        if (cb != (WidgetCallback)0) { unsafe { cb(focused, ud); } }
    }
}

int gui_dispatch_event(struct Widget* root, struct GuiEvent* ev) {
    int evKind; unsafe { evKind = ev->kind; }
    if (evKind == GUI_EVENT_MOUSE_MOVE) {
        // Hover state always updates on move, but the event must still
        // reach __widget_dispatch_one() too — a pressed slider needs
        // MOUSE_MOVE while dragging, not just MOUSE_DOWN/UP.
        __widget_update_hover(root, ev);
    }
    int handled = __widget_dispatch_one(root, ev);

    int prevented; unsafe { prevented = ev->defaultPrevented; }
    if (!prevented && evKind == GUI_EVENT_KEY_DOWN) {
        __widget_apply_default_key_action(root, ev);
    }
    return handled;
}

// ── lifecycle ─────────────────────────────────────────────────────────────────

void widget_free(struct Widget* w) {
    unsigned long n; unsafe { n = w->children.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct Widget* child;
        unsafe { struct Widget** slot = (struct Widget**)w->children.get_raw(i); child = *slot; }
        widget_free(child);
        i = i + 1UL;
    }
    unsafe {
        w->children.free();
        w->text.free();
        dealloc((void*)w);
    }
}

} // namespace std
