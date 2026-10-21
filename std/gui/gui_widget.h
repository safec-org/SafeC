#pragma once
// SafeC Standard Library — GUI: retained-mode widget tree.
//
// Built entirely on gui.h (window/events/drawing) + gui_font.h
// (software rasterizer + fonts) — every widget is drawn by the same
// portable rect/text primitives every gui_* backend shares, so a UI built
// here looks identical on Cocoa/Win32/X11/bare metal.
//
// A widget tree is built once (Widget::vstack(), Widget::button(), ...,
// parent.add_child(child)), then driven once per frame:
//
//   root.layout(0, 0, win.width, win.height);
//   int handled = 0;
//   while (win.poll_event(&ev)) { handled |= root.dispatch_event(&ev); }
//   root.render(&win);
//   win.present();
//
// Every built-in widget constructor returns a heap-allocated
// 'struct Widget*' with sensible defaults already set — customize it with
// its set_*() and on_*() methods (each returns the same widget pointer, so
// calls chain: Widget::button("Go").set_padding(8).set_bg(r,g,b,a)). Every widget is fully independently
// customizable: colors, border, padding, spacing, preferred size, stretch
// weight, cross-axis alignment, font (and font scale), visibility, and
// enabled state.
//
// For anything the built-ins don't cover, Widget::custom() takes your own
// draw/layout functions (and optionally an event hook) plus a userData
// pointer, and participates in the same tree/layout/event pipeline as any
// built-in widget — see the "Custom widgets" section below.

#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_font.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

#define WIDGET_VSTACK    0
#define WIDGET_HSTACK    1
#define WIDGET_BUTTON    2
#define WIDGET_LABEL     3
#define WIDGET_CHECKBOX  4
#define WIDGET_TEXTINPUT 5
#define WIDGET_SPACER    6
#define WIDGET_SLIDER    7
#define WIDGET_CUSTOM    8

#define WIDGET_ALIGN_START   0
#define WIDGET_ALIGN_CENTER  1
#define WIDGET_ALIGN_END     2
#define WIDGET_ALIGN_STRETCH 3

struct WidgetStyle {
    struct GuiColor bg;
    struct GuiColor fg;
    struct GuiColor border;
    int borderWidth;
    int padding;
    int spacing;               // gap between children, for VSTACK/HSTACK
    int fontScale;         // 1, 2, 3, ... (built-in/assigned font's glyph cell * this)
    const ?&GuiFont font;  // empty (null) = built-in default font
};

struct Widget {
    int kind;

    // box, computed by layout() each frame
    int x; int y; int w; int h;

    // sizing hints (all optional — 0/unset means "size to content")
    int prefW; int prefH;
    int stretch;    // 0 = natural size; >0 = flex weight along the parent stack's main axis
    int alignment;  // WIDGET_ALIGN_* — this widget's position on the parent's cross axis

    struct WidgetStyle style;
    struct String text;
    int visible;
    int enabled;

    int checked;                            // WIDGET_CHECKBOX
    double sliderValue; double sliderMin; double sliderMax; // WIDGET_SLIDER
    unsigned long caret;                     // WIDGET_TEXTINPUT cursor position

    int hover;
    int pressed;
    int focused;

    // Callback/hook slots are stored as plain 'void*' — not the typed
    // WidgetCallback/WidgetDrawFn/... aliases below — purely because those
    // aliases reference '&Widget' themselves and safec requires a
    // struct to be fully defined before it's used as a typedef'd function
    // pointer's parameter type, so the typedefs can only be declared right
    // after this struct closes, not before. Every setter/getter still
    // takes/returns the properly typed alias — the void* storage is an
    // implementation detail, cast at the few call sites inside gui_widget.sc.
    void* onClick;      // WidgetCallback
    void* onClickData;
    void* onChange;     // WidgetCallback
    void* onChangeData;

    void* customDraw;   // WidgetDrawFn
    void* customLayout; // WidgetLayoutFn
    void* customEvent;  // WidgetCustomEventFn
    void* userData;

    // onClickData/onChangeData/userData stay 'void*' rather than a typed
    // '?&T' (see README's "Outliving references" section, and
    // std/dma.h's 'generic<T> struct DmaChannel' for the pattern where a
    // concrete T *does* fit) deliberately: 'children' below is one
    // Vec<struct Widget*> holding buttons, sliders, labels, etc.
    // together, each with its own unrelated userData type. Genericizing
    // Widget over T would make 'Widget<ButtonCtx>' and 'Widget<SliderCtx>'
    // distinct, incompatible types — unable to share one tree. Same
    // shape as std/dsp/timer_wheel.h's WheelTimer and std/fs/block.h's
    // BlockDevice (implemented by both the ext and FAT drivers): a
    // single, multi-tenant collection meant to hold many unrelated
    // context types at once, which 'void*' erasure supports and a
    // generic struct's one-type-per-instantiation model does not.
    struct Vec<&Widget> children;
    ?&Widget parent;        // empty (null) for the root widget

    // ── constructors ─────────────────────────────────────────────────────
    // All return a freshly heap-allocated, non-null widget — a region-less
    // '&Widget' (see README's "Outliving references"), since nothing here
    // cares whether the tree it ends up in is heap/stack/static/arena-owned.
    static &Widget vstack();
    static &Widget hstack();
    static &Widget button(const &char text);
    static &Widget label(const &char text);
    static &Widget checkbox(const &char text, int checked);
    static &Widget textinput(const &char placeholder);
    static &Widget spacer();
    static &Widget slider(double minValue, double maxValue, double value);

    // A WIDGET_CUSTOM node drawn and laid out by your own functions (see
    // "Custom widgets" below); userData is the caller's type, stored erased.
    generic<T>
    static &Widget custom(WidgetDrawFn draw, WidgetLayoutFn layout,
                          WidgetCustomEventFn eventFn, ?&T userData);
    static &Widget custom_raw_(WidgetDrawFn draw, WidgetLayoutFn layout,
                               WidgetCustomEventFn eventFn, void* userData);

    // ── tree ─────────────────────────────────────────────────────────────
    void add_child(&Widget child);

    // ── customization (each returns this widget, so calls chain) ─────────
    &Widget set_bg(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    &Widget set_fg(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    &Widget set_border(unsigned char r, unsigned char g, unsigned char b,
                       unsigned char a, int width);
    &Widget set_padding(int padding);
    &Widget set_spacing(int spacing);
    &Widget set_pref_size(int prefW, int prefH);
    &Widget set_stretch(int stretch);
    &Widget set_alignment(int alignment);
    &Widget set_font_scale(int scale);
    &Widget set_font(const ?&GuiFont font);
    &Widget set_visible(int visible);
    &Widget set_enabled(int enabled);
    &Widget set_text(const &char text);

    // ── callbacks (each returns this widget, so calls chain) ─────────────
    // cb(widget, userData) on click / value change. userData's type is the
    // caller's; the widget stores it erased (see the field comment above).
    generic<T> &Widget on_click(fn void(&Widget, ?&T) cb, ?&T userData);
    generic<T> &Widget on_change(fn void(&Widget, ?&T) cb, ?&T userData);
    &Widget on_click_raw_(void* cb, void* userData);
    &Widget on_change_raw_(void* cb, void* userData);

    // ── per-frame pipeline ───────────────────────────────────────────────

    // Positions this widget and its whole subtree within the box
    // (x,y,w,h) — VSTACK/HSTACK divide their box among children by
    // stretch weight (after giving each non-stretched child its
    // natural/preferred size), applying 'padding' inside the container and
    // 'spacing' between children; WIDGET_CUSTOM nodes call their own
    // WidgetLayoutFn instead.
    void layout(int x, int y, int w, int h);

    // Draws this widget and its whole subtree into 'win' (call after
    // layout()).
    void render(&GuiWindow win);

    // Routes one event through the tree rooted here (deepest/topmost
    // widget under the cursor first): updates hover/pressed/focused state,
    // fires onClick on mouse-up-inside-after-mouse-down-inside, fires
    // onChange for checkbox toggle/slider drag/text input edits, and calls
    // a WIDGET_CUSTOM node's customEvent hook if set. Returns 1 if some
    // widget consumed the event.
    //
    // After that, for GUI_EVENT_KEY_DOWN, a default action runs — the same
    // idea as a browser's default DOM action for a key event — *unless*
    // ev.prevent_default() (see gui.h) was called on this event first
    // (typically from a WIDGET_CUSTOM node's WidgetCustomEventFn, the one
    // hook that gets the raw GuiEvent* and can act before the default
    // fires):
    //   Tab          moves focus to the next focusable widget (wraps around)
    //   Enter        activates the focused widget (button click / checkbox toggle)
    //   Escape       clears focus
    //   Left/Right   nudges a focused slider by 5% of its range
    // These are genuinely *default* actions: they still run even if
    // dispatch already returned 1 for this event (e.g. a focused text
    // input consuming Backspace doesn't block Tab from later moving focus
    // off it).
    int dispatch_event(&GuiEvent ev);

    // Recursively frees this widget and its whole subtree (children,
    // text, style), including the widget's own allocation.
    void free();
};

// onClick/onChange(widget, userData).
typedef fn void(&Widget, ?&void) WidgetCallback;
// Custom draw(window, widget) — called instead of the built-in renderer
// for a WIDGET_CUSTOM node; must draw within [widget->x, widget->x+widget->w) x
// [widget->y, widget->y+widget->h) using GuiWindow's drawing methods (win.fill_rect(), win.draw_text(), ...).
typedef fn void(&GuiWindow, &Widget) WidgetDrawFn;
// Custom layout(widget, x, y, w, h) — called instead of the built-in
// layout for a WIDGET_CUSTOM node; must set widget->x/y/w/h (typically to
// the given box) and lay out+recurse into any children itself.
typedef fn void(&Widget, int, int, int, int) WidgetLayoutFn;
// Custom event(widget, event) -> 1 if handled (stops the event from
// reaching this widget's parent/siblings), 0 to let normal dispatch
// (hover/press/click detection) continue.
typedef fn int(&Widget, &GuiEvent) WidgetCustomEventFn;

// Custom widgets: 'draw'/'layout' are required; 'eventFn' may be NULL (the
// widget then just participates in hit-testing/hover/press/click like a
// button would, with onClick/onChange still firing normally — set eventFn
// only when you need to intercept raw events yourself, e.g. for a custom
// slider/canvas that tracks drag deltas).
generic<T>
&Widget Widget::custom(WidgetDrawFn draw, WidgetLayoutFn layout,
                       WidgetCustomEventFn eventFn, ?&T userData) {
    unsafe { return Widget::custom_raw_(draw, layout, eventFn, (void*)userData); }
}

generic<T>
&Widget Widget::on_click(fn void(&Widget, ?&T) cb, ?&T userData) {
    unsafe { return self.on_click_raw_((void*)cb, (void*)userData); }
}

generic<T>
&Widget Widget::on_change(fn void(&Widget, ?&T) cb, ?&T userData) {
    unsafe { return self.on_change_raw_((void*)cb, (void*)userData); }
}

} // namespace std
