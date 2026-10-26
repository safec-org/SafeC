// SafeC Standard Library — GUI: the headless backend (see gui_headless.h).
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_headless.h>
#include <std/mem.h>
#include <std/collections/string.h>
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
extern char* getenv(const char* name);
#endif

namespace std {

struct GuiHeadlessWin {
    unsigned long presents;
};

static struct GuiBackend gHeadless;
static int gHeadlessInit = 0;
static double gHeadlessNowMs = 0.0;
static double gHeadlessScale = 1.0;
static int gHeadlessPrimary = GUI_MOD_CTRL;
static struct String gHeadlessClip;
static int gHeadlessClipSet = 0;
static unsigned long long gHeadlessNextKey = 1ULL;
static unsigned int gHeadlessFocusId = 0U;

static int __hl_init() { return 1; }

int gui_headless_requested() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    unsafe {
        const char* v = getenv("SAFEC_GUI_BACKEND");
        if (v == (const char*)0) { return 0; }
        const char* want = "headless";
        unsigned long i = 0UL;
        while (want[i] != '\0' && v[i] == want[i]) { i = i + 1UL; }
        return (want[i] == '\0' && v[i] == '\0') ? 1 : 0;
    }
#else
    return 0;
#endif
}

static int __hl_create(&GuiWindowState st, const &GuiWindowOptions opts) {
    unsafe {
        struct GuiHeadlessWin* w = (struct GuiHeadlessWin*)alloc_zeroed_raw_(sizeof(struct GuiHeadlessWin));
        st.native = (void*)w;
    }
    st.key = gHeadlessNextKey;
    gHeadlessNextKey = gHeadlessNextKey + 1ULL;
    st.scale = opts.hiDpi ? gHeadlessScale : 1.0;
    st.pixelW = (int)((double)st.logicalW * st.scale + 0.5);
    st.pixelH = (int)((double)st.logicalH * st.scale + 0.5);
    if (st.x == GUI_POS_DEFAULT) { st.x = 0; }
    if (st.y == GUI_POS_DEFAULT) { st.y = 0; }
    return 1;
}

static void __hl_destroy(&GuiWindowState st) {
    if (gHeadlessFocusId == st.id) { gHeadlessFocusId = 0U; }
    unsafe {
        if (!st.native.is_null()) { dealloc((?&heap Erased)st.native); }
    }
    st.native = null;
}

static void __hl_present(&GuiWindowState st, const &unsigned char px, int bw, int bh, int x, int y, int w, int h) {
    unsafe {
        if (st.native.is_null()) { return; }
        struct GuiHeadlessWin* hw = (struct GuiHeadlessWin*)st.native;
        hw->presents = hw->presents + 1UL;
    }
}

static void __hl_set_title(&GuiWindowState st, const &char title) { }

static int __hl_clamp(int v, int lo, int hi) {
    if (lo > 0 && v < lo) { v = lo; }
    if (hi > 0 && v > hi) { v = hi; }
    return v;
}

static void __hl_apply_size(&GuiWindowState st, int w, int h) {
    w = __hl_clamp(w, st.minW, st.maxW);
    h = __hl_clamp(h, st.minH, st.maxH);
    if (w < 1) { w = 1; }
    if (h < 1) { h = 1; }
    st.logicalW = w;
    st.logicalH = h;
    st.pixelW = (int)((double)w * st.scale + 0.5);
    st.pixelH = (int)((double)h * st.scale + 0.5);
    struct GuiEvent ev = GuiEvent::new(GUI_EVENT_RESIZE);
    ev.width = st.pixelW;
    ev.height = st.pixelH;
    gui_backend_post(st, ev);
}

static void __hl_set_size(&GuiWindowState st, int w, int h) { __hl_apply_size(st, w, h); }

static void __hl_set_position(&GuiWindowState st, int x, int y) {
    st.x = x;
    st.y = y;
    struct GuiEvent ev = GuiEvent::new(GUI_EVENT_MOVE);
    ev.x = (double)x;
    ev.y = (double)y;
    gui_backend_post(st, ev);
}

static void __hl_update_limits(&GuiWindowState st) {
    int w = __hl_clamp(st.logicalW, st.minW, st.maxW);
    int h = __hl_clamp(st.logicalH, st.minH, st.maxH);
    if (w != st.logicalW || h != st.logicalH) { __hl_apply_size(st, w, h); }
}

static void __hl_set_visible(&GuiWindowState st, int v) { st.visible = v; }

static void __hl_focus(&GuiWindowState st) {
    if (gHeadlessFocusId == st.id) { return; }
    unsigned long n = gui_backend_window_count();
    unsigned long i = 0UL;
    while (i < n) {
        ?&GuiWindowState o = gui_backend_window_at(i);
        match (o) {
            case some(other): {
                if (other.id == gHeadlessFocusId && other.focused) {
                    other.focused = 0;
                    gui_backend_post(other, GuiEvent::new(GUI_EVENT_FOCUS_OUT));
                }
            }
            case null: { }
        }
        i = i + 1UL;
    }
    gHeadlessFocusId = st.id;
    st.focused = 1;
    gui_backend_post(st, GuiEvent::new(GUI_EVENT_FOCUS_IN));
}

static void __hl_set_cursor(&GuiWindowState st, int c) { }

static int __hl_pump(int timeoutMs) {
    // Nothing to read from anywhere; a wait just lets virtual time pass.
    if (timeoutMs > 0 && !gui_backend_has_events()) {
        gHeadlessNowMs = gHeadlessNowMs + (double)timeoutMs;
    }
    return 0;
}

static void __hl_wake() { }

static long long __hl_now_ns() {
    return (long long)(gHeadlessNowMs * 1000000.0);
}

static int __hl_clip_set(const &char utf8, unsigned long n) {
    if (gHeadlessClipSet) { gHeadlessClip.free(); }
    gHeadlessClip = String::from_n_(utf8, n);
    gHeadlessClipSet = 1;
    return 1;
}

static int __hl_clip_get(&String out) {
    out.clear();
    if (!gHeadlessClipSet) { return 0; }
    unsafe { out.push_n_((const char*)gHeadlessClip.data, gHeadlessClip.len); }
    return 1;
}

static double __hl_display_scale() { return gHeadlessScale; }

static struct GuiNativeHandle __hl_native(&GuiWindowState st) {
    struct GuiNativeHandle h;
    h.kind = GUI_NATIVE_HEADLESS;
    h.display = 0ULL;
    h.window = st.key;
    h.view = 0ULL;
    h.visual = 0ULL;
    return h;
}

&GuiBackend gui_headless_backend() {
    if (!gHeadlessInit) {
        gHeadless.name = "headless";
        gHeadless.primaryMod = gHeadlessPrimary;
        gHeadless.init = __hl_init;
        gHeadless.window_create = __hl_create;
        gHeadless.window_destroy = __hl_destroy;
        gHeadless.window_present = __hl_present;
        gHeadless.window_set_title = __hl_set_title;
        gHeadless.window_set_size = __hl_set_size;
        gHeadless.window_set_position = __hl_set_position;
        gHeadless.window_update_limits = __hl_update_limits;
        gHeadless.window_set_visible = __hl_set_visible;
        gHeadless.window_focus = __hl_focus;
        gHeadless.window_set_cursor = __hl_set_cursor;
        gHeadless.pump = __hl_pump;
        gHeadless.wake = __hl_wake;
        gHeadless.now_ns = __hl_now_ns;
        gHeadless.clipboard_set = __hl_clip_set;
        gHeadless.clipboard_get = __hl_clip_get;
        gHeadless.display_scale = __hl_display_scale;
        gHeadless.native_handle = __hl_native;
        gHeadlessInit = 1;
    }
    return &gHeadless;
}

int gui_use_headless() {
    return gui_set_backend(gui_headless_backend());
}

void gui_headless_advance(double ms) {
    if (ms > 0.0) { gHeadlessNowMs = gHeadlessNowMs + ms; }
}

double gui_headless_now_ms() { return gHeadlessNowMs; }

void gui_headless_set_scale(double scale) {
    gHeadlessScale = scale > 0.0 ? scale : 1.0;
}

void gui_headless_set_primary_mod(int mod) {
    gHeadlessPrimary = mod;
    gui_headless_backend();
    gHeadless.primaryMod = mod;
}

int gui_headless_inject(&GuiWindow win, struct GuiEvent ev) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return 0;
        case some(st): {
            gui_backend_post(st, ev);
            return 1;
        }
    }
    return 0;
}

static struct GuiEvent __hl_mouse(int kind, double x, double y, int button) {
    struct GuiEvent e = GuiEvent::new(kind);
    e.x = x;
    e.y = y;
    e.button = button;
    return e;
}

void gui_headless_mouse_move(&GuiWindow win, double x, double y) {
    gui_headless_inject(win, __hl_mouse(GUI_EVENT_MOUSE_MOVE, x, y, 0));
}

void gui_headless_mouse_down(&GuiWindow win, double x, double y, int button) {
    gui_headless_inject(win, __hl_mouse(GUI_EVENT_MOUSE_DOWN, x, y, button));
}

void gui_headless_mouse_up(&GuiWindow win, double x, double y, int button) {
    gui_headless_inject(win, __hl_mouse(GUI_EVENT_MOUSE_UP, x, y, button));
}

void gui_headless_click(&GuiWindow win, double x, double y) {
    gui_headless_mouse_move(win, x, y);
    gui_headless_mouse_down(win, x, y, GUI_BUTTON_LEFT);
    gui_headless_mouse_up(win, x, y, GUI_BUTTON_LEFT);
}

void gui_headless_scroll(&GuiWindow win, double x, double y, double dx, double dy) {
    struct GuiEvent e = __hl_mouse(GUI_EVENT_SCROLL, x, y, 0);
    e.scrollDx = dx;
    e.scrollDy = dy;
    gui_headless_inject(win, e);
}

void gui_headless_key(&GuiWindow win, int keycode, int modifiers, unsigned int codepoint) {
    struct GuiEvent d = GuiEvent::new(GUI_EVENT_KEY_DOWN);
    d.keycode = keycode;
    d.modifiers = modifiers;
    d.codepoint = codepoint;
    gui_headless_inject(win, d);
    struct GuiEvent u = GuiEvent::new(GUI_EVENT_KEY_UP);
    u.keycode = keycode;
    u.modifiers = modifiers;
    gui_headless_inject(win, u);
}

void gui_headless_type(&GuiWindow win, const &char text) {
    unsigned long n = 0UL;
    unsafe { while (text[n] != '\0') { n = n + 1UL; } }
    unsigned long i = 0UL;
    while (i < n) {
        unsigned int cp = gui_utf8_next_n_(text, n, &i);
        int key = GUI_KEY_UNKNOWN;
        if (cp >= 97U && cp <= 122U) { key = (int)(cp - 32U); }
        else if ((cp >= 65U && cp <= 90U) || (cp >= 48U && cp <= 57U)) { key = (int)cp; }
        else if (cp == 32U) { key = GUI_KEY_SPACE; }
        int mods = (cp >= 65U && cp <= 90U) ? GUI_MOD_SHIFT : 0;
        if (cp == 10U) { gui_headless_key(win, GUI_KEY_ENTER, 0, 0U); }
        else if (cp == 9U) { gui_headless_key(win, GUI_KEY_TAB, 0, 0U); }
        else { gui_headless_key(win, key, mods, cp); }
    }
}

void gui_headless_resize(&GuiWindow win, int width, int height) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return;
        case some(st): { __hl_apply_size(st, width, height); }
    }
}

void gui_headless_close(&GuiWindow win) {
    gui_headless_inject(win, GuiEvent::new(GUI_EVENT_CLOSE));
}

unsigned long gui_headless_present_count(const &GuiWindow win) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return 0UL;
        case some(st): {
            unsafe {
                if (st.native.is_null()) { return 0UL; }
                struct GuiHeadlessWin* hw = (struct GuiHeadlessWin*)st.native;
                return hw->presents;
            }
        }
    }
    return 0UL;
}

int gui_headless_cursor(const &GuiWindow win) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) { case null: return GUI_CURSOR_ARROW; case some(st): return st.cursor; }
    return GUI_CURSOR_ARROW;
}

} // namespace std
