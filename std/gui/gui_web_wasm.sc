// SafeC Standard Library — GUI backend: WebAssembly <canvas> (see
// gui_web.h for the model and gui_backend.h for the contract). Every
// operation is an import from std/gui/gui_web.js ("env" module, the
// std_gui_web_* names below); the page's events come back through the
// exports at the end of this file.
// safeguard: implements <std/gui/gui.h>   (the wasm32/wasm64 platform backend)
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_app.h>
#include <std/gui/gui_web.h>
#include <std/mem.h>
#include <std/collections/string.h>

namespace std {

// ── imports (std/gui/gui_web.js) ────────────────────────────────────────
extern double std_gui_web_now();
extern double std_gui_web_dpr();
// GUI_MOD_META on Apple systems (Cmd), GUI_MOD_CTRL elsewhere.
extern int  std_gui_web_primary_mod();
// Creates the canvas (and frame) of window 'id'. opts: width, height,
// hiDpi, visible, x, y, decorated, resizable, minW, minH, maxW, maxH;
// out: logical width, height, scale * 1000, x, y. 1 = ok.
extern int  std_gui_web_create(int id, const char* title, int titleLen, const int* opts, int* out);
extern void std_gui_web_destroy(int id);
extern void std_gui_web_present(int id, const unsigned char* pixels, int bufW, int bufH, int x, int y, int w, int h);
extern void std_gui_web_set_title(int id, const char* title, int len);
extern void std_gui_web_set_size(int id, int width, int height);
extern void std_gui_web_set_position(int id, int x, int y);
extern void std_gui_web_update_limits(int id, int minW, int minH, int maxW, int maxH);
extern void std_gui_web_say(const char* text, int len);
extern int  std_gui_web_dom_handle(int id);
extern int  std_gui_web_mount_in(int id, int parent);
extern void std_gui_web_set_visible(int id, int visible);
extern void std_gui_web_focus(int id);
extern void std_gui_web_set_cursor(int id, int cursor);
extern int  std_gui_web_clipboard_set(const char* text, int len);
// The clipboard text as the page last knew it: copies up to 'cap' bytes,
// returns its full length (-1: none).
extern int  std_gui_web_clipboard_get(char* buf, int cap);

struct GuiWebWindow {
    unsigned char* staging;   // straight-alpha copy for ImageData
    int stagingW; int stagingH;
};

static struct GuiBackend gWebBackend;
static int gWebBackendInit = 0;
static struct String gWebClip;
static int gWebClipInit = 0;

static int __web_init() {
    unsafe { gWebBackend.primaryMod = std_gui_web_primary_mod(); }
    return 1;
}

static int __web_create(&GuiWindowState st, const &GuiWindowOptions opts) {
    unsafe {
        int o[12];
        o[0] = st.logicalW; o[1] = st.logicalH; o[2] = opts.hiDpi; o[3] = opts.visible;
        o[4] = opts.x; o[5] = opts.y; o[6] = opts.decorated; o[7] = opts.resizable;
        o[8] = opts.minWidth; o[9] = opts.minHeight; o[10] = opts.maxWidth; o[11] = opts.maxHeight;
        int out[5];
        out[0] = 0; out[1] = 0; out[2] = 1000; out[3] = GUI_POS_DEFAULT; out[4] = GUI_POS_DEFAULT;
        int n = (int)st.title.len;
        if (!std_gui_web_create((int)st.id, (const char*)st.title.data, n, (const int*)&o[0], &out[0])) {
            return 0;
        }
        st.x = out[3];
        st.y = out[4];
        struct GuiWebWindow* w = (struct GuiWebWindow*)alloc_zeroed_raw_(sizeof(struct GuiWebWindow));
        st.native = (void*)w;
        st.key = (unsigned long long)st.id;
        if (out[0] > 0) { st.logicalW = out[0]; }
        if (out[1] > 0) { st.logicalH = out[1]; }
        st.scale = (double)out[2] / 1000.0;
        if (st.scale <= 0.0) { st.scale = 1.0; }
        st.pixelW = (int)((double)st.logicalW * st.scale + 0.5);
        st.pixelH = (int)((double)st.logicalH * st.scale + 0.5);
        return 1;
    }
}

static void __web_destroy(&GuiWindowState st) {
    unsafe {
        std_gui_web_destroy((int)st.id);
        struct GuiWebWindow* w = (struct GuiWebWindow*)st.native;
        if (w != (struct GuiWebWindow*)0) {
            if (w->staging != (unsigned char*)0) { dealloc(w->staging); }
            dealloc(w);
        }
        st.native = null;
    }
}

static void __web_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                          int x, int y, int rw, int rh) {
    unsafe {
        struct GuiWebWindow* w = (struct GuiWebWindow*)st.native;
        if (w == (struct GuiWebWindow*)0) { return; }
        if (w->stagingW != bw || w->stagingH != bh) {
            if (w->staging != (unsigned char*)0) { dealloc(w->staging); }
            w->staging = (unsigned char*)alloc_zeroed_raw_((unsigned long)bw * (unsigned long)bh * 4UL);
            w->stagingW = bw;
            w->stagingH = bh;
        }
        // ImageData is straight (not premultiplied) RGBA.
        int yy = y;
        while (yy < y + rh) {
            unsigned long off = ((unsigned long)yy * (unsigned long)bw + (unsigned long)x) * 4UL;
            const unsigned char* s = (const unsigned char*)px + off;
            unsigned char* d = w->staging + off;
            int xx = 0;
            while (xx < rw) {
                unsigned int a = (unsigned int)s[3];
                if (a == 255U || a == 0U) {
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                } else {
                    unsigned int half = a / 2U;
                    unsigned int r = ((unsigned int)s[0] * 255U + half) / a;
                    unsigned int g = ((unsigned int)s[1] * 255U + half) / a;
                    unsigned int b = ((unsigned int)s[2] * 255U + half) / a;
                    d[0] = (unsigned char)(r > 255U ? 255U : r);
                    d[1] = (unsigned char)(g > 255U ? 255U : g);
                    d[2] = (unsigned char)(b > 255U ? 255U : b);
                    d[3] = (unsigned char)a;
                }
                s = s + 4; d = d + 4;
                xx = xx + 1;
            }
            yy = yy + 1;
        }
        std_gui_web_present((int)st.id, (const unsigned char*)w->staging, bw, bh, x, y, rw, rh);
    }
}

static void __web_set_title(&GuiWindowState st, const &char title) {
    unsafe {
        int n = 0;
        while (title[n] != '\0') { n = n + 1; }
        std_gui_web_set_title((int)st.id, title, n);
    }
}

static void __web_set_size(&GuiWindowState st, int lw, int lh) {
    unsafe { std_gui_web_set_size((int)st.id, lw, lh); }
    // The canvas takes the size at once.
    st.logicalW = lw;
    st.logicalH = lh;
    st.pixelW = (int)((double)lw * st.scale + 0.5);
    st.pixelH = (int)((double)lh * st.scale + 0.5);
    struct GuiEvent ev = GuiEvent::new(GUI_EVENT_RESIZE);
    ev.width = st.pixelW;
    ev.height = st.pixelH;
    gui_backend_post(st, ev);
}

static void __web_set_position(&GuiWindowState st, int x, int y) { unsafe { std_gui_web_set_position((int)st.id, x, y); } }
static void __web_update_limits(&GuiWindowState st) {
    unsafe { std_gui_web_update_limits((int)st.id, st.minW, st.minH, st.maxW, st.maxH); }
}
static void __web_set_visible(&GuiWindowState st, int v) { unsafe { std_gui_web_set_visible((int)st.id, v); } }
static void __web_focus(&GuiWindowState st) { unsafe { std_gui_web_focus((int)st.id); } }
static void __web_set_cursor(&GuiWindowState st, int c) { unsafe { std_gui_web_set_cursor((int)st.id, c); } }

// The page delivers events through the exports below, between frames; a
// browser can't wait here.
static int __web_pump(int timeoutMs) { return gui_backend_has_events(); }

static long long __web_now_ns() {
    double ms = 0.0;
    unsafe { ms = std_gui_web_now(); }
    return (long long)(ms * 1000000.0);
}

static int __web_clip_set(const &char utf8, unsigned long n) {
    if (!gWebClipInit) { gWebClip = String::new(); gWebClipInit = 1; }
    gWebClip.free();
    gWebClip = String::from_n_(utf8, n);
    unsafe { return std_gui_web_clipboard_set(utf8, (int)n); }
}

static int __web_clip_get(&String out) {
    out.clear();
    unsafe {
        int n = std_gui_web_clipboard_get((char*)0, 0);
        if (n < 0) { return 0; }
        if (n == 0) { return 1; }
        char* buf = (char*)alloc_raw_((unsigned long)n + 1UL);
        int got = std_gui_web_clipboard_get(buf, n);
        if (got > n) { got = n; }
        out.push_n_((const char*)buf, (unsigned long)got);
        dealloc(buf);
        return 1;
    }
}

static double __web_display_scale() {
    unsafe {
        double d = std_gui_web_dpr();
        return d > 0.0 ? d : 1.0;
    }
}

static struct GuiNativeHandle __web_native(&GuiWindowState st) {
    struct GuiNativeHandle h;
    h.kind = GUI_NATIVE_WEB;
    h.display = 0ULL;
    h.window = st.key;      // the window id the glue maps to its canvas
    h.view = 0ULL;
    h.visual = 0ULL;
    return h;
}

&GuiBackend gui_platform_backend() {
    if (!gWebBackendInit) {
        gWebBackend.name = "web";
        gWebBackend.primaryMod = GUI_MOD_CTRL;   // init() asks the page (Cmd on Apple systems)
        gWebBackend.init = __web_init;
        gWebBackend.window_create = __web_create;
        gWebBackend.window_destroy = __web_destroy;
        gWebBackend.window_present = __web_present;
        gWebBackend.window_set_title = __web_set_title;
        gWebBackend.window_set_size = __web_set_size;
        gWebBackend.window_set_position = __web_set_position;
        gWebBackend.window_update_limits = __web_update_limits;
        gWebBackend.window_set_visible = __web_set_visible;
        gWebBackend.window_focus = __web_focus;
        gWebBackend.window_set_cursor = __web_set_cursor;
        gWebBackend.pump = __web_pump;
        gWebBackend.wake = (fn void())0;
        gWebBackend.now_ns = __web_now_ns;
        gWebBackend.clipboard_set = __web_clip_set;
        gWebBackend.clipboard_get = __web_clip_get;
        gWebBackend.display_scale = __web_display_scale;
        gWebBackend.native_handle = __web_native;
        gWebBackendInit = 1;
    }
    return &gWebBackend;
}

// ── exports (called by the glue) ─────────────────────────────────────────

// One input event for window 'id'. kind is GUI_EVENT_*; x/y in buffer
// pixels; for RESIZE/SCALE, width/height are the new buffer size and
// 'scale' the new scale factor (logical size = buffer / scale).
__attribute__((export_name("std_gui_web_event")))
int gui_web_event_(int id, int kind, double x, double y, int button, int keycode, int codepoint,
                   int mods, double dx, double dy, int width, int height, double scale) {
    ?&GuiWindowState ost = gui_backend_find((unsigned long long)(unsigned int)id);
    match (ost) {
        case null: return 0;
        case some(st): {
            struct GuiEvent ev = GuiEvent::new(kind);
            ev.x = x;
            ev.y = y;
            ev.button = button;
            ev.keycode = keycode;
            ev.codepoint = (unsigned int)codepoint;
            ev.modifiers = mods;
            ev.scrollDx = dx;
            ev.scrollDy = dy;
            ev.width = width;
            ev.height = height;
            if (kind == GUI_EVENT_RESIZE || kind == GUI_EVENT_SCALE) {
                if (scale > 0.0) { st.scale = scale; }
                st.pixelW = width;
                st.pixelH = height;
                st.logicalW = (int)((double)width / st.scale + 0.5);
                st.logicalH = (int)((double)height / st.scale + 0.5);
            }
            if (kind == GUI_EVENT_MOVE) { st.x = (int)x; st.y = (int)y; }
            if (kind == GUI_EVENT_FOCUS_IN) { st.focused = 1; }
            if (kind == GUI_EVENT_FOCUS_OUT) { st.focused = 0; }
            gui_backend_post(st, ev);
            return 1;
        }
    }
    return 0;
}

void gui_web_announce(const &char text) {
    unsafe {
        int n = 0;
        while (text[n] != '\0') { n = n + 1; }
        std_gui_web_say(text, n);
    }
}

int gui_web_window_element(const &GuiWindow win) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return -1;
        case some(st): { unsafe { return std_gui_web_dom_handle((int)st.id); } }
    }
    return -1;
}

int gui_web_mount(&GuiWindow win, int parent) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return 0;
        case some(st): { unsafe { return std_gui_web_mount_in((int)st.id, parent); } }
    }
    return 0;
}

static ?&GuiApp gWebApp = null;

void gui_web_run(&GuiApp app) {
    gWebApp = app;
}

// One frame (requestAnimationFrame): app.pump(0). 0 once the app is done
// (or none was handed to gui_web_run()).
__attribute__((export_name("std_gui_web_frame")))
int gui_web_frame_(double timestamp) {
    match (gWebApp) {
        case null: return 0;
        case some(app): {
            if (app.pump(0)) { return 1; }
            gWebApp = null;
            return 0;
        }
    }
    return 0;
}

} // namespace std
