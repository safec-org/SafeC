// SafeC Standard Library — GUI backend: bare metal (a linear framebuffer,
// as set up by a bootloader — e.g. Multiboot2's framebuffer tag, a VESA
// VBE mode, or a board's fixed display-controller memory map). See gui.h
// for the portable API this implements.
//
// Unlike the OS-native backends (Cocoa/Win32/X11), there is no windowing
// system here — 'GuiWindow::create()' doesn't open anything, it just
// validates that gui_fb_configure() was called first; the portable layer
// allocates the same CPU-side RGBA8888 draw buffer every backend uses, and
// GuiWindow::present() copies (and pixel-format-converts) that buffer into
// the real framebuffer memory at the window's position (set_position(),
// default 0,0 — several windows may tile one screen). There is also no
// OS input stack — a bare-metal target's own keyboard/mouse/touch ISR is responsible for calling
// gui_fb_inject_event() with whatever it decodes; GuiWindow::poll_event() just
// drains the small queue that fills. Wire up your own PS/2/USB-HID/etc.
// driver to call gui_fb_inject_event() — this file has no opinion on
// which one you use.
//
// Call gui_fb_configure() exactly once, before GuiWindow::create(), with
// the real framebuffer's base address/pitch/pixel format — typically
// values your bootloader handed you (e.g. a Multiboot2 framebuffer tag's
// framebuffer_addr/framebuffer_pitch/framebuffer_bpp fields).
// safeguard: implements <std/gui/gui.h>   (the bare-metal backend; gui_fb.h is its setup)
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_fb.h>
#include <std/mem.h>

namespace std {

static struct GuiFbConfig gFbConfig;
static int gFbConfigured = 0;
static struct GuiBackend gFbBackend;
static int gFbBackendInit = 0;
static double gFbVirtualMs = 0.0;
static fn long long() gFbClock;
static int gFbClockSet = 0;

void gui_fb_configure(struct GuiFbConfig cfg) {
    gFbConfig = cfg;
    gFbConfigured = 1;
}

void gui_fb_set_clock(fn long long() nowNs) {
    gFbClock = nowNs;
    gFbClockSet = 1;
}

// The ring an input ISR fills (no allocation on that path); pump() moves
// it into the portable queue.
struct GuiFbPlatform {
    struct GuiEvent queue[GUI_FB_QUEUE_CAP];
    int queueHead;
    int queueTail;
};

// A target's own input driver/ISR calls this with whatever it decodes
// (e.g. a PS/2 scancode translated to a GuiEvent) — GuiWindow::poll_event() then
// hands it back out through the normal API, same as every other backend.
void gui_fb_inject_event(&GuiWindow win, struct GuiEvent ev) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return;
        case some(st): {
            unsafe {
                if (st.native.is_null()) { return; }
                struct GuiFbPlatform* p = (struct GuiFbPlatform*)st.native;
                int next = (p->queueTail + 1) % GUI_FB_QUEUE_CAP;
                if (next == p->queueHead) { return; } // full — drop
                p->queue[p->queueTail] = ev;
                p->queueTail = next;
            }
        }
    }
}

static int __gui_fb_init() { return gFbConfigured; }

static int __gui_fb_create(&GuiWindowState st, const &GuiWindowOptions opts) {
    if (!gFbConfigured) { return 0; }
    unsafe {
        struct GuiFbPlatform* p = (struct GuiFbPlatform*)alloc_zeroed_raw_(sizeof(struct GuiFbPlatform));
        if (p == (struct GuiFbPlatform*)0) { return 0; }
        st.native = (?&Erased)p;
        st.key = (unsigned long long)(unsigned long)p;
    }
    st.scale = 1.0;
    st.pixelW = st.logicalW;
    st.pixelH = st.logicalH;
    if (st.x == GUI_POS_DEFAULT) { st.x = 0; }
    if (st.y == GUI_POS_DEFAULT) { st.y = 0; }
    st.focused = 1;
    return 1;
}

static void __gui_fb_destroy(&GuiWindowState st) {
    unsafe {
        if (!st.native.is_null()) { dealloc((?&heap Erased)st.native); }
    }
    st.native = null;
}

// The portable layer draws RGBA8888 into the window's buffer; this
// backend's job is turning that into whatever the real hardware
// framebuffer's pixel format is when presenting.
static void __gui_fb_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                             int rx, int ry, int rw, int rh) {
    if (!gFbConfigured) { return; }
    unsafe {
        unsigned char* fb = (unsigned char*)gFbConfig.base;
        const unsigned char* pixels = (const unsigned char*)px;
        int y = ry;
        while (y < ry + rh) {
            int sy = st.y + y;
            if (sy >= 0 && sy < gFbConfig.height) {
                unsigned long srcRow = (unsigned long)y * (unsigned long)bw * 4UL;
                unsigned long dstRow = (unsigned long)sy * gFbConfig.pitch;
                int x = rx;
                while (x < rx + rw) {
                    int sx = st.x + x;
                    if (sx >= 0 && sx < gFbConfig.width) {
                        unsigned long si = srcRow + (unsigned long)x * 4UL;
                        unsigned char r = pixels[si + 0UL];
                        unsigned char g = pixels[si + 1UL];
                        unsigned char b = pixels[si + 2UL];
                        if (gFbConfig.bpp == 32) {
                            unsigned int packedColor = ((unsigned int)r << gFbConfig.redShift) |
                                                   ((unsigned int)g << gFbConfig.greenShift) |
                                                   ((unsigned int)b << gFbConfig.blueShift);
                            unsigned long di = dstRow + (unsigned long)sx * 4UL;
                            fb[di + 0UL] = (unsigned char)(packedColor & 0xFFU);
                            fb[di + 1UL] = (unsigned char)((packedColor >> 8) & 0xFFU);
                            fb[di + 2UL] = (unsigned char)((packedColor >> 16) & 0xFFU);
                            fb[di + 3UL] = (unsigned char)((packedColor >> 24) & 0xFFU);
                        } else if (gFbConfig.bpp == 16) {
                            // RGB565.
                            unsigned int r5 = ((unsigned int)r >> 3) & 0x1FU;
                            unsigned int g6 = ((unsigned int)g >> 2) & 0x3FU;
                            unsigned int b5 = ((unsigned int)b >> 3) & 0x1FU;
                            unsigned int packed16 = (r5 << 11) | (g6 << 5) | b5;
                            unsigned long di = dstRow + (unsigned long)sx * 2UL;
                            fb[di + 0UL] = (unsigned char)(packed16 & 0xFFU);
                            fb[di + 1UL] = (unsigned char)((packed16 >> 8) & 0xFFU);
                        }
                    }
                    x = x + 1;
                }
            }
            y = y + 1;
        }
    }
}

static void __gui_fb_set_title(&GuiWindowState st, const &char title) { }  // no window chrome on bare metal

static void __gui_fb_set_size(&GuiWindowState st, int w, int h) {
    st.logicalW = w; st.logicalH = h;
    st.pixelW = w; st.pixelH = h;
    struct GuiEvent ev = GuiEvent::new(GUI_EVENT_RESIZE);
    ev.width = w;
    ev.height = h;
    gui_backend_post(st, ev);
}

static void __gui_fb_set_position(&GuiWindowState st, int x, int y) {
    st.x = x;
    st.y = y;
    struct GuiEvent ev = GuiEvent::new(GUI_EVENT_MOVE);
    ev.x = (double)x;
    ev.y = (double)y;
    gui_backend_post(st, ev);
}

static void __gui_fb_noop(&GuiWindowState st) { }
static void __gui_fb_set_int(&GuiWindowState st, int v) { }

static int __gui_fb_pump(int timeoutMs) {
    int any = 0;
    unsigned long n = gui_backend_window_count();
    unsigned long i = 0UL;
    while (i < n) {
        ?&GuiWindowState ost = gui_backend_window_at(i);
        match (ost) {
            case some(st): {
                unsafe {
                    if (!st.native.is_null()) {
                        struct GuiFbPlatform* p = (struct GuiFbPlatform*)st.native;
                        while (p->queueHead != p->queueTail) {
                            struct GuiEvent ev = p->queue[p->queueHead];
                            p->queueHead = (p->queueHead + 1) % GUI_FB_QUEUE_CAP;
                            gui_backend_post(st, ev);
                            any = 1;
                        }
                    }
                }
            }
            case null: { }
        }
        i = i + 1UL;
    }
    // No clock of its own to sleep on: a wait just lets the virtual
    // clock move (see gui_fb_set_clock for a real one).
    if (!any && timeoutMs > 0 && !gFbClockSet) { gFbVirtualMs = gFbVirtualMs + (double)timeoutMs; }
    return any;
}

static long long __gui_fb_now_ns() {
    if (gFbClockSet) { return gFbClock(); }
    return (long long)(gFbVirtualMs * 1000000.0);
}

static int __gui_fb_clip_set(const &char utf8, unsigned long n) { return 0; }
static int __gui_fb_clip_get(&String out) { return 0; }
static double __gui_fb_scale() { return 1.0; }

static struct GuiNativeHandle __gui_fb_native(&GuiWindowState st) {
    struct GuiNativeHandle h;
    h.kind = GUI_NATIVE_FB;
    unsafe { h.display = (unsigned long long)(unsigned long)gFbConfig.base; }
    h.window = st.key;
    h.view = 0ULL;
    h.visual = 0ULL;
    return h;
}

&GuiBackend gui_platform_backend() {
    if (!gFbBackendInit) {
        gFbBackend.name = "fb";
        gFbBackend.primaryMod = GUI_MOD_CTRL;
        gFbBackend.init = __gui_fb_init;
        gFbBackend.window_create = __gui_fb_create;
        gFbBackend.window_destroy = __gui_fb_destroy;
        gFbBackend.window_present = __gui_fb_present;
        gFbBackend.window_set_title = __gui_fb_set_title;
        gFbBackend.window_set_size = __gui_fb_set_size;
        gFbBackend.window_set_position = __gui_fb_set_position;
        gFbBackend.window_update_limits = __gui_fb_noop;
        gFbBackend.window_set_visible = __gui_fb_set_int;
        gFbBackend.window_focus = __gui_fb_noop;
        gFbBackend.window_set_cursor = __gui_fb_set_int;
        gFbBackend.pump = __gui_fb_pump;
        gFbBackend.wake = (fn void())0;
        gFbBackend.now_ns = __gui_fb_now_ns;
        gFbBackend.clipboard_set = __gui_fb_clip_set;
        gFbBackend.clipboard_get = __gui_fb_clip_get;
        gFbBackend.display_scale = __gui_fb_scale;
        gFbBackend.native_handle = __gui_fb_native;
        gFbBackendInit = 1;
    }
    return &gFbBackend;
}

} // namespace std
