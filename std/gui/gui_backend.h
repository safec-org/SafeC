#pragma once
// SafeC Standard Library — GUI: the windowing-backend interface.
//
// Everything platform-specific in std/gui sits behind one table of
// function pointers, struct GuiBackend below. The portable layer
// (gui_window.sc: GuiWindow, the clipboard, the clock; gui_app.sc: the
// event loop) calls only through it, and a backend calls back only into
// the few gui_backend_*() functions declared here. A backend is therefore
// one .sc file that fills in a GuiBackend and never touches the canvas,
// text or widget code — those only ever see a GuiWindow's pixel buffer.
//
// Backends in std/gui today:
//   gui_x11.sc      Xlib (Linux, BSD)          GUI_NATIVE_X11
//   gui_win32.sc    Win32 (user32/gdi32)       GUI_NATIVE_WIN32
//   gui_cocoa.sc    AppKit via the ObjC runtime GUI_NATIVE_COCOA
//   gui_fb.sc       bare-metal framebuffer     GUI_NATIVE_FB
//   gui_headless.sc in memory (any target)     GUI_NATIVE_HEADLESS
//
// ── Choosing a backend ────────────────────────────────────────────────────
// Each platform file defines gui_platform_backend(); safeguard links the
// one whose filename suffix matches the target (safeguard/src/
// StdModules.cpp, stdVariantMatches: _x11 on Linux/BSD, _win32 on Windows,
// _cocoa on macOS, _fb on bare metal). gui_backend() returns the backend in
// use: the one installed with gui_set_backend() if any, else
// gui_platform_backend(). gui_set_backend() works only while no window
// exists — a program (or a backend module's own setup call, like
// gui_use_headless()) picks its backend before creating the first window.
//
// So a new backend is added one of two ways:
//   * As the platform's default: a new suffix rule in stdVariantMatches
//     (e.g. "_wasm" for wasm32/wasm64 targets, which today link none) and
//     a gui_<suffix>.sc defining gui_platform_backend().
//   * As an alternative on a platform that already has a default — e.g.
//     Wayland next to X11 on Linux: a gui_wayland.sc that defines
//     gui_wayland_backend() (NOT gui_platform_backend(), or both would
//     define it) and a setup call gui_use_wayland() that calls
//     gui_set_backend(gui_wayland_backend()). To make it automatic, the X11
//     file's gui_platform_backend() can prefer it at run time
//     ($WAYLAND_DISPLAY set and the Wayland unit linked: guard it behind a
//     '// safeguard: feature wayland' unit and a SAFEC_FEATURE_WAYLAND
//     check). Keep libwayland's link/deb lines in the Wayland file's own
//     '// safeguard: link linux ...' / '// safeguard: deb linux ...'
//     directives.
//
// ── The contract ──────────────────────────────────────────────────────────
// Threading: every entry is called on the thread that created the first
// window ("the UI thread"), except wake(), which may be called from any
// thread.
//
// init()       Called once, lazily, before the first window_create (or the
//              first clipboard/clock call). Connect to the display server.
//              Return 1, or 0 if there is none (then every window creation
//              fails cleanly: GuiWindow.platform stays null).
// window_create(st, opts)
//              Create and (if opts.visible) show a window whose client area
//              is opts.width x opts.height logical units, honoring the
//              other options as far as the platform can. Before returning
//              1, set:
//                st.key       a nonzero value identifying the native window
//                             (X11 Window id, HWND, NSWindow*), used with
//                             gui_backend_find() to route native events;
//                st.native    the backend's own per-window data, if any;
//                st.scale     buffer pixels per logical unit (1.0 unless
//                             opts.hiDpi and the display is high-DPI);
//                st.pixelW/H  the buffer size in pixels (logical * scale);
//                st.x/st.y    the position if known (else GUI_POS_DEFAULT).
//              The portable layer allocates the pixel buffer itself.
// window_destroy(st)
//              Destroy the native window and free st.native. Events still
//              queued for it are discarded by the caller.
// window_present(st, pixels, bufW, bufH, x, y, w, h)
//              Show the rectangle (x,y,w,h) of 'pixels' (premultiplied
//              RGBA8888, R first in memory, bufW*4 bytes per row) in the
//              window. Convert to the native pixel order here (X11 and
//              Win32 want BGRA; Cocoa takes RGBA). The buffer may be
//              smaller or larger than the window for a moment around a
//              resize; present what overlaps.
// window_set_title / window_set_size (logical) / window_set_position
//              (logical) / window_set_visible / window_focus /
//              window_set_cursor (GUI_CURSOR_*) / window_update_limits
//              (st.minW/minH/maxW/maxH changed, logical units, 0 = none)
//              Apply the change. A resize the window system carries out is
//              reported back like a user resize: post GUI_EVENT_RESIZE.
// pump(timeoutMs)
//              Read every pending native event and post the resulting
//              GuiEvents with gui_backend_post(). If none was pending, wait
//              up to timeoutMs for one (0 = don't wait, negative = until an
//              event or wake()). Return 1 if anything was posted.
//              The events to post, all in buffer pixels:
//                CLOSE (the close button; do NOT destroy the window — the
//                application decides), RESIZE (width/height = new buffer
//                size; also update st.pixelW/H, st.logicalW/H), MOVE,
//                FOCUS_IN/OUT (also set st.focused), MOUSE_DOWN/UP/MOVE/
//                ENTER/LEAVE, SCROLL, KEY_DOWN/UP (keycode as GUI_KEY_*,
//                modifiers, and on KEY_DOWN the typed character in
//                codepoint), CHAR (text with no key event, e.g. from an
//                input method), SCALE (the scale factor changed; set
//                st.scale and st.pixelW/H first), EXPOSE (contents lost).
//              gui_backend_post() fills in windowId, timeMs and the
//              double-click count; the backend doesn't.
// wake()       Make a blocked pump() return soon. May be null.
// now_ns()     A monotonic clock in nanoseconds.
// clipboard_set(utf8, len) / clipboard_get(out)
//              The system clipboard as UTF-8 text (1 = ok / 0 = no text).
// display_scale()
//              The scale factor a hiDpi window would get now.
// native_handle(st)
//              The handles a GPU presenter needs (struct GuiNativeHandle).
// primaryMod   GUI_MOD_META on macOS, GUI_MOD_CTRL elsewhere.
//
// ── Presenters: GPU-accelerated presentation ─────────────────────────────
// Rendering stays on the CPU (that is what makes it identical everywhere),
// but getting the finished buffer onto the screen can go through the GPU
// instead of the backend's software blit (XPutImage, StretchDIBits, a
// CALayer's contents): upload it into a texture and draw one quad, with
// OpenGL/EGL (X11, Wayland), Metal (a CAMetalLayer on the Cocoa view) or
// Direct3D 11 (a DXGI swap chain on the HWND). That is a GuiPresenter: a
// separate table attached per window with gui_window_set_presenter(); the
// portable present() calls presenter.present instead of
// backend.window_present while one is attached. attach() gets the native
// handles from backend.native_handle() and keeps its own state in
// st.presenterData; present() gets exactly window_present's arguments
// (premultiplied RGBA8888, the damaged rectangle — upload only that);
// detach() runs when the presenter is replaced or the window destroyed.
// A presenter must cope with the buffer size changing between presents
// (recreate its texture / resize its swap chain when bufW/bufH change).
//
// ── Web (WebAssembly <canvas>) ────────────────────────────────────────────
// A browser can't block: its backend's pump() never waits, and the program
// drives GuiApp::pump(0) from requestAnimationFrame instead of calling
// GuiApp::run() (gui_app.h's "Driving the loop yourself"). window_present
// hands the RGBA buffer to an imported JS function (ImageData is
// RGBA, non-premultiplied: un-premultiply or use an opaque window); the
// DOM's input events reach the program through exported functions that
// call gui_backend_post() on the window found with gui_backend_find().

#include <std/gui/gui.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

// GuiNativeHandle.kind
#define GUI_NATIVE_NONE     0
#define GUI_NATIVE_X11      1   // display = Display*, window = Window, visual = Visual*
#define GUI_NATIVE_WIN32    2   // display = HINSTANCE, window = HWND
#define GUI_NATIVE_COCOA    3   // display = NSApplication*, window = NSWindow*, view = NSView*
#define GUI_NATIVE_WAYLAND  4   // display = wl_display*, window = wl_surface*
#define GUI_NATIVE_WEB      5   // window = the canvas' id in the host page
#define GUI_NATIVE_FB       6   // display = framebuffer base address
#define GUI_NATIVE_HEADLESS 7

namespace std {

// Pointer-sized native handles, widened to 64 bits so the struct is the
// same on every target (Win64's unsigned long is 32 bits).
struct GuiNativeHandle {
    int kind;                    // GUI_NATIVE_*
    unsigned long long display;
    unsigned long long window;
    unsigned long long view;
    unsigned long long visual;
};

// The backend-neutral state behind GuiWindow.platform. Backends read and
// update the fields their contract above names; the rest is the portable
// layer's.
struct GuiWindowState {
    unsigned int id;              // == GuiWindow.id
    unsigned long long key;       // backend: native window identifier (see window_create)
    ?&void native;                // backend-private per-window data
    ?&void presenterData;         // presenter-private per-window data
    ?&void presenter;             // the attached struct GuiPresenter (gui_window_set_presenter); null = the backend's own window_present

    struct Vec<struct GuiEvent> queue;  // pending events, FIFO from 'qhead'
    unsigned long qhead;

    struct String title;
    int x; int y;                 // logical position, GUI_POS_DEFAULT if unknown
    int logicalW; int logicalH;   // logical client size
    int pixelW; int pixelH;       // buffer size the window currently wants
    double scale;
    int hiDpi;
    int minW; int minH; int maxW; int maxH;
    int visible;
    int focused;
    int hovered;
    int cursor;
    int resizable;
    int decorated;
    int topmost;

    // double-click detection (gui_backend_post)
    double lastDownMs; double lastDownX; double lastDownY; int lastDownButton; int clickCount;
};

struct GuiBackend {
    const char* name;
    int primaryMod;   // GUI_MOD_META on macOS, GUI_MOD_CTRL elsewhere

    fn int() init;
    fn int(&GuiWindowState, const &GuiWindowOptions) window_create;
    fn void(&GuiWindowState) window_destroy;
    fn void(&GuiWindowState, const &unsigned char, int, int, int, int, int, int) window_present;
    fn void(&GuiWindowState, const &char) window_set_title;
    fn void(&GuiWindowState, int, int) window_set_size;
    fn void(&GuiWindowState, int, int) window_set_position;
    fn void(&GuiWindowState) window_update_limits;
    fn void(&GuiWindowState, int) window_set_visible;
    fn void(&GuiWindowState) window_focus;
    fn void(&GuiWindowState, int) window_set_cursor;
    fn int(int) pump;
    fn void() wake;               // may be null
    fn long long() now_ns;
    fn int(const &char, unsigned long) clipboard_set;
    fn int(&String) clipboard_get;
    fn double() display_scale;
    fn struct GuiNativeHandle(&GuiWindowState) native_handle;
};

struct GuiPresenter {
    const char* name;
    fn int(&GuiWindowState, struct GuiNativeHandle) attach;   // 1 = ok
    fn void(&GuiWindowState, const &unsigned char, int, int, int, int, int, int) present;
    fn void(&GuiWindowState) detach;
};

// Defined by the platform backend file linked for the target (see above).
&GuiBackend gui_platform_backend();

// The backend in use (installed one, else the platform's), initialized.
&GuiBackend gui_backend();
// Installs 'b' as the backend for this process. Returns 0 (and changes
// nothing) once a window exists.
int gui_set_backend(&GuiBackend b);
// 1 once the backend's init() succeeded (calls it if it hasn't run yet).
int gui_backend_ready();

// For backends: queue 'ev' for the window (fills windowId, timeMs and, for
// GUI_EVENT_MOUSE_DOWN, clicks).
void gui_backend_post(&GuiWindowState st, struct GuiEvent ev);
// For backends: the live window whose st.key is 'key'.
?&GuiWindowState gui_backend_find(unsigned long long key);
// For backends and the event loop: the live windows, in creation order.
unsigned long gui_backend_window_count();
?&GuiWindowState gui_backend_window_at(unsigned long i);
// 1 if any live window has an event queued.
int gui_backend_has_events();

// Attaches 'p' (null: back to the backend's software present) to the
// window, detaching the previous presenter. Returns 0 if p.attach failed
// (the window then keeps the backend's present).
int gui_window_set_presenter(&GuiWindow win, ?&GuiPresenter p);
// The window's state (null for an offscreen window).
?&GuiWindowState gui_window_state(const &GuiWindow win);

} // namespace std
