#pragma once
// SafeC Standard Library — GUI: cross-platform windows + software framebuffer.
//
// The std/gui stack, bottom to top (include only what you use):
//   gui.h          windows, events, clipboard, cursors, the pixel buffer
//   gui_draw.h     the original integer primitives + 8x8 bitmap font
//   gui_canvas.h   anti-aliased 2D vector canvas (paths, gradients, images,
//                  transforms, clipping, blend modes, TrueType text)
//   gui_text.h     TrueType/OpenType fonts and text layout
//   gui_widget.h   retained widget toolkit (layout, theming, ~35 widgets)
//   gui_app.h      application object: many windows, one event loop,
//                  timers, animation frames
//   gui_backend.h  the interface a windowing backend implements
//   gui_headless.h an in-memory backend (tests, CI, offscreen rendering)
//
// Exactly one platform backend is linked — safeguard picks it by the
// file's suffix for the target (see gui_backend.h for the contract):
//   std/gui/gui_cocoa.sc   macOS, via the Objective-C runtime + Core Graphics
//   std/gui/gui_win32.sc   Windows, via the Win32 API
//   std/gui/gui_x11.sc     Linux/BSD/Unix, via Xlib
//   std/gui/gui_fb.sc      bare metal, via a linear framebuffer (see gui_fb.h)
// A program can also install another backend at run time before its first
// window (gui_set_backend(), gui_backend.h) — gui_headless.h's in-memory
// one, or a later Wayland/WebAssembly one.
//
// Every backend presents the same model: a window owns a CPU-side pixel
// buffer, 'pixels' — RGBA8888, premultiplied alpha, top-left origin,
// row-major, 4 bytes per pixel, R first in memory on every platform — that
// the application draws into (directly, through gui_draw.h, through
// gui_canvas.h, or through the widget toolkit), and win.present() puts it
// on the screen (the backend converts to its native pixel order on the
// way, or hands it to a GPU presenter: see gui_backend.h's GuiPresenter).
// Rasterization is always on the CPU, deterministic and identical on every
// backend: the same frame is the same bytes on Cocoa, Win32, X11, bare
// metal and the headless backend.
//
// Sizes: a window's size is given in *logical* units (points on macOS,
// device-independent pixels on Windows, pixels on X11). With
// GuiWindowOptions.hiDpi set, the pixel buffer is the window's size times
// the display's scale factor (win.scale, e.g. 2.0 on a Retina display), so
// drawing is sharp; otherwise scale is 1 and the OS scales the window up.
// Event coordinates and win.width/height are always in buffer pixels.

// Event/key constants are plain #defines (not enums) to match the rest of
// the standard library's convention for cross-namespace constant access
// (see e.g. std/serial/protobuf.h's PB_WIRE_* ) — a real 'enum' declared
// inside 'namespace std' can't be referenced from user code as
// 'std::SOME_CONSTANT' the way a free function or #define can.
#define GUI_EVENT_NONE        0
#define GUI_EVENT_CLOSE       1   // the user asked to close the window (title-bar close, Alt+F4, ...)
#define GUI_EVENT_RESIZE      2   // width/height: the new buffer size (already applied to win.pixels)
#define GUI_EVENT_MOUSE_DOWN  3
#define GUI_EVENT_MOUSE_UP    4
#define GUI_EVENT_MOUSE_MOVE  5
#define GUI_EVENT_KEY_DOWN    6   // keycode + modifiers; codepoint = the typed character, if any
#define GUI_EVENT_KEY_UP      7
#define GUI_EVENT_CHAR        8   // a typed character with no KEY_DOWN to carry it (IME, compose)
#define GUI_EVENT_SCROLL      9   // scrollDx/scrollDy in "lines" (+y = away from the user / up)
#define GUI_EVENT_FOCUS_IN    10  // the window became the keyboard-focus window
#define GUI_EVENT_FOCUS_OUT   11
#define GUI_EVENT_MOVE        12  // x/y: the window's new position on screen (logical units)
#define GUI_EVENT_MOUSE_ENTER 13
#define GUI_EVENT_MOUSE_LEAVE 14
#define GUI_EVENT_SCALE       15  // the scale factor changed (moved to another display); width/height = new buffer size
#define GUI_EVENT_EXPOSE      16  // the OS lost the window's contents: present again

// Portable key codes for GUI_EVENT_KEY_DOWN/UP's 'keycode' field — each
// backend translates its own native key codes into these, so application
// code never sees a platform-specific key code. Letter keys are their
// upper-case ASCII value ('A'..'Z' = 65..90) and digit keys theirs
// ('0'..'9' = 48..57), whatever the keyboard layout's shift state.
#define GUI_KEY_UNKNOWN   0
#define GUI_KEY_LEFT      1
#define GUI_KEY_RIGHT     2
#define GUI_KEY_UP        3
#define GUI_KEY_DOWN      4
#define GUI_KEY_ENTER     5
#define GUI_KEY_ESCAPE    6
#define GUI_KEY_TAB       7
#define GUI_KEY_BACKSPACE 8
#define GUI_KEY_SPACE     9
#define GUI_KEY_DELETE    10
#define GUI_KEY_HOME      11
#define GUI_KEY_END       12
#define GUI_KEY_PAGE_UP   13
#define GUI_KEY_PAGE_DOWN 14
#define GUI_KEY_INSERT    15
#define GUI_KEY_F1        16  // F1..F12 = 16..27
#define GUI_KEY_F12       27
#define GUI_KEY_SHIFT     28
#define GUI_KEY_CONTROL   29
#define GUI_KEY_ALT       30
#define GUI_KEY_META      31  // Command (macOS), Windows/Super key
#define GUI_KEY_MENU      32

// GuiEvent.modifiers bits (held while the event happened).
#define GUI_MOD_SHIFT 1
#define GUI_MOD_CTRL  2
#define GUI_MOD_ALT   4   // Option on macOS
#define GUI_MOD_META  8   // Command on macOS, Windows/Super key elsewhere

// GuiEvent.button.
#define GUI_BUTTON_LEFT   0
#define GUI_BUTTON_RIGHT  1
#define GUI_BUTTON_MIDDLE 2

// Mouse cursor shapes (GuiWindow::set_cursor).
#define GUI_CURSOR_ARROW       0
#define GUI_CURSOR_IBEAM       1
#define GUI_CURSOR_HAND        2
#define GUI_CURSOR_CROSSHAIR   3
#define GUI_CURSOR_RESIZE_EW   4   // ↔
#define GUI_CURSOR_RESIZE_NS   5   // ↕
#define GUI_CURSOR_RESIZE_NWSE 6
#define GUI_CURSOR_RESIZE_NESW 7
#define GUI_CURSOR_MOVE        8
#define GUI_CURSOR_WAIT        9
#define GUI_CURSOR_NOT_ALLOWED 10
#define GUI_CURSOR_HIDDEN      11
#define GUI_CURSOR_COUNT       12

// GuiWindowOptions.x/y: let the window system place the window.
#define GUI_POS_DEFAULT (-2147483647)

#include <std/collections/string.h>
#include <std/gui/gui_utf8.h>

namespace std {

// An RGBA color, straight (not premultiplied) alpha — the canvas
// premultiplies when it blends. Lives here (rather than in gui_draw.h) so
// GuiWindow can declare its drawing methods without the core header
// depending on any add-on header.
struct GuiColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;

    // Defined in gui_draw.sc. rgb() is opaque (a = 255).
    static struct GuiColor rgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    static struct GuiColor rgb(unsigned char r, unsigned char g, unsigned char b);
    // 0xRRGGBB (opaque) / 0xRRGGBBAA.
    static struct GuiColor hex(unsigned int rrggbb);
    static struct GuiColor hexa(unsigned int rrggbbaa);
    // From floats in 0..1 (clamped).
    static struct GuiColor rgbaf(float r, float g, float b, float a);
    // Hue in degrees (0..360), saturation/value/alpha in 0..1.
    static struct GuiColor hsv(float h, float s, float v, float a);
    // This color with alpha multiplied by 'f' (0..1).
    struct GuiColor with_alpha(float f) const;
    // The straight mix (1-t)*self + t*other, alpha included.
    struct GuiColor mix(struct GuiColor other, float t) const;
    // Lighter (amount > 0) or darker (amount < 0), amount in -1..1.
    struct GuiColor shade(float amount) const;
    // 0xRRGGBBAA.
    unsigned int packed() const;
    // h (degrees), s, v in 0..1.
    void to_hsv(&float h, &float s, &float v) const;
};

struct GuiEvent {
    int          kind;       // GUI_EVENT_*
    double       x;          // mouse position, window-local, top-left origin, buffer pixels
    double       y;          // (GUI_EVENT_MOVE: the window's position)
    int          button;     // GUI_BUTTON_* (mouse events)
    int          keycode;    // GUI_KEY_* (key events)
    unsigned int codepoint;  // Unicode codepoint (GUI_EVENT_CHAR, and KEY_DOWN of a printable key)
    double       scrollDx;
    double       scrollDy;
    int          width;      // new buffer size (GUI_EVENT_RESIZE / GUI_EVENT_SCALE)
    int          height;
    int          defaultPrevented; // see prevent_default() below
    int          modifiers;  // GUI_MOD_* bits
    int          clicks;     // MOUSE_DOWN: 1, 2 for a double click, 3 for a triple click
    unsigned int windowId;   // the window this event is for (GuiWindow.id)
    double       timeMs;     // gui_time_ms() when the backend received it

    // A zeroed event of 'kind'.
    static struct GuiEvent new(int kind);

    // Suppresses the default action for this event: Widget::dispatch_event()'s
    // default key handling (gui_widget.h), and GuiApp closing a window on
    // GUI_EVENT_CLOSE (gui_app.h). Defined in gui_window.sc.
    void prevent_default();
    // 1 if 'mod' (GUI_MOD_*) was held. primary_mod(): Command on macOS,
    // Ctrl elsewhere — the modifier of Copy/Paste/Select-all shortcuts.
    int  has_mod(int mod) const;
    int  primary_mod() const;
};

// Everything GuiWindow::create_with() can be told. Start from
// GuiWindowOptions::new(title, w, h) and change fields.
struct GuiWindowOptions {
    const char* title;     // UTF-8
    int x;                 // position in logical units, or GUI_POS_DEFAULT
    int y;
    int width;             // client-area size, logical units
    int height;
    int minWidth;          // 0 = no limit
    int minHeight;
    int maxWidth;
    int maxHeight;
    int resizable;         // default 1
    int decorated;         // title bar + border; default 1
    int visible;           // shown on creation; default 1
    int hiDpi;             // buffer in physical pixels (scale > 1 on high-DPI displays); default 1
    int topmost;           // stays above other windows (tool windows, popups); default 0

    static struct GuiWindowOptions new(const &char title, int width, int height);
};

struct GuiWindow {
    ?&void               platform; // backend-neutral window state (struct GuiWindowState, gui_backend.h); null for an offscreen window or on failure
    &heap unsigned char  pixels;   // width*height*4 bytes, premultiplied RGBA8888
    int                  width;    // buffer size in pixels
    int                  height;
    int                  shouldClose;
    double               scale;    // buffer pixels per logical unit (1.0 unless hiDpi on a high-DPI display)
    unsigned int         id;       // unique per window (0 for an offscreen window)

    // Creates and shows a window with a zeroed (transparent black) pixel
    // buffer of 'width'x'height'. On failure, returned window's 'platform' is
    // null and 'width'/'height' are 0 — callers should check before use.
    // Equivalent to create_with(GuiWindowOptions::new(title, width, height))
    // with hiDpi off (the buffer is exactly width x height pixels).
    static struct GuiWindow create(const &char title, int width, int height);
    static struct GuiWindow create_with(const &GuiWindowOptions opts);

    // A window that is only a pixel buffer — nothing on screen, platform
    // null, poll_event() never returns anything, present() does nothing.
    // For rendering to memory (thumbnails, tests, exporting a frame).
    static struct GuiWindow offscreen(int width, int height);

    // Pumps the platform's event queue (never blocks) and returns the next
    // event for this window in 'outEvent' (1), or 0 if there is none —
    // call in a loop ("while (win.poll_event(&ev)) { ... }") once per
    // frame. Updates shouldClose on GUI_EVENT_CLOSE, and on
    // GUI_EVENT_RESIZE/GUI_EVENT_SCALE reallocates 'pixels' (zeroed) to
    // the new size before returning the event. Several windows can be
    // polled independently: each gets only its own events.
    int  poll_event(&GuiEvent outEvent);

    // Like poll_event(), but if nothing is pending, waits up to
    // 'timeoutMs' (negative = forever) for an event to arrive.
    int  wait_event(&GuiEvent outEvent, int timeoutMs);

    // Puts 'pixels' on the screen. Call once per frame after drawing.
    // present_rect() updates only that part (the rest is assumed unchanged
    // since the last present).
    void present();
    void present_rect(int x, int y, int w, int h);

    // Sets pixel (x,y) to 'rgba' (0xRRGGBBAA), unblended, no-op if out of
    // bounds. (The value is stored as given; for anything but opaque
    // colors draw through gui_canvas.h, which blends and premultiplies.)
    void set_pixel(int x, int y, unsigned int rgba);
    // The stored 0xRRGGBBAA at (x,y); 0 if out of bounds.
    unsigned int get_pixel(int x, int y) const;
    // Fills the whole buffer with 'color' (premultiplied on the way).
    void clear(struct GuiColor color);

    // Releases the window and its pixel buffer.
    void drop();
    void destroy() consume;   // = drop()

    // ── window management (no-ops for an offscreen window) ───────────────
    void set_title(const &char title);
    // Client-area size in logical units. The buffer follows with a
    // GUI_EVENT_RESIZE from poll_event().
    void set_size(int width, int height);
    void set_position(int x, int y);
    // Last known position in logical units (GUI_POS_DEFAULT if unknown).
    int  pos_x() const;
    int  pos_y() const;
    // Logical size limits; 0 = none.
    void set_min_size(int width, int height);
    void set_max_size(int width, int height);
    void show();
    void hide();
    int  is_visible() const;
    // Raises the window and asks for keyboard focus.
    void focus();
    // 1 while this window has keyboard focus.
    int  is_focused() const;
    // GUI_CURSOR_* shown while the pointer is over this window.
    void set_cursor(int cursor);
    // Logical size (buffer size / scale).
    int  logical_width() const;
    int  logical_height() const;
    // Asks the backend for the window to be closed, as if the user clicked
    // its close button: a GUI_EVENT_CLOSE is queued.
    void request_close();
    // Native window identifier for GPU presenters and interop: the X11
    // Window id, the HWND, or the NSWindow pointer; 0 if none.
    unsigned long long native_handle() const;

    // ── drawing (software rasterizer; defined in gui_draw.sc / gui_svg.sc)
    // All of these write through set_pixel(), so they work on every
    // backend; out-of-bounds pixels are clipped. For anti-aliased shapes,
    // gradients, images and TrueType text use gui_canvas.h instead.
    void fill_rect(int x, int y, int w, int h, struct GuiColor color);
    void draw_rect_border(int x, int y, int w, int h, struct GuiColor color, int thickness);
    void draw_line(int x0, int y0, int x1, int y1, struct GuiColor color);

    // Draws 'text' in the built-in 8x8 font with its top-left at (x,y),
    // each glyph cell 8*scale by 8*scale pixels. Codepoints outside
    // printable ASCII (0x20-0x7E) draw as a blank cell. Returns the total
    // pixel width drawn. For a loaded/custom font use GuiFont::draw_text()
    // (gui_font.h); for its metrics see gui_draw.h's gui_text_width().
    int  draw_text(int x, int y, const &char text, struct GuiColor color, int scale);

    // Parses 'svgText' (a complete <svg>...</svg> document, see gui_svg.h
    // for the supported subset) and draws it with its top-left at
    // (ox, oy). Returns 1 if the <svg> root parsed, 0 on malformed or
    // unsupported input (shapes before a parse error are already drawn).
    // Defined in gui_svg.sc.
    int  draw_svg(int ox, int oy, const &char svgText);
};

// ── process-wide services (defined in gui_window.sc, through the backend) ──

// Milliseconds on a monotonic clock (the headless backend's is virtual and
// only moves with gui_headless_advance()).
double gui_time_ms();

// The system clipboard, as UTF-8 text. set returns 1 on success; get
// replaces 'out' with the clipboard's text and returns 1, or returns 0
// (leaving 'out' empty) when there is no text on it. Needs a window to
// exist on X11 (the clipboard is owned by a window there).
int  gui_clipboard_set_text(const &char utf8);
int  gui_clipboard_get_text(&String out);

// GUI_MOD_META on macOS, GUI_MOD_CTRL elsewhere (for the backend in use).
int  gui_primary_mod();

// The display scale factor new hiDpi windows will get (1.0, 1.25, 2.0, ...).
double gui_display_scale();

} // namespace std
