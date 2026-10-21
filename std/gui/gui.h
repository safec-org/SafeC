#pragma once
// SafeC Standard Library — GUI: cross-platform window + software framebuffer.
//
// One backend must be linked in alongside this header — pick exactly one:
//   std/gui/gui_cocoa.sc   macOS, via the Objective-C runtime + Core Graphics
//   std/gui/gui_win32.sc   Windows, via the Win32 API
//   std/gui/gui_x11.sc     Linux/BSD/Unix, via Xlib
//   std/gui/gui_fb.sc      bare metal, via a linear framebuffer (see its own
//                          header comment for how the base address is supplied)
// (same convention as std/sched/io_nb.h's backend files — the caller picks
// the right one for the target, since SafeC has no OS-detection-driven
// automatic file selection). std/gui/gui_auto.sc picks one of the three
// desktop backends via '#ifdef __APPLE__ / __linux__ / _WIN32' for the
// common case of "just build for whatever this machine is".
//
// Every backend presents the exact same model: a window owns a CPU-side
// RGBA8888 pixel buffer (top-left origin, row-major, 4 bytes/pixel) that
// the application draws into directly (or via std/gui/gui_widget.h's
// retained-mode layer, built on top of this); win.present() blits it to the
// screen.
// There is no GPU-accelerated rendering path — this is a software
// framebuffer, the same scope as Rust's minifb/softbuffer crates, chosen
// so all four backends (three OS-native windowing systems plus bare metal,
// which has no GPU driver at all) share one real, working rendering model
// instead of three different ones.

// Event/key constants are plain #defines (not enums) to match the rest of
// the standard library's convention for cross-namespace constant access
// (see e.g. std/serial/protobuf.h's PB_WIRE_* ) — a real 'enum' declared
// inside 'namespace std' can't be referenced from user code as
// 'std::SOME_CONSTANT' the way a free function or #define can.
#define GUI_EVENT_NONE       0
#define GUI_EVENT_CLOSE      1
#define GUI_EVENT_RESIZE     2
#define GUI_EVENT_MOUSE_DOWN 3
#define GUI_EVENT_MOUSE_UP   4
#define GUI_EVENT_MOUSE_MOVE 5
#define GUI_EVENT_KEY_DOWN   6
#define GUI_EVENT_KEY_UP     7
#define GUI_EVENT_CHAR       8
#define GUI_EVENT_SCROLL     9

// Portable key codes for GUI_EVENT_KEY_DOWN/UP's 'keycode' field — each
// backend translates its own native key codes into these, so application
// code never sees a platform-specific key code.
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

namespace std {

// An RGBA color for the drawing methods below. Lives here (rather than in
// gui_draw.h) so GuiWindow can declare its drawing methods without the
// core header depending on any add-on header.
struct GuiColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

// Defined in gui_draw.sc.
struct GuiColor gui_rgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
struct GuiColor gui_rgb(unsigned char r, unsigned char g, unsigned char b);

struct GuiEvent {
    int          kind;       // GuiEventKind
    double       x;          // mouse position, window-local, top-left origin
    double       y;
    int          button;     // 0=left, 1=right, 2=other (mouse events)
    int          keycode;    // GuiKey (key events)
    unsigned int codepoint;  // Unicode codepoint (GUI_EVENT_CHAR)
    double       scrollDx;
    double       scrollDy;
    int          width;      // new size (GUI_EVENT_RESIZE)
    int          height;
    int          defaultPrevented; // see prevent_default() below

    // Suppresses Widget::dispatch_event()'s default key action for this
    // event (see gui_widget.h) — call from a WIDGET_CUSTOM node's
    // WidgetCustomEventFn when you're handling a key yourself and don't
    // want e.g. Tab/Enter's default behavior on top. Defined in
    // gui_widget.sc.
    void prevent_default();
};

struct GuiWindow {
    ?&void               platform; // backend-private handle; null on failure
    &heap unsigned char  pixels;   // width*height*4 bytes, RGBA8888
    int                  width;
    int                  height;
    int                  shouldClose;

    // Each backend (gui_cocoa/gui_win32/gui_x11/gui_fb.sc) defines these.

    // Pumps exactly one pending platform event into 'outEvent' and returns
    // 1, or returns 0 immediately if none is pending (never blocks) — call
    // in a loop ("while (win.poll_event(&ev)) { ... }") once per frame to
    // drain everything queued since the last call. Also updates
    // shouldClose when the platform's own close control is used.
    int  poll_event(&GuiEvent outEvent);

    // Uploads 'pixels' to the screen. Call once per frame after drawing.
    void present();

    // Sets pixel (x,y) to 'rgba' (0xRRGGBBAA), no-op if out of bounds.
    void set_pixel(int x, int y, unsigned int rgba);

    // Releases the window and its pixel buffer.
    void destroy();

    // ── drawing (software rasterizer; defined in gui_draw.sc / gui_svg.sc)
    // All of these write through set_pixel(), so they work on every
    // backend; out-of-bounds pixels are clipped.
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

// Creates and shows a window with a zeroed (transparent black) pixel
// buffer of 'width'x'height'. On failure, returned window's 'platform' is
// null and 'width'/'height' are 0 — callers should check before use.
struct GuiWindow gui_create_window(const &char title, int width, int height);

} // namespace std
