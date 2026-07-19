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
// the application draws into directly (or via std/gui/widget.h's retained-
// mode layer, built on top of this); gui_present() blits it to the screen.
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
    int          defaultPrevented; // see gui_widget.h's gui_event_prevent_default()
};

struct GuiWindow {
    void*                platform; // backend-private handle
    &heap unsigned char  pixels;   // width*height*4 bytes, RGBA8888
    int                  width;
    int                  height;
    int                  shouldClose;
};

// Creates and shows a window with a zeroed (transparent black) pixel
// buffer of 'width'x'height'. On failure, returned window's 'platform' is
// null and 'width'/'height' are 0 — callers should check before use.
struct GuiWindow gui_create_window(const char* title, int width, int height);

// Pumps exactly one pending platform event into 'outEvent' and returns 1,
// or returns 0 immediately if none is pending (never blocks) — call in a
// loop ("while (gui_poll_event(&win, &ev)) { ... }") once per frame to
// drain everything queued since the last call. Also updates
// win->shouldClose when the platform's own close control is used.
int gui_poll_event(struct GuiWindow* win, struct GuiEvent* outEvent);

// Uploads win->pixels to the screen. Call once per frame after drawing.
void gui_present(struct GuiWindow* win);

// Sets pixel (x,y) to 'rgba' (0xRRGGBBAA), no-op if out of bounds.
void gui_set_pixel(struct GuiWindow* win, int x, int y, unsigned int rgba);

// Releases the window and its pixel buffer.
void gui_destroy_window(struct GuiWindow* win);

} // namespace std
