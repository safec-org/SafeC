#pragma once
// SafeC Standard Library — GUI: the headless backend.
//
// A complete struct GuiBackend (gui_backend.h) that keeps windows in
// memory only: nothing is shown, presenting just counts frames, the
// clipboard is a process-local string, and the clock is virtual — it
// moves only when gui_headless_advance() moves it (or when an event loop
// waits: a pump with a timeout and nothing queued advances the clock by
// that timeout instead of sleeping). That makes everything above the
// backend — multi-window event routing, timers, animation frames, the
// widget toolkit — testable deterministically, on any target, with no
// display server; it's also how to render GUI frames on a server.
//
//   gui_use_headless();                     // before the first window
//   struct GuiWindow win = GuiWindow::create("t", 200, 100);
//   gui_headless_click(&win, 10.0, 10.0);   // MOUSE_DOWN + MOUSE_UP
//   gui_headless_type(&win, "héllo");        // KEY_DOWN/UP per character
//   while (win.poll_event(&ev)) { ... }
//
// Events injected for a window arrive through that window's poll_event()
// (or GuiApp's loop) exactly as a platform backend's would, with windowId,
// timeMs and double-click counts filled in. Window operations behave like
// a cooperative window manager: set_size() posts GUI_EVENT_RESIZE,
// set_position() GUI_EVENT_MOVE, focus() FOCUS_OUT to the old focus window
// and FOCUS_IN to the new one, request_close() GUI_EVENT_CLOSE.

#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>

namespace std {

// The headless backend's table.
&GuiBackend gui_headless_backend();
// Installs it (gui_set_backend). Returns 0 if a window already exists.
int  gui_use_headless();

// The virtual clock, in milliseconds (starts at 0).
void   gui_headless_advance(double ms);
double gui_headless_now_ms();
// The scale factor hiDpi windows created from now on get (default 1.0).
void   gui_headless_set_scale(double scale);
// GUI_MOD_* the primary modifier reports (default GUI_MOD_CTRL).
void   gui_headless_set_primary_mod(int mod);

// Queues 'ev' for 'win' as if the platform had delivered it. Returns 0 for
// an offscreen window.
int  gui_headless_inject(&GuiWindow win, struct GuiEvent ev);
// Conveniences built on gui_headless_inject (coordinates in buffer pixels).
void gui_headless_mouse_move(&GuiWindow win, double x, double y);
void gui_headless_mouse_down(&GuiWindow win, double x, double y, int button);
void gui_headless_mouse_up(&GuiWindow win, double x, double y, int button);
void gui_headless_click(&GuiWindow win, double x, double y);
void gui_headless_scroll(&GuiWindow win, double x, double y, double dx, double dy);
// KEY_DOWN then KEY_UP; 'codepoint' (0 for none) rides on the KEY_DOWN.
void gui_headless_key(&GuiWindow win, int keycode, int modifiers, unsigned int codepoint);
// One key press per UTF-8 character of 'text'.
void gui_headless_type(&GuiWindow win, const &char text);
// The user resizing the window to a new logical size.
void gui_headless_resize(&GuiWindow win, int width, int height);
// The user clicking the close button.
void gui_headless_close(&GuiWindow win);

// How many times 'win' was presented.
unsigned long gui_headless_present_count(const &GuiWindow win);
// The cursor the window last asked for (GUI_CURSOR_*).
int gui_headless_cursor(const &GuiWindow win);

} // namespace std
