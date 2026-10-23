#pragma once
// SafeC Standard Library — GUI: the Wayland backend (Linux).
//
// A complete struct GuiBackend (gui_backend.h) on the Wayland protocol.
// Nothing is linked: libwayland-client.so.0 is dlopen()ed on first use,
// together with libxkbcommon.so.0 (keymaps, UTF-8 text, dead keys) and
// libwayland-cursor.so.0 (cursor themes) when they are installed, so the
// same binary still runs, through X11, on a system without Wayland.
//
// Which backend a Linux program uses is decided at its first window (or
// first clipboard/clock call):
//   SAFEC_GUI_BACKEND=wayland   Wayland (fails cleanly if there is none)
//   SAFEC_GUI_BACKEND=x11       X11 (on a Wayland desktop: through Xwayland)
//   SAFEC_GUI_BACKEND=headless  the in-memory backend (gui_headless.h)
//   unset / anything else       Wayland if $WAYLAND_DISPLAY is set, the
//                               library loads and the compositor answers;
//                               X11 otherwise.
// A program can also force it with gui_use_wayland() before its first
// window.
//
// What it does (the protocol objects in parentheses):
//   windows      xdg-shell toplevels (xdg_wm_base, xdg_surface,
//                xdg_toplevel), any number of them, each its own surface.
//                Title, app id ($SAFEC_GUI_APP_ID, else the program's
//                name), min/max size, close button, compositor-driven
//                resizes (tiling, maximize) arrive as GUI_EVENT_RESIZE.
//   presenting   wl_shm buffers, double-buffered (a third or fourth one
//                only while the compositor still holds both), damage-only
//                copies, throttled by frame callbacks: a present while the
//                previous frame is still pending is coalesced into the next
//                frame. A GPU presenter (gui_gl.h) replaces all of this
//                with EGL.
//   HiDPI        wp_fractional_scale_v1 + wp_viewporter (e.g. 1.25 or 1.5
//                — the buffer is exactly the surface's size in device
//                pixels), else wl_surface.preferred_buffer_scale, else the
//                scale of the outputs the window is on; GUI_EVENT_SCALE
//                when it changes. $SAFEC_GUI_SCALE overrides it.
//   decorations  server-side via zxdg_decoration_manager_v1 when the
//                compositor offers it (sway, KDE, wlroots compositors,
//                Weston >= 14). Without it (GNOME's Mutter, older Weston)
//                windows have NO decorations: std/gui draws no client-side
//                title bar — put your own close/move controls in the UI if
//                you target those compositors. decorated = 0 asks for none.
//   input        wl_seat: pointer (enter/leave, motion, buttons, wheel —
//                high-resolution value120 / discrete steps / smooth
//                scrolling), keyboard (xkbcommon keymaps from the
//                compositor, Shift/Ctrl/Alt/Super, dead keys and compose
//                sequences, client-side key repeat at the compositor's
//                rate). Without libxkbcommon keys are mapped as a US
//                layout. Seats that gain or lose devices are followed.
//   cursors      wp_cursor_shape_v1 when offered, else the XCursor theme
//                ($XCURSOR_THEME / $XCURSOR_SIZE) through libwayland-cursor.
//   clipboard    wl_data_device text selection (UTF-8; reads wait up to a
//                second for the other client). Setting it needs a recent
//                input event's serial — like every Wayland client, a
//                program can only take the clipboard while it has focus.
// Not available on Wayland by design: window positions (set_position()
// is ignored and pos_x()/pos_y() stay GUI_POS_DEFAULT unless set), raising
// or focusing a window (focus() is a no-op), and always-on-top
// (topmost is ignored).
//
// Verified against real compositors (tests/stdlib/wayland/run.sh: sway
// headless with grim/wtype/wlrctl, and Weston headless).

#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>

namespace std {

// The Wayland backend's table; null on a target without Wayland support
// (anything but Linux).
?&GuiBackend gui_wayland_backend();
// 1 if libwayland-client loads and a compositor accepts the connection
// (connects on the first call; the connection is then the backend's).
int gui_wayland_available();
// Installs the Wayland backend (gui_set_backend) if it is available.
// Returns 0, changing nothing, if it isn't or a window already exists.
int gui_use_wayland();

} // namespace std
