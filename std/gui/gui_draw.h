#pragma once
// SafeC Standard Library — GUI: software rasterizer.
//
// Draws directly into a struct GuiWindow's RGBA8888 pixel buffer (see
// gui.h) — rectangles, borders, lines, and text via a built-in 8x8 bitmap
// font. This is the same portable drawing surface every gui_* backend
// shares, so a UI drawn with these calls looks identical on Cocoa, Win32,
// X11, and bare metal alike (no native widget rendering involved). The
// retained-mode widget layer in gui_widget.h is built entirely on top of
// this — application code can also call these directly for fully custom
// drawing (see gui_widget.h's WidgetDrawFn hook).

#include <std/gui/gui.h>

namespace std {

// The drawing primitives themselves are GuiWindow methods, declared in
// gui.h next to the GuiColor type they take and defined in gui_draw.sc:
//   win.fill_rect(x, y, w, h, color)
//   win.draw_rect_border(x, y, w, h, color, thickness)
//   win.draw_line(x0, y0, x1, y1, color)
//   win.draw_text(x, y, text, color, scale)   // built-in 8x8 font
// GuiColor::rgb()/GuiColor::rgba() are declared there too.

// Width/height in pixels 'text' would occupy if drawn with the built-in
// font at the given scale, without actually drawing it — used by widgets
// to size themselves to their label/text content.
int gui_text_width(const &char text, int scale);
int gui_text_height(int scale);

} // namespace std
