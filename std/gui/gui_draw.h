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

namespace std {

struct GuiColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

struct GuiColor gui_rgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
struct GuiColor gui_rgb(unsigned char r, unsigned char g, unsigned char b);

void gui_fill_rect(struct GuiWindow* win, int x, int y, int w, int h, struct GuiColor color);
void gui_draw_rect_border(struct GuiWindow* win, int x, int y, int w, int h,
                           struct GuiColor color, int thickness);
void gui_draw_line(struct GuiWindow* win, int x0, int y0, int x1, int y1, struct GuiColor color);

// Draws 'text' with its top-left at (x,y), each glyph cell 8*scale by
// 8*scale pixels (scale=1 -> 8x8, scale=2 -> 16x16, ...). Unsupported
// codepoints (outside the printable-ASCII range the built-in font covers,
// 0x20-0x7E) draw as a blank cell. Returns the total pixel width drawn.
int gui_draw_text(struct GuiWindow* win, int x, int y, const char* text,
                   struct GuiColor color, int scale);

// Width/height in pixels 'text' would occupy if drawn at the given scale,
// without actually drawing it — used by widgets to size themselves to
// their label/text content.
int gui_text_width(const char* text, int scale);
int gui_text_height(int scale);

} // namespace std
