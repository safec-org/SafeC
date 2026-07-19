#pragma once
// SafeC Standard Library — GUI: loadable/serializable bitmap fonts.
//
// gui_draw.h's gui_draw_text() always uses the single built-in 8x8 font
// baked into the binary. This header adds a real font *asset*: a generic
// bit-packed glyph bitmap (any glyph size, not just 8x8) that can be
// built programmatically, saved to disk, and reloaded — and a
// '_ex'-suffixed drawing API that takes one explicitly, for UIs that want
// a custom look instead of (or alongside) the built-in font.
//
// On-disk format ("SCXF", version 1) — little-endian, no external
// dependency (not TTF/OTF — a font must already be rasterized to a
// GuiFont, e.g. via gui_font_new()+gui_font_set_bit(), before it can be
// saved):
//   offset 0  (4 bytes)  magic "SCXF"
//   offset 4  (1 byte)   version = 1
//   offset 5  (1 byte)   glyphW (pixels, 1-255)
//   offset 6  (1 byte)   glyphH (pixels, 1-255)
//   offset 7  (1 byte)   reserved = 0
//   offset 8  (4 bytes)  first codepoint (uint32 LE)
//   offset 12 (4 bytes)  glyph count (uint32 LE)
//   offset 16 ...        count * ceil(glyphW*glyphH/8) bytes: each glyph's
//                        pixels, row-major, MSB-first-per-byte, packed
//                        with no padding between rows or glyphs.

namespace std {

struct GuiFont {
    int glyphW;
    int glyphH;
    int first;                  // first codepoint this font covers
    int count;                  // number of glyphs (codepoints first..first+count-1)
    &heap unsigned char bitmap; // count * bytesPerGlyph(glyphW,glyphH) bytes
};

// Allocates a blank (all-zero) font of the given shape, ready for
// gui_font_set_bit() calls to draw each glyph programmatically.
struct GuiFont gui_font_new(int glyphW, int glyphH, int first, int count);

// A heap-owned copy of the built-in 8x8 font (see gui_draw.sc), in the
// same generic GuiFont representation — e.g. as a starting point to edit
// with gui_font_set_bit() and save a modified variant.
struct GuiFont gui_font_default();

// Bit accessors, (0,0) = top-left of the glyph for codepoint 'cp'.
// Reads return 0 for an out-of-range codepoint or pixel; writes are a
// no-op in that case.
int  gui_font_get_bit(const struct GuiFont* font, int cp, int px, int py);
void gui_font_set_bit(struct GuiFont* font, int cp, int px, int py, int value);

// Serialization.
int gui_font_save(const struct GuiFont* font, const char* path); // 1 = ok
int gui_font_load(struct GuiFont* outFont, const char* path);    // 1 = ok

void gui_font_free(struct GuiFont* font);

// Drawing with an explicit font (see gui_draw.h for the built-in-font
// versions these mirror).
int gui_draw_text_ex(struct GuiWindow* win, int x, int y, const char* text,
                      struct GuiColor color, int scale, const struct GuiFont* font);
int gui_text_width_ex(const char* text, int scale, const struct GuiFont* font);
int gui_text_height_ex(int scale, const struct GuiFont* font);

} // namespace std
