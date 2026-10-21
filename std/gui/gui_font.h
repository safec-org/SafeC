#pragma once
// SafeC Standard Library — GUI: loadable/serializable bitmap fonts.
//
// GuiWindow::draw_text() (gui.h) always uses the single built-in 8x8 font
// baked into the binary. This header adds a real font *asset*: a generic
// bit-packed glyph bitmap (any glyph size, not just 8x8) that can be
// built programmatically, saved to disk, and reloaded — and a
// drawing API on the font itself (font.draw_text(&win, ...)), for UIs that want
// a custom look instead of (or alongside) the built-in font.
//
// On-disk format ("SCXF", version 1) — little-endian, no external
// dependency (not TTF/OTF — a font must already be rasterized to a
// GuiFont, e.g. via GuiFont::new()+font.set_bit(), before it can be
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

#include <std/gui/gui.h>

namespace std {

struct GuiFont {
    int glyphW;
    int glyphH;
    int first;                  // first codepoint this font covers
    int count;                  // number of glyphs (codepoints first..first+count-1)
    &heap unsigned char bitmap; // count * bytesPerGlyph(glyphW,glyphH) bytes

    // Allocates a blank (all-zero) font of the given shape, ready for
    // set_bit() calls to draw each glyph programmatically.
    static struct GuiFont new(int glyphW, int glyphH, int first, int count);

    // A heap-owned copy of the built-in 8x8 font (see gui_draw.sc), in the
    // same generic GuiFont representation — e.g. as a starting point to edit
    // with set_bit() and save a modified variant. Release it with free().
    static struct GuiFont default();

    // Bit accessors, (0,0) = top-left of the glyph for codepoint 'cp'.
    // Reads return 0 for an out-of-range codepoint or pixel; writes are a
    // no-op in that case.
    int  get_bit(int cp, int px, int py) const;
    void set_bit(int cp, int px, int py, int value);

    // Serialization. save() writes this font as an SCXF file; load()
    // reads one into this font in place (overwriting it without freeing —
    // call it on a fresh 'struct GuiFont f;', or free() first). On failure
    // load() leaves the font untouched. Both return 1 = ok.
    int  save(const &char path) const;
    int  load(const &char path);

    void free();

    // Drawing with this font (mirrors GuiWindow::draw_text() and
    // gui_draw.h's gui_text_width()/gui_text_height() for the built-in
    // font): each glyph cell is glyphW*scale by glyphH*scale pixels;
    // codepoints the font doesn't cover draw as a blank cell. draw_text()
    // returns the total pixel width drawn.
    int  draw_text(&GuiWindow win, int x, int y, const &char text,
                   struct GuiColor color, int scale) const;
    int  text_width(const &char text, int scale) const;
    int  text_height(int scale) const;
};

} // namespace std
