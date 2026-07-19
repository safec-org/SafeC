#pragma once
// SafeC Standard Library — GUI: PNG image loading.
//
// A real decoder — a from-scratch DEFLATE (RFC 1951) inflate implementation
// (stored/fixed-Huffman/dynamic-Huffman blocks, canonical Huffman decode,
// LZ77 back-references) plus PNG chunk parsing (RFC 2083) and the standard
// five scanline filters (None/Sub/Up/Average/Paeth) — not a stub. Scope:
// 8-bit-per-channel, non-interlaced PNGs with color type 0 (grayscale), 2
// (RGB), or 6 (RGBA) — the overwhelming majority of PNGs in practice.
// Palette (color type 3), 16-bit depth, interlacing (Adam7), and ancillary
// chunks (gAMA, tRNS, etc.) are not implemented; gui_load_png() fails
// cleanly (returns 0) rather than misdecoding them. Chunk CRCs are not
// verified (decoding, not validating, is the goal here).

namespace std {

struct GuiImage {
    &heap unsigned char pixels; // width*height*4 bytes, RGBA8888 (matches GuiWindow's own buffer format)
    int width;
    int height;
};

// Decodes a PNG file's bytes (already read into memory) into 'out'.
// Returns 1 on success, 0 on failure (unsupported feature or malformed
// input) — 'out' is left zeroed on failure.
int gui_load_png(const unsigned char* data, unsigned long size, struct GuiImage* out);

// Draws 'img' with its top-left at (x, y) into 'win' (see gui_draw.h) —
// works with any GuiImage, PNG-decoded or otherwise constructed.
void gui_draw_image(struct GuiWindow* win, int x, int y, const struct GuiImage* img);

void gui_image_free(struct GuiImage* img);

} // namespace std
