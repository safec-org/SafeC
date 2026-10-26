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
// chunks (gAMA, tRNS, etc.) are not implemented; GuiImage::load_png() fails
// cleanly (returns 0) rather than misdecoding them. Chunk CRCs are not
// verified (decoding, not validating, is the goal here).

#include <std/gui/gui.h>

namespace std {

struct GuiImage {
    &heap unsigned char pixels; // width*height*4 bytes, RGBA8888, R first
    int width;
    int height;
    int premultiplied;          // 1: color channels are premultiplied by alpha (an image a
                                // GuiCanvas drew into); 0: straight alpha (as decoded from PNG)

    // A transparent width x height image (premultiplied; ready for
    // GuiCanvas::for_image). Free with free().
    static struct GuiImage new(int width, int height);
    // A premultiplied copy of a window's pixel buffer (a screenshot).
    static struct GuiImage from_window(const &GuiWindow win);
    // Reads and decodes a PNG file. 1 = ok.
    int  load_png_file(const &char path);
    // Writes the image as an RGBA PNG file (stored, uncompressed DEFLATE;
    // straight alpha). 1 = ok.
    int  save_png(const &char path) const;
    // Converts between straight and premultiplied alpha in place (no-op
    // when already in that form).
    void premultiply();
    void unpremultiply();
    struct GuiImage copy() const;
    // A resampled copy: area-averaging when shrinking (no aliasing),
    // bilinear when enlarging. Keeps the alpha form.
    struct GuiImage scaled(int width, int height) const;
    // 0xRRGGBBAA as stored / unblended store; 0 / no-op out of bounds.
    unsigned int get_pixel(int x, int y) const;
    void set_pixel(int x, int y, unsigned int rgba);

    // Decodes a PNG file's bytes (already read into memory) into this
    // image in place ('struct GuiImage img; img.load_png(data)').
    // Returns 1 on success, 0 on failure (unsupported feature or malformed
    // input) — the image is left zeroed on failure.
    int  load_png([]const unsigned char data);
    // internal: the (reference, length) core of load_png()
    int  load_png_n_(const &unsigned char data, unsigned long size);

    // Draws this image with its top-left at (x, y) into 'win' (straight
    // copy through GuiWindow::set_pixel(), no blending) — works with any
    // GuiImage, PNG-decoded or otherwise constructed.
    void draw(&GuiWindow win, int x, int y) const;

    void free();
};

} // namespace std
