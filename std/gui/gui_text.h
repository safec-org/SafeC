#pragma once
// SafeC Standard Library — GUI: TrueType/OpenType fonts and text layout.
//
// GuiTypeface parses a font file in memory — a from-scratch reader for the
// sfnt tables a renderer needs (head, hhea, hmtx, maxp, cmap formats
// 0/4/6/12, loca, glyf with simple and composite glyphs, kern format 0,
// GPOS pair-adjustment kerning formats 1 and 2, OS/2, name) — and hands out
// glyph outlines as GuiPaths (quadratic curves), metrics and kerning.
// GuiCanvas::fill_text() (gui_canvas.h) rasterizes them anti-aliased and
// caches the result per glyph, size and subpixel position.
//
// Scope: TrueType outlines ('glyf'). OpenType fonts with CFF/CFF2 (cubic,
// PostScript-flavored) outlines load for their metrics but have no glyph
// shapes — load_* returns 0 for them unless allowed with
// GUI_FONT_ALLOW_NO_OUTLINES. No hinting (glyphs are drawn unhinted, like
// macOS and most modern renderers), no color glyphs, no variable-font
// axes (the default instance is used). Shaping is "simple": one glyph per
// codepoint through the cmap, pair kerning, no ligatures or
// complex-script reordering — right for Latin, Greek, Cyrillic, CJK and
// symbols; not for Arabic or Indic scripts.
//
//   struct GuiTypeface face;
//   if (!face.load_file("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) { ... }
//   float w = face.measure("Hello", 16.0f);         // pixels at 16 px
//   canvas.set_font(&face, 16.0f);
//   canvas.fill_text("Hello", 10.0f, 30.0f);
//   face.free();
//
// GuiTypeface::fallback() is always available: a sans-serif face compiled
// into the library (a subset of DejaVu Sans — printable ASCII, Latin-1, and
// common punctuation such as curly quotes, dashes, ellipsis and bullet;
// see gui_text_fallback.sc for its license), so text renders even with no
// font files at all (bare metal, a minimal container). GuiTypeface::system()
// looks for a system font by family name in the usual places on Linux,
// macOS and Windows.
//
// Sizes are in pixels (the em height), so a size-16 face is 16 px tall
// from ascender-ish to descender-ish, depending on the font's design.

#include <std/gui/gui_path.h>
#include <std/gui/gui_utf8.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

// load_* flags
#define GUI_FONT_COPY               1   // load_memory: keep a private copy of the bytes
#define GUI_FONT_ALLOW_NO_OUTLINES  2   // accept CFF-outline fonts (metrics only)

// Text alignment (GuiTextLayout, GuiCanvas::set_text_align)
#define GUI_TEXT_LEFT   0
#define GUI_TEXT_CENTER 1
#define GUI_TEXT_RIGHT  2

namespace std {

// One cached anti-aliased glyph image (owned by its typeface's cache).
struct GuiGlyphBitmap {
    int glyph;
    int sizeKey;        // size in 1/64 px
    int subpixel;       // 0..3: horizontal offset in quarter pixels
    int width;
    int height;
    int left;           // bitmap's top-left relative to the pen position, pixels
    int top;
    ?&heap unsigned char coverage;   // width*height bytes, 0..255
};

struct GuiTypeface {
    const ?&unsigned char data;   // the font file
    unsigned long size;
    int owned;                    // data was copied/read by us: free() releases it
    int faceIndex;                // within a TrueType collection (.ttc)

    // table offsets from the start of 'data' (0 = absent)
    unsigned int tHead; unsigned int tHhea; unsigned int tHmtx; unsigned int tMaxp;
    unsigned int tCmap; unsigned int tLoca; unsigned int tGlyf; unsigned int tKern;
    unsigned int tGpos; unsigned int tOs2; unsigned int tName;
    unsigned int lenGlyf; unsigned int lenLoca;

    int numGlyphs;
    int unitsPerEm;
    int indexToLocFormat;
    int numHMetrics;
    int ascender;      // font units, positive up
    int descender;     // font units, negative (below the baseline)
    int lineGap;
    int capHeight;
    int xHeight;
    int weight;        // OS/2 usWeightClass (400 regular, 700 bold)
    int italic;
    int hasOutlines;   // 0 for a CFF font loaded with GUI_FONT_ALLOW_NO_OUTLINES

    unsigned int cmapSub;   // chosen cmap subtable (offset from data)
    int cmapFormat;
    unsigned int kernSub;   // 'kern' format 0 subtable: pairs at kernSub+14
    int kernPairs;
    struct Vec<unsigned int> gposPairs;   // GPOS PairPos subtables (offsets from data)

    struct String family;   // name table family name (UTF-8), may be empty

    // glyph cache (see GuiCanvas::fill_text): entries + open-addressed index
    struct Vec<struct GuiGlyphBitmap> cache;
    struct Vec<int> cacheIndex;

    // ── loading ──────────────────────────────────────────────────────────
    // Parse a font in place ('struct GuiTypeface f; f.load_file(...)').
    // Return 1, or 0 for a missing/malformed/unsupported file (the face is
    // then empty: free() is still fine). A .ttc collection loads its face
    // 'index' with load_memory_index / load_file_index.
    int  load_file(const &char path);
    int  load_file_index(const &char path, int index);
    // 'flags': GUI_FONT_COPY to keep a copy (otherwise 'bytes' must outlive
    // the face), GUI_FONT_ALLOW_NO_OUTLINES.
    int  load_memory([]const unsigned char bytes, int flags);
    // internal: the (reference, length) core of load_memory()
    int  load_memory_n_(const &unsigned char bytes, unsigned long size, int flags);
    int  load_memory_index([]const unsigned char bytes, int index, int flags);
    // internal: the (reference, length) core of load_memory_index()
    int  load_memory_index_n_(const &unsigned char bytes, unsigned long size, int index, int flags);

    // The built-in fallback face (process lifetime; never free it).
    static &GuiTypeface fallback();
    // A system font: "sans" / "serif" / "mono" or a family name such as
    // "DejaVu Sans", "Helvetica", "Segoe UI", searched in the system font
    // directories (and $SAFEC_GUI_FONT_DIR). Faces are loaded once and
    // kept for the process; null when nothing matches.
    static ?&GuiTypeface system(const &char family);
    // Like system(), but a bold variant when the family has one.
    static ?&GuiTypeface system_bold(const &char family);

    // ── glyphs and metrics ───────────────────────────────────────────────
    // The glyph for codepoint 'cp' (0 = .notdef: not in this font).
    int   glyph_index(unsigned int cp) const;
    int   has_glyph(unsigned int cp) const;
    // In pixels at 'size' px.
    float advance(int glyph, float size) const;
    float kerning(int left, int right, float size) const;
    float ascent(float size) const;       // above the baseline, positive
    float descent(float size) const;      // below the baseline, positive
    float line_gap(float size) const;
    float line_height(float size) const;  // ascent + descent + line gap
    // Appends glyph's outline to 'out', scaled to 'size' px with the pen
    // (baseline origin) at (x, y), y down. Returns 0 for an empty glyph
    // (a space) or one that can't be read.
    int   glyph_path(int glyph, &GuiPath out, float size, float x, float y) const;
    // Glyph bounding box in font units (y up); 0 if it has none.
    int   glyph_box(int glyph, &int x0, &int y0, &int x1, &int y1) const;
    // Width in pixels of one line of UTF-8 text at 'size' (kerning applied,
    // glyphs this face lacks measured from the fallback face).
    float measure(const &char text, float size) const;
    float measure_n([]const char text, float size) const;
    // internal: the (reference, length) core of measure_n()
    float measure_n_n_(const &char text, unsigned long len, float size) const;

    // ── the glyph cache (used by GuiCanvas; rarely needed directly) ──────
    ?&GuiGlyphBitmap cache_find(int glyph, int sizeKey, int subpixel) const;
    // Takes ownership of 'b.coverage'. Returns the stored entry.
    &GuiGlyphBitmap cache_insert(struct GuiGlyphBitmap b);
    void  cache_clear();

    void free();
};

// One positioned glyph of a GuiTextLayout.
struct GuiTextGlyph {
    const ?&GuiTypeface face;     // the face it comes from (fallback for missing glyphs)
    int glyph;
    float x;                      // pen position relative to the layout's origin (top-left)
    float y;                      // baseline
    float advance;
    int space;                    // a space or tab (hangs at a line end)
    unsigned long byteStart;      // the UTF-8 bytes it was made from
    unsigned long byteEnd;
};

struct GuiTextLine {
    unsigned long firstGlyph;
    unsigned long glyphCount;
    unsigned long byteStart;
    unsigned long byteEnd;        // excluding the line break
    float x;                      // left edge after alignment
    float width;                  // trailing spaces not counted
    float top;
    float baseline;
};

// Text broken into lines and positioned: word wrap at spaces and
// hyphens (anywhere for CJK, and inside words too long for a line),
// hard breaks at '\n', tabs to 4-space stops, alignment within maxWidth.
// Also maps between positions and byte offsets for carets/selection.
struct GuiTextLayout {
    struct Vec<struct GuiTextGlyph> glyphs;
    struct Vec<struct GuiTextLine> lines;
    const ?&GuiTypeface face;
    float size;
    float width;          // widest line
    float height;         // lines * lineHeight
    float lineHeight;
    float ascent;
    float maxWidth;       // as given (0 = no wrapping)
    unsigned long textLen;

    static struct GuiTextLayout new();
    // (Re)builds the layout of text[0..len) — 'len' may be (unsigned
    // long)-1 for a NUL-terminated string. maxWidth <= 0: no wrapping.
    // align: GUI_TEXT_LEFT/CENTER/RIGHT within maxWidth (or the widest line).
    void  build(const &GuiTypeface face, float size, []const char text, float maxWidth, int align);
    // internal: the (reference, length) core of build()
    void  build_n_(const &GuiTypeface face, float size, const &char text, unsigned long len,
                float maxWidth, int align);
    // The byte offset whose caret position is nearest (x, y) (layout
    // coordinates).
    unsigned long hit_test(float x, float y) const;
    // The caret's position for byte offset 'pos': x, the line's top, its height.
    void  caret(unsigned long pos, &float x, &float top, &float height) const;
    // The line holding byte offset 'pos'.
    unsigned long line_of(unsigned long pos) const;
    void  free();
};

} // namespace std
