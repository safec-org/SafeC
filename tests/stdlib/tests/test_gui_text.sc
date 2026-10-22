// std::gui TrueType/OpenType reading, glyph outlines, kerning (kern table,
// GPOS pair formats 1 and 2), measuring, text layout, rasterized text, the
// fallback face and a system font when one is installed. Font fixtures:
// tests/stdlib/fixtures/fonts (subsets of DejaVu Sans).
#include <std/test/test.h>
#include <std/mem.h>
#include <std/gui/gui.h>
#include <std/gui/gui_text.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_path.h>
#include <std/gui/gui_png.h>

static int nearf(float a, float b, float tol) { return (a - b <= tol && b - a <= tol) ? 1 : 0; }

void t_fallback_face() {
    &GuiTypeface f = GuiTypeface::fallback();
    ASSERT_EQ(f.unitsPerEm, 2048);
    ASSERT_TRUE(f.hasOutlines);
    ASSERT_TRUE(f.family.eq_cstr("SafeC Fallback Sans"));
    ASSERT_EQ(f.numGlyphs, 224);
    int a = f.glyph_index((unsigned int)'A');
    int v = f.glyph_index((unsigned int)'V');
    ASSERT_EQ(a, 34);
    ASSERT_TRUE(f.has_glyph(0xE9U));                  // é (Latin-1)
    ASSERT_TRUE(f.has_glyph(0x2026U));                // …
    ASSERT_TRUE(f.has_glyph(0x2713U));                // ✓
    ASSERT_FALSE(f.has_glyph(0x4E00U));               // no CJK
    // Metrics in font units at size = unitsPerEm.
    ASSERT_TRUE(nearf(f.advance(a, 2048.0f), 1401.0f, 0.01f));
    ASSERT_TRUE(nearf(f.kerning(a, v, 2048.0f), -131.0f, 0.01f));   // the 'kern' table
    ASSERT_TRUE(nearf(f.ascent(2048.0f), 1901.0f, 0.01f));
    ASSERT_TRUE(nearf(f.descent(2048.0f), 483.0f, 0.01f));
    ASSERT_TRUE(nearf(f.measure("AV", 2048.0f), 1401.0f * 2.0f - 131.0f, 0.05f));
    ASSERT_TRUE(nearf(f.measure("AV", 20.48f), 26.71f, 0.01f));
    int x0 = 0; int y0 = 0; int x1 = 0; int y1 = 0;
    ASSERT_TRUE(f.glyph_box(f.glyph_index((unsigned int)'o'), &x0, &y0, &x1, &y1));
    ASSERT_EQ(x0, 113); ASSERT_EQ(y0, -29); ASSERT_EQ(x1, 1141); ASSERT_EQ(y1, 1147);
}

void t_outlines() {
    &GuiTypeface f = GuiTypeface::fallback();
    struct GuiPath p = GuiPath::new();
    // 'o' at 2048 px is its font units, y flipped, pen at (0, 2000).
    ASSERT_TRUE(f.glyph_path(f.glyph_index((unsigned int)'o'), &p, 2048.0f, 0.0f, 2000.0f));
    struct GuiRect b = p.bounds();
    ASSERT_TRUE(nearf(b.x, 113.0f, 0.5f) && nearf(b.x + b.w, 1141.0f, 0.5f));
    ASSERT_TRUE(nearf(b.y, 2000.0f - 1147.0f, 0.5f) && nearf(b.y + b.h, 2029.0f, 0.5f));
    int closes = 0;
    unsigned long i = 0UL;
    while (i < p.segs.len) { if (p.segs.at(i).verb == GUI_PATH_CLOSE) closes = closes + 1; i = i + 1UL; }
    ASSERT_EQ(closes, 2);                             // outer and inner contour
    // A composite glyph: Á = A + acute, taller than A.
    p.clear();
    ASSERT_TRUE(f.glyph_path(f.glyph_index(0xC1U), &p, 2048.0f, 0.0f, 2000.0f));
    b = p.bounds();
    ASSERT_TRUE(nearf(b.y, 2000.0f - 1899.0f, 0.5f));
    ASSERT_TRUE(nearf(b.x, 16.0f, 0.5f));
    // A space has no outline.
    p.clear();
    ASSERT_FALSE(f.glyph_path(f.glyph_index(32U), &p, 16.0f, 0.0f, 0.0f));
    p.free();
}

void t_gpos_kerning() {
    struct GuiTypeface k1;
    ASSERT_TRUE(k1.load_file("fixtures/fonts/gpos_kern1.ttf"));    // PairPos format 1
    ASSERT_EQ(k1.kernSub, 0U);                         // no 'kern' table
    ASSERT_EQ(k1.gposPairs.len, 1UL);
    int a = k1.glyph_index((unsigned int)'A');
    int v = k1.glyph_index((unsigned int)'V');
    int t = k1.glyph_index((unsigned int)'T');
    int o = k1.glyph_index((unsigned int)'o');
    ASSERT_TRUE(nearf(k1.kerning(a, v, 2048.0f), -123.0f, 0.01f));
    ASSERT_TRUE(nearf(k1.kerning(t, o, 2048.0f), -77.0f, 0.01f));
    ASSERT_TRUE(nearf(k1.kerning(o, a, 2048.0f), 0.0f, 0.01f));
    k1.free();
    struct GuiTypeface k2;
    ASSERT_TRUE(k2.load_file("fixtures/fonts/gpos_kern2.ttf"));    // PairPos format 2 (classes)
    a = k2.glyph_index((unsigned int)'A');
    v = k2.glyph_index((unsigned int)'V');
    t = k2.glyph_index((unsigned int)'T');
    o = k2.glyph_index((unsigned int)'o');
    ASSERT_TRUE(nearf(k2.kerning(a, v, 2048.0f), -131.0f, 0.01f));
    ASSERT_TRUE(nearf(k2.kerning(t, o, 2048.0f), -348.0f, 0.01f));
    ASSERT_TRUE(nearf(k2.kerning(o, o, 2048.0f), 0.0f, 0.01f));
    ASSERT_FALSE(k2.has_glyph((unsigned int)'B'));
    k2.free();
    struct GuiTypeface missing;
    ASSERT_FALSE(missing.load_file("fixtures/fonts/no_such_font.ttf"));
    missing.free();
}

void t_malformed() {
    &GuiTypeface f = GuiTypeface::fallback();
    // Truncated: the table directory points past the end.
    const &unsigned char bytes;
    unsafe { bytes = (const unsigned char*)f.data; }
    struct GuiTypeface t;
    ASSERT_FALSE(t.load_memory(bytes, 200UL, 0));
    t.free();
    unsigned char junk[64];
    int i = 0;
    while (i < 64) { junk[i] = (unsigned char)(i * 37); i = i + 1; }
    ASSERT_FALSE(t.load_memory(&junk[0], 64UL, GUI_FONT_COPY));
    t.free();
    // A corrupted copy still loads or fails cleanly, and every glyph either
    // has an outline or not, without crashing.
    struct GuiTypeface c;
    ASSERT_TRUE(c.load_memory(bytes, f.size, GUI_FONT_COPY));
    unsafe {
        unsigned char* d = (unsigned char*)c.data;
        unsigned long k = (unsigned long)c.tGlyf + 40UL;
        while (k < (unsigned long)c.tGlyf + (unsigned long)c.lenGlyf) { d[k] = (unsigned char)(d[k] ^ 0x5AU); k = k + 97UL; }
    }
    struct GuiPath p = GuiPath::new();
    int g = 0;
    while (g < c.numGlyphs) { c.glyph_path(g, &p, 12.0f, 0.0f, 0.0f); g = g + 1; }
    p.free();
    c.free();
}

void t_layout() {
    &GuiTypeface f = GuiTypeface::fallback();
    struct GuiTextLayout l = GuiTextLayout::new();
    l.build(f, 20.0f, "Hello", (unsigned long)-1, 0.0f, GUI_TEXT_LEFT);
    ASSERT_EQ(l.lines.len, 1UL);
    ASSERT_EQ(l.glyphs.len, 5UL);
    ASSERT_TRUE(nearf(l.width, f.measure("Hello", 20.0f), 0.01f));
    ASSERT_TRUE(nearf(l.height, f.line_height(20.0f), 0.01f));
    // Hard breaks.
    l.build(f, 20.0f, "one\ntwo\n", (unsigned long)-1, 0.0f, GUI_TEXT_LEFT);
    ASSERT_EQ(l.lines.len, 3UL);
    ASSERT_EQ(l.lines.at(1UL).byteStart, 4UL);
    ASSERT_EQ(l.lines.at(1UL).byteEnd, 7UL);
    ASSERT_EQ(l.lines.at(2UL).glyphCount, 0UL);
    // Word wrap at spaces.
    const &char text = "the quick brown fox jumps over the lazy dog";
    float w = f.measure("the quick brown", 16.0f) + 2.0f;
    l.build(f, 16.0f, text, (unsigned long)-1, w, GUI_TEXT_LEFT);
    ASSERT_TRUE(l.lines.len >= 3UL);
    ASSERT_EQ(l.lines.at(1UL).byteStart, 16UL);        // "fox ..." starts line 2
    unsigned long li = 0UL;
    while (li < l.lines.len) { ASSERT_TRUE(l.lines.at(li).width <= w + 0.01f); li = li + 1UL; }
    // A word longer than the line breaks inside it.
    l.build(f, 16.0f, "abcdefghijklmnopqrstuvwxyz", (unsigned long)-1, 60.0f, GUI_TEXT_LEFT);
    ASSERT_TRUE(l.lines.len >= 3UL);
    // Alignment.
    l.build(f, 16.0f, "ab\nabcdef", (unsigned long)-1, 0.0f, GUI_TEXT_RIGHT);
    ASSERT_TRUE(nearf(l.lines.at(0UL).x + l.lines.at(0UL).width, l.width, 0.01f));
    l.build(f, 16.0f, "ab", (unsigned long)-1, 100.0f, GUI_TEXT_CENTER);
    ASSERT_TRUE(nearf(l.lines.at(0UL).x, (100.0f - f.measure("ab", 16.0f)) * 0.5f, 0.01f));
    // Carets and hit testing round-trip.
    l.build(f, 16.0f, "héllo wörld", (unsigned long)-1, 0.0f, GUI_TEXT_LEFT);
    float cx = 0.0f; float top = 0.0f; float h = 0.0f;
    l.caret(3UL, &cx, &top, &h);                     // after "hé" (é is 2 bytes)
    ASSERT_TRUE(nearf(cx, f.measure("hé", 16.0f), 0.01f));
    ASSERT_EQ(l.hit_test(cx + 0.5f, 5.0f), 3UL);
    ASSERT_EQ(l.hit_test(-5.0f, 5.0f), 0UL);
    ASSERT_EQ(l.hit_test(1000.0f, 5.0f), 13UL);      // the end (13 bytes)
    l.build(f, 16.0f, "aa bb", (unsigned long)-1, f.measure("aa b", 16.0f), GUI_TEXT_LEFT);
    ASSERT_EQ(l.lines.len, 2UL);
    ASSERT_EQ(l.line_of(3UL), 1UL);
    ASSERT_EQ(l.hit_test(2.0f, l.lineHeight + 2.0f), 3UL);
    // Glyphs a face lacks come from the fallback.
    struct GuiTypeface k1;
    ASSERT_TRUE(k1.load_file("fixtures/fonts/gpos_kern1.ttf"));
    l.build(&k1, 16.0f, "AxV", (unsigned long)-1, 0.0f, GUI_TEXT_LEFT);
    unsafe {
        ASSERT_TRUE((const void*)l.glyphs.at(1UL).face == (const void*)f);
        ASSERT_TRUE((const void*)l.glyphs.at(0UL).face == (const void*)&k1);
    }
    l.free();
    k1.free();
}

void t_render() {
    struct GuiImage img = GuiImage::new(160, 40);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    c.clear(GuiColor::hex(0xFFFFFFU));
    c.set_fill_color(GuiColor::hex(0x000000U));
    c.set_font(GuiTypeface::fallback(), 20.0f);
    float w = c.fill_text("Hamburg", 10.0f, 30.0f);
    ASSERT_TRUE(nearf(w, c.measure_text("Hamburg"), 0.01f));
    // Ink inside the text box, nothing far outside it.
    int ink = 0; int stray = 0;
    int y = 0;
    while (y < 40) {
        int x = 0;
        while (x < 160) {
            struct GuiColor p = c.get_pixel(x, y);
            if (p.r < 128) {
                if (x >= 9 && x <= (int)(10.0f + w) + 1 && y >= 10 && y <= 36) { ink = ink + 1; } else { stray = stray + 1; }
            }
            x = x + 1;
        }
        y = y + 1;
    }
    ASSERT_TRUE(ink > 150);
    ASSERT_EQ(stray, 0);
    // Anti-aliased: grey levels on the glyph edges.
    int grey = 0;
    int x = 0;
    while (x < 160) { struct GuiColor p = c.get_pixel(x, 22); if (p.r > 30 && p.r < 225) grey = grey + 1; x = x + 1; }
    ASSERT_TRUE(grey >= 4);
    // The glyph cache filled up, and drawing again is identical.
    &GuiTypeface f = GuiTypeface::fallback();
    ASSERT_TRUE(f.cache.len >= 6UL);
    struct GuiImage copy = img.copy();
    c.clear(GuiColor::hex(0xFFFFFFU));
    c.fill_text("Hamburg", 10.0f, 30.0f);
    int diff = 0;
    y = 0;
    while (y < 40) { x = 0; while (x < 160) { if (img.get_pixel(x, y) != copy.get_pixel(x, y)) diff = diff + 1; x = x + 1; } y = y + 1; }
    ASSERT_EQ(diff, 0);
    // Alignment and baselines.
    c.clear(GuiColor::hex(0xFFFFFFU));
    c.set_text_align(GUI_TEXT_RIGHT);
    c.set_text_baseline(GUI_BASELINE_TOP);
    c.fill_text("Hi", 150.0f, 0.0f);
    int right = 0;
    x = 120;
    while (x < 160) { y = 0; while (y < 20) { if (c.get_pixel(x, y).r < 128) right = right + 1; y = y + 1; } x = x + 1; }
    ASSERT_TRUE(right > 20);
    // Rotated text goes through outline filling.
    c.clear(GuiColor::hex(0xFFFFFFU));
    c.set_text_align(GUI_TEXT_LEFT);
    c.set_text_baseline(GUI_BASELINE_ALPHABETIC);
    c.translate(80.0f, 20.0f);
    c.rotate(0.2f);
    c.fill_text("Rot", -15.0f, 5.0f);
    int rot = 0;
    y = 0;
    while (y < 40) { x = 0; while (x < 160) { if (c.get_pixel(x, y).r < 128) rot = rot + 1; x = x + 1; } y = y + 1; }
    ASSERT_TRUE(rot > 40);
    copy.free();
    c.free();
    img.free();
}

void t_system_font() {
    ?&GuiTypeface sys = GuiTypeface::system("sans");
    match (sys) {
        case null: { }   // no system fonts on this machine: nothing to check
        case some(f): {
            ASSERT_TRUE(f.numGlyphs > 100);
            ASSERT_TRUE(f.has_glyph((unsigned int)'A'));
            ASSERT_TRUE(f.measure("Hello", 16.0f) > 20.0f);
            ?&GuiTypeface again = GuiTypeface::system("sans");      // cached
            unsafe { ASSERT_TRUE((const void*)again == (const void*)f); }
        }
    }
    struct GuiTypeface dv;
    if (dv.load_file("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
        // The full DejaVu Sans: kern table and GPOS, format 12 cmap.
        ASSERT_EQ(dv.cmapFormat, 12);
        ASSERT_TRUE(dv.gposPairs.len > 0UL);
        ASSERT_TRUE(dv.has_glyph(0x0416U));            // Cyrillic Zhe
        int a = dv.glyph_index((unsigned int)'A');
        int v = dv.glyph_index((unsigned int)'V');
        ASSERT_TRUE(dv.kerning(a, v, 2048.0f) < -100.0f);
        ASSERT_TRUE(dv.family.eq_cstr("DejaVu Sans"));
    }
    dv.free();
    ?&GuiTypeface none = GuiTypeface::system("No Such Font Family");
    ASSERT_TRUE(none.is_null());
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("fallback face: cmap, metrics, kern table", t_fallback_face);
        t.add("glyph outlines (simple, composite)", t_outlines);
        t.add("GPOS pair kerning formats 1 and 2", t_gpos_kerning);
        t.add("malformed fonts", t_malformed);
        t.add("text layout: wrap, align, carets", t_layout);
        t.add("rasterized text", t_render);
        t.add("system fonts", t_system_font);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
