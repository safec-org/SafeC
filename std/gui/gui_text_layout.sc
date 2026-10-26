// SafeC Standard Library — GUI: text layout (GuiTextLayout, see gui_text.h):
// glyph positioning with kerning and per-glyph fallback, greedy line
// breaking, alignment, and the position <-> byte-offset mapping text
// editing needs.
#pragma once
#include <std/gui/gui_text.h>
#include <std/math.h>
#include <std/collections/vec.h>

namespace std {

struct GuiTextLayout GuiTextLayout::new() {
    struct GuiTextLayout l;
    l.glyphs = Vec::new();
    l.lines = Vec::new();
    l.face = null;
    l.size = 0.0f;
    l.width = 0.0f;
    l.height = 0.0f;
    l.lineHeight = 0.0f;
    l.ascent = 0.0f;
    l.maxWidth = 0.0f;
    l.textLen = 0UL;
    return l;
}

void GuiTextLayout::free() { self.drop(); }

void GuiTextLayout::drop() {
    self.glyphs.free();
    self.lines.free();
}

// Break opportunities: after spaces and hyphens, and around CJK ideographs,
// kana and Hangul (which break between any two characters).
static int __gui_is_cjk(unsigned int cp) {
    return ((cp >= 0x2E80U && cp <= 0x9FFFU) || (cp >= 0xAC00U && cp <= 0xD7AFU) ||
            (cp >= 0xF900U && cp <= 0xFAFFU) || (cp >= 0xFF00U && cp <= 0xFFEFU) ||
            (cp >= 0x20000U && cp <= 0x2FFFFU)) ? 1 : 0;
}

static void __gui_layout_finish_line(&GuiTextLayout l, unsigned long firstGlyph, unsigned long byteStart,
                                     unsigned long byteEnd) {
    struct GuiTextLine line;
    line.firstGlyph = firstGlyph;
    line.glyphCount = l.glyphs.len - firstGlyph;
    line.byteStart = byteStart;
    line.byteEnd = byteEnd;
    line.x = 0.0f;
    // Width up to the last non-space glyph (trailing spaces hang).
    float w = 0.0f;
    unsigned long i = l.glyphs.len;
    while (i > firstGlyph) {
        i = i - 1UL;
        &GuiTextGlyph g = l.glyphs.at(i);
        if (g.advance > 0.0f && !g.space) {
            w = g.x + g.advance;
            break;
        }
    }
    line.width = w;
    float idx = (float)l.lines.len;
    line.top = idx * l.lineHeight;
    line.baseline = line.top + l.ascent;
    l.lines.push(line);
}

void GuiTextLayout::build(const &GuiTypeface face, float size, const &char text, unsigned long len,
                          float maxWidth, int align) {
    self.glyphs.clear();
    self.lines.clear();
    self.face = face;
    self.size = size;
    self.maxWidth = maxWidth;
    if (len == (unsigned long)-1) {
        len = 0UL;
        unsafe { while (text[len] != '\0') { len = len + 1UL; } }
    }
    self.textLen = len;
    &GuiTypeface fb = GuiTypeface::fallback();
    float gap = face.line_gap(size);
    self.lineHeight = face.ascent(size) + face.descent(size) + gap;
    if (self.lineHeight <= 0.0f) { self.lineHeight = size * 1.2f; }
    self.ascent = face.ascent(size) + gap * 0.5f;
    float spaceW = face.advance(face.glyph_index(32U), size);
    float tabW = spaceW * 4.0f;
    if (tabW <= 0.0f) { tabW = size * 2.0f; }

    unsigned long lineFirst = 0UL;       // first glyph of the line being built
    unsigned long lineByte = 0UL;        // its first byte
    float pen = 0.0f;
    int prevGlyph = 0;
    const ?&GuiTypeface prevFace = null;
    long breakAfter = -1;                // glyph index after which this line may break
    unsigned long i = 0UL;
    while (i < len) {
        unsigned long start = i;
        unsigned int cp = gui_utf8_next(text, len, &i);
        if (cp == 10U) {
            __gui_layout_finish_line(self, lineFirst, lineByte, start);
            lineFirst = self.glyphs.len;
            lineByte = i;
            pen = 0.0f;
            prevGlyph = 0;
            breakAfter = -1;
            continue;
        }
        if (cp == 13U) { continue; }
        const ?&GuiTypeface gface = face;
        int g = face.glyph_index(cp);
        if (g == 0 && cp >= 32U) {
            int g2 = fb.glyph_index(cp);
            if (g2 != 0) { gface = fb; g = g2; }
        }
        float adv = 0.0f;
        if (cp == 9U) {
            adv = tabW - (pen - tabW * floor_f(pen / tabW));
            if (adv < spaceW * 0.5f) { adv = adv + tabW; }
            g = face.glyph_index(32U);
            gface = face;
        } else {
            match (gface) {
                case some(gf): {
                    adv = gf.advance(g, size);
                    unsafe {
                        if ((const void*)prevFace == (const void*)gf && prevGlyph != 0) { pen = pen + gf.kerning(prevGlyph, g, size); }
                    }
                }
                case null: { }
            }
        }
        struct GuiTextGlyph tg;
        tg.face = gface;
        tg.glyph = g;
        tg.x = pen;
        tg.y = 0.0f;
        tg.advance = adv;
        tg.byteStart = start;
        tg.byteEnd = i;
        int isSpace = (cp == 32U || cp == 9U || cp == 0x3000U) ? 1 : 0;
        tg.space = isSpace;
        int cjk = __gui_is_cjk(cp);
        // A CJK character may start a new line.
        if (cjk && self.glyphs.len > lineFirst) { breakAfter = (long)self.glyphs.len - 1L; }
        self.glyphs.push(tg);
        pen = pen + adv;
        prevGlyph = g;
        prevFace = gface;

        if (maxWidth > 0.0f && pen > maxWidth && !isSpace && self.glyphs.len - lineFirst > 1UL) {
            unsigned long cut = 0UL;   // first glyph of the next line
            if (breakAfter >= (long)lineFirst) { cut = (unsigned long)breakAfter + 1UL; }
            else { cut = self.glyphs.len - 1UL; }   // no opportunity: break inside the word
            if (cut > lineFirst && cut < self.glyphs.len + 1UL) {
                // Move glyphs [cut, end) to a new line.
                unsigned long moveCount = self.glyphs.len - cut;
                struct Vec<struct GuiTextGlyph> moved = Vec::new();
                unsigned long k = 0UL;
                while (k < moveCount) { moved.push(*self.glyphs.at(cut + k)); k = k + 1UL; }
                self.glyphs.len = cut;
                unsigned long nextByte = moveCount > 0UL ? moved.at(0UL).byteStart : i;
                __gui_layout_finish_line(self, lineFirst, lineByte, nextByte);
                lineFirst = self.glyphs.len;
                lineByte = nextByte;
                float shift = moveCount > 0UL ? moved.at(0UL).x : 0.0f;
                k = 0UL;
                while (k < moveCount) {
                    struct GuiTextGlyph mg = *moved.at(k);
                    mg.x = mg.x - shift;
                    self.glyphs.push(mg);
                    k = k + 1UL;
                }
                moved.free();
                pen = pen - shift;
                breakAfter = -1;
            }
        }
        if (isSpace || cp == 45U || cp == 0x2013U || cp == 0x2014U || cjk) {
            breakAfter = (long)self.glyphs.len - 1L;
        }
    }
    __gui_layout_finish_line(self, lineFirst, lineByte, len);

    // Alignment and final positions.
    float widest = 0.0f;
    unsigned long li = 0UL;
    while (li < self.lines.len) {
        &GuiTextLine ln = self.lines.at(li);
        if (ln.width > widest) { widest = ln.width; }
        li = li + 1UL;
    }
    self.width = widest;
    self.height = (float)self.lines.len * self.lineHeight;
    float box = maxWidth > 0.0f ? maxWidth : widest;
    li = 0UL;
    while (li < self.lines.len) {
        &GuiTextLine ln = self.lines.at(li);
        float off = 0.0f;
        if (align == GUI_TEXT_CENTER) { off = (box - ln.width) * 0.5f; }
        else if (align == GUI_TEXT_RIGHT) { off = box - ln.width; }
        ln.x = off;
        ln.baseline = ln.top + self.ascent;
        unsigned long k = 0UL;
        while (k < ln.glyphCount) {
            &GuiTextGlyph g = self.glyphs.at(ln.firstGlyph + k);
            g.x = g.x + off;
            g.y = ln.baseline;
            k = k + 1UL;
        }
        li = li + 1UL;
    }
}

unsigned long GuiTextLayout::line_of(unsigned long pos) const {
    unsigned long n = self.lines.len;
    if (n == 0UL) { return 0UL; }
    unsigned long i = 0UL;
    while (i + 1UL < n) {
        if (pos < self.lines.at(i + 1UL).byteStart) { return i; }
        i = i + 1UL;
    }
    return n - 1UL;
}

void GuiTextLayout::caret(unsigned long pos, &float x, &float top, &float height) const {
    *x = 0.0f;
    *top = 0.0f;
    *height = self.lineHeight;
    if (self.lines.len == 0UL) { return; }
    unsigned long li = self.line_of(pos);
    &GuiTextLine ln = self.lines.at(li);
    *top = ln.top;
    float cx = ln.x;
    unsigned long k = 0UL;
    int found = 0;
    while (k < ln.glyphCount) {
        &GuiTextGlyph g = self.glyphs.at(ln.firstGlyph + k);
        if (g.byteStart >= pos) { cx = g.x; found = 1; break; }
        cx = g.x + g.advance;
        k = k + 1UL;
    }
    if (!found && ln.glyphCount == 0UL) { cx = ln.x; }
    *x = cx;
}

unsigned long GuiTextLayout::hit_test(float x, float y) const {
    if (self.lines.len == 0UL) { return 0UL; }
    long li = (long)floor_f(y / (self.lineHeight > 0.0f ? self.lineHeight : 1.0f));
    if (li < 0L) { li = 0L; }
    if (li >= (long)self.lines.len) { li = (long)self.lines.len - 1L; }
    &GuiTextLine ln = self.lines.at((unsigned long)li);
    unsigned long k = 0UL;
    while (k < ln.glyphCount) {
        &GuiTextGlyph g = self.glyphs.at(ln.firstGlyph + k);
        if (x < g.x + g.advance * 0.5f) { return g.byteStart; }
        k = k + 1UL;
    }
    // Past the end of a wrapped line: before its trailing space break.
    if ((unsigned long)li + 1UL < self.lines.len && ln.glyphCount > 0UL) {
        &GuiTextGlyph last = self.glyphs.at(ln.firstGlyph + ln.glyphCount - 1UL);
        if (last.space) { return last.byteStart; }
        return ln.byteEnd;
    }
    return ln.byteEnd;
}

} // namespace std
