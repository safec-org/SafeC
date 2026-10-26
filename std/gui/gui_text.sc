// SafeC Standard Library — GUI: the TrueType/OpenType reader (see
// gui_text.h). Every read is bounds-checked against the font's size, so a
// truncated or malicious file yields missing glyphs, never a wild read.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui_text.h>
#include <std/gui/gui_path.h>
#include <std/mem.h>
#include <std/math.h>
#include <std/io_file.h>
#include <std/io_traits.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>
extern void* memcpy(void* dst, const void* src, size_t n);
extern void* memset(void* ptr, int val, size_t n);

namespace std {

// ── big-endian reads ──────────────────────────────────────────────────────

static unsigned int __tt_u8(const &GuiTypeface f, unsigned long o) {
    if (o >= f.size) { return 0U; }
    unsafe { return (unsigned int)((const unsigned char*)f.data)[o]; }
}

static unsigned int __tt_u16(const &GuiTypeface f, unsigned long o) {
    if (o + 2UL > f.size) { return 0U; }
    unsafe {
        const unsigned char* p = (const unsigned char*)f.data + o;
        return ((unsigned int)p[0] << 8) | (unsigned int)p[1];
    }
}

static int __tt_i16(const &GuiTypeface f, unsigned long o) {
    unsigned int v = __tt_u16(f, o);
    return v >= 0x8000U ? (int)v - 65536 : (int)v;
}

static unsigned int __tt_u32(const &GuiTypeface f, unsigned long o) {
    if (o + 4UL > f.size) { return 0U; }
    unsafe {
        const unsigned char* p = (const unsigned char*)f.data + o;
        return ((unsigned int)p[0] << 24) | ((unsigned int)p[1] << 16) | ((unsigned int)p[2] << 8) | (unsigned int)p[3];
    }
}

static int __tt_tag(const &GuiTypeface f, unsigned long o, const char* tag) {
    unsafe {
        return (__tt_u8(f, o) == (unsigned int)(unsigned char)tag[0] && __tt_u8(f, o + 1UL) == (unsigned int)(unsigned char)tag[1] &&
                __tt_u8(f, o + 2UL) == (unsigned int)(unsigned char)tag[2] && __tt_u8(f, o + 3UL) == (unsigned int)(unsigned char)tag[3]) ? 1 : 0;
    }
}

// ── loading ───────────────────────────────────────────────────────────────

static void __tt_blank(&GuiTypeface f) {
    f.data = null;
    f.size = 0UL;
    f.owned = 0;
    f.faceIndex = 0;
    f.tHead = 0U; f.tHhea = 0U; f.tHmtx = 0U; f.tMaxp = 0U; f.tCmap = 0U; f.tLoca = 0U;
    f.tGlyf = 0U; f.tKern = 0U; f.tGpos = 0U; f.tOs2 = 0U; f.tName = 0U;
    f.lenGlyf = 0U; f.lenLoca = 0U;
    f.numGlyphs = 0; f.unitsPerEm = 1000; f.indexToLocFormat = 0; f.numHMetrics = 0;
    f.ascender = 800; f.descender = -200; f.lineGap = 0; f.capHeight = 700; f.xHeight = 500;
    f.weight = 400; f.italic = 0; f.hasOutlines = 0;
    f.cmapSub = 0U; f.cmapFormat = 0;
    f.kernSub = 0U; f.kernPairs = 0;
    f.gposPairs = Vec::new();
    f.family = String::new();
    f.cache = Vec::new();
    f.cacheIndex = Vec::new();
}

// Picks the best Unicode cmap subtable: full repertoire (format 12) first.
static void __tt_pick_cmap(&GuiTypeface f) {
    unsigned long base = (unsigned long)f.tCmap;
    unsigned int n = __tt_u16(f, base + 2UL);
    int bestScore = 0;
    unsigned int i = 0U;
    while (i < n) {
        unsigned long rec = base + 4UL + (unsigned long)i * 8UL;
        unsigned int plat = __tt_u16(f, rec);
        unsigned int enc = __tt_u16(f, rec + 2UL);
        unsigned long off = base + (unsigned long)__tt_u32(f, rec + 4UL);
        unsigned int fmt = __tt_u16(f, off);
        int score = 0;
        int unicode = (plat == 0U) || (plat == 3U && (enc == 1U || enc == 10U));
        if (unicode && fmt == 12U) { score = 5; }
        else if (unicode && fmt == 4U) { score = 4; }
        else if (unicode && fmt == 6U) { score = 3; }
        else if (plat == 3U && enc == 0U && fmt == 4U) { score = 2; }   // symbol fonts
        else if (fmt == 0U || fmt == 6U) { score = 1; }
        if (score > bestScore) {
            bestScore = score;
            f.cmapSub = (unsigned int)off;
            f.cmapFormat = (int)fmt;
        }
        i = i + 1U;
    }
}

static void __tt_find_kern(&GuiTypeface f) {
    if (f.tKern == 0U) { return; }
    unsigned long base = (unsigned long)f.tKern;
    if (__tt_u16(f, base) != 0U) { return; }       // Apple's version-1 'kern': not read
    unsigned int n = __tt_u16(f, base + 2UL);
    unsigned long off = base + 4UL;
    unsigned int i = 0U;
    while (i < n) {
        unsigned int len = __tt_u16(f, off + 2UL);
        unsigned int coverage = __tt_u16(f, off + 4UL);
        unsigned int format = coverage >> 8;
        // horizontal, not minimum/cross-stream, format 0
        if (format == 0U && (coverage & 1U) != 0U && (coverage & 6U) == 0U) {
            f.kernSub = (unsigned int)off;
            f.kernPairs = (int)__tt_u16(f, off + 6UL);
            return;
        }
        if (len < 6U) { return; }
        off = off + (unsigned long)len;
        i = i + 1U;
    }
}

// The PairPos subtables of the GPOS lookups the 'kern' feature uses.
static void __tt_find_gpos(&GuiTypeface f) {
    if (f.tGpos == 0U) { return; }
    unsigned long base = (unsigned long)f.tGpos;
    unsigned long featList = base + (unsigned long)__tt_u16(f, base + 6UL);
    unsigned long lookList = base + (unsigned long)__tt_u16(f, base + 8UL);
    unsigned int nfeat = __tt_u16(f, featList);
    unsigned int nlook = __tt_u16(f, lookList);
    // lookups referenced by any 'kern' feature (deduplicated)
    struct Vec<unsigned int> wanted = Vec::new();
    unsigned int i = 0U;
    while (i < nfeat) {
        unsigned long rec = featList + 2UL + (unsigned long)i * 6UL;
        if (__tt_tag(f, rec, "kern")) {
            unsigned long ft = featList + (unsigned long)__tt_u16(f, rec + 4UL);
            unsigned int cnt = __tt_u16(f, ft + 2UL);
            unsigned int k = 0U;
            while (k < cnt) {
                unsigned int li = __tt_u16(f, ft + 4UL + (unsigned long)k * 2UL);
                int seen = 0;
                unsigned long w = 0UL;
                while (w < wanted.len) { if (*wanted.at(w) == li) { seen = 1; } w = w + 1UL; }
                if (!seen && li < nlook) { wanted.push(li); }
                k = k + 1U;
            }
        }
        i = i + 1U;
    }
    // Lookup order is the order of the lookup list.
    unsigned int li = 0U;
    while (li < nlook) {
        int use = 0;
        unsigned long w = 0UL;
        while (w < wanted.len) { if (*wanted.at(w) == li) { use = 1; } w = w + 1UL; }
        if (use) {
            unsigned long lk = lookList + (unsigned long)__tt_u16(f, lookList + 2UL + (unsigned long)li * 2UL);
            unsigned int type = __tt_u16(f, lk);
            unsigned int nsub = __tt_u16(f, lk + 4UL);
            unsigned int s = 0U;
            while (s < nsub) {
                unsigned long sub = lk + (unsigned long)__tt_u16(f, lk + 6UL + (unsigned long)s * 2UL);
                unsigned int st = type;
                if (type == 9U) {      // extension: the real subtable is further away
                    st = __tt_u16(f, sub + 2UL);
                    sub = sub + (unsigned long)__tt_u32(f, sub + 4UL);
                }
                if (st == 2U) { f.gposPairs.push((unsigned int)sub); }
                s = s + 1U;
            }
        }
        li = li + 1U;
    }
    wanted.free();
}

// name table: the family name (nameID 16 typographic family, else 1).
static void __tt_read_name(&GuiTypeface f) {
    if (f.tName == 0U) { return; }
    unsigned long base = (unsigned long)f.tName;
    unsigned int count = __tt_u16(f, base + 2UL);
    unsigned long strings = base + (unsigned long)__tt_u16(f, base + 4UL);
    int bestScore = 0;
    unsigned long bestOff = 0UL;
    unsigned int bestLen = 0U;
    int bestWide = 0;
    unsigned int i = 0U;
    while (i < count) {
        unsigned long rec = base + 6UL + (unsigned long)i * 12UL;
        unsigned int plat = __tt_u16(f, rec);
        unsigned int enc = __tt_u16(f, rec + 2UL);
        unsigned int lang = __tt_u16(f, rec + 4UL);
        unsigned int nameId = __tt_u16(f, rec + 6UL);
        unsigned int len = __tt_u16(f, rec + 8UL);
        unsigned int off = __tt_u16(f, rec + 10UL);
        int score = 0;
        if (nameId == 1U || nameId == 16U) {
            score = nameId == 16U ? 2 : 1;
            if (plat == 3U && (enc == 1U || enc == 10U)) { score = score + (lang == 0x409U ? 8 : 6); }
            else if (plat == 0U) { score = score + 5; }
            else if (plat == 1U && enc == 0U) { score = score + 3; }
            else { score = 0; }
        }
        if (score > bestScore) {
            bestScore = score;
            bestOff = strings + (unsigned long)off;
            bestLen = len;
            bestWide = (plat == 3U || plat == 0U) ? 1 : 0;
        }
        i = i + 1U;
    }
    if (bestScore == 0) { return; }
    f.family.clear();
    if (bestWide) {
        unsigned int k = 0U;
        while (k + 1U < bestLen) {
            unsigned int u = __tt_u16(f, bestOff + (unsigned long)k);
            k = k + 2U;
            if (u >= 0xD800U && u <= 0xDBFFU && k + 1U < bestLen) {
                unsigned int lo = __tt_u16(f, bestOff + (unsigned long)k);
                k = k + 2U;
                u = 0x10000U + ((u - 0xD800U) << 10) + (lo - 0xDC00U);
            }
            char enc[4];
            int nb = gui_utf8_encode(u, &enc[0]);
            unsafe { f.family.push_n(&enc[0], (unsigned long)nb); }
        }
    } else {
        unsigned int k = 0U;
        while (k < bestLen) {
            char enc[4];
            int nb = gui_utf8_encode(__tt_u8(f, bestOff + (unsigned long)k), &enc[0]);
            unsafe { f.family.push_n(&enc[0], (unsigned long)nb); }
            k = k + 1U;
        }
    }
}

int GuiTypeface::load_memory_index(const &unsigned char bytes, unsigned long size, int index, int flags) {
    __tt_blank(self);
    if (size < 12UL) { return 0; }
    if ((flags & GUI_FONT_COPY) != 0) {
        unsafe {
            unsigned char* copy = (unsigned char*)alloc_raw_(size);
            if (copy == (unsigned char*)0) { return 0; }
            memcpy((void*)copy, (const void*)bytes, size);
            self.data = (const unsigned char*)copy;
        }
        self.owned = 1;
    } else {
        self.data = bytes;
    }
    self.size = size;
    unsigned long dir = 0UL;
    if (__tt_tag(self, 0UL, "ttcf")) {
        unsigned int nfonts = __tt_u32(self, 8UL);
        if (index < 0 || (unsigned int)index >= nfonts) { self.free(); return 0; }
        dir = (unsigned long)__tt_u32(self, 12UL + (unsigned long)index * 4UL);
        self.faceIndex = index;
    }
    unsigned int version = __tt_u32(self, dir);
    int cff = __tt_tag(self, dir, "OTTO");
    if (version != 0x00010000U && !__tt_tag(self, dir, "true") && !cff) { self.free(); return 0; }
    unsigned int numTables = __tt_u16(self, dir + 4UL);
    unsigned int i = 0U;
    while (i < numTables) {
        unsigned long rec = dir + 12UL + (unsigned long)i * 16UL;
        unsigned int off = __tt_u32(self, rec + 8UL);
        unsigned int len = __tt_u32(self, rec + 12UL);
        if ((unsigned long)off + (unsigned long)len <= size) {
            if (__tt_tag(self, rec, "head")) { self.tHead = off; }
            else if (__tt_tag(self, rec, "hhea")) { self.tHhea = off; }
            else if (__tt_tag(self, rec, "hmtx")) { self.tHmtx = off; }
            else if (__tt_tag(self, rec, "maxp")) { self.tMaxp = off; }
            else if (__tt_tag(self, rec, "cmap")) { self.tCmap = off; }
            else if (__tt_tag(self, rec, "loca")) { self.tLoca = off; self.lenLoca = len; }
            else if (__tt_tag(self, rec, "glyf")) { self.tGlyf = off; self.lenGlyf = len; }
            else if (__tt_tag(self, rec, "kern")) { self.tKern = off; }
            else if (__tt_tag(self, rec, "GPOS")) { self.tGpos = off; }
            else if (__tt_tag(self, rec, "OS/2")) { self.tOs2 = off; }
            else if (__tt_tag(self, rec, "name")) { self.tName = off; }
        }
        i = i + 1U;
    }
    if (self.tHead == 0U || self.tHhea == 0U || self.tHmtx == 0U || self.tMaxp == 0U || self.tCmap == 0U) {
        self.free();
        return 0;
    }
    self.hasOutlines = (self.tGlyf != 0U && self.tLoca != 0U) ? 1 : 0;
    if (!self.hasOutlines && (flags & GUI_FONT_ALLOW_NO_OUTLINES) == 0) { self.free(); return 0; }
    unsigned long head = (unsigned long)self.tHead;
    self.unitsPerEm = (int)__tt_u16(self, head + 18UL);
    if (self.unitsPerEm < 16 || self.unitsPerEm > 16384) { self.unitsPerEm = 1000; }
    self.indexToLocFormat = __tt_i16(self, head + 50UL);
    self.numGlyphs = (int)__tt_u16(self, (unsigned long)self.tMaxp + 4UL);
    unsigned long hhea = (unsigned long)self.tHhea;
    self.ascender = __tt_i16(self, hhea + 4UL);
    self.descender = __tt_i16(self, hhea + 6UL);
    self.lineGap = __tt_i16(self, hhea + 8UL);
    self.numHMetrics = (int)__tt_u16(self, hhea + 34UL);
    if (self.numHMetrics < 1) { self.numHMetrics = 1; }
    self.capHeight = self.ascender * 7 / 10;
    self.xHeight = self.ascender / 2;
    if (self.tOs2 != 0U) {
        unsigned long os2 = (unsigned long)self.tOs2;
        unsigned int ver = __tt_u16(self, os2);
        self.weight = (int)__tt_u16(self, os2 + 4UL);
        unsigned int sel = __tt_u16(self, os2 + 62UL);
        self.italic = (sel & 1U) != 0U ? 1 : 0;
        if ((sel & 0x80U) != 0U) {            // USE_TYPO_METRICS
            self.ascender = __tt_i16(self, os2 + 68UL);
            self.descender = __tt_i16(self, os2 + 70UL);
            self.lineGap = __tt_i16(self, os2 + 72UL);
        }
        if (ver >= 2U) {
            int xh = __tt_i16(self, os2 + 86UL);
            int ch = __tt_i16(self, os2 + 88UL);
            if (xh > 0) { self.xHeight = xh; }
            if (ch > 0) { self.capHeight = ch; }
        }
    }
    if (self.ascender <= 0 && self.descender >= 0) { self.ascender = self.unitsPerEm * 8 / 10; self.descender = -self.unitsPerEm / 5; }
    __tt_pick_cmap(self);
    if (self.cmapSub == 0U) { self.free(); return 0; }
    __tt_find_kern(self);
    __tt_find_gpos(self);
    __tt_read_name(self);
    return 1;
}

int GuiTypeface::load_memory(const &unsigned char bytes, unsigned long size, int flags) {
    return self.load_memory_index(bytes, size, 0, flags);
}

int GuiTypeface::load_file_index(const &char path, int index) {
    __tt_blank(self);
    ?&File fq = File::open(path, "rb");
    match (fq) {
        case null: return 0;
        case some(f): {
            long long n = f.size();
            if (n < 12LL || n > 0x7FFFFFFFLL) { f.close(); return 0; }
            unsigned char* buf;
            unsafe { buf = (unsigned char*)alloc_raw_((unsigned long)n); }
            int ok;
            unsafe { ok = read_exact(f, ((unsigned char*)buf)[0..(unsigned long)n]); }
            f.close();
            if (!ok) { unsafe { dealloc_raw_((void*)buf); } return 0; }
            int r;
            unsafe { r = self.load_memory_index((const unsigned char*)buf, (unsigned long)n, index, 0); }
            if (!r) { unsafe { dealloc_raw_((void*)buf); } return 0; }
            self.owned = 1;    // the buffer is ours now
            return 1;
        }
    }
    return 0;
}

int GuiTypeface::load_file(const &char path) {
    return self.load_file_index(path, 0);
}

void GuiTypeface::cache_clear() {
    unsigned long i = 0UL;
    while (i < self.cache.len) {
        &GuiGlyphBitmap b = self.cache.at(i);
        unsafe { if (!b.coverage.is_null()) { dealloc_raw_((void*)b.coverage); } }
        b.coverage = null;
        i = i + 1UL;
    }
    self.cache.clear();
    self.cacheIndex.clear();
}

void GuiTypeface::free() {
    self.cache_clear();
    self.cache.free();
    self.cacheIndex.free();
    self.gposPairs.free();
    self.family.free();
    unsafe {
        if (self.owned && !self.data.is_null()) { dealloc_raw_((void*)self.data); }
    }
    self.data = null;
    self.size = 0UL;
    self.owned = 0;
    self.cmapSub = 0U;
    self.numGlyphs = 0;
}

// ── cmap ──────────────────────────────────────────────────────────────────

int GuiTypeface::glyph_index(unsigned int cp) const {
    if (self.cmapSub == 0U) { return 0; }
    unsigned long t = (unsigned long)self.cmapSub;
    int g = 0;
    if (self.cmapFormat == 4) {
        if (cp > 0xFFFFU) { return 0; }
        unsigned int segX2 = __tt_u16(self, t + 6UL);
        unsigned int segs = segX2 / 2U;
        unsigned long ends = t + 14UL;
        unsigned long starts = ends + (unsigned long)segX2 + 2UL;
        unsigned long deltas = starts + (unsigned long)segX2;
        unsigned long ranges = deltas + (unsigned long)segX2;
        // binary search for the first segment whose end >= cp
        unsigned int lo = 0U;
        unsigned int hi = segs;
        while (lo < hi) {
            unsigned int mid = (lo + hi) / 2U;
            if (__tt_u16(self, ends + (unsigned long)mid * 2UL) < cp) { lo = mid + 1U; } else { hi = mid; }
        }
        if (lo >= segs) { return 0; }
        unsigned int start = __tt_u16(self, starts + (unsigned long)lo * 2UL);
        if (cp < start) { return 0; }
        unsigned int delta = __tt_u16(self, deltas + (unsigned long)lo * 2UL);
        unsigned int roff = __tt_u16(self, ranges + (unsigned long)lo * 2UL);
        if (roff == 0U) { return (int)((cp + delta) & 0xFFFFU); }
        unsigned long ga = ranges + (unsigned long)lo * 2UL + (unsigned long)roff + (unsigned long)(cp - start) * 2UL;
        unsigned int gv = __tt_u16(self, ga);
        if (gv == 0U) { return 0; }
        return (int)((gv + delta) & 0xFFFFU);
    }
    if (self.cmapFormat == 12) {
        unsigned int n = __tt_u32(self, t + 12UL);
        unsigned int lo = 0U;
        unsigned int hi = n;
        while (lo < hi) {
            unsigned int mid = (lo + hi) / 2U;
            unsigned long gr = t + 16UL + (unsigned long)mid * 12UL;
            unsigned int s = __tt_u32(self, gr);
            unsigned int e = __tt_u32(self, gr + 4UL);
            if (cp < s) { hi = mid; }
            else if (cp > e) { lo = mid + 1U; }
            else { return (int)(__tt_u32(self, gr + 8UL) + (cp - s)); }
        }
        return 0;
    }
    if (self.cmapFormat == 6) {
        unsigned int first = __tt_u16(self, t + 6UL);
        unsigned int count = __tt_u16(self, t + 8UL);
        if (cp < first || cp >= first + count) { return 0; }
        return (int)__tt_u16(self, t + 10UL + (unsigned long)(cp - first) * 2UL);
    }
    if (self.cmapFormat == 0) {
        if (cp > 255U) { return 0; }
        return (int)__tt_u8(self, t + 6UL + (unsigned long)cp);
    }
    return g;
}

int GuiTypeface::has_glyph(unsigned int cp) const {
    return self.glyph_index(cp) != 0 ? 1 : 0;
}

// ── metrics ───────────────────────────────────────────────────────────────

static float __tt_scale(const &GuiTypeface f, float size) {
    return size / (float)f.unitsPerEm;
}

float GuiTypeface::advance(int glyph, float size) const {
    if (glyph < 0 || self.tHmtx == 0U) { return 0.0f; }
    int i = glyph < self.numHMetrics ? glyph : self.numHMetrics - 1;
    unsigned int adv = __tt_u16(self, (unsigned long)self.tHmtx + (unsigned long)i * 4UL);
    return (float)adv * __tt_scale(self, size);
}

static int __tt_coverage(const &GuiTypeface f, unsigned long cov, int glyph) {
    unsigned int fmt = __tt_u16(f, cov);
    unsigned int n = __tt_u16(f, cov + 2UL);
    unsigned int g = (unsigned int)glyph;
    if (fmt == 1U) {
        unsigned int lo = 0U; unsigned int hi = n;
        while (lo < hi) {
            unsigned int mid = (lo + hi) / 2U;
            unsigned int v = __tt_u16(f, cov + 4UL + (unsigned long)mid * 2UL);
            if (v < g) { lo = mid + 1U; } else if (v > g) { hi = mid; } else { return (int)mid; }
        }
        return -1;
    }
    if (fmt == 2U) {
        unsigned int lo = 0U; unsigned int hi = n;
        while (lo < hi) {
            unsigned int mid = (lo + hi) / 2U;
            unsigned long r = cov + 4UL + (unsigned long)mid * 6UL;
            unsigned int s = __tt_u16(f, r);
            unsigned int e = __tt_u16(f, r + 2UL);
            if (g < s) { hi = mid; } else if (g > e) { lo = mid + 1U; }
            else { return (int)(__tt_u16(f, r + 4UL) + (g - s)); }
        }
        return -1;
    }
    return -1;
}

static int __tt_classdef(const &GuiTypeface f, unsigned long cd, int glyph) {
    unsigned int fmt = __tt_u16(f, cd);
    unsigned int g = (unsigned int)glyph;
    if (fmt == 1U) {
        unsigned int start = __tt_u16(f, cd + 2UL);
        unsigned int n = __tt_u16(f, cd + 4UL);
        if (g < start || g >= start + n) { return 0; }
        return (int)__tt_u16(f, cd + 6UL + (unsigned long)(g - start) * 2UL);
    }
    if (fmt == 2U) {
        unsigned int n = __tt_u16(f, cd + 2UL);
        unsigned int lo = 0U; unsigned int hi = n;
        while (lo < hi) {
            unsigned int mid = (lo + hi) / 2U;
            unsigned long r = cd + 4UL + (unsigned long)mid * 6UL;
            unsigned int s = __tt_u16(f, r);
            unsigned int e = __tt_u16(f, r + 2UL);
            if (g < s) { hi = mid; } else if (g > e) { lo = mid + 1U; } else { return (int)__tt_u16(f, r + 4UL); }
        }
    }
    return 0;
}

static unsigned int __tt_popcount(unsigned int v) {
    unsigned int n = 0U;
    while (v != 0U) { n = n + (v & 1U); v = v >> 1; }
    return n;
}

// XAdvance of a value record (0 if the format has none).
static int __tt_xadvance(const &GuiTypeface f, unsigned long rec, unsigned int vf) {
    if ((vf & 4U) == 0U) { return 0; }
    return __tt_i16(f, rec + 2UL * (unsigned long)__tt_popcount(vf & 3U));
}

// 1 and *out = the first value record's xAdvance if a GPOS pair subtable
// covers (left, right).
static int __tt_gpos_pair(const &GuiTypeface f, unsigned long sub, int left, int right, &int out) {
    unsigned int fmt = __tt_u16(f, sub);
    int ci = __tt_coverage(f, sub + (unsigned long)__tt_u16(f, sub + 2UL), left);
    if (ci < 0) { return 0; }
    unsigned int vf1 = __tt_u16(f, sub + 4UL);
    unsigned int vf2 = __tt_u16(f, sub + 6UL);
    unsigned long s1 = 2UL * (unsigned long)__tt_popcount(vf1);
    unsigned long s2 = 2UL * (unsigned long)__tt_popcount(vf2);
    if (fmt == 1U) {
        unsigned int nsets = __tt_u16(f, sub + 8UL);
        if ((unsigned int)ci >= nsets) { return 0; }
        unsigned long set = sub + (unsigned long)__tt_u16(f, sub + 10UL + (unsigned long)ci * 2UL);
        unsigned int n = __tt_u16(f, set);
        unsigned long recSize = 2UL + s1 + s2;
        unsigned int lo = 0U; unsigned int hi = n;
        while (lo < hi) {
            unsigned int mid = (lo + hi) / 2U;
            unsigned long r = set + 2UL + (unsigned long)mid * recSize;
            unsigned int g2 = __tt_u16(f, r);
            if (g2 < (unsigned int)right) { lo = mid + 1U; }
            else if (g2 > (unsigned int)right) { hi = mid; }
            else { *out = __tt_xadvance(f, r + 2UL, vf1); return 1; }
        }
        return 0;
    }
    if (fmt == 2U) {
        unsigned long cd1 = sub + (unsigned long)__tt_u16(f, sub + 8UL);
        unsigned long cd2 = sub + (unsigned long)__tt_u16(f, sub + 10UL);
        unsigned int n1 = __tt_u16(f, sub + 12UL);
        unsigned int n2 = __tt_u16(f, sub + 14UL);
        int c1 = __tt_classdef(f, cd1, left);
        int c2 = __tt_classdef(f, cd2, right);
        if ((unsigned int)c1 >= n1 || (unsigned int)c2 >= n2) { return 0; }
        unsigned long r = sub + 16UL + ((unsigned long)c1 * (unsigned long)n2 + (unsigned long)c2) * (s1 + s2);
        *out = __tt_xadvance(f, r, vf1);
        return 1;
    }
    return 0;
}

float GuiTypeface::kerning(int left, int right, float size) const {
    if (left <= 0 || right <= 0) { return 0.0f; }
    if (self.kernSub != 0U && self.kernPairs > 0) {
        unsigned long pairs = (unsigned long)self.kernSub + 14UL;
        unsigned int key = ((unsigned int)left << 16) | (unsigned int)right;
        unsigned int lo = 0U; unsigned int hi = (unsigned int)self.kernPairs;
        while (lo < hi) {
            unsigned int mid = (lo + hi) / 2U;
            unsigned long r = pairs + (unsigned long)mid * 6UL;
            unsigned int k = __tt_u32(self, r);
            if (k < key) { lo = mid + 1U; } else if (k > key) { hi = mid; }
            else { return (float)__tt_i16(self, r + 4UL) * __tt_scale(self, size); }
        }
        return 0.0f;
    }
    unsigned long i = 0UL;
    while (i < self.gposPairs.len) {
        int v = 0;
        if (__tt_gpos_pair(self, (unsigned long)*self.gposPairs.at(i), left, right, &v)) {
            return (float)v * __tt_scale(self, size);
        }
        i = i + 1UL;
    }
    return 0.0f;
}

float GuiTypeface::ascent(float size) const { return (float)self.ascender * __tt_scale(self, size); }
float GuiTypeface::descent(float size) const { return -(float)self.descender * __tt_scale(self, size); }
float GuiTypeface::line_gap(float size) const {
    return (float)(self.lineGap > 0 ? self.lineGap : 0) * __tt_scale(self, size);
}
float GuiTypeface::line_height(float size) const {
    return self.ascent(size) + self.descent(size) + self.line_gap(size);
}

// ── outlines ──────────────────────────────────────────────────────────────

static int __tt_glyph_range(const &GuiTypeface f, int glyph, &unsigned long start, &unsigned long len) {
    if (!f.hasOutlines || glyph < 0 || glyph >= f.numGlyphs) { return 0; }
    unsigned long loca = (unsigned long)f.tLoca;
    unsigned long a = 0UL;
    unsigned long b = 0UL;
    if (f.indexToLocFormat == 0) {
        a = (unsigned long)__tt_u16(f, loca + (unsigned long)glyph * 2UL) * 2UL;
        b = (unsigned long)__tt_u16(f, loca + (unsigned long)(glyph + 1) * 2UL) * 2UL;
    } else {
        a = (unsigned long)__tt_u32(f, loca + (unsigned long)glyph * 4UL);
        b = (unsigned long)__tt_u32(f, loca + (unsigned long)(glyph + 1) * 4UL);
    }
    if (b <= a || b > (unsigned long)f.lenGlyf) { return 0; }
    *start = (unsigned long)f.tGlyf + a;
    *len = b - a;
    return 1;
}

int GuiTypeface::glyph_box(int glyph, &int x0, &int y0, &int x1, &int y1) const {
    unsigned long g = 0UL; unsigned long len = 0UL;
    if (!__tt_glyph_range(self, glyph, &g, &len) || len < 10UL) { return 0; }
    *x0 = __tt_i16(self, g + 2UL);
    *y0 = __tt_i16(self, g + 4UL);
    *x1 = __tt_i16(self, g + 6UL);
    *y1 = __tt_i16(self, g + 8UL);
    return 1;
}

static void __tt_emit(&GuiPath out, struct GuiMatrix m, int op, float x1, float y1, float x2, float y2) {
    float ax = m.apply_x(x1, y1); float ay = m.apply_y(x1, y1);
    if (op == 0) { out.move_to(ax, ay); return; }
    if (op == 1) { out.line_to(ax, ay); return; }
    float bx = m.apply_x(x2, y2); float by = m.apply_y(x2, y2);
    out.quad_to(ax, ay, bx, by);
}

// Appends glyph's contours with transform m (font units -> output).
static int __tt_outline(const &GuiTypeface f, int glyph, &GuiPath out, struct GuiMatrix m, int depth) {
    if (depth > 8) { return 0; }
    unsigned long g = 0UL; unsigned long len = 0UL;
    if (!__tt_glyph_range(f, glyph, &g, &len) || len < 10UL) { return 0; }
    int nc = __tt_i16(f, g);
    if (nc < 0) {
        // Composite glyph.
        unsigned long p = g + 10UL;
        int any = 0;
        unsigned int flags = 0x20U;
        int guard = 0;
        while ((flags & 0x20U) != 0U && guard < 64) {
            flags = __tt_u16(f, p);
            int sub = (int)__tt_u16(f, p + 2UL);
            p = p + 4UL;
            float dx = 0.0f; float dy = 0.0f;
            if ((flags & 1U) != 0U) {          // ARG_1_AND_2_ARE_WORDS
                dx = (float)__tt_i16(f, p); dy = (float)__tt_i16(f, p + 2UL);
                p = p + 4UL;
            } else {
                unsigned int a = __tt_u8(f, p);
                unsigned int b = __tt_u8(f, p + 1UL);
                dx = (float)(a >= 128U ? (int)a - 256 : (int)a);
                dy = (float)(b >= 128U ? (int)b - 256 : (int)b);
                p = p + 2UL;
            }
            if ((flags & 2U) == 0U) { dx = 0.0f; dy = 0.0f; }   // point matching: not supported
            float sa = 1.0f; float sb = 0.0f; float sc = 0.0f; float sd = 1.0f;
            if ((flags & 8U) != 0U) {           // WE_HAVE_A_SCALE
                sa = (float)__tt_i16(f, p) / 16384.0f; sd = sa;
                p = p + 2UL;
            } else if ((flags & 0x40U) != 0U) { // X_AND_Y_SCALE
                sa = (float)__tt_i16(f, p) / 16384.0f;
                sd = (float)__tt_i16(f, p + 2UL) / 16384.0f;
                p = p + 4UL;
            } else if ((flags & 0x80U) != 0U) { // TWO_BY_TWO
                sa = (float)__tt_i16(f, p) / 16384.0f;
                sb = (float)__tt_i16(f, p + 2UL) / 16384.0f;
                sc = (float)__tt_i16(f, p + 4UL) / 16384.0f;
                sd = (float)__tt_i16(f, p + 6UL) / 16384.0f;
                p = p + 8UL;
            }
            struct GuiMatrix cm = GuiMatrix::new(sa, sb, sc, sd, dx, dy);
            if (__tt_outline(f, sub, out, m.mul(cm), depth + 1)) { any = 1; }
            guard = guard + 1;
        }
        return any;
    }
    if (nc == 0) { return 0; }
    unsigned long endPts = g + 10UL;
    int npts = (int)__tt_u16(f, endPts + (unsigned long)(nc - 1) * 2UL) + 1;
    if (npts <= 0 || npts > 65535) { return 0; }
    unsigned long ins = endPts + (unsigned long)nc * 2UL;
    unsigned long p = ins + 2UL + (unsigned long)__tt_u16(f, ins);
    unsigned char* flg;
    int* xs;
    int* ys;
    unsafe {
        flg = (unsigned char*)alloc_raw_((unsigned long)npts);
        xs = (int*)alloc_raw_((unsigned long)npts * 4UL);
        ys = (int*)alloc_raw_((unsigned long)npts * 4UL);
    }
    unsafe {
        int i = 0;
        while (i < npts) {
            unsigned int fl = __tt_u8(f, p);
            p = p + 1UL;
            flg[i] = (unsigned char)fl;
            i = i + 1;
            if ((fl & 8U) != 0U) {
                unsigned int rep = __tt_u8(f, p);
                p = p + 1UL;
                while (rep > 0U && i < npts) { flg[i] = (unsigned char)fl; i = i + 1; rep = rep - 1U; }
            }
        }
        int v = 0;
        i = 0;
        while (i < npts) {
            unsigned int fl = (unsigned int)flg[i];
            if ((fl & 2U) != 0U) {
                int d = (int)__tt_u8(f, p); p = p + 1UL;
                v = v + ((fl & 16U) != 0U ? d : -d);
            } else if ((fl & 16U) == 0U) {
                v = v + __tt_i16(f, p); p = p + 2UL;
            }
            xs[i] = v;
            i = i + 1;
        }
        v = 0;
        i = 0;
        while (i < npts) {
            unsigned int fl = (unsigned int)flg[i];
            if ((fl & 4U) != 0U) {
                int d = (int)__tt_u8(f, p); p = p + 1UL;
                v = v + ((fl & 32U) != 0U ? d : -d);
            } else if ((fl & 32U) == 0U) {
                v = v + __tt_i16(f, p); p = p + 2UL;
            }
            ys[i] = v;
            i = i + 1;
        }
        // Contours: quadratic B-splines; two off-curve points in a row
        // imply an on-curve point midway.
        int start = 0;
        int c = 0;
        while (c < nc) {
            int end = (int)__tt_u16(f, endPts + (unsigned long)c * 2UL);
            c = c + 1;
            if (end < start || end >= npts) { break; }
            int n = end - start + 1;
            if (n < 2) { start = end + 1; continue; }
            // A starting on-curve point (or the midpoint of the first two).
            int first = -1;
            int k = 0;
            while (k < n) { if ((flg[start + k] & 1U) != 0U) { first = k; break; } k = k + 1; }
            float sx; float sy;
            if (first >= 0) {
                sx = (float)xs[start + first]; sy = (float)ys[start + first];
            } else {
                sx = ((float)xs[start] + (float)xs[start + 1]) * 0.5f;
                sy = ((float)ys[start] + (float)ys[start + 1]) * 0.5f;
                first = 0;
            }
            __tt_emit(out, m, 0, sx, sy, 0.0f, 0.0f);
            int haveOff = 0;
            float ox = 0.0f; float oy = 0.0f;
            k = 1;
            while (k <= n) {
                int idx = start + (first + k) % n;
                float px = (float)xs[idx];
                float py = (float)ys[idx];
                int on = (flg[idx] & 1U) != 0U ? 1 : 0;
                if (on) {
                    if (haveOff) { __tt_emit(out, m, 2, ox, oy, px, py); haveOff = 0; }
                    else { __tt_emit(out, m, 1, px, py, 0.0f, 0.0f); }
                } else {
                    if (haveOff) {
                        float mx = (ox + px) * 0.5f;
                        float my = (oy + py) * 0.5f;
                        __tt_emit(out, m, 2, ox, oy, mx, my);
                    }
                    ox = px; oy = py;
                    haveOff = 1;
                }
                k = k + 1;
            }
            if (haveOff) { __tt_emit(out, m, 2, ox, oy, sx, sy); }
            out.close();
            start = end + 1;
        }
        dealloc_raw_((void*)flg);
        dealloc_raw_((void*)xs);
        dealloc_raw_((void*)ys);
    }
    return 1;
}

int GuiTypeface::glyph_path(int glyph, &GuiPath out, float size, float x, float y) const {
    float s = __tt_scale(self, size);
    // font units (y up) -> pixels (y down) at the pen
    struct GuiMatrix m = GuiMatrix::new(s, 0.0f, 0.0f, -s, x, y);
    return __tt_outline(self, glyph, out, m, 0);
}

// ── measuring ─────────────────────────────────────────────────────────────

float GuiTypeface::measure_n(const &char text, unsigned long len, float size) const {
    &GuiTypeface fb = GuiTypeface::fallback();
    float w = 0.0f;
    int prev = 0;
    unsigned long i = 0UL;
    int self_is_fb;
    unsafe { self_is_fb = ((const void*)self == (const void*)fb) ? 1 : 0; }
    while (i < len) {
        unsigned int cp = gui_utf8_next(text, len, &i);
        if (cp == 0U) { break; }
        int g = self.glyph_index(cp);
        if (g == 0 && !self_is_fb) {
            int g2 = fb.glyph_index(cp);
            if (g2 != 0) { w = w + fb.advance(g2, size); prev = 0; continue; }
        }
        if (prev != 0) { w = w + self.kerning(prev, g, size); }
        w = w + self.advance(g, size);
        prev = g;
    }
    return w;
}

float GuiTypeface::measure(const &char text, float size) const {
    unsigned long n = 0UL;
    unsafe { while (text[n] != '\0') { n = n + 1UL; } }
    return self.measure_n(text, n, size);
}

// ── glyph cache ───────────────────────────────────────────────────────────

static unsigned int __tt_hash(int glyph, int sizeKey, int sub) {
    unsigned int h = (unsigned int)glyph * 2654435761U;
    h = h ^ ((unsigned int)sizeKey * 40503U);
    h = h ^ ((unsigned int)sub * 0x9E3779B9U);
    return h ^ (h >> 15);
}

?&GuiGlyphBitmap GuiTypeface::cache_find(int glyph, int sizeKey, int subpixel) const {
    unsigned long cap = self.cacheIndex.len;
    if (cap == 0UL) { return null; }
    unsigned long mask = cap - 1UL;
    unsigned long h = (unsigned long)__tt_hash(glyph, sizeKey, subpixel) & mask;
    unsigned long probes = 0UL;
    while (probes < cap) {
        int idx = *self.cacheIndex.at(h);
        if (idx < 0) { return null; }
        &GuiGlyphBitmap b = self.cache.at((unsigned long)idx);
        if (b.glyph == glyph && b.sizeKey == sizeKey && b.subpixel == subpixel) { return b; }
        h = (h + 1UL) & mask;
        probes = probes + 1UL;
    }
    return null;
}

static void __tt_cache_reindex(&GuiTypeface f, unsigned long cap) {
    f.cacheIndex.clear();
    unsigned long i = 0UL;
    while (i < cap) { f.cacheIndex.push(-1); i = i + 1UL; }
    unsigned long mask = cap - 1UL;
    unsigned long k = 0UL;
    while (k < f.cache.len) {
        &GuiGlyphBitmap b = f.cache.at(k);
        unsigned long h = (unsigned long)__tt_hash(b.glyph, b.sizeKey, b.subpixel) & mask;
        while (*f.cacheIndex.at(h) >= 0) { h = (h + 1UL) & mask; }
        f.cacheIndex.set(h, (int)k);
        k = k + 1UL;
    }
}

&GuiGlyphBitmap GuiTypeface::cache_insert(struct GuiGlyphBitmap b) {
    // Bounded: a full cache starts over (glyphs are cheap to re-render).
    if (self.cache.len >= 4096UL) { self.cache_clear(); }
    self.cache.push(b);
    unsigned long cap = self.cacheIndex.len;
    if (cap < 64UL || self.cache.len * 2UL > cap) {
        unsigned long ncap = cap < 64UL ? 64UL : cap * 2UL;
        while (ncap < self.cache.len * 2UL) { ncap = ncap * 2UL; }
        __tt_cache_reindex(self, ncap);
    } else {
        unsigned long mask = cap - 1UL;
        unsigned long h = (unsigned long)__tt_hash(b.glyph, b.sizeKey, b.subpixel) & mask;
        while (*self.cacheIndex.at(h) >= 0) { h = (h + 1UL) & mask; }
        self.cacheIndex.set(h, (int)(self.cache.len - 1UL));
    }
    return self.cache.at(self.cache.len - 1UL);
}

// ── system fonts ──────────────────────────────────────────────────────────

struct GuiSystemFace {
    struct String key;
    ?&GuiTypeface face;
};

static struct Vec<struct GuiSystemFace> gGuiSystemFaces;
static int gGuiSystemFacesInit = 0;

static int __tt_try_load(&GuiTypeface f, const &char dir, const &char name) {
    struct String path = String::from(dir);
    path.push("/");
    path.push(name);
    int ok;
    unsafe { ok = f.load_file((const char*)path.data); }
    path.free();
    return ok;
}

extern char* getenv(const char* name);

static ?&GuiTypeface __tt_system_lookup(const &char family, int bold) {
    if (!gGuiSystemFacesInit) { gGuiSystemFaces = Vec::new(); gGuiSystemFacesInit = 1; }
    struct String key = String::from(family);
    key.push(bold ? "#b" : "#r");
    unsigned long i = 0UL;
    while (i < gGuiSystemFaces.len) {
        &GuiSystemFace e = gGuiSystemFaces.at(i);
        if (e.key.eq(&key)) { key.free(); return e.face; }
        i = i + 1UL;
    }
    // Candidate file names for the family.
    struct Vec<struct String> names = Vec::new();
    struct String fam = String::from(family);
    struct String lower = fam.to_lower();
    int isGeneric = 0;
    if (lower.eq_cstr("sans") || lower.eq_cstr("sans-serif") || lower.eq_cstr("ui")) {
        isGeneric = 1;
        if (bold) {
            names.push(String::from("DejaVuSans-Bold.ttf")); names.push(String::from("LiberationSans-Bold.ttf"));
            names.push(String::from("NotoSans-Bold.ttf")); names.push(String::from("FreeSansBold.ttf"));
            names.push(String::from("segoeuib.ttf")); names.push(String::from("arialbd.ttf"));
            names.push(String::from("Arial Bold.ttf"));
        } else {
            names.push(String::from("DejaVuSans.ttf")); names.push(String::from("LiberationSans-Regular.ttf"));
            names.push(String::from("NotoSans-Regular.ttf")); names.push(String::from("FreeSans.ttf"));
            names.push(String::from("segoeui.ttf")); names.push(String::from("arial.ttf"));
            names.push(String::from("Arial.ttf")); names.push(String::from("Geneva.ttf"));
        }
    } else if (lower.eq_cstr("serif")) {
        isGeneric = 1;
        names.push(String::from(bold ? "DejaVuSerif-Bold.ttf" : "DejaVuSerif.ttf"));
        names.push(String::from(bold ? "LiberationSerif-Bold.ttf" : "LiberationSerif-Regular.ttf"));
        names.push(String::from(bold ? "NotoSerif-Bold.ttf" : "NotoSerif-Regular.ttf"));
        names.push(String::from(bold ? "FreeSerifBold.ttf" : "FreeSerif.ttf"));
        names.push(String::from(bold ? "timesbd.ttf" : "times.ttf"));
        names.push(String::from(bold ? "Times New Roman Bold.ttf" : "Times New Roman.ttf"));
    } else if (lower.eq_cstr("mono") || lower.eq_cstr("monospace")) {
        isGeneric = 1;
        names.push(String::from(bold ? "DejaVuSansMono-Bold.ttf" : "DejaVuSansMono.ttf"));
        names.push(String::from(bold ? "LiberationMono-Bold.ttf" : "LiberationMono-Regular.ttf"));
        names.push(String::from(bold ? "NotoSansMono-Bold.ttf" : "NotoSansMono-Regular.ttf"));
        names.push(String::from(bold ? "FreeMonoBold.ttf" : "FreeMono.ttf"));
        names.push(String::from(bold ? "consolab.ttf" : "consola.ttf"));
        names.push(String::from(bold ? "courbd.ttf" : "cour.ttf"));
        names.push(String::from(bold ? "Courier New Bold.ttf" : "Courier New.ttf"));
    }
    // From the family name itself: "DejaVu Sans" -> DejaVuSans.ttf,
    // DejaVuSans-Regular.ttf, "DejaVu Sans.ttf", dejavusans.ttf ...
    struct String compact = String::new();
    unsafe {
        unsigned long k = 0UL;
        while (k < fam.len) { char ch = fam.data[k]; if (ch != ' ') { compact.push_char(ch); } k = k + 1UL; }
    }
    if (!isGeneric) {
        struct String a = compact.clone(); a.push(bold ? "-Bold.ttf" : ".ttf"); names.push(a);
        struct String b = compact.clone(); b.push(bold ? "Bold.ttf" : "-Regular.ttf"); names.push(b);
        struct String c = fam.clone(); c.push(bold ? " Bold.ttf" : ".ttf"); names.push(c);
        struct String d = compact.to_lower(); d.push(bold ? "bd.ttf" : ".ttf"); names.push(d);
        struct String e = compact.clone(); e.push(".ttc"); names.push(e);
        struct String g = fam.clone(); g.push(".ttc"); names.push(g);
    }
    // Directories to look in.
    struct Vec<struct String> dirs = Vec::new();
    unsafe {
        char* env = getenv("SAFEC_GUI_FONT_DIR");
        if (env != (char*)0) { dirs.push(String::from(env)); }
    }
    struct String firstWord = String::new();
    unsafe {
        unsigned long k = 0UL;
        while (k < lower.len && lower.data[k] != ' ') { firstWord.push_char(lower.data[k]); k = k + 1UL; }
    }
    const char* fixed[17];
    fixed[0] = "/usr/share/fonts/truetype/dejavu"; fixed[1] = "/usr/share/fonts/truetype/liberation";
    fixed[2] = "/usr/share/fonts/truetype/liberation2"; fixed[3] = "/usr/share/fonts/truetype/noto";
    fixed[4] = "/usr/share/fonts/truetype/freefont"; fixed[5] = "/usr/share/fonts/TTF";
    fixed[6] = "/usr/share/fonts/truetype"; fixed[7] = "/usr/share/fonts/dejavu";
    fixed[8] = "/usr/share/fonts/noto"; fixed[9] = "/usr/local/share/fonts";
    fixed[10] = "/usr/local/share/fonts/dejavu"; fixed[11] = "/System/Library/Fonts";
    fixed[12] = "/System/Library/Fonts/Supplemental"; fixed[13] = "/Library/Fonts";
    fixed[14] = "C:/Windows/Fonts"; fixed[15] = "/usr/share/fonts/opentype/noto";
    fixed[16] = "/usr/share/fonts/liberation";
    if (firstWord.len > 0UL) {
        struct String d1 = String::from("/usr/share/fonts/truetype/"); d1.push_str(&firstWord); dirs.push(d1);
        struct String d2 = String::from("/usr/share/fonts/"); d2.push_str(&firstWord); dirs.push(d2);
    }
    int fi = 0;
    while (fi < 17) { unsafe { dirs.push(String::from(fixed[fi])); } fi = fi + 1; }

    ?&GuiTypeface found = null;
    unsafe {
        &GuiTypeface face = (struct GuiTypeface*)alloc_zeroed_raw_(sizeof(struct GuiTypeface));
        int ok = 0;
        unsigned long ni = 0UL;
        while (ni < names.len && !ok) {
            unsigned long di = 0UL;
            while (di < dirs.len && !ok) {
                if (__tt_try_load(face, (const char*)dirs.at(di).data, (const char*)names.at(ni).data)) { ok = 1; }
                di = di + 1UL;
            }
            ni = ni + 1UL;
        }
        if (ok) { found = face; } else { face.free(); dealloc_raw_((void*)face); }
    }
    unsigned long k = 0UL;
    while (k < names.len) { names.at(k).free(); k = k + 1UL; }
    names.free();
    k = 0UL;
    while (k < dirs.len) { dirs.at(k).free(); k = k + 1UL; }
    dirs.free();
    fam.free();
    lower.free();
    compact.free();
    firstWord.free();
    struct GuiSystemFace entry;
    entry.key = key;
    entry.face = found;
    gGuiSystemFaces.push(entry);
    return found;
}

?&GuiTypeface GuiTypeface::system(const &char family) {
    return __tt_system_lookup(family, 0);
}

?&GuiTypeface GuiTypeface::system_bold(const &char family) {
    return __tt_system_lookup(family, 1);
}

} // namespace std
