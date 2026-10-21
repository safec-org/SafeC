// SafeC Standard Library — GUI: loadable/serializable bitmap fonts (see
// gui_font.h for the on-disk "SCXF" format).
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_font.h>
#include <std/mem.h>
#include <std/io_file.h>
#include <std/io_traits.h>
#include <std/panic.h>
extern void* memset(void* ptr, int val, size_t n);

namespace std {

static int __gui_font_bytes_per_glyph(int glyphW, int glyphH) {
    int bits = glyphW * glyphH;
    return (bits + 7) / 8;
}

struct GuiFont GuiFont::new(int glyphW, int glyphH, int first, int count) {
    struct GuiFont f;
    f.glyphW = glyphW;
    f.glyphH = glyphH;
    f.first = first;
    f.count = count;
    unsigned long total = (unsigned long)(__gui_font_bytes_per_glyph(glyphW, glyphH) * count);
    unsafe {
        f.bitmap = (&heap unsigned char)alloc_raw_(total);
        memset((void*)f.bitmap, 0, total);
    }
    return f;
}

int GuiFont::get_bit(int cp, int px, int py) const {
    int idx = 0; int glyphW = 0; int glyphH = 0; int count = 0;
    unsafe {
        idx = cp - self.first;
        glyphW = self.glyphW;
        glyphH = self.glyphH;
        count = self.count;
    }
    if (idx < 0 || idx >= count) { return 0; }
    if (px < 0 || py < 0 || px >= glyphW || py >= glyphH) { return 0; }
    int bytesPerGlyph = __gui_font_bytes_per_glyph(glyphW, glyphH);
    int bitIndex = py * glyphW + px;
    unsigned long byteOff = (unsigned long)(idx * bytesPerGlyph) + (unsigned long)(bitIndex / 8);
    int bitInByte = 7 - (bitIndex % 8);
    unsigned char byteVal;
    unsafe { byteVal = self.bitmap[byteOff]; }
    return (int)((byteVal >> bitInByte) & 1U);
}

void GuiFont::set_bit(int cp, int px, int py, int value) {
    int idx = 0; int glyphW = 0; int glyphH = 0; int count = 0;
    unsafe {
        idx = cp - self.first;
        glyphW = self.glyphW;
        glyphH = self.glyphH;
        count = self.count;
    }
    if (idx < 0 || idx >= count) { return; }
    if (px < 0 || py < 0 || px >= glyphW || py >= glyphH) { return; }
    int bytesPerGlyph = __gui_font_bytes_per_glyph(glyphW, glyphH);
    int bitIndex = py * glyphW + px;
    unsigned long byteOff = (unsigned long)(idx * bytesPerGlyph) + (unsigned long)(bitIndex / 8);
    int bitInByte = 7 - (bitIndex % 8);
    unsafe {
        unsigned char cur = self.bitmap[byteOff];
        if (value) {
            self.bitmap[byteOff] = cur | (unsigned char)(1U << bitInByte);
        } else {
            self.bitmap[byteOff] = cur & (unsigned char)(~(1U << bitInByte));
        }
    }
}

// The built-in 8x8 font's raw table (gFont8x8) is private to gui_draw.sc;
// re-derive an equivalent GuiFont here by calling the public
// GuiWindow::draw_text() (gui.h, defined in gui_draw.sc) for each glyph
// instead of reaching into its internals.
struct GuiFont GuiFont::default() {
    struct GuiFont f = GuiFont::new(8, 8, 32, 95);
    int cp = 32;
    while (cp <= 126) {
        // Render this one glyph at scale 1 into a throwaway 8x8 window
        // buffer, then copy its pixels into the GuiFont — avoids needing
        // access to gui_draw.sc's private table while still producing a
        // real, correct copy of the exact same built-in glyphs.
        struct GuiWindow tmp;
        tmp.platform = null;
        tmp.width = 8;
        tmp.height = 8;
        tmp.shouldClose = 0;
        unsafe {
            tmp.pixels = (&heap unsigned char)alloc_raw_(8UL * 8UL * 4UL);
            memset((void*)tmp.pixels, 0, 8UL * 8UL * 4UL);
        }
        char one[2];
        unsafe { one[0] = (char)cp; one[1] = '\0'; }
        struct GuiColor white = GuiColor::rgba(255U, 255U, 255U, 255U);
        unsafe { tmp.draw_text(0, 0, (const char*)one, white, 1); }

        int py = 0;
        while (py < 8) {
            int px = 0;
            while (px < 8) {
                unsigned char a;
                unsafe {
                    unsigned long off = ((unsigned long)py * 8UL + (unsigned long)px) * 4UL;
                    a = tmp.pixels[off + 3UL];
                }
                f.set_bit(cp, px, py, (a > 0U) ? 1 : 0);
                px = px + 1;
            }
            py = py + 1;
        }
        unsafe { dealloc_raw_((void*)tmp.pixels); }
        cp = cp + 1;
    }
    return f;
}

void GuiFont::free() {
    unsafe {
        if ((void*)self.bitmap != (void*)0) { dealloc_raw_((void*)self.bitmap); }
        self.bitmap = (&heap unsigned char)0;
        self.count = 0;
    }
}

int GuiFont::save(const &char path) const {
    ?&File fq = File::open(path, "wb");
    if (fq.is_null()) { return 0; }
    &File f = expect_ref(fq, "GuiFont::save: file");

    unsigned char header[16];
    unsafe {
        header[0] = (unsigned char)'S'; header[1] = (unsigned char)'C';
        header[2] = (unsigned char)'X'; header[3] = (unsigned char)'F';
        header[4] = 1U;
        header[5] = (unsigned char)self.glyphW;
        header[6] = (unsigned char)self.glyphH;
        header[7] = 0U;
        unsigned int first = (unsigned int)self.first;
        header[8]  = (unsigned char)(first & 0xFFU);
        header[9]  = (unsigned char)((first >> 8) & 0xFFU);
        header[10] = (unsigned char)((first >> 16) & 0xFFU);
        header[11] = (unsigned char)((first >> 24) & 0xFFU);
        unsigned int count = (unsigned int)self.count;
        header[12] = (unsigned char)(count & 0xFFU);
        header[13] = (unsigned char)((count >> 8) & 0xFFU);
        header[14] = (unsigned char)((count >> 16) & 0xFFU);
        header[15] = (unsigned char)((count >> 24) & 0xFFU);
    }
    int wrote;
    unsafe { wrote = write_all(f, &header[0], 16UL); }
    if (!wrote) { f.close(); return 0; }

    int glyphW; int glyphH; int count;
    unsafe { glyphW = self.glyphW; glyphH = self.glyphH; count = self.count; }
    unsigned long total = (unsigned long)(__gui_font_bytes_per_glyph(glyphW, glyphH) * count);
    int bodyWrote;
    unsafe { bodyWrote = write_all(f, (const &unsigned char)self.bitmap, total); }
    f.close();
    return bodyWrote;
}

int GuiFont::load(const &char path) {
    ?&File fq = File::open(path, "rb");
    if (fq.is_null()) { return 0; }
    &File f = expect_ref(fq, "GuiFont::load: file");

    unsigned char header[16];
    int got;
    unsafe { got = read_exact(f, &header[0], 16UL); }
    if (!got) { f.close(); return 0; }

    int ok = 1;
    unsafe {
        if (header[0] != (unsigned char)'S' || header[1] != (unsigned char)'C' ||
            header[2] != (unsigned char)'X' || header[3] != (unsigned char)'F' ||
            header[4] != 1U) {
            ok = 0;
        }
    }
    if (!ok) { f.close(); return 0; }

    int glyphW; int glyphH; unsigned int first; unsigned int count;
    unsafe {
        glyphW = (int)header[5];
        glyphH = (int)header[6];
        first = (unsigned int)header[8]  | ((unsigned int)header[9] << 8)
              | ((unsigned int)header[10] << 16) | ((unsigned int)header[11] << 24);
        count = (unsigned int)header[12] | ((unsigned int)header[13] << 8)
              | ((unsigned int)header[14] << 16) | ((unsigned int)header[15] << 24);
    }

    struct GuiFont loaded = GuiFont::new(glyphW, glyphH, (int)first, (int)count);
    unsigned long total = (unsigned long)(__gui_font_bytes_per_glyph(glyphW, glyphH) * (int)count);
    int bodyGot;
    unsafe { bodyGot = read_exact(f, (&unsigned char)loaded.bitmap, total); }
    f.close();
    if (!bodyGot) {
        loaded.free();
        return 0;
    }
    unsafe { *self = loaded; }
    return 1;
}

int GuiFont::draw_text(&GuiWindow win, int x, int y, const &char text,
                       struct GuiColor color, int scale) const {
    int cx = x;
    unsigned long i = 0UL;
    unsafe {
        while (text[i] != '\0') {
            int cp = (int)(unsigned char)text[i];
            int py = 0;
            while (py < self.glyphH) {
                int px = 0;
                while (px < self.glyphW) {
                    if (self.get_bit(cp, px, py)) {
                        win.fill_rect(cx + px * scale, y + py * scale, scale, scale, color);
                    }
                    px = px + 1;
                }
                py = py + 1;
            }
            cx = cx + self.glyphW * scale;
            i = i + 1UL;
        }
    }
    return cx - x;
}

int GuiFont::text_width(const &char text, int scale) const {
    unsigned long len = 0UL;
    int glyphW;
    unsafe {
        while (text[len] != '\0') { len = len + 1UL; }
        glyphW = self.glyphW;
    }
    return (int)len * glyphW * scale;
}

int GuiFont::text_height(int scale) const {
    int glyphH;
    unsafe { glyphH = self.glyphH; }
    return glyphH * scale;
}

} // namespace std
