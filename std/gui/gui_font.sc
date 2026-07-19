// SafeC Standard Library — GUI: loadable/serializable bitmap fonts (see
// gui_font.h for the on-disk "SCXF" format).
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_font.h>
#include <std/mem.sc>
#include <std/io_file.sc>

namespace std {

static int __gui_font_bytes_per_glyph(int glyphW, int glyphH) {
    int bits = glyphW * glyphH;
    return (bits + 7) / 8;
}

struct GuiFont gui_font_new(int glyphW, int glyphH, int first, int count) {
    struct GuiFont f;
    f.glyphW = glyphW;
    f.glyphH = glyphH;
    f.first = first;
    f.count = count;
    unsigned long total = (unsigned long)(__gui_font_bytes_per_glyph(glyphW, glyphH) * count);
    unsafe {
        f.bitmap = (&heap unsigned char)alloc(total);
        memset((void*)f.bitmap, 0, total);
    }
    return f;
}

int gui_font_get_bit(const struct GuiFont* font, int cp, int px, int py) {
    int idx = 0; int glyphW = 0; int glyphH = 0; int count = 0;
    unsafe {
        idx = cp - font->first;
        glyphW = font->glyphW;
        glyphH = font->glyphH;
        count = font->count;
    }
    if (idx < 0 || idx >= count) { return 0; }
    if (px < 0 || py < 0 || px >= glyphW || py >= glyphH) { return 0; }
    int bytesPerGlyph = __gui_font_bytes_per_glyph(glyphW, glyphH);
    int bitIndex = py * glyphW + px;
    unsigned long byteOff = (unsigned long)(idx * bytesPerGlyph) + (unsigned long)(bitIndex / 8);
    int bitInByte = 7 - (bitIndex % 8);
    unsigned char byteVal;
    unsafe { byteVal = font->bitmap[byteOff]; }
    return (int)((byteVal >> bitInByte) & 1U);
}

void gui_font_set_bit(struct GuiFont* font, int cp, int px, int py, int value) {
    int idx = 0; int glyphW = 0; int glyphH = 0; int count = 0;
    unsafe {
        idx = cp - font->first;
        glyphW = font->glyphW;
        glyphH = font->glyphH;
        count = font->count;
    }
    if (idx < 0 || idx >= count) { return; }
    if (px < 0 || py < 0 || px >= glyphW || py >= glyphH) { return; }
    int bytesPerGlyph = __gui_font_bytes_per_glyph(glyphW, glyphH);
    int bitIndex = py * glyphW + px;
    unsigned long byteOff = (unsigned long)(idx * bytesPerGlyph) + (unsigned long)(bitIndex / 8);
    int bitInByte = 7 - (bitIndex % 8);
    unsafe {
        unsigned char cur = font->bitmap[byteOff];
        if (value) {
            font->bitmap[byteOff] = cur | (unsigned char)(1U << bitInByte);
        } else {
            font->bitmap[byteOff] = cur & (unsigned char)(~(1U << bitInByte));
        }
    }
}

// The built-in 8x8 font's raw table (gFont8x8) is private to gui_draw.sc;
// re-derive an equivalent GuiFont here by calling gui_draw.h's public
// gui_draw_text() (declared there, already in scope via the #include
// above) for each glyph instead of reaching into its internals.
struct GuiFont gui_font_default() {
    struct GuiFont f = gui_font_new(8, 8, 32, 95);
    int cp = 32;
    while (cp <= 126) {
        // Render this one glyph at scale 1 into a throwaway 8x8 window
        // buffer, then copy its pixels into the GuiFont — avoids needing
        // access to gui_draw.sc's private table while still producing a
        // real, correct copy of the exact same built-in glyphs.
        struct GuiWindow tmp;
        tmp.platform = (void*)0;
        tmp.width = 8;
        tmp.height = 8;
        tmp.shouldClose = 0;
        unsafe {
            tmp.pixels = (&heap unsigned char)alloc(8UL * 8UL * 4UL);
            memset((void*)tmp.pixels, 0, 8UL * 8UL * 4UL);
        }
        char one[2];
        unsafe { one[0] = (char)cp; one[1] = '\0'; }
        struct GuiColor white = gui_rgba(255U, 255U, 255U, 255U);
        unsafe { gui_draw_text(&tmp, 0, 0, (const char*)one, white, 1); }

        int py = 0;
        while (py < 8) {
            int px = 0;
            while (px < 8) {
                unsigned char a;
                unsafe {
                    unsigned long off = ((unsigned long)py * 8UL + (unsigned long)px) * 4UL;
                    a = tmp.pixels[off + 3UL];
                }
                gui_font_set_bit(&f, cp, px, py, (a > 0U) ? 1 : 0);
                px = px + 1;
            }
            py = py + 1;
        }
        unsafe { dealloc((void*)tmp.pixels); }
        cp = cp + 1;
    }
    return f;
}

void gui_font_free(struct GuiFont* font) {
    unsafe {
        if ((void*)font->bitmap != (void*)0) { dealloc((void*)font->bitmap); }
        font->bitmap = (&heap unsigned char)0;
        font->count = 0;
    }
}

int gui_font_save(const struct GuiFont* font, const char* path) {
    void* f;
    unsafe { f = file_open(path, "wb"); }
    if (f == (void*)0) { return 0; }

    unsigned char header[16];
    unsafe {
        header[0] = (unsigned char)'S'; header[1] = (unsigned char)'C';
        header[2] = (unsigned char)'X'; header[3] = (unsigned char)'F';
        header[4] = 1U;
        header[5] = (unsigned char)font->glyphW;
        header[6] = (unsigned char)font->glyphH;
        header[7] = 0U;
        unsigned int first = (unsigned int)font->first;
        header[8]  = (unsigned char)(first & 0xFFU);
        header[9]  = (unsigned char)((first >> 8) & 0xFFU);
        header[10] = (unsigned char)((first >> 16) & 0xFFU);
        header[11] = (unsigned char)((first >> 24) & 0xFFU);
        unsigned int count = (unsigned int)font->count;
        header[12] = (unsigned char)(count & 0xFFU);
        header[13] = (unsigned char)((count >> 8) & 0xFFU);
        header[14] = (unsigned char)((count >> 16) & 0xFFU);
        header[15] = (unsigned char)((count >> 24) & 0xFFU);
    }
    unsigned long wrote;
    unsafe { wrote = file_write(f, (const void*)header, 16UL); }
    if (wrote != 16UL) { unsafe { file_close(f); } return 0; }

    int glyphW; int glyphH; int count;
    unsafe { glyphW = font->glyphW; glyphH = font->glyphH; count = font->count; }
    unsigned long total = (unsigned long)(__gui_font_bytes_per_glyph(glyphW, glyphH) * count);
    unsigned long bodyWrote;
    unsafe { bodyWrote = file_write(f, (const void*)font->bitmap, total); }
    unsafe { file_close(f); }
    return (bodyWrote == total) ? 1 : 0;
}

int gui_font_load(struct GuiFont* outFont, const char* path) {
    void* f;
    unsafe { f = file_open(path, "rb"); }
    if (f == (void*)0) { return 0; }

    unsigned char header[16];
    unsigned long got;
    unsafe { got = file_read(f, (void*)header, 16UL); }
    if (got != 16UL) { unsafe { file_close(f); } return 0; }

    int ok = 1;
    unsafe {
        if (header[0] != (unsigned char)'S' || header[1] != (unsigned char)'C' ||
            header[2] != (unsigned char)'X' || header[3] != (unsigned char)'F' ||
            header[4] != 1U) {
            ok = 0;
        }
    }
    if (!ok) { unsafe { file_close(f); } return 0; }

    int glyphW; int glyphH; unsigned int first; unsigned int count;
    unsafe {
        glyphW = (int)header[5];
        glyphH = (int)header[6];
        first = (unsigned int)header[8]  | ((unsigned int)header[9] << 8)
              | ((unsigned int)header[10] << 16) | ((unsigned int)header[11] << 24);
        count = (unsigned int)header[12] | ((unsigned int)header[13] << 8)
              | ((unsigned int)header[14] << 16) | ((unsigned int)header[15] << 24);
    }

    struct GuiFont loaded = gui_font_new(glyphW, glyphH, (int)first, (int)count);
    unsigned long total = (unsigned long)(__gui_font_bytes_per_glyph(glyphW, glyphH) * (int)count);
    unsigned long bodyGot;
    unsafe { bodyGot = file_read(f, (void*)loaded.bitmap, total); }
    unsafe { file_close(f); }
    if (bodyGot != total) {
        gui_font_free(&loaded);
        return 0;
    }
    unsafe { *outFont = loaded; }
    return 1;
}

int gui_draw_text_ex(struct GuiWindow* win, int x, int y, const char* text,
                      struct GuiColor color, int scale, const struct GuiFont* font) {
    int cx = x;
    unsigned long i = 0UL;
    unsafe {
        while (text[i] != '\0') {
            int cp = (int)(unsigned char)text[i];
            int py = 0;
            while (py < font->glyphH) {
                int px = 0;
                while (px < font->glyphW) {
                    if (gui_font_get_bit(font, cp, px, py)) {
                        gui_fill_rect(win, cx + px * scale, y + py * scale, scale, scale, color);
                    }
                    px = px + 1;
                }
                py = py + 1;
            }
            cx = cx + font->glyphW * scale;
            i = i + 1UL;
        }
    }
    return cx - x;
}

int gui_text_width_ex(const char* text, int scale, const struct GuiFont* font) {
    unsigned long len = 0UL;
    int glyphW;
    unsafe {
        while (text[len] != '\0') { len = len + 1UL; }
        glyphW = font->glyphW;
    }
    return (int)len * glyphW * scale;
}

int gui_text_height_ex(int scale, const struct GuiFont* font) {
    int glyphH;
    unsafe { glyphH = font->glyphH; }
    return glyphH * scale;
}

} // namespace std
