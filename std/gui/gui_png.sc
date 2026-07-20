// SafeC Standard Library — GUI: PNG image loading implementation (see
// gui_png.h). A real DEFLATE (RFC 1951) inflate + PNG (RFC 2083) chunk
// parser, not a stub — see gui_png.h's header comment for exact scope
// (8-bit depth, non-interlaced, grayscale/RGB/RGBA only).
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_png.h>
#include <std/mem.sc>
#include <std/collections/string.h>
#include <std/collections/string.sc>

namespace std {

// ── bit-level DEFLATE reader ─────────────────────────────────────────────────

struct InflateState {
    const unsigned char* data;
    unsigned long size;
    unsigned long bytePos;
    unsigned int bitBuf;
    int bitCount;
    struct String out; // growable decompressed byte buffer
    int error;
};

static unsigned int __inflate_bit(struct InflateState* s) {
    unsafe {
        if (s->bitCount == 0) {
            unsigned char b = 0U;
            if (s->bytePos < s->size) {
                b = s->data[s->bytePos];
                s->bytePos = s->bytePos + 1UL;
            }
            s->bitBuf = (unsigned int)b;
            s->bitCount = 8;
        }
        unsigned int bit = s->bitBuf & 1U;
        s->bitBuf = s->bitBuf >> 1;
        s->bitCount = s->bitCount - 1;
        return bit;
    }
}

// LSB-first multi-bit integer (used for everything except Huffman codes
// themselves — see RFC 1951 3.1.1).
static unsigned int __inflate_bits(struct InflateState* s, int n) {
    unsigned int v = 0U;
    int i = 0;
    while (i < n) {
        v = v | (__inflate_bit(s) << i);
        i = i + 1;
    }
    return v;
}

static void __inflate_align(struct InflateState* s) {
    unsafe { s->bitBuf = 0U; s->bitCount = 0; }
}

// ── canonical Huffman ────────────────────────────────────────────────────────

#define GUI_PNG_MAX_SYMBOLS 320

struct HuffTree {
    int lengths[GUI_PNG_MAX_SYMBOLS];
    int codes[GUI_PNG_MAX_SYMBOLS];
    int numSymbols;
};

static void __build_huffman(struct HuffTree* t, const int* lengths, int numSymbols) {
  unsafe {
    t->numSymbols = numSymbols;
    int i = 0;
    while (i < numSymbols) { t->lengths[i] = lengths[i]; i = i + 1; }

    int blCount[16];
    int j = 0;
    while (j < 16) { blCount[j] = 0; j = j + 1; }
    i = 0;
    while (i < numSymbols) {
        if (lengths[i] > 0) { blCount[lengths[i]] = blCount[lengths[i]] + 1; }
        i = i + 1;
    }
    int nextCode[16];
    int code = 0;
    j = 1;
    while (j <= 15) {
        code = (code + blCount[j - 1]) << 1;
        nextCode[j] = code;
        j = j + 1;
    }
    i = 0;
    while (i < numSymbols) {
        int len = lengths[i];
        if (len > 0) {
            t->codes[i] = nextCode[len];
            nextCode[len] = nextCode[len] + 1;
        } else {
            t->codes[i] = -1;
        }
        i = i + 1;
    }
  }
}

static int __huffman_decode(struct InflateState* s, struct HuffTree* t) {
    int code = 0;
    int len = 0;
    while (len < 16) {
        code = (code << 1) | (int)__inflate_bit(s);
        len = len + 1;
        int i = 0;
        int numSymbols; unsafe { numSymbols = t->numSymbols; }
        while (i < numSymbols) {
            int tlen; int tcode;
            unsafe { tlen = t->lengths[i]; tcode = t->codes[i]; }
            if (tlen == len && tcode == code) { return i; }
            i = i + 1;
        }
    }
    unsafe { s->error = 1; }
    return -1;
}

// RFC 1951 3.2.5 length/distance extra-bits tables.
static const int gLenBase[29]  = { 3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258 };
static const int gLenExtra[29] = { 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0 };
static const int gDistBase[30]  = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577 };
static const int gDistExtra[30] = { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };
static const int gClOrder[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };

static void __inflate_block(struct InflateState* s, struct HuffTree* litTree, struct HuffTree* distTree) {
    while (1) {
        int sym = __huffman_decode(s, litTree);
        int hadError; unsafe { hadError = s->error; }
        if (hadError) { return; }
        if (sym < 256) {
            unsafe { s->out.push_char((char)(unsigned char)sym); }
        } else if (sym == 256) {
            return; // end of block
        } else {
            int lenIdx = sym - 257;
            if (lenIdx < 0 || lenIdx >= 29) { unsafe { s->error = 1; } return; }
            unsigned int length = (unsigned int)gLenBase[lenIdx] + __inflate_bits(s, gLenExtra[lenIdx]);
            int distSym = __huffman_decode(s, distTree);
            unsafe { hadError = s->error; }
            if (hadError || distSym < 0 || distSym >= 30) { unsafe { s->error = 1; } return; }
            unsigned int distance = (unsigned int)gDistBase[distSym] + __inflate_bits(s, gDistExtra[distSym]);
            unsigned long outLen; unsafe { outLen = s->out.len; }
            if ((unsigned long)distance > outLen) { unsafe { s->error = 1; } return; }
            unsigned long srcStart = outLen - (unsigned long)distance;
            unsigned int i = 0U;
            while (i < length) {
                char c;
                unsafe { c = s->out.data[srcStart + (unsigned long)i]; }
                unsafe { s->out.push_char(c); }
                i = i + 1U;
            }
        }
    }
}

static void __inflate_fixed_trees(struct HuffTree* litTree, struct HuffTree* distTree) {
    int litLens[288];
    int i = 0;
    while (i < 288) {
        if (i < 144) { litLens[i] = 8; }
        else if (i < 256) { litLens[i] = 9; }
        else if (i < 280) { litLens[i] = 7; }
        else { litLens[i] = 8; }
        i = i + 1;
    }
    __build_huffman(litTree, litLens, 288);

    int distLens[30];
    i = 0;
    while (i < 30) { distLens[i] = 5; i = i + 1; }
    __build_huffman(distTree, distLens, 30);
}

static void __inflate_dynamic_trees(struct InflateState* s, struct HuffTree* litTree, struct HuffTree* distTree) {
    int hlit = (int)__inflate_bits(s, 5) + 257;
    int hdist = (int)__inflate_bits(s, 5) + 1;
    int hclen = (int)__inflate_bits(s, 4) + 4;

    int clLens[19];
    int i = 0;
    while (i < 19) { clLens[i] = 0; i = i + 1; }
    i = 0;
    while (i < hclen) {
        clLens[gClOrder[i]] = (int)__inflate_bits(s, 3);
        i = i + 1;
    }
    struct HuffTree clTree;
    __build_huffman(&clTree, clLens, 19);

    int total = hlit + hdist;
    int allLens[320];
    i = 0;
    while (i < total) {
        int sym = __huffman_decode(s, &clTree);
        int hadError; unsafe { hadError = s->error; }
        if (hadError) { return; }
        if (sym < 16) {
            allLens[i] = sym;
            i = i + 1;
        } else if (sym == 16) {
            int prev = (i > 0) ? allLens[i - 1] : 0;
            int rep = (int)__inflate_bits(s, 2) + 3;
            int r = 0;
            while (r < rep && i < total) { allLens[i] = prev; i = i + 1; r = r + 1; }
        } else if (sym == 17) {
            int rep = (int)__inflate_bits(s, 3) + 3;
            int r = 0;
            while (r < rep && i < total) { allLens[i] = 0; i = i + 1; r = r + 1; }
        } else if (sym == 18) {
            int rep = (int)__inflate_bits(s, 7) + 11;
            int r = 0;
            while (r < rep && i < total) { allLens[i] = 0; i = i + 1; r = r + 1; }
        } else {
            unsafe { s->error = 1; }
            return;
        }
    }

    int litLens[288];
    i = 0;
    while (i < hlit) { litLens[i] = allLens[i]; i = i + 1; }
    __build_huffman(litTree, litLens, hlit);

    int distLens[32];
    i = 0;
    while (i < hdist) { distLens[i] = allLens[hlit + i]; i = i + 1; }
    __build_huffman(distTree, distLens, hdist);
}

// Inflates a raw DEFLATE stream (the payload after zlib's 2-byte header,
// before its 4-byte Adler32 trailer — see gui_inflate_zlib()). Returns 1
// on success.
static int __inflate_raw(const unsigned char* data, unsigned long size, struct String* outStr) {
    struct InflateState s;
    s.data = data;
    s.size = size;
    s.bytePos = 0UL;
    s.bitBuf = 0U;
    s.bitCount = 0;
    unsafe { s.out = string_new(); }
    s.error = 0;

    int bfinal = 0;
    while (!bfinal) {
        bfinal = (int)__inflate_bit(&s);
        int btype = (int)__inflate_bits(&s, 2);
        if (btype == 0) {
            __inflate_align(&s);
            unsigned char lenLo = 0U; unsigned char lenHi = 0U;
            if (s.bytePos + 1UL < s.size) {
                unsafe { lenLo = s.data[s.bytePos]; lenHi = s.data[s.bytePos + 1UL]; }
            }
            s.bytePos = s.bytePos + 4UL; // skip LEN + NLEN (NLEN unvalidated)
            unsigned int len = (unsigned int)lenLo | ((unsigned int)lenHi << 8);
            unsigned int k = 0U;
            while (k < len) {
                char c = '\0';
                if (s.bytePos < s.size) {
                    unsafe { c = (char)s.data[s.bytePos]; }
                    s.bytePos = s.bytePos + 1UL;
                }
                unsafe { s.out.push_char(c); }
                k = k + 1U;
            }
        } else if (btype == 1) {
            struct HuffTree litTree; struct HuffTree distTree;
            __inflate_fixed_trees(&litTree, &distTree);
            __inflate_block(&s, &litTree, &distTree);
        } else if (btype == 2) {
            struct HuffTree litTree; struct HuffTree distTree;
            __inflate_dynamic_trees(&s, &litTree, &distTree);
            if (s.error) { unsafe { s.out.free(); } return 0; }
            __inflate_block(&s, &litTree, &distTree);
        } else {
            unsafe { s.out.free(); }
            return 0;
        }
        if (s.error) { unsafe { s.out.free(); } return 0; }
    }

    unsafe { *outStr = s.out; }
    return 1;
}

// Inflates a full zlib stream (2-byte header + DEFLATE payload + 4-byte
// Adler32 trailer, RFC 1950) — the format PNG's concatenated IDAT data is.
static int __inflate_zlib(const unsigned char* data, unsigned long size, struct String* outStr) {
    if (size < 6UL) { return 0; }
    unsafe { return __inflate_raw(data + 2UL, size - 6UL, outStr); }
}

// ── PNG filtering ────────────────────────────────────────────────────────────

static unsigned char __paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = p - a; if (pa < 0) { pa = -pa; }
    int pb = p - b; if (pb < 0) { pb = -pb; }
    int pc = p - c; if (pc < 0) { pc = -pc; }
    if (pa <= pb && pa <= pc) { return (unsigned char)a; }
    if (pb <= pc) { return (unsigned char)b; }
    return (unsigned char)c;
}

// ── PNG chunk parsing ────────────────────────────────────────────────────────

static unsigned int __be32(const unsigned char* p) {
    unsigned int r;
    unsafe {
        r = ((unsigned int)p[0] << 24) | ((unsigned int)p[1] << 16) |
            ((unsigned int)p[2] << 8) | (unsigned int)p[3];
    }
    return r;
}

int gui_load_png(const unsigned char* data, unsigned long size, &GuiImage out) {
    unsafe {
        out.pixels = (&heap unsigned char)0;
        out.width = 0;
        out.height = 0;
    }

    if (size < 8UL) { return 0; }
    unsafe {
        if (data[0] != 0x89U || data[1] != (unsigned char)'P' || data[2] != (unsigned char)'N' ||
            data[3] != (unsigned char)'G') {
            return 0;
        }
    }

    int width = 0; int height = 0; int bitDepth = 0; int colorType = -1; int interlace = 0;
    struct String idat = string_new();
    int haveIhdr = 0;

    unsigned long pos = 8UL;
    while (pos + 8UL <= size) {
        unsigned int chunkLen;
        unsigned char type0; unsigned char type1; unsigned char type2; unsigned char type3;
        unsafe {
            chunkLen = __be32(data + pos);
            type0 = data[pos + 4UL]; type1 = data[pos + 5UL];
            type2 = data[pos + 6UL]; type3 = data[pos + 7UL];
        }
        unsigned long dataStart = pos + 8UL;
        if (dataStart + (unsigned long)chunkLen > size) { break; }

        int isIHDR = (type0 == (unsigned char)'I' && type1 == (unsigned char)'H' &&
                      type2 == (unsigned char)'D' && type3 == (unsigned char)'R') ? 1 : 0;
        int isIDAT = (type0 == (unsigned char)'I' && type1 == (unsigned char)'D' &&
                      type2 == (unsigned char)'A' && type3 == (unsigned char)'T') ? 1 : 0;
        int isIEND = (type0 == (unsigned char)'I' && type1 == (unsigned char)'E' &&
                      type2 == (unsigned char)'N' && type3 == (unsigned char)'D') ? 1 : 0;

        if (isIHDR && chunkLen >= 13U) {
            unsafe {
                width = (int)__be32(data + dataStart);
                height = (int)__be32(data + dataStart + 4UL);
                bitDepth = (int)data[dataStart + 8UL];
                colorType = (int)data[dataStart + 9UL];
                interlace = (int)data[dataStart + 12UL];
            }
            haveIhdr = 1;
        } else if (isIDAT) {
            unsigned long k = 0UL;
            while (k < (unsigned long)chunkLen) {
                char c;
                unsafe { c = (char)data[dataStart + k]; }
                unsafe { idat.push_char(c); }
                k = k + 1UL;
            }
        } else if (isIEND) {
            break;
        }

        pos = dataStart + (unsigned long)chunkLen + 4UL; // +4 skips the CRC (unverified)
    }

    if (!haveIhdr || width <= 0 || height <= 0 || bitDepth != 8 || interlace != 0) {
        unsafe { idat.free(); }
        return 0;
    }
    int channels = 0;
    if (colorType == 0) { channels = 1; }
    else if (colorType == 2) { channels = 3; }
    else if (colorType == 6) { channels = 4; }
    else { unsafe { idat.free(); } return 0; } // palette/gray+alpha not supported

    struct String raw;
    int ok;
    unsafe { ok = __inflate_zlib((const unsigned char*)idat.data, idat.len, &raw); }
    unsafe { idat.free(); }
    if (!ok) { return 0; }

    unsigned long expected = (unsigned long)height * (1UL + (unsigned long)width * (unsigned long)channels);
    unsafe {
        if (raw.len < expected) { raw.free(); return 0; }
    }

    unsigned long npix = (unsigned long)width * (unsigned long)height * 4UL;
    unsafe {
        out.pixels = (&heap unsigned char)alloc(npix);
    }
    unsafe { out.width = width; out.height = height; }

    unsigned long rowBytes = (unsigned long)width * (unsigned long)channels;
    // Previous-scanline reconstructed bytes (for Up/Average/Paeth), zeroed
    // for the first row per the PNG spec.
    unsigned char* prevRow;
    unsafe { prevRow = (unsigned char*)alloc(rowBytes); memset((void*)prevRow, 0, rowBytes); }
    unsigned char* curRow;
    unsafe { curRow = (unsigned char*)alloc(rowBytes); }

    unsigned long srcPos = 0UL;
    int y = 0;
    while (y < height) {
        unsigned char filterType;
        unsafe { filterType = raw.data[srcPos]; }
        srcPos = srcPos + 1UL;

        unsigned long x = 0UL;
        while (x < rowBytes) {
            unsigned char filt; int a; int b; int c;
            unsafe {
                filt = raw.data[srcPos + x];
                a = (x >= (unsigned long)channels) ? (int)(unsigned char)curRow[x - (unsigned long)channels] : 0;
                b = (int)(unsigned char)prevRow[x];
                c = (x >= (unsigned long)channels) ? (int)(unsigned char)prevRow[x - (unsigned long)channels] : 0;
            }
            unsigned char recon;
            if (filterType == 0U) { recon = filt; }
            else if (filterType == 1U) { recon = (unsigned char)((int)filt + a); }
            else if (filterType == 2U) { recon = (unsigned char)((int)filt + b); }
            else if (filterType == 3U) { recon = (unsigned char)((int)filt + (a + b) / 2); }
            else if (filterType == 4U) { recon = (unsigned char)((int)filt + (int)__paeth(a, b, c)); }
            else { recon = filt; }
            unsafe { curRow[x] = recon; }
            x = x + 1UL;
        }
        srcPos = srcPos + rowBytes;

        // Expand this reconstructed row into RGBA8888 output.
        unsigned long px = 0UL;
        while (px < (unsigned long)width) {
            unsigned long si = px * (unsigned long)channels;
            unsigned long di = ((unsigned long)y * (unsigned long)width + px) * 4UL;
            unsigned char r; unsigned char g; unsigned char b2; unsigned char a2;
            if (channels == 1) {
                unsafe { r = curRow[si]; }
                g = r; b2 = r; a2 = 255U;
            } else if (channels == 3) {
                unsafe { r = curRow[si]; g = curRow[si + 1UL]; b2 = curRow[si + 2UL]; }
                a2 = 255U;
            } else {
                unsafe { r = curRow[si]; g = curRow[si + 1UL]; b2 = curRow[si + 2UL]; a2 = curRow[si + 3UL]; }
            }
            unsafe {
                out.pixels[di + 0UL] = r;
                out.pixels[di + 1UL] = g;
                out.pixels[di + 2UL] = b2;
                out.pixels[di + 3UL] = a2;
            }
            px = px + 1UL;
        }

        unsigned char* tmp = prevRow;
        prevRow = curRow;
        curRow = tmp;
        y = y + 1;
    }

    unsafe {
        dealloc((void*)prevRow);
        dealloc((void*)curRow);
        raw.free();
    }
    return 1;
}

void gui_draw_image(&GuiWindow win, int x, int y, const &GuiImage img) {
    int iw; int ih;
    unsafe { iw = img.width; ih = img.height; }
    int yy = 0;
    while (yy < ih) {
        int xx = 0;
        while (xx < iw) {
            unsigned long si = ((unsigned long)yy * (unsigned long)iw + (unsigned long)xx) * 4UL;
            unsigned char r; unsigned char g; unsigned char b; unsigned char a;
            unsafe {
                r = img.pixels[si + 0UL]; g = img.pixels[si + 1UL];
                b = img.pixels[si + 2UL]; a = img.pixels[si + 3UL];
            }
            unsigned int rgba = ((unsigned int)r << 24) | ((unsigned int)g << 16) |
                                 ((unsigned int)b << 8) | (unsigned int)a;
            gui_set_pixel(win, x + xx, y + yy, rgba);
            xx = xx + 1;
        }
        yy = yy + 1;
    }
}

void gui_image_free(&GuiImage img) {
    unsafe {
        if ((void*)img.pixels != (void*)0) { dealloc((void*)img.pixels); }
        img.pixels = (&heap unsigned char)0;
        img.width = 0;
        img.height = 0;
    }
}

} // namespace std
