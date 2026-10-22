// SafeC Standard Library — GUI: UTF-8 helpers (see gui_utf8.h).
#pragma once
#include <std/gui/gui_utf8.h>

namespace std {

// ── UTF-8 ─────────────────────────────────────────────────────────────────

unsigned int gui_utf8_next(const &char s, unsigned long len, &unsigned long i) {
    unsafe {
        unsigned long p = *i;
        if (p >= len) { return 0U; }
        unsigned int c0 = (unsigned int)(unsigned char)s[p];
        if (c0 < 0x80U) { *i = p + 1UL; return c0; }
        int need = 0;
        unsigned int cp = 0U;
        if ((c0 & 0xE0U) == 0xC0U) { need = 1; cp = c0 & 0x1FU; }
        else if ((c0 & 0xF0U) == 0xE0U) { need = 2; cp = c0 & 0x0FU; }
        else if ((c0 & 0xF8U) == 0xF0U) { need = 3; cp = c0 & 0x07U; }
        else { *i = p + 1UL; return 0xFFFDU; }
        if (p + (unsigned long)need >= len) { *i = p + 1UL; return 0xFFFDU; }
        int k = 1;
        while (k <= need) {
            unsigned int c = (unsigned int)(unsigned char)s[p + (unsigned long)k];
            if ((c & 0xC0U) != 0x80U) { *i = p + 1UL; return 0xFFFDU; }
            cp = (cp << 6) | (c & 0x3FU);
            k = k + 1;
        }
        *i = p + 1UL + (unsigned long)need;
        if ((need == 1 && cp < 0x80U) || (need == 2 && cp < 0x800U) || (need == 3 && cp < 0x10000U) ||
            cp > 0x10FFFFU || (cp >= 0xD800U && cp <= 0xDFFFU)) {
            return 0xFFFDU;
        }
        return cp;
    }
}

int gui_utf8_encode(unsigned int cp, &char out) {
    unsafe {
        if (cp < 0x80U) { out[0] = (char)cp; return 1; }
        if (cp < 0x800U) {
            out[0] = (char)(0xC0U | (cp >> 6));
            out[1] = (char)(0x80U | (cp & 0x3FU));
            return 2;
        }
        if (cp < 0x10000U) {
            out[0] = (char)(0xE0U | (cp >> 12));
            out[1] = (char)(0x80U | ((cp >> 6) & 0x3FU));
            out[2] = (char)(0x80U | (cp & 0x3FU));
            return 3;
        }
        if (cp > 0x10FFFFU) { cp = 0xFFFDU; return gui_utf8_encode(cp, out); }
        out[0] = (char)(0xF0U | (cp >> 18));
        out[1] = (char)(0x80U | ((cp >> 12) & 0x3FU));
        out[2] = (char)(0x80U | ((cp >> 6) & 0x3FU));
        out[3] = (char)(0x80U | (cp & 0x3FU));
        return 4;
    }
}

unsigned long gui_utf8_prev(const &char s, unsigned long i) {
    if (i == 0UL) { return 0UL; }
    unsafe {
        unsigned long p = i - 1UL;
        int steps = 0;
        while (p > 0UL && steps < 3 && (((unsigned int)(unsigned char)s[p]) & 0xC0U) == 0x80U) {
            p = p - 1UL;
            steps = steps + 1;
        }
        return p;
    }
}

} // namespace std
