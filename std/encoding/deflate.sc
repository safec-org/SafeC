// SafeC Standard Library — DEFLATE / zlib / gzip implementation (see
// deflate.h). The decoder follows the canonical-Huffman scheme of zlib's
// reference "puff" decoder; the encoder is LZ77 with hash chains feeding a
// per-block choice of stored, fixed-Huffman or dynamic-Huffman coding.
#pragma once
#include <std/encoding/deflate.h>
#include <std/collections/vec.h>
#include <std/mem.h>

namespace std {

// ── Checksums ──────────────────────────────────────────────────────────────

static unsigned int gDeflateCrcTable[256];
static int gDeflateCrcReady = 0;

static void deflate_crc_init_() {
    unsigned int n = 0U;
    while (n < 256U) {
        unsigned int c = n;
        int k = 0;
        while (k < 8) {
            if ((c & 1U) != 0U) { c = 0xEDB88320U ^ (c >> 1); } else { c = c >> 1; }
            k = k + 1;
        }
        unsafe { gDeflateCrcTable[n] = c; }
        n = n + 1U;
    }
    gDeflateCrcReady = 1;
}

static unsigned int crc32_update_n_(unsigned int crc, const &unsigned char data, unsigned long len) {
    if (!gDeflateCrcReady) { deflate_crc_init_(); }
    unsigned int c = crc ^ 0xFFFFFFFFU;
    unsafe {
        unsigned long i = 0UL;
        while (i < len) {
            c = gDeflateCrcTable[(c ^ (unsigned int)data[i]) & 0xFFU] ^ (c >> 8);
            i = i + 1UL;
        }
    }
    return c ^ 0xFFFFFFFFU;
}

static unsigned int crc32_n_(const &unsigned char data, unsigned long len) {
    return crc32_update_n_(0U, data, len);
}

static unsigned int adler32_update_n_(unsigned int adler, const &unsigned char data, unsigned long len) {
    unsigned int a = adler & 0xFFFFU;
    unsigned int b = (adler >> 16) & 0xFFFFU;
    unsafe {
        unsigned long i = 0UL;
        while (i < len) {
            // 5552 is the largest run before b can overflow 32 bits.
            unsigned long end = i + 5552UL;
            if (end > len) { end = len; }
            while (i < end) {
                a = a + (unsigned int)data[i];
                b = b + a;
                i = i + 1UL;
            }
            a = a % 65521U;
            b = b % 65521U;
        }
    }
    return (b << 16) | a;
}

static unsigned int adler32_n_(const &unsigned char data, unsigned long len) {
    return adler32_update_n_(1U, data, len);
}

// ── Output buffer ──────────────────────────────────────────────────────────

struct DeflateBuf_ {
    unsigned char* data;
    unsigned long  len;
    unsigned long  cap;
    int            failed;
};

static void dbuf_init_(struct DeflateBuf_* b, unsigned long cap) {
    unsafe {
        if (cap < 64UL) { cap = 64UL; }
        b->data = (unsigned char*)alloc_raw_(cap);
        b->len = 0UL;
        b->cap = cap;
        b->failed = b->data == (unsigned char*)0;
        if (b->failed) { b->cap = 0UL; }
    }
}

static int dbuf_reserve_(struct DeflateBuf_* b, unsigned long extra) {
    unsafe {
        if (b->failed) { return 0; }
        if (b->len + extra <= b->cap) { return 1; }
        unsigned long nc = b->cap * 2UL;
        while (nc < b->len + extra) { nc = nc * 2UL; }
        unsigned char* nd = (unsigned char*)realloc_buf_raw_((void*)b->data, nc);
        if (nd == (unsigned char*)0) { b->failed = 1; return 0; }
        b->data = nd;
        b->cap = nc;
        return 1;
    }
}

static void dbuf_push_(struct DeflateBuf_* b, unsigned char c) {
    unsafe {
        if (b->len == b->cap && !dbuf_reserve_(b, 1UL)) { return; }
        b->data[b->len] = c;
        b->len = b->len + 1UL;
    }
}

static void dbuf_push_n_(struct DeflateBuf_* b, const unsigned char* src, unsigned long n) {
    unsafe {
        if (!dbuf_reserve_(b, n)) { return; }
        unsigned long i = 0UL;
        while (i < n) { b->data[b->len + i] = src[i]; i = i + 1UL; }
        b->len = b->len + n;
    }
}

// Hands the buffer over as a Vec (or an empty Vec after a failure).
static struct Vec<unsigned char> dbuf_to_vec_(struct DeflateBuf_* b) {
    struct Vec<unsigned char> v = Vec::new();
    unsafe {
        if (b->failed || b->data == (unsigned char*)0) {
            if (b->data != (unsigned char*)0) { dealloc(b->data); }
            return v;
        }
        v.data = (?&heap unsigned char)b->data;
        v.len = b->len;
        v.cap = b->cap;
    }
    return v;
}

static void dbuf_free_(struct DeflateBuf_* b) {
    unsafe {
        if (b->data != (unsigned char*)0) { dealloc(b->data); }
        b->data = (unsigned char*)0;
        b->len = 0UL;
        b->cap = 0UL;
    }
}

// ── Shared tables (RFC 1951 §3.2.5) ────────────────────────────────────────

static const unsigned short kLenBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
static const unsigned short kLenExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
static const unsigned short kDistBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577 };
static const unsigned short kDistExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
// Order of the code-length code lengths in a dynamic block header.
static const unsigned char kClOrder[19] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

// ── Inflate ────────────────────────────────────────────────────────────────

#define INFL_MAXBITS 15

struct InflHuff_ {
    unsigned short count[INFL_MAXBITS + 1];
    unsigned short symbol[288];
};

struct InflState_ {
    const unsigned char* in;
    unsigned long inlen;
    unsigned long incnt;
    unsigned int  bitbuf;
    int           bitcnt;
    int           err;
    unsigned long max_out;
    struct DeflateBuf_ out;
};

static int infl_bits_(struct InflState_* s, int need) {
    unsafe {
        unsigned int val = s->bitbuf;
        while (s->bitcnt < need) {
            if (s->incnt >= s->inlen) { s->err = 1; return 0; }
            val = val | ((unsigned int)s->in[s->incnt] << s->bitcnt);
            s->incnt = s->incnt + 1UL;
            s->bitcnt = s->bitcnt + 8;
        }
        s->bitbuf = need >= 32 ? 0U : (val >> need);
        s->bitcnt = s->bitcnt - need;
        if (need == 0) { return 0; }
        return (int)(val & ((1U << need) - 1U));
    }
}

static int infl_stored_(struct InflState_* s) {
    unsafe {
        s->bitbuf = 0U;
        s->bitcnt = 0;
        if (s->incnt + 4UL > s->inlen) { s->err = 1; return 0; }
        unsigned int len = (unsigned int)s->in[s->incnt] | ((unsigned int)s->in[s->incnt + 1UL] << 8);
        unsigned int nlen = (unsigned int)s->in[s->incnt + 2UL] | ((unsigned int)s->in[s->incnt + 3UL] << 8);
        s->incnt = s->incnt + 4UL;
        if (len != (~nlen & 0xFFFFU)) { s->err = 1; return 0; }
        if (s->incnt + (unsigned long)len > s->inlen) { s->err = 1; return 0; }
        if (s->max_out != 0UL && s->out.len + (unsigned long)len > s->max_out) { s->err = 1; return 0; }
        dbuf_push_n_(&s->out, &s->in[s->incnt], (unsigned long)len);
        s->incnt = s->incnt + (unsigned long)len;
        return 1;
    }
}

static int infl_decode_(struct InflState_* s, struct InflHuff_* h) {
    unsafe {
        int code = 0;
        int first = 0;
        int index = 0;
        int len = 1;
        while (len <= INFL_MAXBITS) {
            code = code | infl_bits_(s, 1);
            if (s->err) { return -1; }
            int count = (int)h->count[len];
            if (code - count < first) { return (int)h->symbol[index + (code - first)]; }
            index = index + count;
            first = first + count;
            first = first << 1;
            code = code << 1;
            len = len + 1;
        }
        s->err = 1;
        return -1;
    }
}

// Builds the decoding tables; returns 0 for a complete code, > 0 for an
// incomplete one, < 0 for an over-subscribed one.
static int infl_construct_(struct InflHuff_* h, const unsigned short* length, int n) {
    unsafe {
        int len = 0;
        while (len <= INFL_MAXBITS) { h->count[len] = (unsigned short)0; len = len + 1; }
        int sym = 0;
        while (sym < n) { h->count[length[sym]] = (unsigned short)(h->count[length[sym]] + (unsigned short)1); sym = sym + 1; }
        if ((int)h->count[0] == n) { return 0; }
        int left = 1;
        len = 1;
        while (len <= INFL_MAXBITS) {
            left = left << 1;
            left = left - (int)h->count[len];
            if (left < 0) { return left; }
            len = len + 1;
        }
        unsigned short offs[INFL_MAXBITS + 1];
        offs[1] = (unsigned short)0;
        len = 1;
        while (len < INFL_MAXBITS) {
            offs[len + 1] = (unsigned short)(offs[len] + h->count[len]);
            len = len + 1;
        }
        sym = 0;
        while (sym < n) {
            if (length[sym] != (unsigned short)0) {
                h->symbol[offs[length[sym]]] = (unsigned short)sym;
                offs[length[sym]] = (unsigned short)(offs[length[sym]] + (unsigned short)1);
            }
            sym = sym + 1;
        }
        return left;
    }
}

static int infl_codes_(struct InflState_* s, struct InflHuff_* lencode, struct InflHuff_* distcode) {
    unsafe {
        while (1) {
            int symbol = infl_decode_(s, lencode);
            if (symbol < 0) { return 0; }
            if (symbol < 256) {
                if (s->max_out != 0UL && s->out.len + 1UL > s->max_out) { s->err = 1; return 0; }
                dbuf_push_(&s->out, (unsigned char)symbol);
                if (s->out.failed) { s->err = 1; return 0; }
                continue;
            }
            if (symbol == 256) { return 1; }
            symbol = symbol - 257;
            if (symbol >= 29) { s->err = 1; return 0; }
            int len = (int)kLenBase[symbol] + infl_bits_(s, (int)kLenExtra[symbol]);
            int dsym = infl_decode_(s, distcode);
            if (dsym < 0) { return 0; }
            if (dsym >= 30) { s->err = 1; return 0; }
            unsigned long dist = (unsigned long)kDistBase[dsym] + (unsigned long)infl_bits_(s, (int)kDistExtra[dsym]);
            if (s->err) { return 0; }
            if (dist > s->out.len) { s->err = 1; return 0; }
            if (s->max_out != 0UL && s->out.len + (unsigned long)len > s->max_out) { s->err = 1; return 0; }
            if (!dbuf_reserve_(&s->out, (unsigned long)len)) { s->err = 1; return 0; }
            unsigned long from = s->out.len - dist;
            int k = 0;
            while (k < len) {
                s->out.data[s->out.len] = s->out.data[from + (unsigned long)k];
                s->out.len = s->out.len + 1UL;
                k = k + 1;
            }
        }
    }
    return 0;
}

static int infl_fixed_(struct InflState_* s) {
    struct InflHuff_ lencode;
    struct InflHuff_ distcode;
    unsigned short lengths[320];
    unsafe {
        int sym = 0;
        while (sym < 144) { lengths[sym] = (unsigned short)8; sym = sym + 1; }
        while (sym < 256) { lengths[sym] = (unsigned short)9; sym = sym + 1; }
        while (sym < 280) { lengths[sym] = (unsigned short)7; sym = sym + 1; }
        while (sym < 288) { lengths[sym] = (unsigned short)8; sym = sym + 1; }
        infl_construct_(&lencode, &lengths[0], 288);
        sym = 0;
        while (sym < 30) { lengths[sym] = (unsigned short)5; sym = sym + 1; }
        infl_construct_(&distcode, &lengths[0], 30);
    }
    return infl_codes_(s, &lencode, &distcode);
}

static int infl_dynamic_(struct InflState_* s) {
    struct InflHuff_ lencode;
    struct InflHuff_ distcode;
    unsigned short lengths[320];
    unsafe {
        int nlen = infl_bits_(s, 5) + 257;
        int ndist = infl_bits_(s, 5) + 1;
        int ncode = infl_bits_(s, 4) + 4;
        if (s->err || nlen > 286 || ndist > 30) { s->err = 1; return 0; }
        int index = 0;
        while (index < ncode) {
            lengths[kClOrder[index]] = (unsigned short)infl_bits_(s, 3);
            index = index + 1;
        }
        while (index < 19) { lengths[kClOrder[index]] = (unsigned short)0; index = index + 1; }
        if (s->err) { return 0; }
        if (infl_construct_(&lencode, &lengths[0], 19) != 0) { s->err = 1; return 0; }
        index = 0;
        while (index < nlen + ndist) {
            int symbol = infl_decode_(s, &lencode);
            if (symbol < 0) { return 0; }
            if (symbol < 16) {
                lengths[index] = (unsigned short)symbol;
                index = index + 1;
            } else {
                int len = 0;
                int rep = 0;
                if (symbol == 16) {
                    if (index == 0) { s->err = 1; return 0; }
                    len = (int)lengths[index - 1];
                    rep = 3 + infl_bits_(s, 2);
                } else if (symbol == 17) {
                    rep = 3 + infl_bits_(s, 3);
                } else {
                    rep = 11 + infl_bits_(s, 7);
                }
                if (s->err || index + rep > nlen + ndist) { s->err = 1; return 0; }
                while (rep > 0) {
                    lengths[index] = (unsigned short)len;
                    index = index + 1;
                    rep = rep - 1;
                }
            }
        }
        if (lengths[256] == (unsigned short)0) { s->err = 1; return 0; }
        int err = infl_construct_(&lencode, &lengths[0], nlen);
        if (err < 0 || (err > 0 && nlen - (int)lencode.count[0] != 1)) { s->err = 1; return 0; }
        err = infl_construct_(&distcode, &lengths[nlen], ndist);
        if (err < 0 || (err > 0 && ndist - (int)distcode.count[0] != 1)) { s->err = 1; return 0; }
    }
    return infl_codes_(s, &lencode, &distcode);
}

static struct Vec<unsigned char> inflate_raw_n_(const &unsigned char data, unsigned long len,
                                      unsigned long max_out, ?&unsigned long consumed,
                                      ?&int ok) {
    struct InflState_ s;
    unsafe { s.in = (const unsigned char*)data; }
    s.inlen = len;
    s.incnt = 0UL;
    s.bitbuf = 0U;
    s.bitcnt = 0;
    s.err = 0;
    s.max_out = max_out;
    unsigned long guess = len * 4UL;
    if (max_out != 0UL && guess > max_out) { guess = max_out; }
    dbuf_init_(&s.out, guess);
    if (s.out.failed) { s.err = 1; }
    int last = 0;
    while (!s.err && !last) {
        last = infl_bits_(&s, 1);
        int type = infl_bits_(&s, 2);
        if (s.err) { break; }
        int good = 0;
        if (type == 0) { good = infl_stored_(&s); }
        else if (type == 1) { good = infl_fixed_(&s); }
        else if (type == 2) { good = infl_dynamic_(&s); }
        else { s.err = 1; }
        if (!good) { s.err = 1; }
        if (s.out.failed) { s.err = 1; }
    }
    if (consumed != (unsigned long*)0) { unsafe { *consumed = s.incnt; } }
    if (ok != (int*)0) { unsafe { *ok = !s.err; } }
    if (s.err) {
        dbuf_free_(&s.out);
        return Vec::new();
    }
    return dbuf_to_vec_(&s.out);
}

// ── Deflate ────────────────────────────────────────────────────────────────

#define DEFL_WSIZE      32768UL
#define DEFL_WMASK      32767UL
#define DEFL_HBITS      15
#define DEFL_HSIZE      32768UL
#define DEFL_MIN_MATCH  3UL
#define DEFL_MAX_MATCH  258UL
#define DEFL_BLOCK_SYMS 16384UL

struct DeflState_ {
    struct DeflateBuf_ out;
    unsigned long long bitbuf;
    int bitcnt;
    // Symbols of the current block: lit (0..255) or length (3..258, flagged
    // by a non-zero dist).
    unsigned short* litlen;
    unsigned short* dists;
    unsigned long   nsyms;
    unsigned long   blockStart;    // input offset of the block's first byte
    unsigned int    litFreq[286];
    unsigned int    distFreq[30];
};

static void defl_put_bits_(struct DeflState_* s, unsigned int value, int n) {
    unsafe {
        s->bitbuf = s->bitbuf | ((unsigned long long)value << s->bitcnt);
        s->bitcnt = s->bitcnt + n;
        while (s->bitcnt >= 8) {
            dbuf_push_(&s->out, (unsigned char)(s->bitbuf & 0xFFULL));
            s->bitbuf = s->bitbuf >> 8;
            s->bitcnt = s->bitcnt - 8;
        }
    }
}

static void defl_align_(struct DeflState_* s) {
    unsafe {
        if (s->bitcnt > 0) {
            dbuf_push_(&s->out, (unsigned char)(s->bitbuf & 0xFFULL));
        }
        s->bitbuf = 0ULL;
        s->bitcnt = 0;
    }
}

static int defl_len_code_(unsigned int len) {
    int c = 0;
    unsafe {
        while (c < 28 && (unsigned int)kLenBase[c + 1] <= len) { c = c + 1; }
    }
    return c;
}

static int defl_dist_code_(unsigned int dist) {
    int c = 0;
    unsafe {
        while (c < 29 && (unsigned int)kDistBase[c + 1] <= dist) { c = c + 1; }
    }
    return c;
}

// Huffman code lengths for freq[0..n), limited to maxBits: a plain
// two-smallest-merge build; when the tree is too deep, the frequencies are
// flattened (halved, keeping non-zero ones non-zero) and it is rebuilt.
static void defl_build_lengths_(const unsigned int* freqIn, int n, int maxBits, unsigned char* lens) {
    unsafe {
        unsigned int freq[300];
        int i = 0;
        while (i < n) { freq[i] = freqIn[i]; lens[i] = (unsigned char)0; i = i + 1; }
        while (1) {
            // Nodes: 0..n-1 leaves, n.. internal. parent[] links up.
            unsigned int w[600];
            int parent[600];
            int alive[600];
            int nodes = 0;
            int used = 0;
            i = 0;
            while (i < n) {
                w[i] = freq[i];
                parent[i] = -1;
                alive[i] = freq[i] > 0U;
                if (alive[i]) { used = used + 1; }
                i = i + 1;
            }
            nodes = n;
            if (used == 0) { return; }
            if (used == 1) {
                i = 0;
                while (i < n) { if (freq[i] > 0U) { lens[i] = (unsigned char)1; } i = i + 1; }
                return;
            }
            int remaining = used;
            while (remaining > 1) {
                int a = -1;
                int b = -1;
                int k = 0;
                while (k < nodes) {
                    if (alive[k]) {
                        if (a < 0 || w[k] < w[a]) { b = a; a = k; }
                        else if (b < 0 || w[k] < w[b]) { b = k; }
                    }
                    k = k + 1;
                }
                w[nodes] = w[a] + w[b];
                parent[nodes] = -1;
                alive[nodes] = 1;
                alive[a] = 0;
                alive[b] = 0;
                parent[a] = nodes;
                parent[b] = nodes;
                nodes = nodes + 1;
                remaining = remaining - 1;
            }
            int maxLen = 0;
            i = 0;
            while (i < n) {
                if (freq[i] > 0U) {
                    int d = 0;
                    int p = parent[i];
                    while (p >= 0) { d = d + 1; p = parent[p]; }
                    lens[i] = (unsigned char)d;
                    if (d > maxLen) { maxLen = d; }
                }
                i = i + 1;
            }
            if (maxLen <= maxBits) { return; }
            i = 0;
            while (i < n) {
                if (freq[i] > 0U) { freq[i] = (freq[i] + 1U) / 2U; }
                i = i + 1;
            }
        }
    }
}

// Canonical codes (bit-reversed, ready to send LSB-first) from lengths.
static void defl_build_codes_(const unsigned char* lens, int n, unsigned short* codes) {
    unsafe {
        unsigned short blCount[16];
        unsigned short next[16];
        int i = 0;
        while (i < 16) { blCount[i] = (unsigned short)0; i = i + 1; }
        i = 0;
        while (i < n) { blCount[lens[i]] = (unsigned short)(blCount[lens[i]] + (unsigned short)1); i = i + 1; }
        blCount[0] = (unsigned short)0;
        unsigned int code = 0U;
        int bits = 1;
        while (bits < 16) {
            code = (code + (unsigned int)blCount[bits - 1]) << 1;
            next[bits] = (unsigned short)code;
            bits = bits + 1;
        }
        i = 0;
        while (i < n) {
            int len = (int)lens[i];
            if (len != 0) {
                unsigned int c = (unsigned int)next[len];
                next[len] = (unsigned short)(next[len] + (unsigned short)1);
                unsigned int r = 0U;
                int k = 0;
                while (k < len) { r = (r << 1) | ((c >> k) & 1U); k = k + 1; }
                codes[i] = (unsigned short)r;
            } else {
                codes[i] = (unsigned short)0;
            }
            i = i + 1;
        }
    }
}

// The code-length sequence for a dynamic header, RLE'd with symbols
// 16/17/18; each entry is symbol | (extra << 8).
static int defl_rle_lengths_(const unsigned char* lens, int n, unsigned short* outSyms) {
    int count = 0;
    unsafe {
        int i = 0;
        while (i < n) {
            int l = (int)lens[i];
            int run = 1;
            while (i + run < n && (int)lens[i + run] == l) { run = run + 1; }
            if (l == 0) {
                int r = run;
                while (r >= 11) {
                    int take = r > 138 ? 138 : r;
                    outSyms[count] = (unsigned short)(18 | ((take - 11) << 8)); count = count + 1;
                    r = r - take;
                }
                if (r >= 3) {
                    outSyms[count] = (unsigned short)(17 | ((r - 3) << 8)); count = count + 1;
                    r = 0;
                }
                while (r > 0) { outSyms[count] = (unsigned short)0; count = count + 1; r = r - 1; }
            } else {
                outSyms[count] = (unsigned short)l; count = count + 1;
                int r = run - 1;
                while (r >= 3) {
                    int take = r > 6 ? 6 : r;
                    outSyms[count] = (unsigned short)(16 | ((take - 3) << 8)); count = count + 1;
                    r = r - take;
                }
                while (r > 0) { outSyms[count] = (unsigned short)l; count = count + 1; r = r - 1; }
            }
            i = i + run;
        }
    }
    return count;
}

static void defl_emit_symbols_(struct DeflState_* s, const unsigned short* litCodes, const unsigned char* litLens,
                               const unsigned short* distCodes, const unsigned char* distLens) {
    unsafe {
        unsigned long i = 0UL;
        while (i < s->nsyms) {
            unsigned int d = (unsigned int)s->dists[i];
            unsigned int v = (unsigned int)s->litlen[i];
            if (d == 0U) {
                defl_put_bits_(s, (unsigned int)litCodes[v], (int)litLens[v]);
            } else {
                int lc = defl_len_code_(v);
                defl_put_bits_(s, (unsigned int)litCodes[257 + lc], (int)litLens[257 + lc]);
                if (kLenExtra[lc] != (unsigned short)0) {
                    defl_put_bits_(s, v - (unsigned int)kLenBase[lc], (int)kLenExtra[lc]);
                }
                int dc = defl_dist_code_(d);
                defl_put_bits_(s, (unsigned int)distCodes[dc], (int)distLens[dc]);
                if (kDistExtra[dc] != (unsigned short)0) {
                    defl_put_bits_(s, d - (unsigned int)kDistBase[dc], (int)kDistExtra[dc]);
                }
            }
            i = i + 1UL;
        }
        defl_put_bits_(s, (unsigned int)litCodes[256], (int)litLens[256]);
    }
}

static unsigned long defl_data_bits_(struct DeflState_* s, const unsigned char* litLens, const unsigned char* distLens) {
    unsigned long bits = 0UL;
    unsafe {
        int i = 0;
        while (i < 286) {
            bits = bits + (unsigned long)s->litFreq[i] * (unsigned long)litLens[i];
            if (i >= 257) { bits = bits + (unsigned long)s->litFreq[i] * (unsigned long)kLenExtra[i - 257]; }
            i = i + 1;
        }
        i = 0;
        while (i < 30) {
            bits = bits + (unsigned long)s->distFreq[i] * ((unsigned long)distLens[i] + (unsigned long)kDistExtra[i]);
            i = i + 1;
        }
    }
    return bits;
}

static void defl_stored_block_(struct DeflState_* s, const unsigned char* src, unsigned long n, int final) {
    unsafe {
        unsigned long off = 0UL;
        while (1) {
            unsigned long chunk = n - off;
            if (chunk > 65535UL) { chunk = 65535UL; }
            int lastChunk = off + chunk >= n;
            defl_put_bits_(s, (final && lastChunk) ? 1U : 0U, 1);
            defl_put_bits_(s, 0U, 2);
            defl_align_(s);
            dbuf_push_(&s->out, (unsigned char)(chunk & 0xFFUL));
            dbuf_push_(&s->out, (unsigned char)((chunk >> 8) & 0xFFUL));
            dbuf_push_(&s->out, (unsigned char)(~chunk & 0xFFUL));
            dbuf_push_(&s->out, (unsigned char)((~chunk >> 8) & 0xFFUL));
            dbuf_push_n_(&s->out, &src[off], chunk);
            off = off + chunk;
            if (lastChunk) { break; }
        }
    }
}

// Writes the pending symbols (covering input [blockStart, end)) as one
// block, in whichever encoding is smallest.
static void defl_flush_block_(struct DeflState_* s, const unsigned char* src, unsigned long end, int final) {
    unsafe {
        s->litFreq[256] = 1U;
        unsigned char litLens[286];
        unsigned char distLens[30];
        unsigned short litCodes[286];
        unsigned short distCodes[30];
        defl_build_lengths_(&s->litFreq[0], 286, 15, &litLens[0]);
        defl_build_lengths_(&s->distFreq[0], 30, 15, &distLens[0]);
        // A decoder needs at least one distance code.
        int anyDist = 0;
        int i = 0;
        while (i < 30) { if (distLens[i] != (unsigned char)0) { anyDist = 1; } i = i + 1; }
        if (!anyDist) { distLens[0] = (unsigned char)1; }

        int nlit = 286;
        while (nlit > 257 && litLens[nlit - 1] == (unsigned char)0) { nlit = nlit - 1; }
        int ndist = 30;
        while (ndist > 1 && distLens[ndist - 1] == (unsigned char)0) { ndist = ndist - 1; }
        unsigned char all[316];
        i = 0;
        while (i < nlit) { all[i] = litLens[i]; i = i + 1; }
        i = 0;
        while (i < ndist) { all[nlit + i] = distLens[i]; i = i + 1; }
        unsigned short rle[316];
        int nrle = defl_rle_lengths_(&all[0], nlit + ndist, &rle[0]);
        unsigned int clFreq[19];
        i = 0;
        while (i < 19) { clFreq[i] = 0U; i = i + 1; }
        i = 0;
        while (i < nrle) { clFreq[rle[i] & (unsigned short)0xFF] = clFreq[rle[i] & (unsigned short)0xFF] + 1U; i = i + 1; }
        unsigned char clLens[19];
        unsigned short clCodes[19];
        defl_build_lengths_(&clFreq[0], 19, 7, &clLens[0]);
        int ncl = 19;
        while (ncl > 4 && clLens[kClOrder[ncl - 1]] == (unsigned char)0) { ncl = ncl - 1; }

        unsigned long dynBits = 3UL + 5UL + 5UL + 4UL + 3UL * (unsigned long)ncl;
        i = 0;
        while (i < nrle) {
            int sym = (int)(rle[i] & (unsigned short)0xFF);
            dynBits = dynBits + (unsigned long)clLens[sym];
            if (sym == 16) { dynBits = dynBits + 2UL; }
            if (sym == 17) { dynBits = dynBits + 3UL; }
            if (sym == 18) { dynBits = dynBits + 7UL; }
            i = i + 1;
        }
        dynBits = dynBits + defl_data_bits_(s, &litLens[0], &distLens[0]);

        // All 288 fixed lit/len lengths: the canonical codes of the
        // 9-bit range depend on how many 8-bit codes precede them.
        unsigned char fixLit[288];
        unsigned char fixDist[30];
        i = 0;
        while (i < 288) {
            fixLit[i] = (unsigned char)(i < 144 ? 8 : (i < 256 ? 9 : (i < 280 ? 7 : 8)));
            i = i + 1;
        }
        i = 0;
        while (i < 30) { fixDist[i] = (unsigned char)5; i = i + 1; }
        unsigned long fixBits = 3UL + defl_data_bits_(s, &fixLit[0], &fixDist[0]);
        unsigned long rawLen = end - s->blockStart;
        unsigned long storedBits = (rawLen + 5UL * (rawLen / 65535UL + 1UL)) * 8UL + 16UL;

        if (storedBits <= fixBits && storedBits <= dynBits) {
            defl_stored_block_(s, &src[s->blockStart], rawLen, final);
        } else if (fixBits <= dynBits) {
            defl_put_bits_(s, final ? 1U : 0U, 1);
            defl_put_bits_(s, 1U, 2);
            unsigned short fixLitCodes[288];
            unsigned short fixDistCodes[30];
            defl_build_codes_(&fixLit[0], 288, &fixLitCodes[0]);
            defl_build_codes_(&fixDist[0], 30, &fixDistCodes[0]);
            defl_emit_symbols_(s, &fixLitCodes[0], &fixLit[0], &fixDistCodes[0], &fixDist[0]);
        } else {
            defl_build_codes_(&litLens[0], 286, &litCodes[0]);
            defl_build_codes_(&distLens[0], 30, &distCodes[0]);
            defl_build_codes_(&clLens[0], 19, &clCodes[0]);
            defl_put_bits_(s, final ? 1U : 0U, 1);
            defl_put_bits_(s, 2U, 2);
            defl_put_bits_(s, (unsigned int)(nlit - 257), 5);
            defl_put_bits_(s, (unsigned int)(ndist - 1), 5);
            defl_put_bits_(s, (unsigned int)(ncl - 4), 4);
            i = 0;
            while (i < ncl) { defl_put_bits_(s, (unsigned int)clLens[kClOrder[i]], 3); i = i + 1; }
            i = 0;
            while (i < nrle) {
                int sym = (int)(rle[i] & (unsigned short)0xFF);
                unsigned int extra = (unsigned int)(rle[i] >> 8);
                defl_put_bits_(s, (unsigned int)clCodes[sym], (int)clLens[sym]);
                if (sym == 16) { defl_put_bits_(s, extra, 2); }
                if (sym == 17) { defl_put_bits_(s, extra, 3); }
                if (sym == 18) { defl_put_bits_(s, extra, 7); }
                i = i + 1;
            }
            defl_emit_symbols_(s, &litCodes[0], &litLens[0], &distCodes[0], &distLens[0]);
        }
        s->nsyms = 0UL;
        s->blockStart = end;
        i = 0;
        while (i < 286) { s->litFreq[i] = 0U; i = i + 1; }
        i = 0;
        while (i < 30) { s->distFreq[i] = 0U; i = i + 1; }
    }
}

static void defl_add_lit_(struct DeflState_* s, unsigned char c) {
    unsafe {
        s->litlen[s->nsyms] = (unsigned short)c;
        s->dists[s->nsyms] = (unsigned short)0;
        s->nsyms = s->nsyms + 1UL;
        s->litFreq[c] = s->litFreq[c] + 1U;
    }
}

static void defl_add_match_(struct DeflState_* s, unsigned long len, unsigned long dist) {
    unsafe {
        s->litlen[s->nsyms] = (unsigned short)len;
        s->dists[s->nsyms] = (unsigned short)dist;
        s->nsyms = s->nsyms + 1UL;
        int lc = defl_len_code_((unsigned int)len);
        s->litFreq[257 + lc] = s->litFreq[257 + lc] + 1U;
        int dc = defl_dist_code_((unsigned int)dist);
        s->distFreq[dc] = s->distFreq[dc] + 1U;
    }
}

static unsigned long defl_hash_(const unsigned char* p) {
    unsafe {
        unsigned long h = ((unsigned long)p[0] << 10) ^ ((unsigned long)p[1] << 5) ^ (unsigned long)p[2];
        return (h * 2654435761UL >> 7) & (DEFL_HSIZE - 1UL);
    }
}

// Longest match for position 'pos' along the hash chain; returns its length
// (0 if < MIN_MATCH) and sets *distOut.
static unsigned long defl_longest_(const unsigned char* src, unsigned long n, unsigned long pos,
                                   const int* head, const int* prev, unsigned long chain,
                                   unsigned long nice, unsigned long* distOut) {
    unsafe {
        unsigned long best = 0UL;
        unsigned long maxLen = n - pos;
        if (maxLen > DEFL_MAX_MATCH) { maxLen = DEFL_MAX_MATCH; }
        if (maxLen < DEFL_MIN_MATCH) { return 0UL; }
        int cand = head[defl_hash_(&src[pos])];
        while (cand >= 0 && chain > 0UL) {
            unsigned long c = (unsigned long)cand;
            if (c >= pos || pos - c > DEFL_WSIZE) { break; }
            if (best >= maxLen) { break; }
            if (src[c + best] == src[pos + best]) {
                unsigned long l = 0UL;
                while (l < maxLen && src[c + l] == src[pos + l]) { l = l + 1UL; }
                if (l > best) {
                    best = l;
                    *distOut = pos - c;
                    if (l >= nice) { break; }
                }
            }
            int nx = prev[c & DEFL_WMASK];
            if (nx >= cand) { break; }
            cand = nx;
            chain = chain - 1UL;
        }
        return best >= DEFL_MIN_MATCH ? best : 0UL;
    }
}

static void defl_insert_(const unsigned char* src, unsigned long n, unsigned long pos, int* head, int* prev) {
    unsafe {
        if (pos + 2UL >= n) { return; }
        unsigned long h = defl_hash_(&src[pos]);
        prev[pos & DEFL_WMASK] = head[h];
        head[h] = (int)pos;
    }
}

static struct Vec<unsigned char> deflate_raw_n_(const &unsigned char data, unsigned long len, int level) {
    if (level < 0) { level = 0; }
    if (level > 9) { level = 9; }
    struct DeflState_ s;
    dbuf_init_(&s.out, len / 2UL + 64UL);
    s.bitbuf = 0ULL;
    s.bitcnt = 0;
    s.nsyms = 0UL;
    s.blockStart = 0UL;
    const unsigned char* src;
    unsafe { src = (const unsigned char*)data; }
    if (level == 0 || len == 0UL) {
        defl_stored_block_(&s, src, len, 1);
        defl_align_(&s);
        return dbuf_to_vec_(&s.out);
    }
    unsigned long chainByLevel[10] = { 0UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL, 256UL, 1024UL, 4096UL };
    unsigned long niceByLevel[10] = { 0UL, 8UL, 16UL, 32UL, 64UL, 128UL, 128UL, 258UL, 258UL, 258UL };
    unsigned long chain;
    unsigned long nice;
    unsafe {
        chain = chainByLevel[level];
        nice = niceByLevel[level];
    }
    int lazy = level >= 4;
    int* head;
    int* prev;
    unsafe {
        head = (int*)alloc_raw_(DEFL_HSIZE * sizeof(int));
        prev = (int*)alloc_raw_(DEFL_WSIZE * sizeof(int));
        s.litlen = (unsigned short*)alloc_raw_(DEFL_BLOCK_SYMS * sizeof(unsigned short));
        s.dists = (unsigned short*)alloc_raw_(DEFL_BLOCK_SYMS * sizeof(unsigned short));
        if (head == (int*)0 || prev == (int*)0 || s.litlen == (unsigned short*)0 || s.dists == (unsigned short*)0) {
            s.out.failed = 1;
        }
    }
    if (!s.out.failed) {
        unsafe {
            unsigned long i = 0UL;
            while (i < DEFL_HSIZE) { head[i] = -1; i = i + 1UL; }
            i = 0UL;
            while (i < DEFL_WSIZE) { prev[i] = -1; i = i + 1UL; }
            i = 0UL;
            while (i < 286UL) { s.litFreq[i] = 0U; i = i + 1UL; }
            i = 0UL;
            while (i < 30UL) { s.distFreq[i] = 0U; i = i + 1UL; }

            unsigned long pos = 0UL;
            while (pos < len) {
                unsigned long dist = 0UL;
                unsigned long m = defl_longest_(src, len, pos, head, prev, chain, nice, &dist);
                if (m > 0UL && lazy && m < nice && pos + 1UL < len) {
                    // Lazy evaluation: a longer match one byte later wins.
                    defl_insert_(src, len, pos, head, prev);
                    unsigned long dist2 = 0UL;
                    unsigned long m2 = defl_longest_(src, len, pos + 1UL, head, prev, chain, nice, &dist2);
                    if (m2 > m) {
                        defl_add_lit_(&s, src[pos]);
                        pos = pos + 1UL;
                        m = m2;
                        dist = dist2;
                    } else {
                        // pos was already inserted above; insert the rest.
                        defl_add_match_(&s, m, dist);
                        unsigned long k = 1UL;
                        while (k < m) { defl_insert_(src, len, pos + k, head, prev); k = k + 1UL; }
                        pos = pos + m;
                        if (s.nsyms >= DEFL_BLOCK_SYMS - 2UL) { defl_flush_block_(&s, src, pos, 0); }
                        continue;
                    }
                }
                if (m > 0UL) {
                    defl_add_match_(&s, m, dist);
                    unsigned long k = 0UL;
                    while (k < m) { defl_insert_(src, len, pos + k, head, prev); k = k + 1UL; }
                    pos = pos + m;
                } else {
                    defl_insert_(src, len, pos, head, prev);
                    defl_add_lit_(&s, src[pos]);
                    pos = pos + 1UL;
                }
                if (s.nsyms >= DEFL_BLOCK_SYMS - 2UL) { defl_flush_block_(&s, src, pos, 0); }
            }
            defl_flush_block_(&s, src, len, 1);
            defl_align_(&s);
        }
    }
    unsafe {
        if (head != (int*)0) { dealloc(head); }
        if (prev != (int*)0) { dealloc(prev); }
        if (s.litlen != (unsigned short*)0) { dealloc(s.litlen); }
        if (s.dists != (unsigned short*)0) { dealloc(s.dists); }
    }
    return dbuf_to_vec_(&s.out);
}

// ── Containers ─────────────────────────────────────────────────────────────

static void defl_push_vec_(struct DeflateBuf_* b, &Vec<unsigned char> v) {
    match (v.data) {
        case some(d): { unsafe { dbuf_push_n_(b, (const unsigned char*)d, v.len); } }
        case null: { }
    }
}

static struct Vec<unsigned char> zlib_compress_n_(const &unsigned char data, unsigned long len, int level) {
    struct Vec<unsigned char> body = deflate_raw_n_(data, len, level);
    struct DeflateBuf_ b;
    dbuf_init_(&b, body.len + 8UL);
    unsigned int flevel = level <= 1 ? 0U : (level < 6 ? 1U : (level == 6 ? 2U : 3U));
    unsigned int cmf = 0x78U;
    unsigned int flg = flevel << 6;
    flg = flg + (31U - ((cmf * 256U + flg) % 31U)) % 31U;
    dbuf_push_(&b, (unsigned char)cmf);
    dbuf_push_(&b, (unsigned char)flg);
    defl_push_vec_(&b, &body);
    body.free();
    unsigned int a = adler32_n_(data, len);
    dbuf_push_(&b, (unsigned char)(a >> 24));
    dbuf_push_(&b, (unsigned char)(a >> 16));
    dbuf_push_(&b, (unsigned char)(a >> 8));
    dbuf_push_(&b, (unsigned char)a);
    return dbuf_to_vec_(&b);
}

static struct Vec<unsigned char> zlib_decompress_n_(const &unsigned char data, unsigned long len,
                                          unsigned long max_out, ?&int ok) {
    int good = 1;
    unsigned int cmf = 0U;
    unsigned int flg = 0U;
    if (len < 6UL) { good = 0; }
    if (good) {
        unsafe {
            cmf = (unsigned int)data[0];
            flg = (unsigned int)data[1];
        }
        if ((cmf & 0x0FU) != 8U || (cmf >> 4) > 7U || ((cmf * 256U + flg) % 31U) != 0U || (flg & 0x20U) != 0U) {
            good = 0;
        }
    }
    struct Vec<unsigned char> out = Vec::new();
    if (good) {
        unsigned long used = 0UL;
        int iok = 0;
        unsafe { out = inflate_raw_n_((const &unsigned char)&data[2], len - 2UL, max_out, &used, &iok); }
        if (!iok || 2UL + used + 4UL > len) {
            good = 0;
        } else {
            unsigned int want;
            unsafe {
                unsigned long t = 2UL + used;
                want = ((unsigned int)data[t] << 24) | ((unsigned int)data[t + 1UL] << 16) |
                       ((unsigned int)data[t + 2UL] << 8) | (unsigned int)data[t + 3UL];
            }
            unsigned int got = 1U;
            match (out.data) {
                case some(d): { got = adler32_n_(d, out.len); }
                case null: { }
            }
            if (got != want) { good = 0; }
        }
    }
    if (!good) { out.free(); out = Vec::new(); }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return out;
}

static struct Vec<unsigned char> gzip_compress_n_(const &unsigned char data, unsigned long len, int level) {
    struct Vec<unsigned char> body = deflate_raw_n_(data, len, level);
    struct DeflateBuf_ b;
    dbuf_init_(&b, body.len + 20UL);
    unsigned char hdr[10] = { (unsigned char)0x1F, (unsigned char)0x8B, (unsigned char)8, (unsigned char)0,
                              (unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)0,
                              (unsigned char)0, (unsigned char)255 };
    unsafe {
        if (level >= 9) { hdr[8] = (unsigned char)2; }
        if (level == 1) { hdr[8] = (unsigned char)4; }
        dbuf_push_n_(&b, &hdr[0], 10UL);
    }
    defl_push_vec_(&b, &body);
    body.free();
    unsigned int c = crc32_n_(data, len);
    unsigned int sz = (unsigned int)(len & 0xFFFFFFFFUL);
    int k = 0;
    while (k < 4) { dbuf_push_(&b, (unsigned char)(c >> (8 * k))); k = k + 1; }
    k = 0;
    while (k < 4) { dbuf_push_(&b, (unsigned char)(sz >> (8 * k))); k = k + 1; }
    return dbuf_to_vec_(&b);
}

static int gzip_is_gzip_n_(const &unsigned char data, unsigned long len) {
    if (len < 2UL) { return 0; }
    unsafe { return data[0] == (unsigned char)0x1F && data[1] == (unsigned char)0x8B; }
}

static struct Vec<unsigned char> gzip_decompress_n_(const &unsigned char data, unsigned long len,
                                          unsigned long max_out, ?&int ok) {
    struct DeflateBuf_ b;
    dbuf_init_(&b, len * 3UL);
    int good = !b.failed;
    unsigned long pos = 0UL;
    int members = 0;
    const unsigned char* d;
    unsafe { d = (const unsigned char*)data; }
    while (good && pos < len) {
        unsafe {
            if (members > 0 && (pos + 2UL > len || d[pos] != (unsigned char)0x1F)) { break; }
            if (pos + 18UL > len || d[pos] != (unsigned char)0x1F || d[pos + 1UL] != (unsigned char)0x8B ||
                d[pos + 2UL] != (unsigned char)8) { good = 0; break; }
            unsigned int flags = (unsigned int)d[pos + 3UL];
            if ((flags & 0xE0U) != 0U) { good = 0; break; }
            unsigned long p = pos + 10UL;
            if ((flags & 4U) != 0U) {
                if (p + 2UL > len) { good = 0; break; }
                unsigned long xlen = (unsigned long)d[p] | ((unsigned long)d[p + 1UL] << 8);
                p = p + 2UL + xlen;
            }
            if ((flags & 8U) != 0U) {
                while (p < len && d[p] != (unsigned char)0) { p = p + 1UL; }
                p = p + 1UL;
            }
            if ((flags & 16U) != 0U) {
                while (p < len && d[p] != (unsigned char)0) { p = p + 1UL; }
                p = p + 1UL;
            }
            if ((flags & 2U) != 0U) { p = p + 2UL; }
            if (p >= len) { good = 0; break; }
            unsigned long used = 0UL;
            int iok = 0;
            unsigned long room = 0UL;
            if (max_out != 0UL) {
                if (b.len >= max_out) { good = 0; break; }
                room = max_out - b.len;
            }
            struct Vec<unsigned char> part = inflate_raw_n_((const &unsigned char)&d[p], len - p, room, &used, &iok);
            if (!iok || p + used + 8UL > len) { part.free(); good = 0; break; }
            unsigned long t = p + used;
            unsigned int wantCrc = (unsigned int)d[t] | ((unsigned int)d[t + 1UL] << 8) |
                                   ((unsigned int)d[t + 2UL] << 16) | ((unsigned int)d[t + 3UL] << 24);
            unsigned int wantLen = (unsigned int)d[t + 4UL] | ((unsigned int)d[t + 5UL] << 8) |
                                   ((unsigned int)d[t + 6UL] << 16) | ((unsigned int)d[t + 7UL] << 24);
            unsigned int gotCrc = 0U;
            match (part.data) {
                case some(pd): { gotCrc = crc32_n_(pd, part.len); }
                case null: { }
            }
            if (gotCrc != wantCrc || (unsigned int)(part.len & 0xFFFFFFFFUL) != wantLen) {
                part.free(); good = 0; break;
            }
            defl_push_vec_(&b, &part);
            part.free();
            pos = t + 8UL;
            members = members + 1;
        }
    }
    if (b.failed) { good = 0; }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    if (!good) {
        dbuf_free_(&b);
        return Vec::new();
    }
    return dbuf_to_vec_(&b);
}


// ── Slice API (deflate.h) over the (pointer, length) cores above ──────────

unsigned int crc32([]const unsigned char d) { unsafe { return crc32_n_((const &unsigned char)d.ptr, d.len); } }
unsigned int crc32_update(unsigned int crc, []const unsigned char d) { unsafe { return crc32_update_n_(crc, (const &unsigned char)d.ptr, d.len); } }
unsigned int adler32([]const unsigned char d) { unsafe { return adler32_n_((const &unsigned char)d.ptr, d.len); } }
unsigned int adler32_update(unsigned int adler, []const unsigned char d) { unsafe { return adler32_update_n_(adler, (const &unsigned char)d.ptr, d.len); } }
struct Vec<unsigned char> deflate_raw([]const unsigned char d, int level) { unsafe { return deflate_raw_n_((const &unsigned char)d.ptr, d.len, level); } }
struct Vec<unsigned char> inflate_raw([]const unsigned char d, unsigned long max_out, ?&unsigned long consumed, ?&int ok) {
    unsafe { return inflate_raw_n_((const &unsigned char)d.ptr, d.len, max_out, consumed, ok); }
}
struct Vec<unsigned char> zlib_compress([]const unsigned char d, int level) { unsafe { return zlib_compress_n_((const &unsigned char)d.ptr, d.len, level); } }
struct Vec<unsigned char> zlib_decompress([]const unsigned char d, unsigned long max_out, ?&int ok) {
    unsafe { return zlib_decompress_n_((const &unsigned char)d.ptr, d.len, max_out, ok); }
}
struct Vec<unsigned char> gzip_compress([]const unsigned char d, int level) { unsafe { return gzip_compress_n_((const &unsigned char)d.ptr, d.len, level); } }
struct Vec<unsigned char> gzip_decompress([]const unsigned char d, unsigned long max_out, ?&int ok) {
    unsafe { return gzip_decompress_n_((const &unsigned char)d.ptr, d.len, max_out, ok); }
}
int gzip_is_gzip([]const unsigned char d) { unsafe { return gzip_is_gzip_n_((const &unsigned char)d.ptr, d.len); } }

} // namespace std
