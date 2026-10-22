// SafeC Standard Library — DEFLATE compression (RFC 1951) with the zlib
// (RFC 1950) and gzip (RFC 1952) containers, plus the CRC-32 and Adler-32
// checksums they use.
//
// What HTTP's Content-Encoding: gzip / deflate needs (see std/http/http.h's
// client, which decodes them, and the router's gzip middleware, which
// encodes them), and what PNG/zip-style formats build on. Whole-buffer
// (one-shot) API: the input is one byte range in memory and the result is a
// fresh Vec<unsigned char>.
//
// The compressor is LZ77 over a 32 KiB window (hash chains; lazy matching
// from level 4 up) followed by, per block, whichever of stored / fixed
// Huffman / dynamic Huffman encodes it smallest — the same output format
// and roughly the same ratio class as zlib at a given level, not
// byte-identical to it. The decompressor accepts any conforming stream
// (stored, fixed and dynamic blocks; any window up to 32 KiB).
//
// Decompression takes a 'max_out' limit (0 = none): a stream that would
// expand past it fails (*ok = 0) instead of exhausting memory — set it when
// the input is untrusted (a "zip bomb" request body, say).
//
// Freestanding-safe: no I/O, only std/mem.h's allocator.
#pragma once
#include <std/collections/vec.h>

namespace std {

#define DEFLATE_STORE    0   // no compression (stored blocks)
#define DEFLATE_FAST     1
#define DEFLATE_DEFAULT  6
#define DEFLATE_BEST     9

// ── Checksums ──────────────────────────────────────────────────────────────

// CRC-32 (IEEE 802.3, reflected, as gzip/zip/PNG use): crc32 of "123456789"
// is 0xCBF43926. crc32_update continues a running value (start from 0).
unsigned int crc32(const &unsigned char data, unsigned long len);
unsigned int crc32_update(unsigned int crc, const &unsigned char data, unsigned long len);

// Adler-32 (zlib): adler32_update continues a running value (start from 1).
unsigned int adler32(const &unsigned char data, unsigned long len);
unsigned int adler32_update(unsigned int adler, const &unsigned char data, unsigned long len);

// ── Raw DEFLATE ────────────────────────────────────────────────────────────

// Compresses data[0..len) at 'level' (0..9; out-of-range values are
// clamped). Always succeeds (an empty Vec only if allocation fails).
struct Vec<unsigned char> deflate_raw(const &unsigned char data, unsigned long len, int level);

// Decompresses a raw DEFLATE stream. *ok = 0 on a malformed or truncated
// stream or when the output would exceed max_out (0 = no limit). Bytes
// after the final block are ignored; *consumed (if non-null) receives how
// many input bytes the stream used.
struct Vec<unsigned char> inflate_raw(const &unsigned char data, unsigned long len,
                                      unsigned long max_out, ?&unsigned long consumed,
                                      ?&int ok);

// ── zlib (RFC 1950) ────────────────────────────────────────────────────────

// 2-byte header + raw DEFLATE + big-endian Adler-32 — HTTP's
// "Content-Encoding: deflate".
struct Vec<unsigned char> zlib_compress(const &unsigned char data, unsigned long len, int level);
// Checks the header and the Adler-32; *ok = 0 on any mismatch (a preset
// dictionary, FDICT, is not supported).
struct Vec<unsigned char> zlib_decompress(const &unsigned char data, unsigned long len,
                                          unsigned long max_out, ?&int ok);

// ── gzip (RFC 1952) ────────────────────────────────────────────────────────

// One gzip member (no file name, mtime 0) — "Content-Encoding: gzip".
struct Vec<unsigned char> gzip_compress(const &unsigned char data, unsigned long len, int level);
// Decodes every concatenated member (as gunzip does), checking each
// member's CRC-32 and length; optional header fields (FEXTRA, FNAME,
// FCOMMENT, FHCRC) are skipped. *ok = 0 on any malformed member.
struct Vec<unsigned char> gzip_decompress(const &unsigned char data, unsigned long len,
                                          unsigned long max_out, ?&int ok);

// 1 if data starts with the gzip magic bytes (1f 8b).
int gzip_is_gzip(const &unsigned char data, unsigned long len);

} // namespace std
