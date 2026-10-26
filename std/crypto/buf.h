#pragma once
// SafeC Standard Library — std/crypto's slice-to-reference helpers.
// Internal: included by std/crypto's headers (std-private; not API).
//
// std/crypto's public functions take []const unsigned char / []unsigned char;
// these turn a slice back into the reference the (reference, length) cores
// work on, checking a fixed-size buffer (key, nonce, tag, digest) is long
// enough — a short one is a bug, so it panics instead of reading or writing
// past the end.
#include <std/panic.h>

namespace std {

static unsigned char crypto_buf_zero_[1];

static void crypto_buf_short_() {
    panic_at("std/crypto: buffer shorter than the size this argument needs", "std/crypto/buf.h", __LINE__);
}

// s as a reference, after checking it holds 'need' bytes. An empty s (with
// need 0) gives a reference to a zero byte, never null.
static const &unsigned char crypto_in_([]const unsigned char s, unsigned long need) {
    if (s.len < need) { crypto_buf_short_(); }
    if (s.len == 0UL) { return &crypto_buf_zero_[0]; }
    unsafe { return (const &unsigned char)s.ptr; }
}

static &unsigned char crypto_out_([]unsigned char s, unsigned long need) {
    if (s.len < need) { crypto_buf_short_(); }
    if (s.len == 0UL) { return &crypto_buf_zero_[0]; }
    unsafe { return (&unsigned char)s.ptr; }
}

// s as a nullable reference: null when empty.
static const ?&unsigned char crypto_opt_([]const unsigned char s) {
    if (s.len == 0UL) { return null; }
    unsafe { return (const ?&unsigned char)s.ptr; }
}

} // namespace std
