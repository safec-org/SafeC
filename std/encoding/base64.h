// SafeC Standard Library — Base64 (RFC 4648, standard alphabet + '=' padding)
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

// Encodes the bytes of 'data' into standard Base64 text (with '=' padding,
// no line-wrapping).
struct String base64_encode([]const unsigned char data);

// Decodes standard Base64 text (whitespace between groups is tolerated and
// skipped; anything else invalid — a non-alphabet, non-'=' byte, or '='
// appearing somewhere other than as trailing padding — sets '*ok' (if
// non-NULL) to 0 and returns an empty Vec). On success, '*ok' is set to 1
// and the returned Vec<unsigned char> holds the decoded
// data; free it with Vec::free() same as any other Vec.
struct Vec<unsigned char> base64_decode(const &char s, ?&int ok);

} // namespace std
