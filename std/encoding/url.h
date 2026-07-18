// SafeC Standard Library — URL (percent) encoding, RFC 3986 unreserved set
#pragma once
#include <std/collections/string.h>

namespace std {

// Percent-encodes every byte except the unreserved set (A-Z a-z 0-9 - _ . ~)
// — this matches the "component" flavor used for query-string keys/values
// and path segments (JavaScript's encodeURIComponent, not encodeURI).
// Space becomes "%20" (not "+" — that's an application/x-www-form-urlencoded
// convention some callers layer on top, not part of RFC 3986 itself).
struct String url_encode(const char* s);

// Reverses url_encode: "%XX" becomes the raw byte 0xXX, "+" is left as a
// literal '+' (not decoded to space — see url_encode's note; a form decoder
// that wants "+" as space should replace it before calling this). On a
// malformed "%" escape (not followed by two hex digits), '*ok' (if non-NULL)
// is set to 0 and decoding stops at that point; otherwise '*ok' is set to 1.
struct String url_decode(const char* s, int* ok);

} // namespace std
