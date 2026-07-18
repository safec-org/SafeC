// SafeC Standard Library — UTF-8 encode/decode
//
// A "codepoint" here is a plain Unicode scalar value (0..0x10FFFF, excluding
// the surrogate range 0xD800..0xDFFF) packed into an 'unsigned int'. There is
// no separate 'rune'/'char32_t' type in SafeC — this mirrors Go's 'rune'-as-
// int32 convention rather than introducing a new primitive type.
#pragma once

namespace std {

// Encodes 'cp' into 'out' (caller-provided buffer, must have room for at
// least 4 bytes — UTF-8's longest encoding). Returns the number of bytes
// written (1-4), or 0 if 'cp' is not a valid Unicode scalar value (> 0x10FFFF
// or inside the surrogate range 0xD800..0xDFFF).
int utf8_encode(unsigned int cp, char* out);

// Decodes one codepoint starting at 's[pos]'. On success, writes the
// codepoint to '*cp_out' and returns the number of bytes consumed (1-4). On
// a malformed sequence (bad continuation byte, overlong encoding, encoded
// surrogate, or truncated multi-byte sequence at a NUL/end-of-buffer),
// returns 0 and '*cp_out' is unspecified.
int utf8_decode(const char* s, unsigned long pos, unsigned int* cp_out);

// Returns 1 if the entire NUL-terminated string is well-formed UTF-8.
int utf8_is_valid(const char* s);

// Counts codepoints (not bytes) in a NUL-terminated, well-formed UTF-8
// string. Returns 0 if the string is not valid UTF-8 (use utf8_is_valid
// first if you need to distinguish that from a genuinely empty string).
unsigned long utf8_len(const char* s);

// Number of bytes a codepoint encodes to (1-4), or 0 if invalid — the size
// utf8_encode would need without actually writing anything.
int utf8_encoded_len(unsigned int cp);

} // namespace std
