// SafeC Standard Library — Character classification and conversion
// ASCII / "C" locale semantics, implemented natively: freestanding-safe, and
// inlinable (libc's versions are an out-of-line call per character).
#pragma once
#include <std/ctype.h>

namespace std {

inline int char_is_upper(int c)  { return c >= 'A' && c <= 'Z'; }
inline int char_is_lower(int c)  { return c >= 'a' && c <= 'z'; }
inline int char_is_alpha(int c)  { return char_is_upper(c) || char_is_lower(c); }
inline int char_is_digit(int c)  { return c >= '0' && c <= '9'; }
inline int char_is_alnum(int c)  { return char_is_alpha(c) || char_is_digit(c); }
inline int char_is_xdigit(int c) { return char_is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
inline int char_is_space(int c)  { return c == ' ' || (c >= 9 && c <= 13); }
inline int char_is_blank(int c)  { return c == ' ' || c == 9; }
inline int char_is_print(int c)  { return c >= 32 && c <= 126; }
inline int char_is_graph(int c)  { return c >= 33 && c <= 126; }
inline int char_is_punct(int c)  { return char_is_graph(c) && !char_is_alnum(c); }
inline int char_is_ctrl(int c)   { return (c >= 0 && c < 32) || c == 127; }
inline int char_to_lower(int c)  { return char_is_upper(c) ? c + 32 : c; }
inline int char_to_upper(int c)  { return char_is_lower(c) ? c - 32 : c; }

} // namespace std
