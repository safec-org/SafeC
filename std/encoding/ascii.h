// SafeC Standard Library — ASCII validation and case conversion
#pragma once

namespace std {

// Returns 1 if every byte in the NUL-terminated string is in [0, 127].
int ascii_is_valid(const char* s);

// Returns 1 if every byte in s[0, len) is in [0, 127] (no NUL terminator
// required — for validating a raw byte range, e.g. from a socket read).
int ascii_is_valid_n(const char* s, unsigned long len);

int ascii_is_alpha(char c);
int ascii_is_digit(char c);
int ascii_is_alnum(char c);
int ascii_is_space(char c);
int ascii_is_upper(char c);
int ascii_is_lower(char c);
int ascii_is_print(char c);   // printable, including ' ' (0x20..0x7E)

char ascii_to_upper(char c);  // no-op for non a-z bytes
char ascii_to_lower(char c);  // no-op for non A-Z bytes

} // namespace std
