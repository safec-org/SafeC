#pragma once
// SafeC Standard Library — GUI: UTF-8 helpers shared by the window layer,
// the text engine and the widgets (and usable on their own).

namespace std {

// gui_utf8_next() decodes the codepoint starting at byte *i of s[0..len)
// and advances *i past it (malformed or truncated sequences, overlongs and
// surrogates decode as U+FFFD, one byte at a time); gui_utf8_encode()
// writes cp's 1-4 bytes to out[0..4) and returns how many; gui_utf8_prev()
// returns the start of the codepoint before byte i (0 at the start).
unsigned int  gui_utf8_next(const &char s, unsigned long len, &unsigned long i);
int           gui_utf8_encode(unsigned int cp, &char out);
unsigned long gui_utf8_prev(const &char s, unsigned long i);

} // namespace std
