// SafeC Standard Library — Character classification and conversion (C89–C23)
// Functions are locale-aware (delegate to the current C locale).
// SafeC idiomatic names: char_is_*(c) and char_to_*(c).
#pragma once

// ── Classification ─────────────────────────────────────────────────────────────
// Each function returns non-zero (true) or 0 (false).
// `c` must be in the range [0, 255] or EOF (-1).

// Is `c` a letter (a-z, A-Z)?
int char_is_alpha(int c);

// Is `c` a decimal digit (0-9)?
int char_is_digit(int c);

// Is `c` a letter or decimal digit?
int char_is_alnum(int c);

// Is `c` a hexadecimal digit (0-9, a-f, A-F)?
int char_is_xdigit(int c);

// Is `c` a whitespace character (space, tab, newline, CR, FF, VT)?
int char_is_space(int c);

// Is `c` a blank character (space or horizontal tab only)?
int char_is_blank(int c);

// Is `c` a printable character (including space)?
int char_is_print(int c);

// Is `c` a printable character excluding space (has visible glyph)?
int char_is_graph(int c);

// Is `c` a punctuation character (graphical but not alphanumeric)?
int char_is_punct(int c);

// Is `c` a control character (value < 32 or == 127 in ASCII)?
int char_is_ctrl(int c);

// Is `c` an uppercase letter?
int char_is_upper(int c);

// Is `c` a lowercase letter?
int char_is_lower(int c);

// ── Conversion ────────────────────────────────────────────────────────────────

// Return the lowercase version of `c` (unchanged if not uppercase).
int char_to_lower(int c);

// Return the uppercase version of `c` (unchanged if not lowercase).
int char_to_upper(int c);
