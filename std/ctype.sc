// SafeC Standard Library — Character classification and conversion
// Wraps libc ctype functions for locale-aware character operations.
// Explicit extern declarations are used because ctype.h implements these
// as macros on many platforms (e.g., macOS), which SafeC cannot resolve.
#pragma once
#include "ctype.h"

// ── Explicit extern declarations for libc ctype functions ────────────────────
extern int isalpha(int c);
extern int isdigit(int c);
extern int isalnum(int c);
extern int isxdigit(int c);
extern int isspace(int c);
extern int isblank(int c);
extern int isprint(int c);
extern int isgraph(int c);
extern int ispunct(int c);
extern int iscntrl(int c);
extern int isupper(int c);
extern int islower(int c);
extern int tolower(int c);
extern int toupper(int c);

int char_is_alpha(int c)  { unsafe { return isalpha(c); } }
int char_is_digit(int c)  { unsafe { return isdigit(c); } }
int char_is_alnum(int c)  { unsafe { return isalnum(c); } }
int char_is_xdigit(int c) { unsafe { return isxdigit(c); } }
int char_is_space(int c)  { unsafe { return isspace(c); } }
int char_is_blank(int c)  { unsafe { return isblank(c); } }
int char_is_print(int c)  { unsafe { return isprint(c); } }
int char_is_graph(int c)  { unsafe { return isgraph(c); } }
int char_is_punct(int c)  { unsafe { return ispunct(c); } }
int char_is_ctrl(int c)   { unsafe { return iscntrl(c); } }
int char_is_upper(int c)  { unsafe { return isupper(c); } }
int char_is_lower(int c)  { unsafe { return islower(c); } }
int char_to_lower(int c)  { unsafe { return tolower(c); } }
int char_to_upper(int c)  { unsafe { return toupper(c); } }
