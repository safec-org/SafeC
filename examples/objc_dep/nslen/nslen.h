#pragma once
// A C API over Foundation's NSString (see nslen.m).

// The length of a UTF-8 string in UTF-16 code units, as NSString counts it
// ("héllo" is 5, an emoji outside the BMP is 2); -1 if 's' isn't valid UTF-8.
long nslen_utf16_length(const char* s);

// Writes the uppercased string (NSString's -uppercaseString) into 'out',
// NUL-terminated and truncated to 'cap' bytes; returns the full length in
// bytes, or -1 if 's' isn't valid UTF-8.
long nslen_uppercase(const char* s, char* out, long cap);
