#pragma once
// SafeC Standard Library — Locale (C11 <locale.h>)

// Locale category constants (POSIX values)
const int LC_ALL()      { return 6; }
const int LC_COLLATE()  { return 3; }
const int LC_CTYPE()    { return 0; }
const int LC_MONETARY() { return 4; }
const int LC_NUMERIC()  { return 1; }
const int LC_TIME()     { return 2; }

// Set the locale for a given category.
// name: "" = system default, "C" = POSIX/C locale, "en_US.UTF-8", etc.
// Returns the new locale string, or NULL on failure.
const char* locale_set(int category, const char* name);

// Query the current locale for a category without changing it.
const char* locale_get(int category);
