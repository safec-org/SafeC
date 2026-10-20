#pragma once
// SafeC Standard Library — Locale (C11 <locale.h>)

// Locale category constants (POSIX values)
namespace std {

static const int LC_ALL()      { return 6; }
static const int LC_COLLATE()  { return 3; }
static const int LC_CTYPE()    { return 0; }
static const int LC_MONETARY() { return 4; }
static const int LC_NUMERIC()  { return 1; }
static const int LC_TIME()     { return 2; }

// Set the locale for a given category.
// name: "" = system default, "C" = POSIX/C locale, "en_US.UTF-8", etc.
// Returns the new locale string, or NULL on failure.
const ?&char locale_set(int category, const &char name);

// Query the current locale for a category without changing it.
const ?&char locale_get(int category);

} // namespace std
