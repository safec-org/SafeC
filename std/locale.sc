// SafeC Standard Library — locale implementation
#pragma once
#include <std/locale.h>

namespace std {

extern char* setlocale(int category, const char* locale);

inline const ?&char locale_set(int category, const &char name) {
    unsafe { return setlocale(category, name); }
}

inline const ?&char locale_get(int category) {
    unsafe { return setlocale(category, (const char*)0); }
}

} // namespace std
