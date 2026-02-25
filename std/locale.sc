// SafeC Standard Library — locale implementation
#include "locale.h"

extern char* setlocale(int category, const char* locale);

const char* locale_set(int category, const char* name) {
    unsafe { return (const char*)setlocale(category, name); }
}

const char* locale_get(int category) {
    unsafe { return (const char*)setlocale(category, (const char*)0); }
}
