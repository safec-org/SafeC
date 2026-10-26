// SafeC Standard Library — Path strings (see path.h)
#pragma once
#include <std/path.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

int path_is_sep(char c) {
#ifdef _WIN32
    return c == '/' || c == '\\';
#else
    return c == '/';
#endif
}

int path_is_absolute(const &char p) {
    char c0;
    unsafe { c0 = p[0]; }
    if (path_is_sep(c0)) { return 1; }
#ifdef _WIN32
    unsafe {
        if (((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z')) && p[1] == ':' && path_is_sep(p[2])) {
            return 1;
        }
    }
#endif
    return 0;
}

struct String path_join(const &char a, const &char b) {
    if (path_is_absolute(b)) { return String::from(b); }
    struct String s = String::from(a);
    unsafe {
        if (b[0] == '\0') { return s; }
        if (s.len > 0UL && !path_is_sep((char)s.char_at(s.len - 1UL))) { s.push_char('/'); }
        unsigned long i = 0UL;
        while (b[i] != '\0' && path_is_sep(b[i]) && s.len > 0UL) { i = i + 1UL; }
        s.push((const char*)b + i);
    }
    return s;
}

// Length of 'p' without trailing separators (but keeping a lone root "/").
static unsigned long path_trimmed_len_(const &char p) {
    unsigned long n = 0UL;
    unsafe { while (p[n] != '\0') { n = n + 1UL; } }
    while (n > 1UL) {
        char c;
        unsafe { c = p[n - 1UL]; }
        if (!path_is_sep(c)) { break; }
        n = n - 1UL;
    }
    return n;
}

// Index where the last component of p[0..n) starts.
static unsigned long path_last_start_(const &char p, unsigned long n) {
    unsigned long i = n;
    while (i > 0UL) {
        char c;
        unsafe { c = p[i - 1UL]; }
        if (path_is_sep(c)) { break; }
        i = i - 1UL;
    }
    return i;
}

struct String path_dirname(const &char p) {
    unsigned long n = path_trimmed_len_(p);
    unsigned long start = path_last_start_(p, n);
    if (start == 0UL) { return String::new(); }
    // Drop the separator(s) before the last component, keeping a root.
    unsigned long end = start;
    while (end > 1UL) {
        char c;
        unsafe { c = p[end - 1UL]; }
        if (!path_is_sep(c)) { break; }
        end = end - 1UL;
    }
    unsafe { return String::from_n_(p, end); }
}

struct String path_basename(const &char p) {
    unsigned long n = path_trimmed_len_(p);
    unsigned long start = path_last_start_(p, n);
    char c0;
    unsafe { c0 = p[0]; }
    if (n == 1UL && path_is_sep(c0)) { return String::new(); }
    unsafe { return String::from_n_((const char*)p + start, n - start); }
}

struct String path_ext(const &char p) {
    struct String base = path_basename(p);
    long long dot = base.rfind_char('.');
    struct String out;
    if (dot <= 0LL) { out = String::new(); }
    else { out = base.substr((unsigned long)dot + 1UL, base.len); }
    base.free();
    return out;
}

struct String path_stem(const &char p) {
    struct String base = path_basename(p);
    long long dot = base.rfind_char('.');
    if (dot > 0LL) { base.truncate((unsigned long)dot); }
    return base;
}

// Splits 'p' into its non-empty, non-"." components, resolving "..".
// Returns 1 if 'p' is absolute; 'prefix' gets a Windows drive ("C:").
static int path_components_(const &char p, &Vec<struct String> parts, &String prefix) {
    int abs = path_is_absolute(p);
    unsigned long i = 0UL;
#ifdef _WIN32
    unsafe {
        if (p[0] != '\0' && p[1] == ':') {
            prefix.push_n_(p, 2UL);
            i = 2UL;
        }
    }
#endif
    for (;;) {
        char c;
        unsafe { c = p[i]; }
        if (c == '\0') { break; }
        if (path_is_sep(c)) { i = i + 1UL; continue; }
        unsigned long s = i;
        unsafe { while (p[i] != '\0' && !path_is_sep(p[i])) { i = i + 1UL; } }
        struct String comp;
        unsafe { comp = String::from_n_((const char*)p + s, i - s); }
        if (comp.eq_cstr(".")) { comp.free(); continue; }
        if (comp.eq_cstr("..")) {
            int can_pop = 0;
            if (parts.len > 0UL) { can_pop = !parts.at(parts.len - 1UL).eq_cstr(".."); }
            if (can_pop) {
                ?struct String last = parts.pop();
                match (last) { case some(l): { l.free(); } case none: { } }
                comp.free();
                continue;
            }
            if (abs) { comp.free(); continue; }   // "/.." is "/"
        }
        parts.push(comp);
    }
    return abs;
}

static void path_free_parts_(&Vec<struct String> parts) {
    for (unsigned long i = 0UL; i < parts.len; i = i + 1UL) { parts.at(i).free(); }
    parts.free();
}

struct String path_normalize(const &char p) {
    struct Vec<struct String> parts = Vec::new();
    struct String out = String::new();
    int abs = path_components_(p, &parts, &out);
    if (abs) { out.push_char('/'); }
    for (unsigned long i = 0UL; i < parts.len; i = i + 1UL) {
        if (i > 0UL) { out.push_char('/'); }
        out.push_str(parts.at(i));
    }
    if (out.len == 0UL) { out.push_char('.'); }
    path_free_parts_(&parts);
    return out;
}

struct String path_relative(const &char from, const &char to) {
    struct Vec<struct String> a = Vec::new();
    struct Vec<struct String> b = Vec::new();
    struct String pa = String::new();
    struct String pb = String::new();
    int abs_a = path_components_(from, &a, &pa);
    int abs_b = path_components_(to, &b, &pb);
    struct String out = String::new();
    if (abs_a != abs_b || !pa.eq(&pb)) {
        path_free_parts_(&a); path_free_parts_(&b); pa.free(); pb.free();
        out.free();
        return path_normalize(to);
    }
    unsigned long common = 0UL;
    while (common < a.len && common < b.len && a.at(common).eq(b.at(common))) { common = common + 1UL; }
    for (unsigned long i = common; i < a.len; i = i + 1UL) {
        if (out.len > 0UL) { out.push_char('/'); }
        out.push("..");
    }
    for (unsigned long i = common; i < b.len; i = i + 1UL) {
        if (out.len > 0UL) { out.push_char('/'); }
        out.push_str(b.at(i));
    }
    if (out.len == 0UL) { out.push_char('.'); }
    path_free_parts_(&a); path_free_parts_(&b); pa.free(); pb.free();
    return out;
}

} // namespace std
