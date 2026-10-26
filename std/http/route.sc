// SafeC Standard Library — URL path patterns (see route.h).
#pragma once
#include <std/http/route.h>
#include <std/encoding/url.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

// Segment [*s, *e) of p starting at *pos (skipping one '/'); 0 at the end.
static int rt_next_seg_(const char* p, unsigned long n, &unsigned long pos, &unsigned long s, &unsigned long e) {
    unsafe {
        unsigned long i = *pos;
        if (i < n && p[i] == '/') { i = i + 1UL; }
        if (i >= n) { *pos = i; return 0; }
        *s = i;
        while (i < n && p[i] != '/') { i = i + 1UL; }
        *e = i;
        *pos = i;
    }
    return 1;
}

static unsigned long rt_trim_len_(const char* p) {
    unsigned long n = 0UL;
    unsafe {
        while (p[n] != (char)0) { n = n + 1UL; }
        while (n > 1UL && p[n - 1UL] == '/') { n = n - 1UL; }
    }
    return n;
}

static void rt_push_(&Vec<RouteParam> params, const char* name, unsigned long nameLen,
                     const char* val, unsigned long valLen) {
    struct RouteParam rp;
    unsafe { rp.name = String::from_n(name, nameLen); }
    struct String raw = String::new();
    unsafe { raw.push_n(val, valLen); }
    // '+' stays '+' in a path; only %XX is decoded.
    int ok = 0;
    struct String dec = url_decode(raw.as_ptr(), &ok);
    if (ok) { rp.value = dec; raw.free(); } else { dec.free(); rp.value = raw; }
    params.push(rp);
}

long long http_route_match(const &char pattern, const &char path, &Vec<RouteParam> params) {
    const char* pt = pattern;
    const char* pa = path;
    unsigned long pn = rt_trim_len_(pt);
    unsigned long an = rt_trim_len_(pa);
    unsigned long before = params.length();
    unsigned long ppos = 0UL;
    unsigned long apos = 0UL;
    long long score = 1LL;
    while (1) {
        unsigned long ps = 0UL;
        unsigned long pe = 0UL;
        unsigned long as = 0UL;
        unsigned long ae = 0UL;
        int hasP = rt_next_seg_(pt, pn, &ppos, &ps, &pe);
        if (!hasP) {
            int hasA = rt_next_seg_(pa, an, &apos, &as, &ae);
            if (hasA) { break; }
            return score;
        }
        char first;
        unsafe { first = pt[ps]; }
        if (first == '*') {
            // Rest of the path (after the next '/').
            unsigned long restStart = apos;
            unsafe { if (restStart < an && pa[restStart] == '/') { restStart = restStart + 1UL; } }
            if (restStart > an) { restStart = an; }
            unsafe {
                if (pe - ps > 1UL) { rt_push_(params, &pt[ps + 1UL], pe - ps - 1UL, &pa[restStart], an - restStart); }
                else { rt_push_(params, "*", 1UL, &pa[restStart], an - restStart); }
            }
            return score * 4LL + 1LL;
        }
        int hasA = rt_next_seg_(pa, an, &apos, &as, &ae);
        if (!hasA) { break; }
        if (first == ':') {
            unsafe { rt_push_(params, &pt[ps + 1UL], pe - ps - 1UL, &pa[as], ae - as); }
            score = score * 4LL + 2LL;
        } else {
            if (pe - ps != ae - as) { break; }
            int same = 1;
            unsafe {
                unsigned long k = 0UL;
                while (k < pe - ps) {
                    if (pt[ps + k] != pa[as + k]) { same = 0; break; }
                    k = k + 1UL;
                }
            }
            if (!same) { break; }
            score = score * 4LL + 3LL;
        }
        if (score > 1000000000000000LL) { score = 1000000000000000LL; }
    }
    // No match: drop captures added during the attempt.
    while (params.length() > before) drop(params.pop());
    return 0LL;
}

const ?&char http_route_param(const &Vec<RouteParam> params, const &char name) {
    unsigned long i = 0UL;
    while (i < params.length()) {
        &RouteParam rp = params.at(i);
        if (rp.name.eq_cstr(name)) { return rp.value.as_ptr(); }
        i = i + 1UL;
    }
    return null;
}

void http_route_params_clear(&Vec<RouteParam> params) {
    // clear() drops each param's strings
    params.clear();
}

} // namespace std
