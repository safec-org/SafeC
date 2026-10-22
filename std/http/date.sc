// SafeC Standard Library — HTTP date implementation (see date.h).
#pragma once
#include <std/http/date.h>
#include <std/collections/string.h>
#include <std/time.h>

namespace std {

// Three letters per name; day 0 (1970-01-01) was a Thursday.
static const char* http_wdays_() { return "ThuFriSatSunMonTueWed"; }
static const char* http_months_() { return "JanFebMarAprMayJunJulAugSepOctNovDec"; }

// Howard Hinnant's days_from_civil / civil_from_days.
long long http_days_from_civil(long long year, int month, int day) {
    long long y = month <= 2 ? year - 1LL : year;
    long long era = (y >= 0LL ? y : y - 399LL) / 400LL;
    long long yoe = y - era * 400LL;
    long long m = (long long)month;
    long long doy = (153LL * (m > 2LL ? m - 3LL : m + 9LL) + 2LL) / 5LL + (long long)day - 1LL;
    long long doe = yoe * 365LL + yoe / 4LL - yoe / 100LL + doy;
    return era * 146097LL + doe - 719468LL;
}

void http_civil_from_days(long long days, &long long year, &int month, &int day) {
    long long z = days + 719468LL;
    long long era = (z >= 0LL ? z : z - 146096LL) / 146097LL;
    long long doe = z - era * 146097LL;
    long long yoe = (doe - doe / 1460LL + doe / 36524LL - doe / 146096LL) / 365LL;
    long long y = yoe + era * 400LL;
    long long doy = doe - (365LL * yoe + yoe / 4LL - yoe / 100LL);
    long long mp = (5LL * doy + 2LL) / 153LL;
    long long d = doy - (153LL * mp + 2LL) / 5LL + 1LL;
    long long m = mp < 10LL ? mp + 3LL : mp - 9LL;
    *year = m <= 2LL ? y + 1LL : y;
    *month = (int)m;
    *day = (int)d;
}

static void http_push2_(&String out, long long v) {
    out.push_char((char)((int)'0' + (int)(v / 10LL)));
    out.push_char((char)((int)'0' + (int)(v % 10LL)));
}

struct String http_date_format(long long t) {
    // Clamp to years 1..9999.
    long long lo = -62135596800LL;   // 0001-01-01T00:00:00Z
    long long hi = 253402300799LL;   // 9999-12-31T23:59:59Z
    if (t < lo) { t = lo; }
    if (t > hi) { t = hi; }
    long long days = t >= 0LL ? t / 86400LL : (t - 86399LL) / 86400LL;
    long long secs = t - days * 86400LL;
    long long year = 0LL;
    int month = 1;
    int day = 1;
    http_civil_from_days(days, &year, &month, &day);
    long long wd = days % 7LL;
    if (wd < 0LL) { wd = wd + 7LL; }
    struct String out = String::with_cap(32UL);
    unsafe {
        out.push_n(&http_wdays_()[wd * 3LL], 3UL);
        out.push(", ");
        http_push2_(&out, (long long)day);
        out.push_char(' ');
        out.push_n(&http_months_()[(month - 1) * 3], 3UL);
        out.push_char(' ');
        out.push_char((char)((int)'0' + (int)(year / 1000LL)));
        out.push_char((char)((int)'0' + (int)(year / 100LL % 10LL)));
        http_push2_(&out, year % 100LL);
        out.push_char(' ');
        http_push2_(&out, secs / 3600LL);
        out.push_char(':');
        http_push2_(&out, secs / 60LL % 60LL);
        out.push_char(':');
        http_push2_(&out, secs % 60LL);
        out.push(" GMT");
    }
    return out;
}

struct HttpDateCur_ {
    const char* s;
    unsigned long i;
    int bad;
};

static int hd_is_alpha_(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int hd_is_digit_(char c) { return c >= '0' && c <= '9'; }

static char hd_peek_(&HttpDateCur_ c) {
    unsafe { return c.s[c.i]; }
}

static void hd_expect_(&HttpDateCur_ c, char ch) {
    unsafe {
        if (c.s[c.i] == ch) { c.i = c.i + 1UL; } else { c.bad = 1; }
    }
}

static void hd_spaces_(&HttpDateCur_ c) {
    unsafe {
        if (c.s[c.i] != ' ') { c.bad = 1; return; }
        while (c.s[c.i] == ' ') { c.i = c.i + 1UL; }
    }
}

// Reads exactly 'minDigits'..'maxDigits' digits.
static long long hd_num_(&HttpDateCur_ c, int minDigits, int maxDigits) {
    long long v = 0LL;
    int n = 0;
    unsafe {
        while (n < maxDigits && hd_is_digit_(c.s[c.i])) {
            v = v * 10LL + (long long)((int)c.s[c.i] - (int)'0');
            c.i = c.i + 1UL;
            n = n + 1;
        }
    }
    if (n < minDigits) { c.bad = 1; }
    return v;
}

static unsigned long hd_word_(&HttpDateCur_ c) {
    unsigned long n = 0UL;
    unsafe { while (hd_is_alpha_(c.s[c.i])) { c.i = c.i + 1UL; n = n + 1UL; } }
    return n;
}

static int hd_month_(&HttpDateCur_ c) {
    unsafe {
        unsigned long start = c.i;
        unsigned long n = hd_word_(c);
        if (n != 3UL) { c.bad = 1; return 1; }
        int m = 0;
        while (m < 12) {
            const char* name = &http_months_()[m * 3];
            if (c.s[start] == name[0] && c.s[start + 1UL] == name[1] && c.s[start + 2UL] == name[2]) {
                return m + 1;
            }
            m = m + 1;
        }
    }
    c.bad = 1;
    return 1;
}

static void hd_time_(&HttpDateCur_ c, &long long h, &long long m, &long long s) {
    *h = hd_num_(c, 2, 2);
    hd_expect_(c, ':');
    *m = hd_num_(c, 2, 2);
    hd_expect_(c, ':');
    *s = hd_num_(c, 2, 2);
}

long long http_date_parse(const &char text, ?&int ok) {
    struct HttpDateCur_ c;
    unsafe { c.s = (const char*)text; }
    c.i = 0UL;
    c.bad = 0;
    long long year = 0LL;
    int month = 1;
    long long day = 1LL;
    long long hh = 0LL;
    long long mm = 0LL;
    long long ss = 0LL;
    unsafe { while (c.s[c.i] == ' ') { c.i = c.i + 1UL; } }
    unsigned long wl = hd_word_(&c);
    if (wl < 3UL) { c.bad = 1; }
    if (!c.bad && hd_peek_(&c) == ',') {
        c.i = c.i + 1UL;
        hd_spaces_(&c);
        if (wl == 3UL) {
            // IMF-fixdate: DD Mon YYYY HH:MM:SS GMT
            day = hd_num_(&c, 2, 2);
            hd_spaces_(&c);
            month = hd_month_(&c);
            hd_spaces_(&c);
            year = hd_num_(&c, 4, 4);
        } else {
            // RFC 850: DD-Mon-YY HH:MM:SS GMT
            day = hd_num_(&c, 2, 2);
            hd_expect_(&c, '-');
            month = hd_month_(&c);
            hd_expect_(&c, '-');
            year = hd_num_(&c, 2, 4);
            if (year < 100LL) {
                // RFC 9110: a two-digit year more than 50 years in the
                // future is in the past century — taken relative to 2000.
                year = year < 70LL ? 2000LL + year : 1900LL + year;
            }
        }
        hd_spaces_(&c);
        hd_time_(&c, &hh, &mm, &ss);
        hd_spaces_(&c);
        unsafe {
            if (c.s[c.i] == 'G' && c.s[c.i + 1UL] == 'M' && c.s[c.i + 2UL] == 'T') { c.i = c.i + 3UL; }
            else { c.bad = 1; }
        }
    } else if (!c.bad && wl == 3UL) {
        // asctime: Mon DD HH:MM:SS YYYY   (DD space-padded)
        hd_spaces_(&c);
        month = hd_month_(&c);
        hd_spaces_(&c);
        day = hd_num_(&c, 1, 2);
        hd_spaces_(&c);
        hd_time_(&c, &hh, &mm, &ss);
        hd_spaces_(&c);
        year = hd_num_(&c, 4, 4);
    } else {
        c.bad = 1;
    }
    unsafe { while (!c.bad && c.s[c.i] == ' ') { c.i = c.i + 1UL; } }
    if (!c.bad && hd_peek_(&c) != (char)0) { c.bad = 1; }
    if (!c.bad && (day < 1LL || day > 31LL || hh > 23LL || mm > 59LL || ss > 60LL)) { c.bad = 1; }
    if (ok != (int*)0) { unsafe { *ok = !c.bad; } }
    if (c.bad) { return 0LL; }
    long long days = http_days_from_civil(year, month, (int)day);
    return days * 86400LL + hh * 3600LL + mm * 60LL + ss;
}

struct String http_date_now() { return http_date_format(time_now()); }

} // namespace std
