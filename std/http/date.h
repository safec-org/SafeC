// SafeC Standard Library — HTTP dates (RFC 9110 §5.6.7).
//
// The Date, Last-Modified, Expires, If-Modified-Since and Retry-After
// headers (and cookies' Expires attribute) carry timestamps as
// "IMF-fixdate" text: "Sun, 06 Nov 1994 08:49:37 GMT". Times here are
// Unix seconds (UTC), the same scale as std::time_now() (std/time.h).
// Pure arithmetic — no libc time functions, no time zone database.
#pragma once
#include <std/collections/string.h>

namespace std {

// IMF-fixdate for 'unix_seconds' (UTC). Years before 1 or after 9999 are
// clamped into that range.
struct String http_date_format(long long unix_seconds);

// Parses any of the three formats RFC 9110 requires recipients to accept:
// IMF-fixdate ("Sun, 06 Nov 1994 08:49:37 GMT"), the obsolete RFC 850 form
// ("Sunday, 06-Nov-94 08:49:37 GMT", two-digit years mapped to the nearest
// 50-year window) and asctime ("Sun Nov  6 08:49:37 1994"). The weekday
// name is not checked against the date. *ok = 0 (and 0 returned) on
// anything else.
long long http_date_parse(const &char text, ?&int ok);

// The current time as an IMF-fixdate (hosted targets: std::time_now()).
struct String http_date_now();

// Days since 1970-01-01 for a proleptic-Gregorian civil date (month 1-12),
// and back — the conversions the two functions above are built on.
long long http_days_from_civil(long long year, int month, int day);
void      http_civil_from_days(long long days, &long long year, &int month, &int day);

} // namespace std
