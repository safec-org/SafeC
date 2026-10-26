#pragma once
// SafeC Standard Library — Time (comprehensive C11 <time.h>)
// Complements sys.h (which covers high-resolution clocks).

// Opaque time value: seconds since Unix epoch (1970-01-01 00:00:00 UTC).
// Returned by time_now().
namespace std {

long long time_now();

// Broken-down calendar time (C's struct tm fields, in the same order).
struct TimeFields {
    int sec;     // 0-60
    int min;     // 0-59
    int hour;    // 0-23
    int mday;    // 1-31
    int mon;     // 0-11
    int year;    // years since 1900
    int wday;    // 0 = Sunday
    int yday;    // 0-365
    int isdst;   // > 0 daylight saving, 0 not, < 0 unknown
};

// A Unix timestamp as broken-down UTC time.
struct TimeFields time_gmtime(long long t);

// A Unix timestamp as broken-down local time.
struct TimeFields time_localtime(long long t);

// Formats tm into buf with a strftime format string. Returns the number of
// bytes written (0 on failure or when buf is too small).
unsigned long time_format([]char buf, const &char fmt, const &TimeFields tm);

// Parses a time string with strptime (POSIX) into *tm.
// Returns 1 on success, 0 on failure (and on Windows).
int time_parse(const &char s, const &char fmt, &TimeFields tm);

// Converts broken-down local time back to a Unix timestamp (-1 on error).
long long time_mktime(const &TimeFields tm);

// Difference in seconds between two timestamps.
double time_diff(long long end, long long start);

// Wall-clock time in nanoseconds (CLOCK_REALTIME).
long long time_wall_ns();

// Monotonic clock in nanoseconds (CLOCK_MONOTONIC) — for elapsed time.
long long time_mono_ns();

// Process CPU time in nanoseconds (CLOCK_PROCESS_CPUTIME_ID).
long long time_cpu_ns();

// Elapsed nanoseconds between two mono timestamps.
long long time_elapsed_ns(long long start, long long end);

// Convert nanoseconds to milliseconds.
double time_ns_to_ms(long long ns);

// Sleep for nanoseconds (high precision).
void time_sleep_ns(long long ns);

// Days in a given month (month 0-11, year = actual year e.g. 2024).
int time_days_in_month(int month, int year);

// Return 1 if year is a leap year, 0 otherwise.
int time_is_leap(int year);

} // namespace std
