#pragma once
// SafeC Standard Library — Time (comprehensive C11 <time.h>)
// Complements sys.h (which covers high-resolution clocks).

// Opaque time value: seconds since Unix epoch (1970-01-01 00:00:00 UTC).
// Returned by time_now().
long long time_now();

// Convert a Unix timestamp to a broken-down UTC time.
// out must point to a 9-element int array:
//   [0]=sec, [1]=min, [2]=hour, [3]=mday, [4]=mon(0-11),
//   [5]=year(since 1900), [6]=wday(0=Sun), [7]=yday, [8]=isdst
void time_gmtime(long long t, int* out);

// Convert a Unix timestamp to local broken-down time (same layout as above).
void time_localtime(long long t, int* out);

// Format a time into buf using strftime format string.
// tm_fields: same 9-element array as time_gmtime.
// Returns number of bytes written (0 on failure).
unsigned long time_format(char* buf, unsigned long cap,
                           const char* fmt, int* tm_fields);

// Parse a time string with strptime (POSIX).
// Returns 1 on success, 0 on failure.
int time_parse(const char* s, const char* fmt, int* tm_fields);

// Convert broken-down local time back to Unix timestamp (-1 on error).
long long time_mktime(int* tm_fields);

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
