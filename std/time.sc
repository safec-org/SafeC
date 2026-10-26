// SafeC Standard Library — time implementation
#pragma once
#include <std/stddef.h>
#include <std/time.h>

namespace std {

extern long long time(long long* t);
extern size_t strftime(char* s, size_t max, const char* format, const unsigned char* tm);
extern long long mktime(unsigned char* tm);

// gmtime_r/localtime_r/strptime are POSIX, not in any C standard --
// Windows/MSVC's UCRT doesn't export them (same class of link failure as
// drand48/clock_gettime elsewhere in this file). A real Windows port needs
// SYSTEMTIME/FileTimeToSystemTime plus its own timezone-database story,
// not a one-line substitute the way time_wall_ns/time_sleep_ns below got —
// out of scope here. Stubbed honestly instead of silently wrong: zeroed
// output / a real failure return, not fabricated calendar fields.
// struct tm begins with the 9 int fields TimeFields mirrors; platform
// extras (tm_gmtoff, tm_zone) fit in the 14-int buffers used here.
static struct TimeFields time_from_tm_(const &int tm) {
    struct TimeFields f;
    unsafe {
        f.sec = tm[0]; f.min = tm[1]; f.hour = tm[2]; f.mday = tm[3]; f.mon = tm[4];
        f.year = tm[5]; f.wday = tm[6]; f.yday = tm[7]; f.isdst = tm[8];
    }
    return f;
}

static void time_to_tm_(const &TimeFields f, &int tm) {
    unsafe {
        tm[0] = f.sec; tm[1] = f.min; tm[2] = f.hour; tm[3] = f.mday; tm[4] = f.mon;
        tm[5] = f.year; tm[6] = f.wday; tm[7] = f.yday; tm[8] = f.isdst;
        int i = 9;
        while (i < 14) { tm[i] = 0; i = i + 1; }
    }
}

#ifdef _WIN32
inline struct TimeFields time_gmtime(long long t) {
    int tm_buf[14];
    int i = 0;
    while (i < 14) { tm_buf[i] = 0; i = i + 1; }
    return time_from_tm_(&tm_buf[0]);
}
inline struct TimeFields time_localtime(long long t) {
    return time_gmtime(t);
}
inline int time_parse(const &char s, const &char fmt, &TimeFields tm) {
    return 0; // not implemented on Windows -- see comment above
}
#else
extern unsigned char* gmtime_r(const long long* t, unsigned char* tm);
extern unsigned char* localtime_r(const long long* t, unsigned char* tm);
extern char* strptime(const char* s, const char* format, unsigned char* tm);

inline struct TimeFields time_gmtime(long long t) {
    int tm_buf[14];   // 56 bytes: covers struct tm on all platforms
    unsafe { gmtime_r((const long long*)&t, (unsigned char*)tm_buf); }
    return time_from_tm_(&tm_buf[0]);
}

inline struct TimeFields time_localtime(long long t) {
    int tm_buf[14];
    unsafe { localtime_r((const long long*)&t, (unsigned char*)tm_buf); }
    return time_from_tm_(&tm_buf[0]);
}

inline int time_parse(const &char s, const &char fmt, &TimeFields tm) {
    int tm_buf[14];
    int i = 0;
    while (i < 14) { tm_buf[i] = 0; i = i + 1; }
    tm_buf[8] = -1;
    char* end;
    unsafe { end = strptime(s, fmt, (unsigned char*)tm_buf); }
    if (end == (char*)0) return 0;
    *tm = time_from_tm_(&tm_buf[0]);
    return 1;
}
#endif

// struct tm has 9 int fields (at minimum); we use a 56-byte buffer to be safe

inline long long time_now() {
    unsafe { return time((long long*)0); }
}

inline unsigned long time_format([]char buf, const &char fmt, const &TimeFields tm) {
    if (buf.len == 0UL) { return 0UL; }
    int tm_buf[14];
    time_to_tm_(tm, &tm_buf[0]);
    unsafe { return strftime((char*)buf.ptr, buf.len, fmt, (const unsigned char*)tm_buf); }
}

inline long long time_mktime(const &TimeFields tm) {
    int tm_buf[14];
    time_to_tm_(tm, &tm_buf[0]);
    unsafe { return mktime((unsigned char*)tm_buf); }
}

inline double time_diff(long long end, long long start) {
    return (double)(end - start);
}

// clockid_t values passed to clock_gettime() are NOT standardized by
// POSIX -- only the *names* are; each OS assigns its own integers, and
// they genuinely differ. CLOCK_REALTIME is 0 on both of the two families
// below (coincidence, not a guarantee), but CLOCK_MONOTONIC and
// CLOCK_PROCESS_CPUTIME_ID are not:
//   Darwin (macOS/iOS):  REALTIME=0  MONOTONIC=6   PROCESS_CPUTIME_ID=12
//   Linux (glibc/musl):  REALTIME=0  MONOTONIC=1   PROCESS_CPUTIME_ID=2
// This file used to hardcode the Darwin values unconditionally -- on
// Linux, id 6 isn't an error, which is what made this bug so easy to
// miss: it's CLOCK_MONOTONIC_COARSE, a real, valid clock that just
// trades precision for a cheap VDSO-only read (no syscall trap) by
// updating on the kernel's timer-tick cadence instead of continuously.
// Confirmed as a real, not theoretical, bug: benchmarking a CPU ML
// workload on WSL2 (see benchmarks.md) came back with train/inference
// times landing on suspiciously exact millisecond boundaries
// (100.000ms, 324.000ms to 3 decimal places) -- consistent with a clock
// that only ticks every several milliseconds on WSL2's virtualized timer,
// not with the workload's real, sub-tick-granular duration. id 12 is
// similarly wrong on Linux (some unrelated clock, not
// CLOCK_PROCESS_CPUTIME_ID). time_wall_ns() (CLOCK_REALTIME=0) was never
// affected either way.
#ifdef __APPLE__
#define SAFEC_CLOCK_MONOTONIC_ 6
#define SAFEC_CLOCK_CPUTIME_   12
#else
#define SAFEC_CLOCK_MONOTONIC_ 1
#define SAFEC_CLOCK_CPUTIME_   2
#endif

// GetSystemTimePreciseAsFileTime (Windows 8+) writes a FILETIME -- two
// packed 32-bit words that are just a 64-bit little-endian count of
// 100ns ticks since 1601-01-01 UTC when read as one i64 on this ABI, the
// same "read a multi-field Win32 struct through a same-size scalar
// pointer" trick already used elsewhere in this codebase (e.g.
// WSAPOLLFD). 116444736000000000 is the fixed offset (in 100ns ticks)
// between the Windows epoch (1601) and Unix epoch (1970) -- a standard,
// well-known constant for this exact conversion, not derived here.
#ifdef _WIN32
extern void GetSystemTimePreciseAsFileTime(long long* out);

inline long long time_wall_ns() {
    unsafe {
        long long filetime100ns = 0LL;
        GetSystemTimePreciseAsFileTime(&filetime100ns);
        return (filetime100ns - 116444736000000000LL) * 100LL;
    }
}
#else
// struct timespec { time_t tv_sec; long tv_nsec; }: two 4-byte fields on a
// 32-bit C ABI, two 8-byte fields on a 64-bit one. (SafeC's own 'long' is 64
// bits everywhere, so it can't stand in for C's here.)
#if __SIZEOF_POINTER__ == 4
#define SAFEC_TS_FIELD_ int
#else
#define SAFEC_TS_FIELD_ long long
#endif
struct SafecTimespec_ {
    SAFEC_TS_FIELD_ tv_sec;
    SAFEC_TS_FIELD_ tv_nsec;
};
extern int clock_gettime(int clk, unsigned char* ts);

inline long long time_wall_ns() {
    unsafe {
        struct SafecTimespec_ ts;
        if (clock_gettime(0, (unsigned char*)&ts) != 0) return -1LL;
        return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
    }
}
#endif

// Windows has no clock_gettime() at all (confirmed: a program calling it
// fails to *link*, not just "wrong clock" the way id 6/12 were on Linux —
// the UCRT import libraries this toolchain links against don't export it).
// QueryPerformanceCounter/-Frequency are the real Win32 monotonic timer,
// available since Windows 2000, and are what every other Windows profiler
// (including .NET's own Stopwatch, used elsewhere on this same benchmarks
// page) is built on. Split into whole-seconds/remainder before scaling to
// ns to avoid overflowing a 64-bit intermediate on a long-running process
// (counter * 1e9 alone can exceed 2^63 well within realistic uptimes).
#ifdef _WIN32
extern int QueryPerformanceCounter(long long* out);
extern int QueryPerformanceFrequency(long long* out);

inline long long time_mono_ns() {
    unsafe {
        long long freq = 0LL;
        long long counter = 0LL;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&counter);
        if (freq <= 0LL) return -1LL;
        long long whole = counter / freq;
        long long rem = counter % freq;
        return whole * 1000000000LL + (rem * 1000000000LL) / freq;
    }
}
#else
inline long long time_mono_ns() {
    unsafe {
        struct SafecTimespec_ ts;
        if (clock_gettime(SAFEC_CLOCK_MONOTONIC_, (unsigned char*)&ts) != 0) return -1LL;
        return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
    }
}
#endif

// GetProcessTimes' kernel+user FILETIME outputs (100ns ticks, duration
// not epoch-relative here) are the Win32 equivalent of
// CLOCK_PROCESS_CPUTIME_ID -- total CPU time this process has consumed.
#ifdef _WIN32
extern unsigned char* GetCurrentProcess();
extern int   GetProcessTimes(unsigned char* h, long long* creation, long long* exitT,
                              long long* kernel, long long* user);

inline long long time_cpu_ns() {
    unsafe {
        long long creation = 0LL; long long exitT = 0LL;
        long long kernel = 0LL; long long user = 0LL;
        unsigned char* h = GetCurrentProcess();
        if (GetProcessTimes(h, &creation, &exitT, &kernel, &user) == 0) return -1LL;
        return (kernel + user) * 100LL;
    }
}
#else
inline long long time_cpu_ns() {
    unsafe {
        struct SafecTimespec_ ts;
        if (clock_gettime(SAFEC_CLOCK_CPUTIME_, (unsigned char*)&ts) != 0) return -1LL;
        return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
    }
}
#endif

inline long long time_elapsed_ns(long long start, long long end) {
    return end - start;
}

inline double time_ns_to_ms(long long ns) {
    return (double)ns / 1000000.0;
}

// Win32's Sleep() only takes whole milliseconds -- real precision loss
// versus nanosleep's nanosecond request, rounded up (not down) so a
// caller never sleeps for *less* than asked.
#ifdef _WIN32
extern void Sleep(unsigned int ms);

inline void time_sleep_ns(long long ns) {
    unsafe {
        long long ms = (ns + 999999LL) / 1000000LL;
        if (ms < 0LL) { ms = 0LL; }
        if (ms > 0xFFFFFFFELL) { ms = 0xFFFFFFFELL; }   // INFINITE is 0xFFFFFFFF
        Sleep((unsigned int)ms);
    }
}
#else
extern int nanosleep(unsigned char* req, unsigned char* rem);

inline void time_sleep_ns(long long ns) {
    unsafe {
        struct SafecTimespec_ ts;
        ts.tv_sec = (SAFEC_TS_FIELD_)(ns / 1000000000LL);
        ts.tv_nsec = (SAFEC_TS_FIELD_)(ns % 1000000000LL);
        nanosleep((unsigned char*)&ts, (unsigned char*)0);
    }
}
#endif

inline int time_is_leap(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

inline int time_days_in_month(int month, int year) {
    int days[12];
    days[0]=31; days[1]=28; days[2]=31; days[3]=30;
    days[4]=31; days[5]=30; days[6]=31; days[7]=31;
    days[8]=30; days[9]=31; days[10]=30; days[11]=31;
    if (month == 1 && time_is_leap(year)) return 29;
    if (month < 0 || month > 11) return -1;
    return days[month];
}

} // namespace std
