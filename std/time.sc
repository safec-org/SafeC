// SafeC Standard Library — time implementation
#pragma once
#include <std/time.h>

namespace std {

extern long long time(long long* t);
extern unsigned long strftime(char* s, unsigned long max, const char* format, const void* tm);
extern long long mktime(void* tm);

// gmtime_r/localtime_r/strptime are POSIX, not in any C standard --
// Windows/MSVC's UCRT doesn't export them (same class of link failure as
// drand48/clock_gettime elsewhere in this file). A real Windows port needs
// SYSTEMTIME/FileTimeToSystemTime plus its own timezone-database story,
// not a one-line substitute the way time_wall_ns/time_sleep_ns below got —
// out of scope here. Stubbed honestly instead of silently wrong: zeroed
// output / a real failure return, not fabricated calendar fields.
#ifdef _WIN32
inline void time_gmtime(long long t, int* out) {
    unsafe {
        int i = 0;
        while (i < 9) { out[i] = 0; i = i + 1; }
    }
}
inline void time_localtime(long long t, int* out) {
    unsafe {
        int i = 0;
        while (i < 9) { out[i] = 0; i = i + 1; }
    }
}
inline int time_parse(const char* s, const char* fmt, int* tm_fields) {
    return 0; // not implemented on Windows -- see comment above
}
#else
extern void* gmtime_r(const long long* t, void* tm);
extern void* localtime_r(const long long* t, void* tm);
extern char* strptime(const char* s, const char* format, void* tm);

inline void time_gmtime(long long t, int* out) {
    unsafe {
        // struct tm is platform-specific; use a generous 56-byte buffer
        int tm_buf[14]; // 14 ints = 56 bytes, covers struct tm on all platforms
        gmtime_r((const long long*)&t, (void*)tm_buf);
        // Copy the first 9 standard fields
        out[0] = tm_buf[0]; // tm_sec
        out[1] = tm_buf[1]; // tm_min
        out[2] = tm_buf[2]; // tm_hour
        out[3] = tm_buf[3]; // tm_mday
        out[4] = tm_buf[4]; // tm_mon
        out[5] = tm_buf[5]; // tm_year
        out[6] = tm_buf[6]; // tm_wday
        out[7] = tm_buf[7]; // tm_yday
        out[8] = tm_buf[8]; // tm_isdst
    }
}

inline void time_localtime(long long t, int* out) {
    unsafe {
        int tm_buf[14];
        localtime_r((const long long*)&t, (void*)tm_buf);
        out[0] = tm_buf[0]; out[1] = tm_buf[1]; out[2] = tm_buf[2];
        out[3] = tm_buf[3]; out[4] = tm_buf[4]; out[5] = tm_buf[5];
        out[6] = tm_buf[6]; out[7] = tm_buf[7]; out[8] = tm_buf[8];
    }
}

inline int time_parse(const char* s, const char* fmt, int* tm_fields) {
    unsafe {
        int tm_buf[14];
        tm_buf[0]=0; tm_buf[1]=0; tm_buf[2]=0; tm_buf[3]=0;
        tm_buf[4]=0; tm_buf[5]=0; tm_buf[6]=0; tm_buf[7]=0;
        tm_buf[8]=-1; tm_buf[9]=0; tm_buf[10]=0; tm_buf[11]=0; tm_buf[12]=0; tm_buf[13]=0;
        char* end = strptime(s, fmt, (void*)tm_buf);
        if (end == (char*)0) return 0;
        tm_fields[0] = tm_buf[0]; tm_fields[1] = tm_buf[1]; tm_fields[2] = tm_buf[2];
        tm_fields[3] = tm_buf[3]; tm_fields[4] = tm_buf[4]; tm_fields[5] = tm_buf[5];
        tm_fields[6] = tm_buf[6]; tm_fields[7] = tm_buf[7]; tm_fields[8] = tm_buf[8];
        return 1;
    }
}
#endif

// struct tm has 9 int fields (at minimum); we use a 56-byte buffer to be safe
// struct timespec = { time_t sec; long nsec } = 16 bytes on 64-bit

inline long long time_now() {
    unsafe { return time((long long*)0); }
}

inline unsigned long time_format(char* buf, unsigned long cap, const char* fmt, int* tm_fields) {
    unsafe {
        int tm_buf[14];
        tm_buf[0] = tm_fields[0]; tm_buf[1] = tm_fields[1]; tm_buf[2] = tm_fields[2];
        tm_buf[3] = tm_fields[3]; tm_buf[4] = tm_fields[4]; tm_buf[5] = tm_fields[5];
        tm_buf[6] = tm_fields[6]; tm_buf[7] = tm_fields[7]; tm_buf[8] = tm_fields[8];
        tm_buf[9] = 0; tm_buf[10] = 0; tm_buf[11] = 0; tm_buf[12] = 0; tm_buf[13] = 0;
        return strftime(buf, cap, fmt, (const void*)tm_buf);
    }
}

inline long long time_mktime(int* tm_fields) {
    unsafe {
        int tm_buf[14];
        tm_buf[0] = tm_fields[0]; tm_buf[1] = tm_fields[1]; tm_buf[2] = tm_fields[2];
        tm_buf[3] = tm_fields[3]; tm_buf[4] = tm_fields[4]; tm_buf[5] = tm_fields[5];
        tm_buf[6] = tm_fields[6]; tm_buf[7] = tm_fields[7]; tm_buf[8] = tm_fields[8];
        tm_buf[9] = 0; tm_buf[10] = 0; tm_buf[11] = 0; tm_buf[12] = 0; tm_buf[13] = 0;
        return mktime((void*)tm_buf);
    }
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
extern int clock_gettime(int clk, void* ts);

inline long long time_wall_ns() {
    unsafe {
        long long ts[2];
        if (clock_gettime(0, (void*)ts) != 0) return -1LL;
        return ts[0] * 1000000000LL + ts[1];
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
        long long ts[2];
        if (clock_gettime(SAFEC_CLOCK_MONOTONIC_, (void*)ts) != 0) return -1LL;
        return ts[0] * 1000000000LL + ts[1];
    }
}
#endif

// GetProcessTimes' kernel+user FILETIME outputs (100ns ticks, duration
// not epoch-relative here) are the Win32 equivalent of
// CLOCK_PROCESS_CPUTIME_ID -- total CPU time this process has consumed.
#ifdef _WIN32
extern void* GetCurrentProcess();
extern int   GetProcessTimes(void* h, long long* creation, long long* exitT,
                              long long* kernel, long long* user);

inline long long time_cpu_ns() {
    unsafe {
        long long creation = 0LL; long long exitT = 0LL;
        long long kernel = 0LL; long long user = 0LL;
        void* h = GetCurrentProcess();
        if (GetProcessTimes(h, &creation, &exitT, &kernel, &user) == 0) return -1LL;
        return (kernel + user) * 100LL;
    }
}
#else
inline long long time_cpu_ns() {
    unsafe {
        long long ts[2];
        if (clock_gettime(SAFEC_CLOCK_CPUTIME_, (void*)ts) != 0) return -1LL;
        return ts[0] * 1000000000LL + ts[1];
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
extern void Sleep(unsigned long ms);

inline void time_sleep_ns(long long ns) {
    unsafe {
        long long ms = (ns + 999999LL) / 1000000LL;
        if (ms < 0LL) { ms = 0LL; }
        Sleep((unsigned long)ms);
    }
}
#else
extern int nanosleep(void* req, void* rem);

inline void time_sleep_ns(long long ns) {
    unsafe {
        long long ts[2];
        ts[0] = ns / 1000000000LL;
        ts[1] = ns % 1000000000LL;
        nanosleep((void*)ts, (void*)0);
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
