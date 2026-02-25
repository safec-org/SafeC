// SafeC Standard Library — time implementation
#include "time.h"

extern long long time(long long* t);
extern void* gmtime_r(const long long* t, void* tm);
extern void* localtime_r(const long long* t, void* tm);
extern unsigned long strftime(char* s, unsigned long max, const char* format, const void* tm);
extern char* strptime(const char* s, const char* format, void* tm);
extern long long mktime(void* tm);
extern int  clock_gettime(int clk, void* ts);
extern int  nanosleep(void* req, void* rem);

// struct tm has 9 int fields (at minimum); we use a 56-byte buffer to be safe
// struct timespec = { time_t sec; long nsec } = 16 bytes on 64-bit

long long time_now() {
    unsafe { return time((long long*)0); }
}

void time_gmtime(long long t, int* out) {
    unsafe {
        // struct tm is platform-specific; use a generous 56-byte buffer
        int tm_buf[14]; // 14 ints = 56 bytes, covers struct tm on all platforms
        gmtime_r(&t, (void*)tm_buf);
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

void time_localtime(long long t, int* out) {
    unsafe {
        int tm_buf[14];
        localtime_r(&t, (void*)tm_buf);
        out[0] = tm_buf[0]; out[1] = tm_buf[1]; out[2] = tm_buf[2];
        out[3] = tm_buf[3]; out[4] = tm_buf[4]; out[5] = tm_buf[5];
        out[6] = tm_buf[6]; out[7] = tm_buf[7]; out[8] = tm_buf[8];
    }
}

unsigned long time_format(char* buf, unsigned long cap, const char* fmt, int* tm_fields) {
    unsafe {
        int tm_buf[14];
        tm_buf[0] = tm_fields[0]; tm_buf[1] = tm_fields[1]; tm_buf[2] = tm_fields[2];
        tm_buf[3] = tm_fields[3]; tm_buf[4] = tm_fields[4]; tm_buf[5] = tm_fields[5];
        tm_buf[6] = tm_fields[6]; tm_buf[7] = tm_fields[7]; tm_buf[8] = tm_fields[8];
        tm_buf[9] = 0; tm_buf[10] = 0; tm_buf[11] = 0; tm_buf[12] = 0; tm_buf[13] = 0;
        return strftime(buf, cap, fmt, (const void*)tm_buf);
    }
}

int time_parse(const char* s, const char* fmt, int* tm_fields) {
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

long long time_mktime(int* tm_fields) {
    unsafe {
        int tm_buf[14];
        tm_buf[0] = tm_fields[0]; tm_buf[1] = tm_fields[1]; tm_buf[2] = tm_fields[2];
        tm_buf[3] = tm_fields[3]; tm_buf[4] = tm_fields[4]; tm_buf[5] = tm_fields[5];
        tm_buf[6] = tm_fields[6]; tm_buf[7] = tm_fields[7]; tm_buf[8] = tm_fields[8];
        tm_buf[9] = 0; tm_buf[10] = 0; tm_buf[11] = 0; tm_buf[12] = 0; tm_buf[13] = 0;
        return mktime((void*)tm_buf);
    }
}

double time_diff(long long end, long long start) {
    return (double)(end - start);
}

long long time_wall_ns() {
    unsafe {
        long long ts[2];
        if (clock_gettime(0, (void*)ts) != 0) return -1LL;
        return ts[0] * 1000000000LL + ts[1];
    }
}

long long time_mono_ns() {
    unsafe {
        long long ts[2];
        if (clock_gettime(1, (void*)ts) != 0) return -1LL;
        return ts[0] * 1000000000LL + ts[1];
    }
}

long long time_cpu_ns() {
    unsafe {
        long long ts[2];
        if (clock_gettime(2, (void*)ts) != 0) return -1LL;
        return ts[0] * 1000000000LL + ts[1];
    }
}

long long time_elapsed_ns(long long start, long long end) {
    return end - start;
}

double time_ns_to_ms(long long ns) {
    return (double)ns / 1000000.0;
}

void time_sleep_ns(long long ns) {
    unsafe {
        long long ts[2];
        ts[0] = ns / 1000000000LL;
        ts[1] = ns % 1000000000LL;
        nanosleep((void*)ts, (void*)0);
    }
}

int time_is_leap(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

int time_days_in_month(int month, int year) {
    int days[12];
    days[0]=31; days[1]=28; days[2]=31; days[3]=30;
    days[4]=31; days[5]=30; days[6]=31; days[7]=31;
    days[8]=30; days[9]=31; days[10]=30; days[11]=31;
    if (month == 1 && time_is_leap(year)) return 29;
    if (month < 0 || month > 11) return -1;
    return days[month];
}
