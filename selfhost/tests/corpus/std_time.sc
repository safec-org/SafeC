// std/time's calendar functions (gmtime_r, strftime, strptime in the
// playground runtime). Local time and the clocks differ by design (no time
// zone, no clock in the playground), so only UTC conversions are printed.
#include <std/time.h>
extern int printf(const char* fmt, ...);

static void show(long long t) {
    struct TimeFields f = std::time_gmtime(t);
    char buf[96];
    unsigned long n = std::time_format(buf[0..96], "%Y-%m-%d %H:%M:%S %a %A %b %B %j %w %u %y %C %e %I %p", &f);
    unsafe { printf("%lld -> [%lu] %s | wday=%d yday=%d isdst=%d\n", t, n, &buf[0], f.wday, f.yday, f.isdst); }
}

int main() {
    show(0LL);
    show(951782400LL + 3661LL);        // 2000-02-29 01:01:01 (leap day)
    show(1709210096LL);                // 2024-02-29 12:34:56
    show(4102444799LL);                // 2099-12-31 23:59:59
    show(-1LL);                        // 1969-12-31 23:59:59
    show(-2208988800LL);               // 1900-01-01
    show(253402300799LL);              // 9999-12-31 23:59:59

    struct TimeFields f = std::time_gmtime(1709210096LL);
    char buf[64];
    unsigned long n = std::time_format(buf[0..64], "%F %T|%D|%R|%%|%c", &f);
    unsafe { printf("[%lu] %s\n", n, &buf[0]); }
    n = std::time_format(buf[0..8], "%Y-%m-%d", &f);   // doesn't fit: 0
    unsafe { printf("small: %lu\n", n); }

    struct TimeFields p;
    int ok = std::time_parse("2024-02-29 12:34:56", "%Y-%m-%d %H:%M:%S", &p);
    unsafe { printf("parse %d: %d %d %d %d:%d:%d wday=%d yday=%d\n", ok, p.year, p.mon, p.mday, p.hour, p.min, p.sec, p.wday, p.yday); }
    ok = std::time_parse("31 Dec 1999 11:59:00 PM", "%d %b %Y %I:%M:%S %p", &p);
    unsafe { printf("parse %d: %d %d %d %d:%d:%d wday=%d yday=%d\n", ok, p.year, p.mon, p.mday, p.hour, p.min, p.sec, p.wday, p.yday); }
    ok = std::time_parse("Tuesday, March 05 2019", "%A, %B %d %Y", &p);
    unsafe { printf("parse %d: %d %d %d wday=%d yday=%d\n", ok, p.year, p.mon, p.mday, p.wday, p.yday); }
    ok = std::time_parse("2024-13-01", "%Y-%m-%d", &p);
    unsafe { printf("bad month: %d\n", ok); }
    ok = std::time_parse("12:00", "%H:%M:%S", &p);
    unsafe { printf("short: %d\n", ok); }

    unsafe { printf("leap %d %d %d %d\n", std::time_is_leap(2000), std::time_is_leap(1900), std::time_is_leap(2024), std::time_is_leap(2023)); }
    unsafe { printf("days %d %d %d\n", std::time_days_in_month(1, 2024), std::time_days_in_month(1, 2023), std::time_days_in_month(11, 2023)); }
    return 0;
}
