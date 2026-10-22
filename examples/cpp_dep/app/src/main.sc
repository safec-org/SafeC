// Calls a C++ library through its extern "C" API (cppstats.h).
#include <cppstats.h>

extern int printf(const char* fmt, ...);

int main() {
    double samples[5] = { 7.0, 1.0, 4.0, 9.0, 3.0 };
    int ids[6] = { 3, 1, 3, 2, 1, 3 };
    unsafe {
        double median = cppstats_median(&samples[0], 5);
        int distinct = cppstats_distinct(&ids[0], 6);
        printf("median %.1f, distinct %d\n", median, distinct);
        if (median != 4.0 || distinct != 3) return 1;
    }
    return 0;
}
