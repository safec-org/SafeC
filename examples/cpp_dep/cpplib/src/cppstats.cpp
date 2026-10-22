#include "cppstats.h"

#include <algorithm>
#include <set>
#include <vector>

extern "C" double cppstats_median(const double* values, int n) {
    if (n <= 0) return 0.0;
    std::vector<double> v(values, values + n);
    std::sort(v.begin(), v.end());
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

extern "C" int cppstats_distinct(const int* values, int n) {
    return n <= 0 ? 0 : (int)std::set<int>(values, values + n).size();
}
