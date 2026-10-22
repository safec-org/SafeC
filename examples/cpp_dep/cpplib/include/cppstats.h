#pragma once
// The C API of a small C++ library: the functions SafeC (or C) can call.
// Everything behind them is C++ (see src/cppstats.cpp).
#ifdef __cplusplus
extern "C" {
#endif

// The median of 'n' values (the mean of the middle two for even n); 0 for n == 0.
double cppstats_median(const double* values, int n);

// The number of distinct values among 'n'.
int cppstats_distinct(const int* values, int n);

#ifdef __cplusplus
}
#endif
