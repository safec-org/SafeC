#pragma once
// Included by repeat_include.sc after it has imported <math.h> itself.
#include <math.h>
static inline double hyp2(double a, double b) { unsafe { return sqrt(a * a + b * b); } }
