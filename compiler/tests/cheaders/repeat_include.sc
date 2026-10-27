// REQUIRES-HOST: Linux
// A C header imported through clang a second time in one translation unit
// (again directly, or from a header included later) is a no-op. It used
// to be treated as a failed import and fall back to pasting the raw
// /usr/include/math.h, whose glibc internals this parser rejects.
#include <math.h>
#include <math.h>
#include "inc/uses_math.h"
#include <math.h>

// CHECK-LABEL: define {{.*}}@main(
int main() {
    double r = hyp2(3.0, 4.0);
    unsafe { r = r + floor(0.5); }
    return r == 5.0 ? 0 : 1;
}
