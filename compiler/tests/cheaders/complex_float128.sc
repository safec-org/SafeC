// REQUIRES-HOST: Linux
// System headers imported through clang (no <complex.h> / <stdio.h> on the
// include path): declarations using types SafeC has no counterpart for are
// left out rather than emitted as C it can't parse. <complex.h> declares
// '_Complex double' functions; glibc 2.43+ <stdio.h>/<stdlib.h> declare
// 'typedef _Complex __float128 __cfloat128;' and '__float128 _Float128'
// for clang (bits/floatn.h), which broke every <stdio.h> include there.
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: define {{.*}}@main(
// CHECK: call {{.*}}@puts(
int main() {
    unsafe { puts("ok"); }
    return 0;
}
