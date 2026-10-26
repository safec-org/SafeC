#include <stdio.h>
#include <math.h>
int main() {
    double d = 1.0 / 3.0;
    printf("%f %.2f %.10f %e %g\n", d, d, d, d, d);
    printf("%g %g %g %g\n", 100000.0, 1000000.0, 0.0001, 0.00001);
    printf("%.3e %E %G\n", 123456.789, 0.000123, 1e-10);
    printf("%f %f %.0f %.0f %.0f\n", 0.5, 2.5, 0.5, 1.5, 2.5);
    printf("%.1f %.1f %.1f\n", 0.05, 0.15, 0.25);
    float f = 3.14159f;
    printf("%f %.3f\n", f, f * 2.0f);
    printf("%10.3f|%-10.2f|%+.1f\n", 3.14159, 2.5, 1.0);
    printf("%f %f\n", sqrt(2.0), pow(2.0, 10.0));
    printf("%.6f %.6f %.6f\n", sin(1.0), cos(1.0), tan(0.5));
    printf("%.6f %.6f %.6f\n", exp(1.0), log(10.0), atan2(1.0, 1.0));
    printf("%.4f %.4f %.4f\n", floor(-2.5), ceil(-2.5), fabs(-3.25));
    double big = 1e300;
    printf("%e %g\n", big, big * 10.0);
    printf("%f\n", 123456789.123456789);
    int i = (int)7.9;
    long l = (long)-7.9;
    printf("%d %ld\n", i, l);
    double c = 25.0;
    double fh = c * 9.0 / 5.0 + 32.0;
    printf("%.1f C = %.1f F\n", c, fh);
    printf("%.15g\n", 0.1 + 0.2);
    printf("%g %g\n", 1.5e-7, 123.0);
    return 0;
}
