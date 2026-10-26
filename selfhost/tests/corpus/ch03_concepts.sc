#include <stdio.h>
int main() {
    int age = 30;
    auto name_len = 5;
    const double pi = 3.14159;
    unsigned long size = 10UL * sizeof(int);
    int x = 42;
    double dx = x;
    long long big = 100LL;
    int small = (int)big;
    printf("%d %d %.5f %lu %f %d\n", age, name_len, pi, size, dx, small);
    unsigned int flags = 0b1100U;
    unsigned int mask = 0b1010U;
    printf("%u %u %u\n", flags & mask, flags | mask, flags ^ mask);
    int a = 7;
    int b = 3;
    printf("%d %d %d %d\n", a + b, a - b, a * b, a / b);
    printf("%d\n", (a > b) && (b > 0));
    int max = 2147483647;
    int wrapped = max +| 1;
    int saturated = max +% 1;
    printf("%d %d\n", wrapped, saturated);
    int i = 0;
    while (i < 3) { printf("while: %d\n", i); i = i + 1; }
    for (int j = 0; j < 3; j = j + 1) { printf("for: %d\n", j); }
    int k = 5;
    if (k > 10) { printf("big\n"); } else if (k > 0) { printf("small positive\n"); } else { printf("non-positive\n"); }
    printf("%d\n", 0777);
    return 0;
}
