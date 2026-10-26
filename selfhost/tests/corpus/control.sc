#include <stdio.h>
int main() {
    int attempts = 0;
    do { attempts = attempts + 1; } while (attempts < 3);
    printf("%d\n", attempts);
    outer: for (int i = 0; i < 10; i = i + 1) {
        for (int j = 0; j < 10; j = j + 1) {
            if (i + j > 15) { break outer; }
            if (j % 2 == 0) { continue; }
            if (i == 8) { printf("%d %d\n", i, j); }
        }
    }
    for (int n = 0; n < 6; n = n + 1) {
        switch (n) {
        case 1:
        case 2:
            printf("low %d\n", n);
            break;
        case 3:
            printf("three\n");
        case 4:
            printf("three or four\n");
            break;
        default:
            printf("other %d\n", n);
        }
    }
    int k = 0;
    while (1) {
        k = k + 1;
        if (k == 5) { break; }
        match (k) {
            case 2: continue;
            default: printf("k=%d\n", k);
        }
    }
    int x = 5;
    int y = x > 3 ? 10 : 20;
    printf("%d\n", y);
    const int PLATFORM = 1;
    if const (PLATFORM == 1) { printf("platform one\n"); } else { printf("other platform\n"); }
    static_assert(sizeof(int) == 4, "int is 32-bit");
    int i = 0;
    int total = 0;
    for (i = 0; i < 100; i++) { total += i; }
    printf("%d %d\n", i, total);
    return 0;
}
