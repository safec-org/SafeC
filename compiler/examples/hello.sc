// hello.sc — minimal SafeC program
// Demonstrates: C-ABI compatibility, const functions, basic types

#include <stdio.h>

const int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int add(int a, int b) {
    return a + b;
}

int main() {
    int x = add(3, 4);
    int f = factorial(5);
    printf("3 + 4 = %d\n", x);
    printf("5! = %d\n", f);
    return 0;
}
