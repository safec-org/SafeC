#include <stdio.h>
int inc(int x) { return x + 1; }
int dbl(int x) { return x * 2; }
struct Op { fn int(int) f; const char* name; };
int fold(fn int(int, int) f, []int xs, int init) {
    int acc = init;
    for (unsigned long i = 0UL; i < xs.len; i = i + 1UL) { acc = f(acc, xs[i]); }
    return acc;
}
int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }
int fib(int n) { if (n < 2) { return n; } return fib(n - 1) + fib(n - 2); }
int main() {
    struct Op ops[2];
    ops[0].f = inc; ops[0].name = "inc";
    ops[1].f = dbl; ops[1].name = "dbl";
    for (int i = 0; i < 2; i = i + 1) { printf("%s(10) = %d\n", ops[i].name, ops[i].f(10)); }
    int xs[4] = {1, 2, 3, 4};
    printf("%d %d\n", fold(add, xs[0..4], 0), fold(mul, xs[0..4], 1));
    printf("%d\n", fib(20));
    return 0;
}
