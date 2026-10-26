#include <stdio.h>
int add(int a, int b) { return a + b; }
pure double square(double x) { return x * x; }
must_use int compute(int x) { return x * x + 1; }
int apply(fn int(int, int) op, int a, int b) { return op(a, b); }
generic<T>
T my_max(T a, T b) {
    if (a > b) { return a; }
    return b;
}
const int factorial(int n) {
    if (n <= 1) { return 1; }
    return n * factorial(n - 1);
}
consteval int table_size() { return 256; }
int lookup[table_size()];
const int val = factorial(5);
int main() {
    fn int(int, int) opRef = add;
    int result = opRef(3, 4);
    int r2 = compute(5);
    printf("%d %d %d %f\n", result, apply(add, 10, 20), r2, square(1.5));
    printf("%d %f\n", my_max(3, 7), my_max(1.5, 2.7));
    int x = 6;
    printf("%d %d\n", val, factorial(x));
    return 0;
}
