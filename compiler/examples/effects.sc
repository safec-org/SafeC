// effects.sc — pure function / static effect system demo
// Compile: ./build/safec examples/effects.sc --emit-llvm -o /dev/null

// Pure functions: the compiler enforces no side effects.
// LLVM marks them as readonly + nounwind, enabling aggressive optimization.

pure int add(int a, int b) {
    return a + b;
}

pure int multiply(int a, int b) {
    return a * b;
}

// Pure functions can call other pure functions
pure int dot_product(int x1, int y1, int x2, int y2) {
    return add(multiply(x1, x2), multiply(y1, y2));
}

// Pure function with more complex logic
pure int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0;
    int b = 1;
    int i = 2;
    while (i <= n) {
        int tmp = b;
        b = a + b;
        a = tmp;
        i = i + 1;
    }
    return b;
}

pure int abs_val(int x) {
    if (x < 0) return 0 - x;
    return x;
}

pure int max(int a, int b) {
    if (a > b) return a;
    return b;
}

pure int min(int a, int b) {
    if (a < b) return a;
    return b;
}

pure int clamp(int val, int lo, int hi) {
    return min(max(val, lo), hi);
}

int main() {
    int result = dot_product(3, 4, 5, 6);
    // result = 3*5 + 4*6 = 15 + 24 = 39

    int fib10 = fibonacci(10);
    // fib10 = 55

    int clamped = clamp(150, 0, 100);
    // clamped = 100

    return 0;
}
