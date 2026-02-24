// consteval_demo.sc — SafeC compile-time evaluation demonstration
//
// Shows (CONST_EVAL.md):
//   - const functions: may execute at compile time or runtime
//   - consteval functions: must execute at compile time only
//   - const global variables: evaluated at compile time
//   - static_assert: compile-time assertions
//   - if const (...): compile-time branch selection (CONST_EVAL.md §14)

#include <stdio.h>

// ── Compile-time constants ────────────────────────────────────────────────────
// These are evaluated by ConstEvalEngine before codegen.
const int CACHE_LINE    = 64;
const int PAGE_SIZE     = 4096;
const int PAGES_PER_MB  = (1024 * 1024) / PAGE_SIZE;

// ── consteval functions ───────────────────────────────────────────────────────
// Must be called only in compile-time contexts.

consteval int pow2(int x) {
    return 1 << x;
}

consteval int kilobytes(int n) {
    return n * 1024;
}

consteval int align_up(int val, int align) {
    return (val + align - 1) & (~(align - 1));
}

// ── const functions ───────────────────────────────────────────────────────────
// May execute at compile time (when called in const context) or at runtime.

const int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

const int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

const int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

const int sum_range(int lo, int hi) {
    int s = 0;
    int i = lo;
    while (i <= hi) {
        s = s + i;
        i = i + 1;
    }
    return s;
}

// ── Compile-time constants derived from const functions ───────────────────────
const int FACT_10   = factorial(10);   // evaluated at compile time
const int FIB_15    = fib(15);         // evaluated at compile time
const int GCD_DEMO  = gcd(48, 36);     // evaluated at compile time
const int SUM_1_100 = sum_range(1, 100); // = 5050, compile time

// ── consteval-derived constants ───────────────────────────────────────────────
const int STACK_SIZE   = pow2(12);       // 4096
const int HEAP_BUF     = kilobytes(64);  // 65536
const int ALIGNED_SIZE = align_up(1000, CACHE_LINE);  // 1024

// ── static_assert ─────────────────────────────────────────────────────────────
// Checked at compile time by ConstEvalEngine.
static_assert(FACT_10 == 3628800, "10! must be 3628800");
static_assert(FIB_15  == 610,     "fib(15) must be 610");
static_assert(GCD_DEMO == 12,     "gcd(48,36) must be 12");
static_assert(SUM_1_100 == 5050,  "sum(1..100) must be 5050");
static_assert(STACK_SIZE == 4096, "2^12 must be 4096");
static_assert(PAGES_PER_MB == 256, "1MB / 4096 must be 256");

// ── if const — compile-time branch selection ──────────────────────────────────
// The branch is selected at compile time; the other branch is dead code.

const int WORD_SIZE = 8;  // 64-bit system

int describeWordSize() {
    if const (WORD_SIZE == 8) {
        printf("64-bit architecture\n");
        return 64;
    } else {
        printf("32-bit architecture\n");
        return 32;
    }
}

// ── Runtime functions (cannot call consteval) ─────────────────────────────────
// These call const functions in a runtime context → runs at runtime normally.

int main() {
    printf("=== SafeC Compile-Time Evaluation Demo ===\n\n");

    // Compile-time constants printed at runtime
    printf("Compile-time constants:\n");
    printf("  10!             = %d\n", FACT_10);
    printf("  fib(15)         = %d\n", FIB_15);
    printf("  gcd(48, 36)     = %d\n", GCD_DEMO);
    printf("  sum(1..100)     = %d\n", SUM_1_100);
    printf("  pow2(12)        = %d\n", STACK_SIZE);
    printf("  kilobytes(64)   = %d\n", HEAP_BUF);
    printf("  align_up(1000, 64) = %d\n", ALIGNED_SIZE);
    printf("  pages_per_MB    = %d\n", PAGES_PER_MB);
    printf("  cache_line      = %d\n", CACHE_LINE);
    printf("\n");

    // if const: compile-time branch
    printf("Architecture: ");
    int bits = describeWordSize();
    printf("  -> %d-bit confirmed\n", bits);
    printf("\n");

    // const functions called at runtime also work normally
    printf("Runtime const function calls:\n");
    printf("  factorial(7)  = %d\n", factorial(7));
    printf("  fib(10)       = %d\n", fib(10));
    printf("  gcd(100, 75)  = %d\n", gcd(100, 75));
    printf("\n");

    printf("All static_asserts passed at compile time.\n");
    return 0;
}
