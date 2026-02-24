// generics.sc — SafeC: tagged unions, consteval, compile-time design
//
// NOTE: Generic function monomorphization is a Phase 2 feature.
// This example demonstrates:
//   - Tagged union (Result style) implemented as a struct
//   - const functions evaluated at compile time when possible
//   - static_assert for layout validation
//   - Error handling without exceptions

extern int printf(&static char fmt, ...);

// ── Compile-time functions ────────────────────────────────────────────────────
// 'const' fn: can run at compile time when called in a const context
const int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

const int power(int base, int exp) {
    if (exp == 0) return 1;
    return base * power(base, exp - 1);
}

// ── Tagged result struct ──────────────────────────────────────────────────────
// SafeC error handling pattern: explicit result type, no exceptions
struct ResultInt {
    bool is_ok;
    int  value;
    int  err_code;
}

ResultInt make_ok(int v) {
    ResultInt r;
    r.is_ok    = true;
    r.value    = v;
    r.err_code = 0;
    return r;
}

ResultInt make_err(int code) {
    ResultInt r;
    r.is_ok    = false;
    r.value    = 0;
    r.err_code = code;
    return r;
}

ResultInt safe_div(int a, int b) {
    if (b == 0) return make_err(-1);
    return make_ok(a / b);
}

ResultInt safe_sqrt_int(int x) {
    if (x < 0) return make_err(-2);
    // Newton's method (integer)
    if (x == 0) return make_ok(0);
    int r = x;
    int r1 = (r + x / r) / 2;
    while (r1 < r) {
        r = r1;
        r1 = (r + x / r) / 2;
    }
    return make_ok(r);
}

// ── static_assert: layout validation ─────────────────────────────────────────
// static_assert(sizeof(ResultInt) == 12, "ResultInt must be 12 bytes");

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    // Compile-time computable values
    int f10 = fib(10);
    int p2_8 = power(2, 8);
    printf("fib(10)   = %d\n", f10);
    printf("2^8       = %d\n", p2_8);

    // Safe error handling — no exceptions, no hidden control flow
    ResultInt r1 = safe_div(100, 7);
    ResultInt r2 = safe_div(100, 0);
    ResultInt r3 = safe_sqrt_int(144);
    ResultInt r4 = safe_sqrt_int(-5);

    if (r1.is_ok) printf("100 / 7   = %d\n", r1.value);
    if (!r2.is_ok) printf("100 / 0   = error(%d)\n", r2.err_code);
    if (r3.is_ok) printf("sqrt(144) = %d\n", r3.value);
    if (!r4.is_ok) printf("sqrt(-5)  = error(%d)\n", r4.err_code);

    return 0;
}
