// advanced_features.sc — Demonstrates 7 new compiler features
// Compile: ./build/safec examples/advanced_features.sc --emit-llvm -o /dev/null

extern int printf(const char* fmt, ...);

// ── 1. Enum underlying type ─────────────────────────────────────────────────
enum Color : unsigned char {
    Red,
    Green = 5,
    Blue
}

// ── 2. Newtype — distinct type wrapping ─────────────────────────────────────
newtype Celsius = int;
newtype UserId = long;

// ── 3. Static local variables ───────────────────────────────────────────────
int next_id() {
    static int counter = 100;
    counter = counter + 1;
    return counter;
}

// ── 4. Overflow-aware arithmetic ────────────────────────────────────────────
int test_overflow() {
    int a = 2000000000;
    int b = 2000000000;

    // Wrapping add: wraps on overflow (no UB)
    int wrap_sum = a +| b;
    printf("wrapping add: %d\n", wrap_sum);

    // Saturating add: clamps to INT_MAX on overflow
    int sat_sum = a +% b;
    printf("saturating add: %d\n", sat_sum);

    // Wrapping multiply
    int wrap_mul = a *| 3;
    printf("wrapping mul: %d\n", wrap_mul);

    // Saturating subtract: clamps to INT_MIN on underflow
    int x = -2000000000;
    int sat_sub = x -% b;
    printf("saturating sub: %d\n", sat_sub);

    return 0;
}

// ── 5. goto codegen ─────────────────────────────────────────────────────────
int test_goto() {
    int sum = 0;
    int i = 0;

loop:
    if (i >= 5) goto done;
    sum = sum + i;
    i = i + 1;
    goto loop;

done:
    printf("goto sum = %d\n", sum);  // 0+1+2+3+4 = 10
    return sum;
}

// ── 6. Align annotation ────────────────────────────────────────────────────
int test_align() {
    align(16) int aligned_val = 42;
    printf("aligned value = %d\n", aligned_val);
    return aligned_val;
}

// ── 7. Slices: arr[start..end] ──────────────────────────────────────────────
int test_slices() {
    int data[4];
    data[0] = 100;
    data[1] = 200;
    data[2] = 300;
    data[3] = 400;

    []int s = data[1..3];
    printf("slice: len=%ld, [0]=%d, [1]=%d\n", s.len, s[0], s[1]);
    return 0;
}

int main() {
    printf("=== Enum underlying type ===\n");
    int c = Blue;
    printf("Blue = %d\n", c);  // 6

    printf("\n=== Newtype ===\n");
    Celsius temp = 36;
    UserId uid = 12345LL;
    printf("temp = %d C\n", temp);
    printf("uid = %lld\n", uid);

    printf("\n=== Static locals ===\n");
    printf("id1 = %d\n", next_id());  // 101
    printf("id2 = %d\n", next_id());  // 102
    printf("id3 = %d\n", next_id());  // 103

    printf("\n=== Overflow arithmetic ===\n");
    test_overflow();

    printf("\n=== goto ===\n");
    test_goto();

    printf("\n=== Align ===\n");
    test_align();

    printf("\n=== Slices ===\n");
    test_slices();

    return 0;
}
