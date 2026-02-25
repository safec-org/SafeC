// new_features.sc — demo for SafeC new language features
// Tests: defer, match (C-style), packed struct, labeled break/continue,
//        must_use keyword, fn function pointer type
#include <stdio.h>
#include <stdlib.h>

// ── must_use keyword ──────────────────────────────────────────────────────────

must_use int important_calc(int x) {
    return x * x + 1;
}

// ── packed struct ─────────────────────────────────────────────────────────────

packed struct PackedHeader {
    int  magic;
    int  version;
    int  size;
};

// ── defer ─────────────────────────────────────────────────────────────────────

void demo_defer() {
    printf("-- defer demo --\n");
    defer printf("  defer 1 (last)\n");
    defer printf("  defer 2 (middle)\n");
    defer printf("  defer 3 (first)\n");
    printf("  body executes first\n");
    // defers fire in LIFO order on return
}

// ── match statement (C-style) ─────────────────────────────────────────────────

void demo_match(int x) {
    printf("-- match: x = %d -> ", x);
    match (x) {
        case 0:          printf("zero\n");
        case 1, 2, 3:    printf("small (1-3)\n");
        case 4..10:      printf("medium (4-10)\n");
        default:         printf("other\n");
    }
}

// ── fn function pointer type ──────────────────────────────────────────────────

// Function that squares an integer
int square(int x)   { return x * x; }
int cube(int x)     { return x * x * x; }
int identity(int x) { return x; }

void demo_fn_ptr() {
    printf("-- fn function pointer --\n");

    // Declare a function pointer using 'fn' keyword
    fn int(int) op;

    op = square;
    printf("  square(5) = %d\n", op(5));

    op = cube;
    printf("  cube(3)   = %d\n", op(3));

    // fn pointer as a local variable reassigned
    fn int(int) transform = identity;
    printf("  identity(7) = %d\n", transform(7));
}

// ── labeled break/continue ────────────────────────────────────────────────────

void demo_labeled_break() {
    printf("-- labeled break --\n");
    int found_i = -1;
    int found_j = -1;

    outer: for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (i * j == 6) {
                found_i = i;
                found_j = j;
                break outer;
            }
        }
    }
    printf("  first i*j==6: i=%d j=%d\n", found_i, found_j);
}

void demo_labeled_continue() {
    printf("-- labeled continue (skip diagonal) --\n");
    outer: for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) continue outer;   // skip when on diagonal
            printf("  (%d,%d)", i, j);
        }
    }
    printf("\n");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    demo_defer();
    printf("\n");

    demo_match(0);
    demo_match(2);
    demo_match(7);
    demo_match(99);
    printf("\n");

    demo_fn_ptr();
    printf("\n");

    demo_labeled_break();
    demo_labeled_continue();
    printf("\n");

    // packed struct
    PackedHeader hdr;
    hdr.magic   = 0x53414645;
    hdr.version = 1;
    hdr.size    = 256;
    printf("-- packed struct --\n");
    printf("  magic=0x%X version=%d size=%d\n", hdr.magic, hdr.version, hdr.size);
    printf("\n");

    // must_use — use the return value (no warning)
    int val = important_calc(5);
    printf("-- must_use keyword --\n");
    printf("  important_calc(5) = %d\n", val);

    return 0;
}
