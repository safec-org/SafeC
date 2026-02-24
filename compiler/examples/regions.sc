// regions.sc — SafeC region-based memory safety
//
// Demonstrates:
//   - Region-qualified references
//   - Nullable references
//   - Struct + field access via safe reference
//   - Region escape detection (the commented-out bad example)
//   - Static references to string literals

extern int printf(&static char fmt, ...);

// User-defined arena region
region AudioPool {
    capacity: 4096
}

// ── Value types (structs) ──────────────────────────────────────────────────────
struct Point {
    int x;
    int y;
}

struct Node {
    int value;
    ?&stack Node next;   // nullable self-referential stack ref
}

// ── Functions taking region-qualified references ──────────────────────────────

// Takes a non-null stack reference — caller guarantees liveness
int sumPoint(&stack Point p) {
    return p.x + p.y;
}

// Mutable heap reference — allows modification
void scalePoint(&heap Point p, int factor) {
    p.x = p.x * factor;
    p.y = p.y * factor;
}

// Immutable static reference — pointer to read-only data
void printMsg(&static char msg) {
    printf("%s\n", msg);
}

// Nullable reference usage — must check before deref
int nodeValue(?&stack Node n) {
    if (n == null) return -1;
    // After null check: safe to use (flow-sensitive)
    // In a full impl, the compiler tracks the checked branch
    return 0;
}

// ── Stack reference: legal local use ──────────────────────────────────────────
void stackRefDemo() {
    Point p;
    p.x = 10;
    p.y = 20;
    &stack Point ref = &p;      // non-null stack reference
    int sum = sumPoint(ref);
    printf("Point sum: %d\n", sum);
}

// ── Compile-time assertions ───────────────────────────────────────────────────
// static_assert(sizeof(Point) == 8, "Point must be 8 bytes");

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    stackRefDemo();
    printMsg("SafeC region demo");
    return 0;
}
