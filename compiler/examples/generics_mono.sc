// generics_mono.sc — SafeC: generic function monomorphization
//
// Demonstrates:
//   - Generic functions using 'generic<T>' syntax
//   - Monomorphization: distinct IR functions emitted for each concrete type
//   - Multiple instantiations: int and double versions
//   - Template functions are omitted from the IR; only concrete instances appear
//
// After compilation you should see distinct LLVM functions:
//   __safec_min_int,    __safec_min_double
//   __safec_max_int,    __safec_max_double
//   __safec_myabs_int
//
// extern signatures use raw C types (see README §9.1).

extern int printf(char* fmt, ...);

// ── Generic min ───────────────────────────────────────────────────────────────
// T is inferred from argument types at each call site.
generic<T> T min(T a, T b) {
    if (a < b) return a;
    return b;
}

// ── Generic max ───────────────────────────────────────────────────────────────
generic<T> T max(T a, T b) {
    if (a > b) return a;
    return b;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    // int instantiations  → __safec_min_int, __safec_max_int
    int ia = 3;
    int ib = 7;
    int imin = min(ia, ib);
    int imax = max(ia, ib);
    printf("min(3, 7)     = %d\n", imin);
    printf("max(3, 7)     = %d\n", imax);

    // double instantiations → __safec_min_double, __safec_max_double
    double da = 1.5;
    double db = 2.5;
    double dmin = min(da, db);
    double dmax = max(da, db);
    printf("min(1.5, 2.5) = %f\n", dmin);
    printf("max(1.5, 2.5) = %f\n", dmax);

    return 0;
}
