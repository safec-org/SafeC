// bounds.sc — SafeC bounds check insertion
//
// Demonstrates:
//   - Static (compile-time) array bounds checking on constant indices
//   - Runtime bounds check on dynamic indices (emitted in IR as abort() + unreachable)
//   - unsafe{} scope suppresses runtime bounds checks
//
// extern signatures use raw C types (see README §9.1).

extern int printf(char* fmt, ...);

int main() {
    int data[8];
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;
    data[5] = 60;
    data[6] = 70;
    data[7] = 80;

    // Compile-time constant indices — no bounds check needed (statically safe)
    int v0 = data[0];
    int v7 = data[7];
    printf("data[0] = %d\n", v0);
    printf("data[7] = %d\n", v7);

    // Dynamic index: the compiler emits a runtime bounds check
    // that calls abort() if idx >= 8 at runtime.
    int idx = 3;
    int v3 = data[idx];
    printf("data[3] = %d\n", v3);

    return 0;
}

// Uncommenting the following would produce a compile-time error:
//   void bad_access() {
//       int arr[4];
//       int x = arr[4];  // error: array index 4 out of bounds for array of size 4
//   }
