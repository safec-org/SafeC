// slices.sc — Slice expressions, member access, and bounds-checked subscript
// Compile: ./build/safec examples/slices.sc --emit-llvm -o /dev/null

extern int printf(const char* fmt, ...);

int main() {
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;

    // ── Slice from array: arr[1..4] → {ptr, len=3}
    []int s = arr[1..4];
    printf("slice len = %ld\n", s.len);     // 3
    printf("s[0] = %d\n", s[0]);            // 20
    printf("s[1] = %d\n", s[1]);            // 30
    printf("s[2] = %d\n", s[2]);            // 40

    // ── Full array slice: arr[0..5]
    []int full = arr[0..5];
    printf("full len = %ld\n", full.len);   // 5

    // ── Tail slice: arr[2..5] → from index 2 to end
    []int tail = arr[2..5];
    printf("tail len = %ld\n", tail.len);   // 3
    printf("tail[0] = %d\n", tail[0]);      // 30

    // ── Re-slicing a slice
    []int sub = s[0..2];
    printf("sub len = %ld\n", sub.len);     // 2
    printf("sub[0] = %d\n", sub[0]);        // 20
    printf("sub[1] = %d\n", sub[1]);        // 30

    return 0;
}
