// A struct parameter over 16 bytes (not an all-float HFA) is passed as a
// pointer to a caller-made copy, so objects built at different optimization
// levels agree on it. As a raw LLVM aggregate, the -O0 and -O2 instruction
// selectors laid out '{ [8 x i32], [64 x i8], i64, i32 }' differently: an
// -O2 library called from an -O0 program read garbage (hmac_sha256 in a
// release std computed the wrong MAC).
struct Mixed { int a[8]; unsigned char b[64]; long c; int d; };

long mixed_sum(struct Mixed m, long k) {
    long s = m.c + (long)m.d;
    int i = 0;
    while (i < 8) { s = s + (long)m.a[i]; i = i + 1; }
    return s * k;
}

// The callee works on the caller's copy: changing it leaves the caller's
// value alone.
long bump(struct Mixed m) { m.c = m.c + 100L; return m.c; }

// The rewritten functions follow main in the module.
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 0
// CHECK-LABEL: define {{.*}}@mixed_sum(ptr
int main() {
    struct Mixed m;
    int i = 0;
    while (i < 8) { m.a[i] = i; i = i + 1; }
    m.c = 1L; m.d = 2;
    long r = bump(m);
    if (r != 101L || m.c != 1L) return 1;
    if (mixed_sum(m, 2L) != 62L) return 2;
    return 0;
}
