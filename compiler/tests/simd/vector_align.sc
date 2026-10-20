// A vec<T,N> loaded or stored through a pointer into an element array is
// only element-aligned, so the access must not claim the vector's own ABI
// alignment (32 for vec<float,8>): an aligned vmovaps on such an address
// faults. Locals the compiler lays out keep the full alignment.
typedef vec<float, 8> f8;

void axpy8(&float y, const &float x, f8 a, unsigned long n) {
    unsigned long i = 0UL;
    unsafe {
        while (i + 8UL <= n) {
            *(f8*)&y[i] = *(f8*)&y[i] + a * *(const f8*)&x[i];
            i = i + 8UL;
        }
    }
}
// CHECK-LABEL: define {{.*}}@axpy8(
// CHECK: load <8 x float>, ptr {{.*}}, align 4
// CHECK-NOT: align 32
// CHECK: store <8 x float> {{.*}}, align 4

f8 twice(f8 v) {
    f8 t = v + v;
    return t;
}

// A local *array* of scalars is element-aligned too: &tab[t] at a variable
// index is only 4-aligned, though the array itself is a local.
f8 load_local(const &float x, unsigned long t) {
    float tab[64];
    unsigned long i = 0UL;
    while (i < 64UL) { unsafe { tab[i] = x[i]; } i = i + 1UL; }
    f8 v;
    unsafe { v = *(const f8*)&tab[t]; }
    return v + v;
}
// CHECK-LABEL: define {{.*}}@load_local(
// CHECK: load <8 x float>, ptr {{.*}}, align 4
