// Strict aliasing (TBAA): a float store can't change an integer or pointer
// field, so 'b->len' and 'b->data' stay in registers across the loop's
// stores and the loop vectorizes — without 'noalias' on the parameter.
struct Buf { float* data; unsigned long len; };

void scale(&Buf b, float k) {
    unsafe {
        unsigned long i = 0UL;
        while (i < b->len) { b->data[i] = b->data[i] * k; i = i + 1UL; }
    }
}
// CHECK-LABEL: define {{.*}}@scale(
// CHECK: fmul <
// CHECK: store <

// Union members share storage across types: their accesses carry no tag.
union Bits { float f; unsigned int u; };
unsigned int bits_of(float x) {
    union Bits b;
    b.f = x;
    unsafe { return b.u; }
}
// CHECK-LABEL: define {{.*}}@bits_of(
// CHECK-NOT: !tbaa
// CHECK: ret i32

int main() { return 0; }
