// C's default argument promotions for the '...' tail of a variadic call:
// a bool (i1) zero-extends to int, char/short sign- or zero-extend by
// their type, float widens to double. A raw i1 left the upper bits of the
// int register undefined: printf("%d", !x) printed 254.
extern int printf(const char* fmt, ...);

int show(int x, char c, unsigned char u, short s, float f) {
    unsafe { return printf("%d %d %d %d %d %f\n", !x, x == 1, c, u, s, f); }
}

// CHECK-LABEL: define {{.*}}@show(
// CHECK: zext i1 {{.*}} to i32
// CHECK: zext i1 {{.*}} to i32
// CHECK: sext i8 {{.*}} to i32
// CHECK: zext i8 {{.*}} to i32
// CHECK: sext i16 {{.*}} to i32
// CHECK: fpext float {{.*}} to double
// CHECK: call {{.*}}@printf(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, double
int main() {
    return show(1, (char)-3, (unsigned char)250, (short)-7, 1.5f) > 0 ? 0 : 1;
}
