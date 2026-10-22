// A floating literal with an 'f'/'F' suffix is a float, as in C — so it
// combines with float operands without a cast, and the arithmetic stays in
// single precision. Unsuffixed literals stay double. A float literal still
// initializes a double (exact widening) and an unsuffixed one a float.
float scale(float x) { return x * 255.0f + 0.5f; }
int positive(float x) { return x > 0.0f ? 1 : 0; }
double widen() { double d = 0.25f; return d * 2.0; }
float narrow() { float f = 0.75; return f - 0.25F; }

// CHECK-LABEL: define {{.*}}@scale(
// CHECK: fmul float {{.*}}2.550000e+02
// CHECK: fadd float {{.*}}5.000000e-01
// CHECK-LABEL: define {{.*}}@positive(
// CHECK: fcmp ogt float
// CHECK-LABEL: define {{.*}}@main(
int main() {
    if (scale(1.0f) != 255.5f) return 1;
    if (positive(-1.0f) != 0) return 2;
    if (widen() != 0.5) return 3;
    if (narrow() != 0.5f) return 4;
    return 0;
}
