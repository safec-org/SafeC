// A static method called through a type parameter ('E::load') resolves to
// the concrete type's static method in each instantiation; 'if const'
// selects code by a property of the type argument.
struct F32 { float v; static float load(F32 x); static struct F32 store(float f); };
float F32::load(F32 x) { return x.v; }
struct F32 F32::store(float f) { struct F32 r; r.v = f; return r; }
struct Q8 { int bits; static float load(Q8 x); static struct Q8 store(float f); };
float Q8::load(Q8 x) { return (float)x.bits / (float)8.0; }
struct Q8 Q8::store(float f) { struct Q8 r; r.bits = (int)(f * (float)8.0); return r; }

generic<E>
int sum2(E a, E b) {
    int bias = 0;
    if const (sizeof(E) == 4UL) { bias = 100; }
    return (int)(E::load(a) + E::load(b)) + bias;
}

// F32: 1.5 + 2.5 + 100 = 104; Q8: 1 + 2 + 100 (also 4 bytes) = 103
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 207
int main() {
    return sum2(F32::store(1.5f), F32::store(2.5f)) + sum2(Q8::store(1.0f), Q8::store(2.0f));
}
