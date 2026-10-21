// A raw pointer argument infers T for a 'const &T' parameter, as a
// non-generic 'const &double' parameter accepts the pointer.
generic<T> T first(const &T xs) { return *xs; }
generic<T> T sum2(const &T xs) { unsafe { return xs[0] + xs[1]; } }
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 11
int main() {
    double v[2]; v[0] = 1.5; v[1] = 2.5;
    double r = 0.0;
    unsafe { const double* p = (const double*)&v[0]; r = first(p) + sum2(p); }
    return (int)(r * 2.0);
}
