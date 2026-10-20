// A mutable reference parameter is not 'noalias': here 'a' points at 'g',
// which f also writes directly, so 'return g' must reload it after
// '*a = 1' instead of returning the constant 5.
int g = 0;
int f(&int a) {
    g = 5;
    *a = 1;
    return g;
}
int main() { return f(&g); }
// CHECK-LABEL: define {{.*}}@f(
// CHECK-NOT: noalias
// CHECK: store i32 1
// CHECK: load i32, ptr @g
