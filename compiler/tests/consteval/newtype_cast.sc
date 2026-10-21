// 'if const' folds a cast to a newtype like a cast to its base type, so a
// generic body can tell a float sample type from a fixed-point newtype.
newtype Q = int;
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 11
int main() {
    int a = 0; int b = 0;
    if const ((int)0.5 == 0) { a = 1; }
    if const ((Q)0.5 == (Q)0) { b = 1; }
    return a * 10 + b;
}
