// A declaration in a nested block shadows an outer variable of the same
// name only until that block ends. These fold to constants at -O2; with
// the inner binding leaking out of its block they would give 6 and 22.
int after_block() {
    int r = 0;
    { int r = 5; r = r + 1; }
    r = r + 1;
    return r;
}
// CHECK-LABEL: define {{.*}}@after_block(
// CHECK: ret i32 1

int both() {
    int x = 1;
    int out = 0;
    { int x = 2; out = x; }
    return out * 10 + x;
}
// CHECK-LABEL: define {{.*}}@both(
// CHECK: ret i32 21

int main() { return 0; }
