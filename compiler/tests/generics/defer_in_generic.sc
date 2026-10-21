// 'defer' inside a generic function body survives monomorphization and
// behaves as in a non-generic function. It used to be cloned as a 'break':
// an error outside a loop, a silent early exit inside one.
generic<T> int put(&T slot, T v) {
    int k = 0;
    defer *slot = v;
    k = 1;
    return k;
}
generic<T> T loop_defer(T limit) {
    T i = (T)0;
    while (i < limit) {
        defer i = i;          // must not end the loop early
        i = i + (T)1;
    }
    return i;
}
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 113
int main() {
    int s = 0;
    int r = put(&s, 3);                       // r = 1, then s = 3
    return r * 100 + s * 3 + loop_defer(4);   // 100 + 9 + 4
}
