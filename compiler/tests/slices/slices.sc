// Slice types in every declaration position: a function returning '[]T',
// an optional slice '?[]T', a read-only '[]const T' that '[]T' converts to,
// and a range-checked re-slice that traps when 'end' passes the length or
// 'start' passes 'end'.
struct Buf { []int data; };

[]int middle([]int s) { return s[1..3]; }
?[]int maybe(int k, []int s) { if (k) return s; return null; }
int sum([]const int s) {
    int t = 0;
    for (unsigned long i = 0UL; i < s.len; i = i + 1UL) t = t + s[i];
    return t;
}
// CHECK: define {{.*}}@middle(
// CHECK: slice.trap
// CHECK: call void @abort

int main() {
    int a[4] = {1, 2, 3, 4};
    struct Buf b;
    b.data = a[0..4];
    []const int ro = middle(b.data);
    return sum(ro) + sum(b.data) - 15;
}

// Member access, indexing and re-slicing on a slice-valued call.
[]int whole([]int s) { return s; }
int rvalues() {
    int a[3] = {7, 8, 9};
    return (int)whole(a[0..3]).len + whole(a[0..3])[2] + whole(a[0..3])[1..3][0];
}
