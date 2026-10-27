// '#line N' sets the line of the next source line for __LINE__ (it was
// ignored), and '#line N "file"' the file too, for __FILE__ and the
// locations of what follows.
// EXEC
int a = __LINE__;
#line 100
int b = __LINE__;
#line 200 "renamed.sc"
int c = __LINE__;
int d =
    __LINE__;

// CHECK: @a = {{.*}}i32 5
// CHECK: @b = {{.*}}i32 100
// CHECK: @c = {{.*}}i32 200
// CHECK: @d = {{.*}}i32 202
int main() {
    const char *f = __FILE__;
    unsafe { if (f[0] != 'r' || f[7] != '.' || f[10] != 0) return 1; }
    return a == 5 && b == 100 && c == 200 && d == 202 ? 0 : 2;
}
