// a '[]const T' is read-only: no writes through it, and no conversion back
// to a writable '[]T'
// expect-error: cannot assign through a read-only slice '[]const int'
// expect-error: argument 1: cannot convert '[]const int' to '[]int'
void fill([]int s) { s[0] = 1; }
int main() {
    int a[2] = {1, 2};
    []const int ro = a[0..2];
    ro[0] = 5;
    fill(ro);
    return 0;
}
