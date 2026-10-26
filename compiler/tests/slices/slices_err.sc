// Elements of a '[]const T' can't be written, and a read-only slice
// doesn't convert back to a writable one.
void fill([]int s) { s[0] = 1; }
int main() {
    int a[2] = {1, 2};
    []const int ro = a[0..2];
    ro[0] = 5;          // EXPECT-ERROR: cannot assign through a read-only slice '[]const int'
    fill(ro);           // EXPECT-ERROR: argument 1: cannot convert '[]const int' to '[]int'
    return 0;
}

void slice_to_pointer_cast([]const unsigned char s) {
    unsafe {
        const unsigned char* p = (const unsigned char*)s; // EXPECT-ERROR: cannot cast slice
    }
}
