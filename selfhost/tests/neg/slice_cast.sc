// a slice is a (pointer, length) pair: casting it to a pointer is an error
// expect-error: cannot cast slice
int first([]const int s) {
    unsafe {
        const int* p = (const int*)s;
        return *p;
    }
}
int main() {
    int a[2] = {1, 2};
    return first(a[..]);
}
