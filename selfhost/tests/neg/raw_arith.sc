// raw pointer arithmetic needs unsafe
// expect-error: raw pointer arithmetic requires 'unsafe' block
int main() {
    int a[4];
    a[0] = 1;
    int* p = null;
    unsafe { p = (int*)&a[0]; }
    int* q = p + 1;
    return 0;
}
