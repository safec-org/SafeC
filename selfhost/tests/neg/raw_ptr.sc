// raw pointer dereference needs unsafe
// expect-error: dereference of raw pointer requires 'unsafe' block
int main() {
    int x = 4;
    int* p = null;
    unsafe { p = (int*)&x; }
    return *p;
}
