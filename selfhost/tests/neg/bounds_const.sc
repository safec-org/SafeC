// a constant index out of an array's bounds
// expect-error: array index 4 out of bounds for array of size 4
int main() { int a[4]; a[0] = 0; return a[4]; }
