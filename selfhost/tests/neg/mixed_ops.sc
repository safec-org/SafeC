// binary operators need operands of the same type
// expect-error: implicit type conversion not allowed in SafeC: 'int' and 'long' differ
int main() {
    int a = 1;
    long b = 2L;
    long c = a + b;
    return (int)c;
}
