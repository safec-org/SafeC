// implicit narrowing conversion
// expect-error: type mismatch in variable declaration: cannot convert 'long' to 'int'
int main() {
    long big = 5000000000L;
    int x = big;
    return x;
}
