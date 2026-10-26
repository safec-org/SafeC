// use of an uninitialized variable
// expect-error: use of possibly uninitialized variable 'x'
int main() { int x; return x + 1; }
