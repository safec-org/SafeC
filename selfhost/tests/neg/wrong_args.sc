// wrong number of arguments
// expect-error: wrong number of arguments: expected 2, got 1
int add(int a, int b) { return a + b; }
int main() { return add(1); }
