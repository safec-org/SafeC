// a nullable reference must be checked before use
// expect-error: dereference of nullable reference '?&int' requires 'unsafe', or use match/is_null()/.default(value) instead
int get(?&int p) { return *p; }
int main() { int x = 3; return get(&x); }
