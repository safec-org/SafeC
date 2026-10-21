// An unknown type in a struct-body method declaration is an error (it used
// to pass silently, or reach LLVM verification with a definition).
struct S { int x; void f(Bogus c); };   // EXPECT-ERROR: unknown type 'Bogus' in the declaration of 'S::f'
int main() { return 0; }
