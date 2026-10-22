// 'const &R T' is a reference to const T, the same type as '&R const T', in
// every declaration position: a local, a static local and a file-scope
// function's return type each used to take the 'const' as the variable's
// (or function's) own and leave a mutable reference behind, so the writes
// below compiled, and two such borrows of one variable were rejected as two
// mutable borrows.
const &int first(&int a) { return a; }
int main() {
    int x = 1;
    const &stack int b = &x;
    const &stack int c = &x;      // a second shared borrow is fine
    *b = 2;                       // EXPECT-ERROR: left side of assignment must be lvalue
    static const ?&int s;
    *first(&x) = 3;               // EXPECT-ERROR: left side of assignment must be lvalue
    b = &x;                       // rebinding the reference itself stays allowed
    return x + *c;
}
