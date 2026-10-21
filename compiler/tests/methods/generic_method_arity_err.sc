// The struct body and the definition must agree on a method's own type
// parameters.
struct Q {
    int x;
    generic<T> static struct Q new(&T c);
    static struct Q make();
};
struct Q Q::new(&int c) { struct Q q; q.x = *c; return q; }   // EXPECT-ERROR: 'Q::new' is declared with 1 type parameter(s) in the struct body but defined with 0
generic<T> struct Q Q::make() { struct Q q; q.x = 1; return q; }   // EXPECT-ERROR: 'Q::make' is declared with 0 type parameter(s) in the struct body but defined with 1
int main() { return 0; }
