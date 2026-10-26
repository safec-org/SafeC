// a generic call's type argument must satisfy the bound
// expect-error: type 'A' does not satisfy constraint 'Show' for generic parameter 'T' (no method 'show')
trait Show { void show() const; }
struct A { int v; };
generic<T: Show> void display(T x) { x.show(); }
int main() { struct A a; a.v = 1; display(a); return 0; }
