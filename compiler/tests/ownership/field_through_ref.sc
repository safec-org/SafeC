// A field of a struct reached through a reference-returning call is a
// place in the referent: a method on it gets that address, not a copy's.
struct S2 { int n; void bump(); };
void S2::bump() { self.n = self.n + 1; }
struct W2 { struct S2 s; };
&W2 pass_w(&W2 w);   // defined elsewhere: the call stays at -O2

// CHECK-LABEL: define {{.*}}@bump_through(
// CHECK: call {{.*}}@pass_w(
// CHECK-NOT: alloca
// CHECK: ret
void bump_through(&W2 w) { pass_w(w).s.bump(); }

int main() { return 0; }
