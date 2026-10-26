// A method call (or assignment) on a field of a struct returned by
// reference ('h.get().c.bump()') must act on the referent, the same as
// through a bound reference ('&Outer r = h.get(); r.c.bump();'). The field
// used to be treated as a non-lvalue because the call itself isn't one, so
// the method got 'self' pointing at a temporary copy and the update was lost.
struct Counter {
    int n;
    void bump();
};
void Counter::bump() { self.n = self.n + 1; }

struct Outer { struct Counter c; };

struct Holder {
    struct Outer o;
    &Outer get();
};
&Outer Holder::get() { return &self.o; }

int chained() {
    struct Holder h;
    h.o.c.n = 0;
    h.get().c.bump();
    h.get().c.bump();
    return h.get().c.n;
}
// CHECK-LABEL: define {{.*}}@chained(
// CHECK: ret i32 2

int chained_assign() {
    struct Holder h;
    h.o.c.n = 0;
    h.get().c.n = 40;
    h.get().c.bump();
    return h.o.c.n;
}
// CHECK-LABEL: define {{.*}}@chained_assign(
// CHECK: ret i32 41

