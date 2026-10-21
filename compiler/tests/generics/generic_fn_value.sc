// A generic function used as a value takes its type arguments from the
// function type it meets: a variable, an assignment, an argument (also of
// another generic function) and a return value.
struct A { int x; };
struct B { long y; };
generic<E> struct Node { E val; fn void(&Node<E>) back; int tag; };
generic<E> void mark_(&Node<E> n) { n.tag = (int)sizeof(E) * 10 + 1; }
generic<E> void other_(&Node<E> n) { n.tag = (int)sizeof(E) * 10 + 2; }
generic<E> void link_(&Node<E> n, fn void(&Node<E>) f) { n.back = f; }
generic<E> int run_(&Node<E> n) { n.back(n); return n.tag; }
generic<E> fn void(&Node<E>) pick_(&Node<E> n, int k) {
    if (k == 1) return other_;
    return mark_;
}

// 41 + 81 + 42 + 82 = 246
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 246
int main() {
    struct Node<struct A> a;
    struct Node<struct B> b;
    fn void(&Node<struct A>) f = mark_;
    a.back = f;
    b.back = mark_;
    int r1 = run_(&a);
    int r2 = run_(&b);
    link_(&a, other_);
    int r3 = run_(&a);
    b.back = pick_(&b, 1);
    int r4 = run_(&b);
    return r1 + r2 + r3 + r4;
}
