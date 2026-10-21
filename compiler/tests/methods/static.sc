// Static methods: 'static T make(...);' in the struct body has no self and
// is called as 'T::make(...)' (or 'ns::T::make(...)'), lowered to the
// plain function T_make. On a generic struct it is a generic function over
// the struct's parameters, inferred from the arguments or the target type.
namespace geo {
struct Pt {
    int x;
    static struct Pt make(int v);
    static struct Pt new();
    int get() const;
};
struct Pt Pt::make(int v) { struct Pt p; p.x = v; return p; }
struct Pt Pt::new() { return Pt::make(7); }   // unqualified inside the namespace
int Pt::get() const { return self.x; }
}
// CHECK-LABEL: define {{.*}}@Pt_make(
// CHECK-NOT: ptr
// CHECK: ret

generic<T>
struct Box {
    T v;
    static struct Box<T> of(T x);
    static struct Box<T> new();
    T get() const;
};
generic<T>
struct Box<T> Box::of(T x) { struct Box<T> b; b.v = x; return b; }
generic<T>
struct Box<T> Box::new() { struct Box<T> b; b.v = (T)3; return b; }
generic<T>
T Box::get() const { return self.v; }

int qualified() {
    struct Pt a = geo::Pt::make(21);
    struct Pt b = geo::Pt::new();
    return a.get() + b.get();
}
// CHECK-LABEL: define {{.*}}@qualified(
// CHECK: ret i32 28

long generic_static() {
    struct Box<int> c = Box::of(5);          // T from the argument
    struct Box<long> d = Box::new();         // T from the target type
    return (long)c.get() * 10L + d.get();
}
// CHECK-LABEL: define {{.*}}@generic_static(
// CHECK: ret i64 53

// Keyword-spelled method names ('join', 'spawn') and inference from the
// return type in 'return Box::new();'.
struct Kw { int x; static struct Kw join(int a); int spawn() const; };
struct Kw Kw::join(int a) { struct Kw k; k.x = a; return k; }
int Kw::spawn() const { return self.x + 1; }
struct Box<int> made() { return Box::new(); }

int keyword_names() {
    struct Kw k = Kw::join(4);
    struct Box<int> b = made();
    return k.spawn() * 10 + b.get();
}
// CHECK-LABEL: define {{.*}}@keyword_names(
// CHECK: ret i32 53

int main() { return 0; }
