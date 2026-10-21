// Trait composition: 'trait T : A + B { ... }' requires A's and B's methods
// as well as its own, transitively, and a generic body bounded by T may
// call all of them. Stacked bounds ('M: A + B') keep working alongside.
trait Sized { int size() const; }
trait Named { int name_len() const; }
trait Shape : Sized + Named { int area() const; }
trait Solid : Shape { int volume() const; }     // two levels deep
trait Labeled : Named {}                         // a pure alias of Named

struct Box {
    int w;
    int h;
    int d;
    int size() const;
    int name_len() const;
    int area() const;
    int volume() const;
};
int Box::size() const { return self.w; }
int Box::name_len() const { return 3; }
int Box::area() const { return self.w * self.h; }
int Box::volume() const { return self.w * self.h * self.d; }

generic<T: Solid>
int describe(const &T s) { return s.size() + s.name_len() + s.area() + s.volume(); }

generic<T: Labeled + Sized>
int label(const &T s) { return s.name_len() * s.size(); }

// Every method resolved through the composed traits: 2 + 3 + 6 + 24 + 3*2.
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 41
int main() {
    struct Box b;
    b.w = 2; b.h = 3; b.d = 4;
    return describe(&b) + label(&b);
}
