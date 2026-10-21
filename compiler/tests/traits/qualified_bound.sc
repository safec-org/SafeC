// A trait can be named with its namespace in a bound or supertrait list.
namespace geo {
trait Area { int area() const; }
trait Named : geo::Area { int id() const; }
}
struct Sq { int s; int area() const; int id() const; };
int Sq::area() const { return self.s * self.s; }
int Sq::id() const { return 7; }
generic<T: geo::Named> int both(const &T t) { return t.area() + t.id(); }
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 16
int main() { struct Sq q; q.s = 3; return both(&q); }
