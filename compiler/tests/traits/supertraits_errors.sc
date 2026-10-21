// A type that has a trait's own methods but not a supertrait's doesn't
// satisfy it; unknown supertraits and cycles are rejected where the trait
// is declared; primitives don't satisfy user traits.
trait Sized { int size() const; }
trait Shape : Sized { int area() const; }
trait Ghost : Missing {}                      // EXPECT-ERROR: unknown supertrait 'Missing' of trait 'Ghost'
trait Loop1 : Loop2 {}
trait Loop2 : Loop1 {}
// EXPECT-ERROR: is its own supertrait

struct Flat { int a; int area() const; };
int Flat::area() const { return self.a; }

generic<T: Shape>
int area_of(const &T s) { return s.area(); }

generic<T: Sized>
int size_of(T v) { return 0; }

int main() {
    struct Flat f;
    f.a = 1;
    int x = area_of(&f);                      // EXPECT-ERROR: does not satisfy constraint 'Shape'
    int y = size_of(5);                       // EXPECT-ERROR: does not satisfy constraint 'Sized'
    return x + y;
}
