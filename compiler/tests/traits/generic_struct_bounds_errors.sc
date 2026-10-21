// A generic struct instantiated with a type that misses a supertrait's
// method is rejected where it is used.
trait Sized { int size() const; }
trait Shape : Sized { int area() const; }

struct Flat { int a; int area() const; };
int Flat::area() const { return self.a; }

generic<T: Shape>
struct Holder { T item; };

int main() {
    struct Holder<struct Flat> h;   // EXPECT-ERROR: type 'Flat' does not satisfy constraint 'Shape' for generic parameter 'T' of struct 'Holder'
    h.item.a = 1;
    return 0;
}
