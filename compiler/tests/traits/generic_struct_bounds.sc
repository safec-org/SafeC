// A generic struct's parameters take trait bounds, stacked ('T: A + B')
// and composed through supertraits, and the struct's methods may call
// every bound method on its fields.
trait Sized { int size() const; }
trait Named { int name_len() const; }
trait Shape : Sized + Named { int area() const; }

struct Box { int w; int size() const; int name_len() const; int area() const; };
int Box::size() const { return self.w; }
int Box::name_len() const { return 3; }
int Box::area() const { return self.w * self.w; }

generic<T: Shape + Sized>
struct Holder {
    T item;
    int k;
    int total() const;
};
generic<T: Shape + Sized>
int Holder::total() const { return self.item.area() + self.item.size() + self.item.name_len() + self.k; }

// 4*4 + 4 + 3 + 1
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 24
int main() {
    struct Holder<struct Box> h;
    h.item.w = 4;
    h.k = 1;
    return h.total();
}
