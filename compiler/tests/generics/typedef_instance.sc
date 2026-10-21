// A typedef of a generic struct's instance: static methods called through
// it instantiate the template's static method with the instance's type
// arguments; a generic method declared outside the struct body (as another
// header would) reaches instances created before the declaration; a method
// declared through the typedef belongs to that instance only.
struct A { int v; };
struct B { long w; };
generic<T>
struct Box {
    T item;
    int tag;
    int get() const;
    static struct Box<T> make(int tag);
};
typedef struct Box<struct A> BoxA;   // an instance exists before 'twice' is declared
typedef struct Box<struct B> BoxB;

generic<T> int Box::twice() const;   // declared outside the body
generic<T> int Box::get() const { return self.tag + (int)sizeof(T); }
generic<T> int Box::twice() const { return self.get() * 2; }
generic<T> struct Box<T> Box::make(int tag) { struct Box<T> b; b.tag = tag; return b; }
// a method of one instance only, declared through its typedef
int BoxA::only_a() const;
int BoxA::only_a() const { return self.tag * 100; }

// Through a type parameter bound to an instance: T::make.
trait Getter { int get() const; }
generic<T: Getter> int remake(const &T sample) { T t = T::make(7); return t.get() + sample.get(); }

// a: get = 1 + 4 = 5, twice = 10, only_a = 100;  b: get = 2 + 8 = 10, twice = 20;
// remake(&b) = (7 + 8) + 10 = 25
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 170
int main() {
    BoxA a = BoxA::make(1);
    BoxB b = BoxB::make(2);
    return a.get() + a.twice() + a.only_a() + b.get() + b.twice() + remake(&b);
}
