#include <stdio.h>
generic<T: Numeric>
T clamp(T val, T lo, T hi) {
    if (val < lo) { return lo; }
    if (val > hi) { return hi; }
    return val;
}
generic<T>
struct Container {
    T value;
    int count;
    T get() const;
    void set(T new_value);
};
generic<T>
T Container::get() const { return self.value; }
generic<T>
void Container::set(T new_value) { self.value = new_value; self.count = self.count + 1; }
generic<T>
struct Pair { T first; T second; };
generic<T>
T sum_pair(struct Pair<T> p) { return p.first + p.second; }
trait Drawable { void draw() const; }
struct Circle { double radius; void draw() const; };
void Circle::draw() const { printf("circle r=%.1f\n", self.radius); }
generic<T: Drawable> void render(T shape) { shape.draw(); }
generic<T>
struct Box {
    T v;
    static struct Box<T> of(T x);
};
generic<T>
struct Box<T> Box::of(T x) { struct Box<T> b; b.v = x; return b; }
generic<T...>
unsigned long count_args(T... args) { return sizeof...(T); }
int main() {
    printf("%d %f\n", clamp(15, 0, 10), clamp(-2.5, -1.0, 1.0));
    struct Container<int> c;
    c.count = 0;
    c.set(42);
    c.set(43);
    printf("%d %d\n", c.get(), c.count);
    struct Container<double> d;
    d.set(1.5);
    printf("%.2f\n", d.get());
    struct Pair<long> p;
    p.first = 40L; p.second = 2L;
    printf("%ld\n", sum_pair(p));
    struct Circle circle = {2.0};
    render(circle);
    struct Box<int> b = Box::of(5);
    printf("%d\n", b.v);
    return 0;
}
