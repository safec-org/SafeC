// A method can be declared outside its struct's body (as another header
// adding to the type does) and is then called like any other method.
struct Counter { int n; int get() const; };
int Counter::get() const { return self.n; }

// "another header"
int Counter::twice() const;
void Counter::bump(int k);

int Counter::twice() const { return self.n * 2; }
void Counter::bump(int k) { self.n = self.n + k; }

// 3 + 2*3
// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 9
int main() {
    struct Counter c;
    c.n = 1;
    c.bump(2);
    return c.get() + c.twice();
}
