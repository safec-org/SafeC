// SAFEC-RUN
// A method called through a member chain on a call that returns a
// reference ('l.at(i).vals.push(x)', 'a.b().c.d()') gets the referenced
// object as self, not a temporary copy of it — the mutation sticks. So do
// assignments through such a chain.
struct Buf {
    int data[8];
    unsigned long len;
    void push(int x);
    unsigned long count() const;
};
void Buf::push(int x) {
    if (self.len < 8UL) { self.data[self.len] = x; self.len = self.len + 1UL; }
}
unsigned long Buf::count() const { return self.len; }

struct Inner { struct Buf d; };
struct Item {
    int id;
    struct Buf vals;
    struct Inner in;
    &struct Inner inner();
};
&struct Inner Item::inner() { return &self.in; }

struct List {
    struct Item items[4];
    &struct Item at(unsigned long i);
    const &struct Item get(unsigned long i) const;
};
&struct Item List::at(unsigned long i) { return &self.items[i]; }
const &struct Item List::get(unsigned long i) const { return &self.items[i]; }

// Pushed into a copy, the stores were dead and -O2 dropped them.
// CHECK-LABEL: define {{.*}}@push_field(
// CHECK: store i32 %x, ptr
// CHECK: ret void
void push_field(&struct List l, int x) { l.at(1UL).vals.push(x); }

// CHECK-LABEL: define {{.*}}@push_nested(
// CHECK: store i32 %x, ptr
// CHECK: %add = add i32 %x, 1
// CHECK: store i32 %add, ptr
// CHECK: ret void
void push_nested(&struct List l, int x) {
    l.at(2UL).in.d.push(x);          // call, field, field, method
    l.at(2UL).inner().d.push(x + 1); // call, call, field, method
}

// A const method through a const-reference chain reads the element in
// place too.
unsigned long count_const(const &struct List l) { return l.get(1UL).vals.count(); }

int main() {
    struct List l;
    for (unsigned long i = 0UL; i < 4UL; i = i + 1UL) {
        l.items[i].id = 0;
        l.items[i].vals.len = 0UL;
        l.items[i].in.d.len = 0UL;
    }
    push_field(&l, 7);
    push_field(&l, 8);
    if (l.items[1].vals.len != 2UL || l.items[1].vals.data[1] != 8) return 1;
    if (count_const(&l) != 2UL) return 2;

    push_nested(&l, 20);
    if (l.items[2].in.d.len != 2UL) return 3;
    if (l.items[2].in.d.data[0] != 20 || l.items[2].in.d.data[1] != 21) return 4;

    l.at(3UL).id = 5;                // assignment through the reference
    l.at(3UL).in.d.len = 1UL;
    l.at(3UL).inner().d.data[0] = 9;
    if (l.items[3].id != 5 || l.items[3].in.d.len != 1UL || l.items[3].in.d.data[0] != 9) return 5;
    return 0;
}
