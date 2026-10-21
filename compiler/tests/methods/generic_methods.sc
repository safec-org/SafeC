// A method can have type parameters of its own ('generic<T>' before it in
// the struct body and on its definition), static or not; each call
// instantiates it like a generic function, inferring T from the
// arguments (self is not an argument) or the target type.
struct Pt { int x; int y; };
struct Node {
    void* ctx;
    int   tag;
    generic<T> static struct Node new(&T ctx, int tag);
    generic<T> int on(fn int(&T, int) cb, &T data);
    generic<T> T pick(T a, T b) const;
};
generic<T>
struct Node Node::new(&T ctx, int tag) { struct Node n; unsafe { n.ctx = (void*)ctx; } n.tag = tag; return n; }
generic<T>
int Node::on(fn int(&T, int) cb, &T data) { return cb(data, self.tag); }
generic<T>
T Node::pick(T a, T b) const { return self.tag > 0 ? a : b; }

static int add_x(&Pt p, int t) { return p.x + t; }
static int mul_n(&int p, int t) { return *p * t; }

int generic_methods() {
    struct Pt p; p.x = 40; p.y = 0;
    int k = 3;
    struct Node a = Node::new(&p, 2);   // T = Pt
    struct Node b = Node::new(&k, 5);   // T = int
    long l = b.pick(7L, 9L);            // T = long
    return a.on(add_x, &p) * 100 + b.on(mul_n, &k) + (int)l;
}
// CHECK-LABEL: define {{.*}}@generic_methods(
// CHECK: ret i32 4222

int main() { return 0; }
