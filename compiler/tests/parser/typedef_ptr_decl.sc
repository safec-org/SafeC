// A statement starting '<typedef-name> **...* <name>' declares a
// multi-level pointer local, like 'T *x' and 'struct T **x' already did
// (it used to parse as the expression 'T * *x' and fail with "undeclared
// identifier"). The same tokens followed by anything but a declarator
// follower ('=', ';', ',', '[') stay an expression, and nested prefix
// operators ('**pp', '-*p', '!!x') parse in expressions.
// EXEC

struct Foo { int v; };
typedef struct Foo Foo_t;

int read2(Foo_t** pp) {
    Foo_t** x = pp;
    Foo_t** z;
    Foo_t** arr[2];
    unsafe {
        Foo_t*** y = (Foo_t***)&x;
        z = *y;
        arr[0] = z;
        arr[1] = pp;
        return (**arr[1]).v + (*arr[0])->v;
    }
}
// CHECK-LABEL: define {{.*}}@read2(
// CHECK: load ptr, ptr %pp
// CHECK: load i32
// CHECK: shl i32
// CHECK: ret i32

int read3(Foo_t*** ppp) {
    Foo_t*** w = ppp;
    unsafe { return (***w).v; }
}
// CHECK-LABEL: define {{.*}}@read3(
// CHECK: load ptr, ptr %ppp
// CHECK: load ptr
// CHECK: load i32
// CHECK: ret i32

// 'a * **pp + c;' / 'a * **pp' with a variable 'a' is multiplication.
int mul(int a, int c, int** pp) {
    unsafe {
        int r = a * **pp;
        a * **pp + c;
        r = r + a * **pp;
        r = r + - -c - -*pp[0];
        if (!!(a > 0)) r = r + 1;
        return r;
    }
}
// CHECK-LABEL: define {{.*}}@mul(
// CHECK: mul i32
// CHECK: ret i32

int main() {
    Foo_t f;
    f.v = 21;
    Foo_t* p = null;
    int n = 3;
    int* np = null;
    unsafe {
        p = (Foo_t*)&f;
        np = (int*)&n;
        if (read2(&p) != 42) return 1;
        Foo_t** pp = (Foo_t**)&p;
        if (read3(&pp) != 21) return 2;
        // 2*3 + 2*3 + 5 + 1 + 3 = 21
        if (mul(2, 5, &np) != 21) return 3;
    }
    return 0;
}
