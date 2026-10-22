// Allocations are tracked through copies (any number of hops, and joins),
// struct fields and array elements: frees reach every alias, and a pointer
// rebound to a fresh allocation is live again.
void dealloc(&heap int p);
struct Box { &heap int v; int n; };
struct Pair { struct Box a; struct Box b; };

// Freed and replaced on one path, untouched on the other.
int replaced_on_one_path(int k) {
    &heap int p = new int;
    *p = 1;
    &heap int r = p;
    if (k) { dealloc(r); r = new int; *r = 2; }
    int x = *r;
    dealloc(r);
    return x;
}

// Two fields of one struct are different allocations.
int sibling_fields() {
    struct Pair pr;
    pr.a.v = new int;
    pr.b.v = new int;
    *pr.a.v = 1;
    *pr.b.v = 2;
    dealloc(pr.a.v);
    int x = *pr.b.v;
    dealloc(pr.b.v);
    return x;
}

// Constant indices are different elements.
int array_elements() {
    &heap int arr[3];
    arr[0] = new int;
    arr[1] = new int;
    *arr[0] = 5;
    dealloc(arr[1]);
    int x = *arr[0];
    dealloc(arr[0]);
    return x;
}

// A struct copied before its field is replaced keeps the old pointer;
// freeing the replacement leaves the copy alone.
int copy_then_replace() {
    struct Box b;
    b.v = new int;
    *b.v = 3;
    struct Box saved = b;
    b.v = new int;
    dealloc(b.v);
    int x = *saved.v;
    dealloc(saved.v);
    return x;
}

// CHECK-LABEL: define {{.*}}@main(
int main() { return replaced_on_one_path(1) + sibling_fields() + array_elements() + copy_then_replace(); }
