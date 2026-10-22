// Function summaries: what a callee frees, returns and leaves in its
// parameters' memory is applied at each call site.
void dealloc(&heap int p);
struct Box { &heap int v; int n; };
struct Node { ?&heap struct Node next; int val; };

&heap int make() { &heap int p = new int; *p = 7; return p; }
void refill(&struct Box b) { dealloc(b.v); b.v = new int; *b.v = 1; }
int peek(&struct Box b) { return *b.v; }

// Two calls, two allocations.
int fresh_results() {
    &heap int a = make();
    &heap int b = make();
    dealloc(a);
    int x = *b;
    dealloc(b);
    return x;
}

// The callee frees the field and stores a new allocation: the field is live.
int replaced_by_callee() {
    struct Box b;
    b.v = new int;
    refill(&b);
    int x = peek(&b);
    dealloc(b.v);
    return x;
}

// A callee that only reads: freeing afterwards is fine.
int read_then_free() {
    struct Box b;
    b.v = new int;
    *b.v = 2;
    int x = peek(&b);
    dealloc(b.v);
    return x;
}

// Recursion: the summary of free_list is a fixpoint.
void free_list(?&heap struct Node n) {
    match (n) {
        case some(node): { free_list(node.next); dealloc((&heap int)node); }
        case null: { }
    }
}

int sum_then_free(?&heap struct Node head) {
    int s = 0;
    match (head) {
        case some(h): { s = h.val; }
        case null: { }
    }
    free_list(head);
    return s;
}

// Generics and methods get summaries too.
generic<T> &heap T pass(&heap T p) { return p; }
void Box::clear() { dealloc(self.v); self.v = new int; }

int generic_and_method() {
    struct Box b;
    b.v = new int;
    &heap int same = pass(b.v);
    *same = 3;
    b.clear();
    int x = *b.v;
    dealloc(b.v);
    return x;
}

// CHECK-LABEL: define {{.*}}@main(
int main() {
    return fresh_results() + replaced_by_callee() + read_then_free() + sum_then_free(null) +
           generic_and_method();
}
