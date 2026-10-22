// Frees inside a callee (directly, through a field, transitively, through
// recursion, in a generic instance or a method) reach the caller's aliases,
// and so does a callee returning its parameter.
void dealloc(&heap int p);
struct Box { &heap int v; int n; };
struct Node { ?&heap struct Node next; int val; };

void release(&heap int p) { dealloc(p); }
&heap int same(&heap int p) { return p; }
void release_box(&struct Box b) { dealloc(b.v); }
void maybe_release(&heap int p, int k) { if (k) release(p); }
int peek(&struct Box b) { return *b.v; }

int callee_frees_param() {
    &heap int p = new int;
    release(p);
    return *p;                            // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}

int callee_returns_param() {
    &heap int p = new int;
    &heap int q = same(p);
    dealloc(q);
    return *p;                            // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}

int callee_frees_field() {
    struct Box b;
    b.v = new int;
    release_box(&b);
    return *b.v;                          // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

int transitive(int k) {
    &heap int p = new int;
    maybe_release(p, k);
    return *p;                            // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}

// The callee reads the freed field / frees it again.
int callee_reads_freed() {
    struct Box b;
    b.v = new int;
    dealloc(b.v);
    return peek(&b);                      // EXPECT-ERROR: call to 'peek' reads a &heap reference after std::dealloc() freed it
}

int callee_double_free() {
    struct Box b;
    b.v = new int;
    dealloc(b.v);
    release_box(&b);                      // EXPECT-ERROR: call to 'release_box' frees memory after std::dealloc() freed it
    return 0;
}

// Recursion.
void free_list(?&heap struct Node n) {
    match (n) {
        case some(node): { free_list(node.next); dealloc((&heap int)node); }
        case null: { }
    }
}

int use_after_recursive_free(&heap struct Node head) {
    free_list(head);
    return head.val;                      // EXPECT-ERROR: use of 'head' (&heap reference) after std::dealloc() freed it
}

void pong(&heap int p, int n);
void ping(&heap int p, int n) { if (n == 0) { dealloc(p); return; } pong(p, n - 1); }
void pong(&heap int p, int n) { ping(p, n); }

int mutual_recursion() {
    &heap int p = new int;
    ping(p, 3);
    return *p;                            // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}

// Generic instance and method.
generic<T> void drop_it(&heap T p) { dealloc((&heap int)p); }
void Box::clear() { dealloc(self.v); }

int generic_free() {
    &heap int p = new int;
    drop_it(p);
    return *p;                            // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}

int method_free() {
    struct Box b;
    b.v = new int;
    b.clear();
    return *b.v;                          // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

// A fresh allocation the callee hands back holds the caller's pointer.
&heap struct Box wrap(&heap int p) { &heap struct Box b = new struct Box; b.v = p; return b; }

int fresh_result_holds_alias() {
    &heap int p = new int;
    &heap struct Box b = wrap(p);
    dealloc(p);
    return *b.v;                          // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

int main() { return 0; }
