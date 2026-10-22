// arena_free_to<R>(m) invalidates exactly the references allocated after
// mark m: each mark is its own record, so nested marks are precise. The old
// one-depth-per-region tracking rejected the reads marked "was rejected".
region R { capacity: 4096 }

int nested_inner() {
    &arena<R> int a = new<R> int;
    *a = 1;
    unsigned long m1 = arena_mark<R>();
    &arena<R> int b = new<R> int;
    *b = 2;
    unsigned long m2 = arena_mark<R>();
    &arena<R> int c = new<R> int;
    *c = 3;
    arena_free_to<R>(m2);                  // frees c only
    int s = *a + *b;                       // was rejected (b)
    arena_free_to<R>(m1);                  // frees b
    return s + *a;
}

// A scratch mark per iteration leaves the outer allocation alone.
int scratch_loop(int n) {
    &arena<R> int keep = new<R> int;
    *keep = 1;
    int s = 0;
    for (int i = 0; i < n; i = i + 1) {
        unsigned long m = arena_mark<R>();
        &arena<R> int t = new<R> int;
        *t = i;
        s = s + *t + *keep;                // was rejected (keep)
        arena_free_to<R>(m);
    }
    return s + *keep;                      // was rejected
}

// A mark handed to a callee: the callee's free_to is resolved against the
// caller's own mark.
void rewind(unsigned long m) { arena_free_to<R>(m); }

int callee_rewind() {
    &arena<R> int a = new<R> int;
    *a = 4;
    unsigned long m = arena_mark<R>();
    &arena<R> int t = new<R> int;
    *t = 5;
    rewind(m);
    return *a;
}

// CHECK-LABEL: define {{.*}}@main(
int main() { return nested_inner() + scratch_loop(3) + callee_rewind(); }
