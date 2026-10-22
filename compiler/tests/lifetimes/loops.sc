// Loops are iterated to a fixpoint, so a free or reset only stales what it
// can actually reach on a later iteration or after the loop. Every function
// here used to be rejected by the syntactic loop-body pre-scan, which treated
// a reset/free anywhere in the body as happening before every statement.
region R { capacity: 4096 }
region S { capacity: 4096 }
void dealloc(&heap int p);

// The reset happens only on the iteration that leaves the loop.
int reset_on_exit(int n) {
    &arena<R> int p = new<R> int;
    *p = 1;
    int s = 0;
    while (1) {
        s = s + *p;
        if (s > n) { arena_reset<R>(); break; }
    }
    return s;
}

// Resetting an unrelated region doesn't touch R.
int unrelated_region(int n) {
    &arena<R> int p = new<R> int;
    *p = 0;
    int i = 0;
    while (i < n) {
        &arena<S> int q = new<S> int;
        *q = i;
        arena_reset<S>();
        i = i + 1;
    }
    return *p;
}

// A free followed by leaving the loop.
int free_then_break(int n) {
    &heap int p = new int;
    *p = n;
    int i = 0;
    while (i < n) {
        if (*p == i) { dealloc(p); break; }
        i = i + 1;
    }
    return i;
}

// Rebound before every use: the previous iteration's free can't reach it.
int fresh_each_iteration(int n) {
    int s = 0;
    for (int i = 0; i < n; i = i + 1) {
        &heap int p = new int;
        *p = i;
        s = s + *p;
        dealloc(p);
    }
    return s;
}

// The same buffer swap as a nested loop: buffers from earlier outer
// iterations are kept apart from the ones the inner loop frees.
int swap_buffers(int rounds, int n) {
    &heap int cur = new int;
    *cur = 0;
    int r = 0;
    while (r < rounds) {
        &heap int next = new int;
        *next = 0;
        int j = 0;
        while (j < n) {
            &heap int grown = new int;
            *grown = *next + *cur;
            dealloc(next);
            next = grown;
            j = j + 1;
        }
        dealloc(cur);
        cur = next;
        r = r + 1;
    }
    int out = *cur;
    dealloc(cur);
    return out;
}

// '&&' only frees on the path where the whole condition is true.
int short_circuit(int k) {
    &heap int p = new int;
    *p = 3;
    if (k && (dealloc(p), 1)) { return 0; }
    int v = *p;
    dealloc(p);
    return v;
}

// CHECK-LABEL: define {{.*}}@main(
int main() {
    return reset_on_exit(3) + unrelated_region(2) + free_then_break(2) +
           fresh_each_iteration(3) + swap_buffers(2, 2) + short_circuit(0);
}
