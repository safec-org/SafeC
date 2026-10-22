// A free or reset anywhere in a loop body (any expression nesting) reaches
// the statements before it on the next iteration, and the code after the
// loop through whichever exits it can take.
region R { capacity: 4096 }
void dealloc(&heap int p);
int pick(int x);

// The free is buried in a subscript index inside a match expression;
// the old syntactic pre-scan never looked there.
int nested_expression(int n) {
    &heap int p = new int;
    int vals[2] = {1, 2};
    int s = 0;
    int i = 0;
    while (i < n) {
        s = s + *p;                       // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
        s = s + match (pick(i)) { case 0: vals[(dealloc(p), 0)], default: 0 };   // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
        i = i + 1;
    }
    return s;
}

// A reset at the end of the body stales the reference at the top.
int reset_end_of_body(int n) {
    &arena<R> int p = new<R> int;
    int s = 0;
    for (int i = 0; i < n; i = i + 1) {
        s = s + *p;                       // EXPECT-ERROR: use of 'p' (&arena<R> reference) after arena_reset<R>()
        arena_reset<R>();
    }
    return s;
}

// The reset on the exit path stales the use after the loop only.
int reset_on_exit(int n) {
    &arena<R> int p = new<R> int;
    int s = 0;
    while (1) {
        s = s + *p;
        if (s > n) { arena_reset<R>(); break; }
    }
    return s + *p;                        // EXPECT-ERROR: use of 'p' (&arena<R> reference) after arena_reset<R>()
}

// A free in the loop's increment expression.
int free_in_increment(int n) {
    &heap int p = new int;
    int s = 0;
    for (int i = 0; i < n; i = i + pick((dealloc(p), 1))) {   // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
        s = s + *p;                       // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
    }
    return s;
}

// A backward goto is a loop too.
int goto_loop(int n) {
    &heap int p = new int;
    int i = 0;
again:
    i = i + *p;                           // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
    if (i < n) { dealloc(p); goto again; }   // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
    return i;
}

// Freed on one switch arm, which falls through into the next.
int switch_fallthrough(int k) {
    &heap int p = new int;
    switch (k) {
    case 1: dealloc(p);
    case 2: return *p;                    // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
    default: break;
    }
    return *p;
}

int main() { return 0; }
