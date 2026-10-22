// Per-mark arena_free_to tracking, and its conservative cases: a mark the
// function didn't take (or can't tell apart) invalidates the whole region.
region R { capacity: 4096 }

int nested_inner_and_outer() {
    &arena<R> int a = new<R> int;
    unsigned long m1 = arena_mark<R>();
    &arena<R> int b = new<R> int;
    unsigned long m2 = arena_mark<R>();
    &arena<R> int c = new<R> int;
    arena_free_to<R>(m2);
    int s = *a + *b;
    s = s + *c;                            // EXPECT-ERROR: use of 'c' (&arena<R> reference) after arena_reset<R>(), arena_destroy<R>(), or arena_free_to<R>() invalidated it
    arena_free_to<R>(m1);
    s = s + *a;
    return s + *b;                         // EXPECT-ERROR: use of 'b' (&arena<R> reference) after arena_reset<R>(), arena_destroy<R>(), or arena_free_to<R>() invalidated it
}

// Rewinding to the outer mark frees what came after the inner one too.
int outer_rewinds_inner() {
    unsigned long m1 = arena_mark<R>();
    unsigned long m2 = arena_mark<R>();
    &arena<R> int c = new<R> int;
    arena_free_to<R>(m1);
    return *c;                             // EXPECT-ERROR: use of 'c' (&arena<R> reference) after arena_reset<R>(), arena_destroy<R>(), or arena_free_to<R>() invalidated it
}

// A mark from the caller: anything in R may be gone.
int unknown_mark(unsigned long m) {
    &arena<R> int a = new<R> int;
    arena_free_to<R>(m);
    return *a;                             // EXPECT-ERROR: use of 'a' (&arena<R> reference) after arena_reset<R>(), arena_destroy<R>(), or arena_free_to<R>() invalidated it
}

// Either mark: whatever followed the earlier one.
int joined_mark(int k) {
    &arena<R> int a = new<R> int;
    unsigned long m1 = arena_mark<R>();
    &arena<R> int b = new<R> int;
    unsigned long m2 = arena_mark<R>();
    unsigned long m = k ? m1 : m2;
    arena_free_to<R>(m);
    return *a + *b;                        // EXPECT-ERROR: use of 'b' (&arena<R> reference) after arena_reset<R>(), arena_destroy<R>(), or arena_free_to<R>() invalidated it
}

// The callee rewinds to the caller's mark.
void rewind(unsigned long m) { arena_free_to<R>(m); }
int callee_rewind() {
    &arena<R> int a = new<R> int;
    unsigned long m = arena_mark<R>();
    &arena<R> int b = new<R> int;
    rewind(m);
    return *a + *b;                        // EXPECT-ERROR: use of 'b' (&arena<R> reference) after arena_reset<R>(), arena_destroy<R>(), or arena_free_to<R>() invalidated it
}

// A callee that resets the region.
void clear_all() { arena_reset<R>(); }
int callee_reset() {
    &arena<R> int a = new<R> int;
    clear_all();
    return *a;                             // EXPECT-ERROR: use of 'a' (&arena<R> reference) after arena_reset<R>(), arena_destroy<R>(), or arena_free_to<R>() invalidated it
}

int main() { return 0; }
