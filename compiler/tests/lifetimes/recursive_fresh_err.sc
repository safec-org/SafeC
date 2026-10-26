// Keeping a recursive call's fresh allocations apart by site still keeps
// real aliases: a pointer copied from one field to another (in this call
// or in the recursive one) is the same object, and freeing it through one
// field dangles the other.
void dealloc(&heap int p);
struct Tables { &heap int buf; &heap int table; int n; };
static struct Tables gT;

static void rehash(&struct Tables t) {
    &heap int fresh = new int;
    dealloc(t.table);
    t.table = fresh;
}

int shared(&struct Tables t, int depth) {
    if (depth > 0) { shared(t, depth - 1); }
    else { rehash(t); t.buf = t.table; }
    rehash(t);
    return *t.buf;                        // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

// The alias exists only in what the recursive call left behind.
int via_recursion(&struct Tables t, int depth) {
    if (depth == 0) { rehash(t); t.buf = t.table; return 0; }
    via_recursion(t, depth - 1);
    rehash(t);
    return *t.buf;                        // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

static void grehash() {
    &heap int fresh = new int;
    dealloc(gT.table);
    gT.table = fresh;
}

int gshared(int depth) {
    if (depth == 0) { grehash(); gT.buf = gT.table; return 0; }
    gshared(depth - 1);
    grehash();
    return *gT.buf;                       // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}
