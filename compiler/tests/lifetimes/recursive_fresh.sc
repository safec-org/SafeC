// Fresh allocations a recursive call leaves in its caller's memory keep
// their own allocation sites (merged only across recursion depths): a
// buffer the callee may have grown and a table it rehashed stay distinct
// objects, so rehashing again after the call — freeing the old table —
// leaves the buffer valid. (Reduced from a StrMap insert next to a Vec
// push; see strmap_recursive.sc for the std version.)
void dealloc(&heap int p);
struct Tables { &heap int buf; &heap int table; int n; };
static struct Tables gT;

static void grow(&struct Tables t) { if (t.n > 0) { t.buf = new int; } }
static void rehash(&struct Tables t) {
    &heap int fresh = new int;
    dealloc(t.table);
    t.table = fresh;
}

// Through a parameter.
int ensure(&struct Tables t, int depth) {
    if (depth > 0) { ensure(t, depth - 1); }
    grow(t);
    rehash(t);
    return *t.buf;
}

// Through a global.
static void ggrow() { if (gT.n > 0) { gT.buf = new int; } }
static void grehash() {
    &heap int fresh = new int;
    dealloc(gT.table);
    gT.table = fresh;
}

int gensure(int depth) {
    if (depth > 0) { gensure(depth - 1); }
    ggrow();
    grehash();
    return *gT.buf;
}

// Mutual recursion: the same holds for a call back into the component
// through another function.
int ping(&struct Tables t, int depth);
int pong(&struct Tables t, int depth) { grow(t); return ping(t, depth - 1); }
int ping(&struct Tables t, int depth) {
    if (depth > 0) { pong(t, depth); }
    rehash(t);
    return *t.buf;
}
