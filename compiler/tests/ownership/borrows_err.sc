// A reference a method returns into its owned receiver can't be used after
// the receiver is dropped or moved.
extern void dealloc_int(&heap int p);
struct Box1 { int v; void drop(); &int get(); };
void Box1::drop() { }
&int Box1::get() { return &self.v; }

int dangling_alias() {
    struct Box1 b; b.v = 7;
    &int r = b.get();
    b.drop();
    return *r;                            // EXPECT-ERROR: use of 'b.v' after 'b' was dropped
}

// The same through memory the value owns (a heap buffer): the reference
// borrows from the owner.
struct Cell1 { &heap int data; void drop(); &int get(); };
void Cell1::drop() { }
&int Cell1::get() { return self.data; }
struct Cell1 cell_new() { struct Cell1 c; c.data = new int; return c; }

int dangling_owned() {
    struct Cell1 c = cell_new();
    &int r = c.get();
    c.drop();
    return *r;                            // EXPECT-ERROR: use of a reference borrowed from 'c' after 'c' was dropped
}

int fine() {
    struct Cell1 c = cell_new();
    &int r = c.get();
    int x = *r;
    c.drop();
    return x;
}

// A container over raw memory: the summary can't see where at() points,
// so a plain '&T' result of a method with no other reference parameter
// borrows from the receiver (lifetime elision).
struct Raw1 { int *buf; void drop(); &int at(int i); };
void Raw1::drop() { }
&int Raw1::at(int i) { unsafe { return (&int)&self.buf[i]; } }
int dangling_raw(struct Raw1 r) {
    &int p = r.at(0);
    r.drop();
    return *p;                            // EXPECT-ERROR: use of a reference borrowed from 'r' after 'r' was dropped
}

int main() { return fine(); }
