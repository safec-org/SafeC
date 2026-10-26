// SAFEC-ARGS: -I @STD@/std -I @STD@ --compat-preprocessor
// Use after drop or move, double drop, overwriting a live value, and moving
// a value out of memory this function only borrows.
extern int printf(const char* fmt, ...);
#include <std/ownership.h>
struct Buf { int n; void drop(); int len() const; };
void Buf::drop() { }
int Buf::len() const { return self.n; }
struct Buf mk(int n) { struct Buf b; b.n = n; return b; }
void take(struct Buf b) { b.drop(); }

int use_after_drop() {
    struct Buf b = mk(1);
    b.drop();
    return b.len();                       // EXPECT-ERROR: use of 'b' after it was dropped
}

void double_drop() {
    struct Buf b = mk(1);
    b.drop();
    b.drop();                             // EXPECT-ERROR: double drop: 'b' was dropped already
}

int use_after_move() {
    struct Buf b = mk(1);
    take(b);
    return b.len();                       // EXPECT-ERROR: use of 'b' after it was moved
}

void maybe_dropped(int c) {
    struct Buf b = mk(1);
    if (c) b.drop();
    b.drop();                             // EXPECT-ERROR: double drop: 'b' may have been dropped already
}

void overwrite() {
    struct Buf b = mk(1);
    b = mk(2);                            // EXPECT-ERROR: assigning to 'b' overwrites a live value of Drop type
    b.drop();
}

// A dropped or moved variable may be assigned again.
void reassign() {
    struct Buf b = mk(1);
    take(b);
    b = mk(2);
    b.drop();
}

struct Holder { struct Buf b; };
struct Buf steal(&Holder h) {
    struct Buf out = h.b;                 // EXPECT-ERROR: cannot move a value of Drop type 'Buf' out of memory this function doesn't own
    return out;
}

void drop_borrowed(&Buf b) {
    b.drop();                             // EXPECT-ERROR: cannot drop a value this function only borrows
}

// A field split off a Drop value must be put back before the value is used
// or dropped as a whole.
struct Two { struct Buf a; int x; void drop(); };
void Two::drop() { self.a.drop(); }
void split_ok() {
    struct Two t; t.a = mk(1); t.x = 0;
    t.a.drop();
    t.a = mk(2);
    t.drop();
}
void split_bad() {
    struct Two t; t.a = mk(1); t.x = 0;
    t.a.drop();
    t.drop();                             // EXPECT-ERROR: 't' is dropped while its field 't.a' is dropped
}

// A drop through a reference into memory someone else owns: the owner
// drops the value again. Swap it out instead.
struct Pair2 { struct Buf a; };
struct List2 { struct Pair2 *items; &Pair2 at(int i); void drop(); };
&Pair2 List2::at(int i) { unsafe { return (&Pair2)&self.items[i]; } }
void List2::drop() { }
void drop_element(&List2 l) {
    l.at(0).a.drop();                     // EXPECT-ERROR: cannot drop a value this function only borrows (it is reached through a reference
}
void drop_field_of_borrowed(&Pair2 p) {
    p.a.drop();                           // EXPECT-ERROR: cannot drop a value this function only borrows (it is reached through a reference
}
void swap_out_ok(&Pair2 p) {
    std::replace(&p.a, mk(9)).drop();
}

int main() { return 0; }
