// Assignment through a call returning a reference follows the reference's
// mutability: a '&T' result's fields are assignable, a 'const &T' one's
// are not.
struct P { int x; };
&struct P id(&struct P r) { return r; }
const &struct P cid(const &struct P r) { return r; }

void ok(&struct P r) { id(r).x = 6; }
void bad(&struct P r) { cid(r).x = 6; }  // EXPECT-ERROR: left side of assignment must be lvalue

// The storage behind a returned reference outlives this function, so a
// '&stack' reference stored into it escapes.
struct H { &stack int p; };
&struct H hid(&struct H r) { return r; }
void escape(&struct H h) {
    int x = 1;
    hid(h).p = &x;                         // EXPECT-ERROR: cannot assign '&stack int' to variable in outer scope: reference would escape
}
