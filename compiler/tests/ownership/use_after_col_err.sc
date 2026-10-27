// SAFEC-ARGS: -I %root/std -I %root --compat-preprocessor
// A use after drop/move through a method call is reported at the receiver
// variable, not at the call's '('.
struct Buf { int n; void drop(); int len() const; };
void Buf::drop() { }
int Buf::len() const { return self.n; }
struct Buf mk(int n) { struct Buf b; b.n = n; return b; }
void take(struct Buf b) { b.drop(); }

int use_after_drop() {
    struct Buf b = mk(1);
    b.drop();
    return b.len();
}
// EXPECT-ERROR-IN: use_after_col_err.sc:13:12: use of 'b' after it was dropped

int use_after_move() {
    struct Buf buf = mk(1);
    take(buf);
    return 1 + buf.len();
}
// EXPECT-ERROR-IN: use_after_col_err.sc:20:16: use of 'buf' after it was moved
