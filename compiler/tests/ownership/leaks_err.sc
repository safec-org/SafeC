// A value of a Drop type must be dropped or moved on every path before its
// scope ends: at the block's end, at break/continue/goto out of it, and at
// every return (after the defers ran).
extern int printf(const char* fmt, ...);
struct Buf { int n; void drop(); int len() const; };
void Buf::drop() { printf("drop %d\n", self.n); }
int Buf::len() const { return self.n; }
struct Buf mk(int n) { struct Buf b; b.n = n; return b; }

void never() {
    struct Buf b = mk(1);
}                                         // EXPECT-ERROR: value of Drop type 'Buf' leaks at the end of its scope: 'b' is never dropped

void some_paths(int c) {
    struct Buf b = mk(1);
    if (c) b.drop();
}                                         // EXPECT-ERROR: 'b' is dropped or moved on some paths but not on others

int early(int c) {
    struct Buf b = mk(1);
    if (c) return 0;                      // EXPECT-ERROR: leaks when the function returns: 'b' is never dropped
    b.drop();
    return 1;
}

void on_break() {
    for (int i = 0; i < 3; i = i + 1) {
        struct Buf b = mk(i);
        if (i == 1) break;                // EXPECT-ERROR: leaks when 'break' leaves its scope
        b.drop();
    }
}

void on_continue() {
    for (int i = 0; i < 3; i = i + 1) {
        struct Buf b = mk(i);
        if (i == 1) continue;             // EXPECT-ERROR: leaks when 'continue' leaves its scope
        b.drop();
    }
}

void on_goto(int c) {
    {
        struct Buf b = mk(1);
        if (c) goto out;                  // EXPECT-ERROR: leaks when 'goto' leaves its scope
        b.drop();
    }
out:
    return;
}

void temporary() {
    printf("%d\n", mk(5).len());          // EXPECT-ERROR: this temporary of Drop type 'Buf' is never dropped
}

void defer_in_loop() {
    for (int i = 0; i < 3; i = i + 1) {
        struct Buf b = mk(i);             // EXPECT-ERROR: from the previous iteration is still live here
        defer b.drop();
    }
}

// A by-value parameter is owned by the callee.
void param(struct Buf b) {
    printf("%d\n", b.len());
}                                         // EXPECT-ERROR: leaks when the function returns: 'b' is never dropped

// An aggregate holding a Drop value owns it: each field is dropped (or the
// whole thing with drop(x)).
struct Pair { struct Buf a; struct Buf b; };
void pair_half() {
    struct Pair p;
    p.a = mk(1);
    p.b = mk(2);
    p.a.drop();
}                                         // EXPECT-ERROR: 'p.b' (a field of Drop type in 'p') leaks at the end of its scope

int main() { return 0; }
