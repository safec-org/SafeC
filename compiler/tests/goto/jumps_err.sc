// goto/label rules Sema enforces: one definition per label, and no jump
// into the scope of a local past its declaration (it would be read
// uninitialized) -- what rejects a std/async local that lives across an
// await point outside the future's state struct.
extern int printf(const char* fmt, ...);

int dup(int x) {
    if (x == 1) goto A;
A:
    x = x + 1;
A:                              // EXPECT-ERROR: duplicate label 'A'
    return x;
}

int skip_decl(int x) {
    if (x == 1) goto B;         // EXPECT-ERROR: goto 'B' jumps over the declaration of 'y'
    int y = 5;
B:
    return y;
}

int into_block(int x) {
    if (x == 1) goto C;         // EXPECT-ERROR: goto 'C' jumps over the declaration of 'z'
    while (x < 10) {
        int z = x * 2;
C:
        x = x + z;
    }
    return x;
}

int for_var(int x) {
    goto D;                     // EXPECT-ERROR: goto 'D' jumps over the declaration of 'i'
    for (int i = 0; i < 3; i = i + 1) {
D:
        x = x + 1;
    }
    return x;
}

// Allowed: the declaration is in scope at the goto too, or its block has
// ended before the label, or the jump is backward.
int fine(int x) {
    int a = 1;
    if (x == 1) goto E;
    { int b = 2; a = a + b; }
E:
    a = a + 1;
    int c = a;
    if (c < 5) goto E;
    static int calls = 0;
    calls = calls + 1;
    return c;
}

int main() { return dup(0) + skip_decl(0) + into_block(0) + for_var(0) + fine(0); }
