// Function references print as the 'fn R(Args)' / '?fn R(Args)' that spell
// them (not '&static const R(Args)').
int sq(int x) { return x * x; }
int main() {
    fn int(int) f = sq;
    ?fn int(int, ...) g = null;
    int y = f;                     // EXPECT-ERROR: cannot convert 'fn int(int)' to 'int'
    int z = g;                     // EXPECT-ERROR: cannot convert '?fn int(int, ...)' to 'int'
    return y + z;
}
